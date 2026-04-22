import glob
import importlib.util
import json
import os
import shutil
import time
from subprocess import DEVNULL, call

from rich.console import Console
from torch.utils.cpp_extension import _get_build_directory, load

# DDP 支持：检测当前进程的本地秩
def get_local_rank():
    """获取当前进程的 LOCAL_RANK，如果未设置则返回 0"""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_distributed():
    """检查是否在分布式训练中"""
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ

PATH = os.path.dirname(os.path.abspath(__file__))
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.7;8.9"


def cuda_toolkit_available():
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def cuda_toolkit_version():
    cuda_home = os.path.join(os.path.dirname(shutil.which("nvcc")), "..")
    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the cuda version.")
    return cuda_version


def load_module_from_path(module_name: str, so_path: str):
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


name = "gsplat_cuda"
build_dir = _get_build_directory(name, verbose=False)
extra_include_paths = [os.path.join(PATH, "csrc/third_party/glm")]
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]

_C = None
sources = list(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + list(
    glob.glob(os.path.join(PATH, "csrc/*.cpp"))
)

so_path = os.path.join(build_dir, "gsplat_cuda.so")
lib_path = os.path.join(build_dir, "gsplat_cuda.lib")

def _load_extension_ddp_safe():
    """
    DDP 安全的扩展加载函数：
    - rank 0 负责编译/加载
    - 其他 rank 等待并加载预编译的 .so 文件
    """
    global _C
    local_rank = get_local_rank()
    is_ddp = is_distributed()
    
    try:
        # 1) 优先尝试正式安装的扩展
        from gsplat import csrc as _C_installed
        _C = _C_installed
        console = Console()
        console.print("[green]gsplat: Using officially installed extension.[/green]")
        return
    except ImportError:
        pass
    
    # 2) 如果是分布式训练，只让 rank 0 进行编译/加载
    if is_ddp and local_rank != 0:
        # 其他 rank 等待 rank 0 完成编译，然后从缓存加载
        console = Console()
        console.print(
            f"[cyan]gsplat (rank {local_rank}): Waiting for rank 0 to build extension...[/cyan]"
        )
        
        # 超时等待机制：最多等待 10 分钟
        max_wait = 600  # 秒
        wait_interval = 1  # 秒
        elapsed = 0
        
        while elapsed < max_wait:
            if os.path.exists(so_path):
                console.print(
                    f"[cyan]gsplat (rank {local_rank}): Extension built. Loading from {so_path}[/cyan]"
                )
                _C = load_module_from_path(name, so_path)
                return
            time.sleep(wait_interval)
            elapsed += wait_interval
        
        raise RuntimeError(
            f"gsplat: rank {local_rank} timed out waiting for rank 0 to build extension"
        )
    
    # 3) rank 0（或单进程）：检查缓存中是否有 .so
    if os.path.exists(so_path):
        console = Console()
        console.print(
            f"[yellow]gsplat: CUDA extension already built. Loading from {so_path}[/yellow]"
        )
        _C = load_module_from_path(name, so_path)
        return
    
    # 4) Windows 上检查 .lib
    if os.path.exists(lib_path):
        console = Console()
        console.print(
            f"[yellow]gsplat: CUDA extension already built. Loading via torch from {build_dir}[/yellow]"
        )
        _C = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
        )
        return
    
    # 5) 没有缓存，需要 JIT 编译
    if cuda_toolkit_available():
        # 只在不存在已编译产物时，清理可能残留的构建目录
        if os.path.isdir(build_dir):
            try:
                shutil.rmtree(build_dir)
            except OSError:
                pass
        
        console = Console()
        with console.status(
            "[bold yellow]gsplat: Setting up CUDA (This may take a few minutes the first time)",
            spinner="bouncingBall",
        ):
            _C = load(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
            )
    else:
        console = Console()
        console.print(
            "[yellow]gsplat: No CUDA toolkit found. gsplat will be disabled.[/yellow]"
        )


try:
    _load_extension_ddp_safe()
except Exception as e:
    console = Console()
    console.print(f"[red]gsplat: Failed to load CUDA extension: {e}[/red]")
    _C = None


__all__ = ["_C"]