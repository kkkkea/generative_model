import glob
import os
from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import Function
from torch.utils.cpp_extension import load

build_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "build")
os.makedirs(build_path, exist_ok=True)

file_path = os.path.split(os.path.abspath(__file__))[0]
csrc_path = os.path.join(file_path, "csrc")
glm_path = os.path.join(csrc_path, "third_party", "glm")

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    if torch.cuda.is_available():
        archs = {
            f"{major}.{minor}"
            for major, minor in (
                torch.cuda.get_device_capability(i)
                for i in range(torch.cuda.device_count())
            )
        }
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(archs))
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.7;8.9"

sources = [os.path.join(file_path, "gswrapper.cpp")]
sources += glob.glob(os.path.join(csrc_path, "*.cu"))

GSWrapper = load(
    name="gscuda_tiled",
    sources=sources,
    build_directory=build_path,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"],
    extra_include_paths=[csrc_path, glm_path],
    verbose=True,
)

MAX_BLOCK_SIZE = 256
FIXED_BLOCK_H = 16
FIXED_BLOCK_W = 16


def _check_block_size(block_h: int, block_w: int) -> int:
    block_size = int(block_h) * int(block_w)
    if block_h <= 0 or block_w <= 0:
        raise ValueError("BLOCK_H and BLOCK_W must be positive")
    if block_size > MAX_BLOCK_SIZE:
        raise ValueError(f"BLOCK_H * BLOCK_W must be <= {MAX_BLOCK_SIZE}")
    return block_size


def _check_fixed_block(block_h: int, block_w: int) -> None:
    if block_h != FIXED_BLOCK_H or block_w != FIXED_BLOCK_W:
        raise ValueError(
            f"gs_cuda_tiled uses fixed {FIXED_BLOCK_H}x{FIXED_BLOCK_W} tiles"
        )


def compute_cumulative_intersects(num_tiles_hit: Tensor) -> Tuple[int, Tensor]:
    # This prefix sum is metadata for binning and is not differentiable. Keeping
    # it on CPU avoids CUDA device-ordinal issues seen in some local setups.
    cum_tiles_hit_cpu = torch.cumsum(
        num_tiles_hit.detach().cpu(), dim=0, dtype=torch.int32
    )
    cum_tiles_hit = cum_tiles_hit_cpu.to(num_tiles_hit.device, non_blocking=True)
    num_intersects = cum_tiles_hit[-1].item() if cum_tiles_hit.numel() > 0 else 0
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Tensor,
    radii: Tensor,
    cum_tiles_hit: Tensor,
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    isect_ids, gaussian_ids = GSWrapper.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = GSWrapper.get_tile_bin_edges(
        num_intersects,
        tile_bounds[0] * tile_bounds[1],
        isect_ids_sorted.contiguous(),
    )
    return (
        isect_ids,
        gaussian_ids,
        isect_ids_sorted,
        gaussian_ids_sorted,
        tile_bins,
    )


def project_gaussians_2d_scale_rot(
    means2d: Tensor,
    scales2d: Tensor,
    rotation: Tensor,
    img_height: int,
    img_width: int,
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return _ProjectGaussians2dScaleRot.apply(
        means2d.contiguous(),
        scales2d.contiguous(),
        rotation.contiguous(),
        img_height,
        img_width,
        tile_bounds,
    )


class _ProjectGaussians2dScaleRot(Function):
    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        scales2d: Tensor,
        rotation: Tensor,
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int],
    ):
        num_points = means2d.shape[-2]
        if num_points < 1 or means2d.shape[-1] != 2:
            raise ValueError(f"Invalid shape for means2d: {means2d.shape}")

        xys, radii, conics, num_tiles_hit = (
            GSWrapper.project_gaussians_2d_scale_rot_forward(
                num_points,
                means2d,
                scales2d,
                rotation,
                img_height,
                img_width,
                tile_bounds,
            )
        )

        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.save_for_backward(means2d, scales2d, rotation, radii, conics)
        return xys, radii, conics, num_tiles_hit

    @staticmethod
    def backward(ctx, v_xys, v_radii, v_conics, v_num_tiles_hit):
        means2d, scales2d, rotation, radii, conics = ctx.saved_tensors
        _, v_mean2d, v_scale, v_rot = GSWrapper.project_gaussians_2d_scale_rot_backward(
            ctx.num_points,
            means2d,
            scales2d,
            rotation,
            ctx.img_height,
            ctx.img_width,
            radii,
            conics,
            v_xys,
            v_conics,
        )
        return v_mean2d, v_scale, v_rot, None, None, None


def rasterize_gaussians_sum(
    xys: Tensor,
    radii: Tensor,
    conics: Tensor,
    num_tiles_hit: Tensor,
    colors: Tensor,
    img_height: int,
    img_width: int,
    BLOCK_H: int = 16,
    BLOCK_W: int = 16,
    topk_norm: bool = False,
) -> Tensor:
    if colors.dtype == torch.uint8:
        colors = colors.float() / 255

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")
    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")
    _check_fixed_block(BLOCK_H, BLOCK_W)

    return _RasterizeGaussiansSum.apply(
        xys.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        img_height,
        img_width,
        BLOCK_H,
        BLOCK_W,
        topk_norm,
    )


class _RasterizeGaussiansSum(Function):
    @staticmethod
    def forward(
        ctx,
        xys: Tensor,
        radii: Tensor,
        conics: Tensor,
        num_tiles_hit: Tensor,
        colors: Tensor,
        img_height: int,
        img_width: int,
        BLOCK_H: int = 16,
        BLOCK_W: int = 16,
        topk_norm: bool = False,
    ) -> Tensor:
        num_points = xys.size(0)
        block = (BLOCK_W, BLOCK_H, 1)
        img_size = (img_width, img_height, 1)
        tile_bounds = (
            (img_width + BLOCK_W - 1) // BLOCK_W,
            (img_height + BLOCK_H - 1) // BLOCK_H,
            1,
        )

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)
        if num_intersects < 1:
            gaussian_ids_sorted = torch.zeros(0, device=xys.device, dtype=torch.int32)
            tile_bins = torch.zeros(
                tile_bounds[0] * tile_bounds[1],
                2,
                device=xys.device,
                dtype=torch.int32,
            )
        else:
            _, _, _, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                radii,
                cum_tiles_hit,
                tile_bounds,
            )

        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.BLOCK_H = BLOCK_H
        ctx.BLOCK_W = BLOCK_W
        ctx.num_intersects = num_intersects
        ctx.topk_norm = topk_norm
        _check_block_size(BLOCK_H, BLOCK_W)

        if not topk_norm:
            (out_img,) = GSWrapper.nd_rasterize_forward(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
            )
            ctx.save_for_backward(gaussian_ids_sorted, tile_bins, xys, conics, colors)
        else:
            out_img, pixel_topk = GSWrapper.nd_rasterize_forward_topk_norm(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
            )
            ctx.save_for_backward(
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                pixel_topk.contiguous(),
            )

        return out_img

    @staticmethod
    def backward(ctx, v_out_img):
        img_height = ctx.img_height
        img_width = ctx.img_width
        BLOCK_H = ctx.BLOCK_H
        BLOCK_W = ctx.BLOCK_W
        topk_norm = ctx.topk_norm

        if not topk_norm:
            gaussian_ids_sorted, tile_bins, xys, conics, colors = ctx.saved_tensors
            v_xy, v_conic, v_colors = GSWrapper.nd_rasterize_backward(
                img_height,
                img_width,
                BLOCK_H,
                BLOCK_W,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                v_out_img.contiguous(),
            )
        else:
            gaussian_ids_sorted, tile_bins, xys, conics, colors, pixel_topk = (
                ctx.saved_tensors
            )
            v_xy, v_conic, v_colors = GSWrapper.nd_rasterize_backward_topk_norm(
                img_height,
                img_width,
                BLOCK_H,
                BLOCK_W,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                v_out_img.contiguous(),
                pixel_topk,
            )

        return (
            v_xy,
            None,
            v_conic,
            None,
            v_colors,
            None,
            None,
            None,
            None,
            None,
        )


def gaussiansplatting_render(
    means2d: Tensor,
    scales2d: Tensor,
    rotation: Tensor,
    colors: Tensor,
    image_size: Tuple[int, int],
    BLOCK_H: int = FIXED_BLOCK_H,
    BLOCK_W: int = FIXED_BLOCK_W,
    topk_norm: bool = True,
) -> Tensor:
    h, w = image_size[:2]
    tile_bounds = ((w + BLOCK_W - 1) // BLOCK_W, (h + BLOCK_H - 1) // BLOCK_H, 1)
    xys, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
        means2d.contiguous(),
        scales2d.contiguous(),
        rotation.contiguous(),
        h,
        w,
        tile_bounds,
    )
    render_img = rasterize_gaussians_sum(
        xys,
        radii,
        conics,
        num_tiles_hit,
        colors.contiguous(),
        h,
        w,
        BLOCK_H,
        BLOCK_W,
        topk_norm,
    )

    render_img = render_img.view(-1, h, w, 3).permute(0, 3, 1, 2).contiguous()

    return render_img


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("gs_cuda_tiled requires a visible CUDA device.")

    num_devices = torch.cuda.device_count()
    print(f"warming up gs_cuda_tiled on {num_devices} visible CUDA device(s)")

    for device_idx in range(num_devices):
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)

        mean = torch.rand((1000, 2), device=device)
        scale = torch.ones((1000, 2), device=device)
        rota = torch.randn((1000, 1), device=device)
        color = torch.rand((1000, 3), device=device)

        img = gaussiansplatting_render(
            means2d=mean,
            scales2d=scale,
            rotation=rota,
            colors=color,
            image_size=(256, 256),
        )

        torch.cuda.synchronize(device)
        print(f"{device}: {tuple(img.shape)}")
