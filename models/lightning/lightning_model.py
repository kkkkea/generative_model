import os
import wandb
import torch
import torch.nn as nn
import lightning.pytorch as pl
import copy

from typing import Callable, Iterable, Optional, Union, Sequence
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import LinearLR, LRScheduler, SequentialLR
from torch.optim import Optimizer
from torchvision.utils import make_grid
from models.gaussian_ae.gs_encoder import calc_loss
from utils.util import (
    SimpleEMA,
    copy_params,
    filter_nograd_tensors,
    no_grad,
    record_metric,
    TimeProfiler,
)

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


class GaussianAEModel(pl.LightningModule):
    def __init__(
        self,
        ae: nn.Module,
        ema_tracker: SimpleEMA = None,
        optimizer: OptimizerCallable = None,
        upsample_ratio: float = 8.0,
        lr_scheduler: LRSchedulerCallable = None,
        warmup_steps: int = 2000,
        warmup_start_factor: float = 1e-3,
        lr_scheduler_interval: str = "step",
        lr_scheduler_frequency: int = 1,
        lr_scheduler_monitor: Optional[str] = None,
        override_lr_on_resume: Optional[float] = None,
        override_ema_decay_on_resume: Optional[float] = None,
        eval_original_model: bool = False,
        enable_profiling: bool = True,
        profiling_print_freq: int = 10,
        profiling_warmup_steps: int = 5,
        enable_log_images: bool = True,
        num_val_log_images: int = 4,
        enable_compile: bool = False,
    ):
        super().__init__()
        self.ae = ae
        self.ema_ae = copy.deepcopy(self.ae)
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.upsample_ratio = upsample_ratio
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_frequency = lr_scheduler_frequency
        self.lr_scheduler_monitor = lr_scheduler_monitor
        self.override_lr_on_resume = override_lr_on_resume
        self.override_ema_decay_on_resume = override_ema_decay_on_resume

        self.eval_original_model = eval_original_model

        self.enable_profiling = enable_profiling
        self.profiling_print_freq = profiling_print_freq
        self.profiling_warmup_steps = profiling_warmup_steps
        self.profiler = None

        self.enable_log_images = enable_log_images
        self.num_val_log_images = num_val_log_images
        self.enable_compile = enable_compile

    def configure_model(self) -> None:
        copy_params(self.ae, self.ema_ae)

        no_grad(self.ema_ae)

        if self.enable_compile:
            self.ae.compile()
            self.ema_ae.compile()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker] if self.ema_tracker is not None else []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_ae = filter_nograd_tensors(self.ae.parameters())
        param_groups = [{"params": params_ae}]
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None and self.warmup_steps <= 0:
            return dict(optimizer=optimizer)

        lr_scheduler = (
            self.lr_scheduler(optimizer) if self.lr_scheduler is not None else None
        )
        if self.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            if lr_scheduler is None:
                lr_scheduler = warmup_scheduler
            else:
                lr_scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, lr_scheduler],
                    milestones=[self.warmup_steps],
                )

        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": self.lr_scheduler_interval,
            "frequency": self.lr_scheduler_frequency,
        }
        if self.lr_scheduler_monitor is not None:
            scheduler_config["monitor"] = self.lr_scheduler_monitor
        return dict(optimizer=optimizer, lr_scheduler=scheduler_config)

    def on_validation_start(self) -> None:
        self.ema_ae.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_ae.to(torch.float32)

    def on_train_start(self) -> None:
        self.ema_ae.to(torch.float32)
        if self.ema_tracker is not None:
            self.ema_tracker.setup_models(net=self.ae, ema_net=self.ema_ae)

        resumed = (getattr(self, "global_step", 0) or 0) > 0 or getattr(
            self.trainer, "ckpt_path", None
        ) is not None
        if resumed and self.override_lr_on_resume is not None:
            self._apply_override_lr(float(self.override_lr_on_resume))

        if (
            resumed
            and self.override_ema_decay_on_resume is not None
            and self.ema_tracker is not None
        ):
            self._apply_override_ema_decay(float(self.override_ema_decay_on_resume))

        if self.enable_profiling and self.profiler is None:
            export_path = os.path.join(
                self.trainer.default_root_dir, "profiling_results.jsonl"
            )
            self.profiler = TimeProfiler(
                enabled=True,
                print_freq=self.profiling_print_freq,
                warmup_steps=self.profiling_warmup_steps,
                window_size=100,
                detailed=True,
                track_memory=True,
                export_json=True,
                export_path=export_path,
                rank=self.global_rank if hasattr(self, "global_rank") else 0,
                world_size=(
                    self.trainer.world_size
                    if hasattr(self.trainer, "world_size")
                    else 1
                ),
            )

    def on_train_end(self):
        if self.profiler:
            self.profiler.summary()

    def training_step(self, batch, batch_idx):
        if self.profiler:
            self.profiler.start_step()

        x, y, meta = batch
        upsample_ratio = self.upsample_ratio

        if self.profiler:
            with self.profiler.profile("training/ae_forward"):
                loss = self.ae(
                    x,
                    gt_pixels=y,
                    upsample_ratio=upsample_ratio,
                )
        else:
            loss = self.ae(
                x,
                gt_pixels=y,
                upsample_ratio=upsample_ratio,
            )

        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)

        if self.profiler:
            for key, value in loss.items():
                if isinstance(value, (float, int, torch.Tensor)):
                    val = value.item() if isinstance(value, torch.Tensor) else value
                    record_metric(f"loss/{key}", val)

            self.profiler.end_step()

        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        upsample_ratio = self.upsample_ratio

        with torch.no_grad():
            if self.eval_original_model:
                recon_imgs = self.ae.forward_inference(x, upsample_ratio=upsample_ratio)
            else:
                recon_imgs = self.ema_ae.forward_inference(
                    x, upsample_ratio=upsample_ratio
                )

        return recon_imgs, y

    def validation_step(self, batch, batch_idx):
        recon_imgs, gt_imgs = self.predict_step(batch, batch_idx)
        l1, psnr, ssim = calc_loss(recon_imgs, gt_imgs)
        self.log_dict(
            {
                "val/l1": l1,
                "val/psnr": psnr,
                "val/ssim": ssim,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.enable_log_images and batch_idx == 0 and self.trainer.is_global_zero:
            num_imgs = min(self.num_val_log_images, gt_imgs.shape[0])
            vis = torch.stack(
                [
                    img
                    for pair in zip(gt_imgs[:num_imgs], recon_imgs[:num_imgs])
                    for img in pair
                ],
                dim=0,
            )
            grid = make_grid(vis.clamp(0.0, 1.0), nrow=2)

            self.logger.experiment.log(
                {
                    "val/reconstructions": wandb.Image(
                        grid.permute(1, 2, 0).detach().cpu().numpy()
                    ),
                    "global_step": self.global_step,
                }
            )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.ae.state_dict(
            destination=destination, prefix=prefix + "ae.", keep_vars=keep_vars
        )
        self.ema_ae.state_dict(
            destination=destination,
            prefix=prefix + "ema_ae.",
            keep_vars=keep_vars,
        )
        return destination

    def _apply_override_lr(self, new_lr: float) -> None:
        for optimizer in self.trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
        try:
            lr_sched_configs = getattr(self.trainer, "lr_scheduler_configs", None)
            if lr_sched_configs is not None:
                for cfg in lr_sched_configs:
                    scheduler = getattr(cfg, "scheduler", None) or cfg
                    base_lrs = getattr(scheduler, "base_lrs", None)
                    if isinstance(base_lrs, list) and len(base_lrs) > 0:
                        scheduler.base_lrs = [new_lr for _ in base_lrs]
        except Exception:
            pass

    def _apply_override_ema_decay(self, new_decay: float) -> None:
        if self.ema_tracker is None:
            return
        self.ema_tracker.decay = new_decay

    def _compute_grad_norm(self) -> float:
        total_norm_sq = None
        for parameter in self.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if total_norm_sq is None:
                total_norm_sq = torch.zeros(1, device=grad.device, dtype=torch.float32)
            total_norm_sq += torch.norm(grad.float(), 2) ** 2
        if total_norm_sq is None:
            return 0.0
        return torch.sqrt(total_norm_sq).item()

    def on_before_optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: Optional[int] = None
    ) -> None:
        if optimizer_idx not in (None, 0):
            return

        grad_norm = self._compute_grad_norm()
        self.log(
            "train/grad_norm", grad_norm, on_step=True, prog_bar=False, sync_dist=False
        )

        lrs = [group.get("lr", None) for group in optimizer.param_groups]
        lrs = [lr for lr in lrs if lr is not None]
        if lrs:
            mean_lr = sum(lrs) / len(lrs)
            self.log("train/lr", mean_lr, on_step=True, prog_bar=False, sync_dist=False)

        if self.ema_tracker is not None:
            self.log(
                "train/ema_decay",
                float(self.ema_tracker.decay),
                on_step=True,
                prog_bar=False,
                sync_dist=False,
            )
