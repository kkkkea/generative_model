import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.transformer import (
    TransformerArgs,
    Cross_Self_Decoder,
    precompute_freqs_cis_2d,
)
from einops import rearrange
from pytorch_msssim import ssim
from utils.gaussian_splatting import generate_2D_gaussian_splatting_step


class GaussianAutoEncoder(nn.Module):
    def __init__(
        self,
        transformer_config: TransformerArgs,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        gaussian_channel: int = 192,
        num_gaussian_per_patch: int = 256,
        l1_loss_ratio: float = 1.0,
        ssim_loss_ratio: float = 0.1,
    ):
        super().__init__()

        self.l1_loss_ratio = l1_loss_ratio
        self.ssim_loss_ratio = ssim_loss_ratio
        self.gaussian_channel = gaussian_channel
        self.num_gaussian_per_patch = num_gaussian_per_patch
        self.num_gaussian_sqrt = int(math.sqrt(num_gaussian_per_patch))
        self.img_size = img_size
        self.in_channels = in_channels

        # hardcode: Must match hardcoded value in CUDA kernel, modify with caution
        self.block_w, self.block_h = 16, 16
        self.tile_bounds = (
            (self.img_size + self.block_w - 1) // self.block_w,
            (self.img_size + self.block_h - 1) // self.block_h,
            1,
        )

        self.grid_size = img_size // patch_size
        img_seq_len = self.grid_size**2
        gaussian_dim = transformer_config.dim
        self.gaussian_embedding = nn.Parameter(torch.randn(img_seq_len, gaussian_dim))

        self.decoder_decoder = Cross_Self_Decoder(config=transformer_config)

        self.gaussian_norm = nn.RMSNorm(gaussian_dim)

        mlp_ratio_sqrt = int(
            math.sqrt(
                self.gaussian_channel * self.num_gaussian_per_patch / gaussian_dim
            )
        )
        self.gaussian_proj = nn.Sequential(
            nn.Linear(
                gaussian_dim,
                gaussian_dim * mlp_ratio_sqrt,
            ),
            nn.ReLU(),
            nn.Linear(
                gaussian_dim * mlp_ratio_sqrt,
                self.gaussian_channel * self.num_gaussian_per_patch,
            ),
        )

        # GS sigma_x, sigma_y
        self.mlp_sigma = nn.Sequential(
            nn.Linear(self.gaussian_channel, self.gaussian_channel),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel, self.gaussian_channel * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel * 4, 2),
        )

        # GS rho
        self.mlp_rho = nn.Sequential(
            nn.Linear(self.gaussian_channel, self.gaussian_channel),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel, self.gaussian_channel * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel * 4, 1),
        )

        # GS alpha
        self.mlp_alpha = nn.Sequential(
            nn.Linear(self.gaussian_channel, self.gaussian_channel),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel, self.gaussian_channel * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel * 4, 1),
        )

        # GS RGB values
        self.mlp_rgb = nn.Sequential(
            nn.Linear(self.gaussian_channel, self.gaussian_channel),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel, self.gaussian_channel * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel * 4, 3),
        )

        # GS mean_x, mean_y
        self.mlp_mean = nn.Sequential(
            nn.Linear(self.gaussian_channel, self.gaussian_channel),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel, self.gaussian_channel * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channel * 4, 2),
        )

        self.freq_cis = precompute_freqs_cis_2d(
            grid_size=self.grid_size,
            n_elem=transformer_config.dim // transformer_config.n_head,
            base=transformer_config.rope_base,
        )

    def calc_loss(self, imgs: torch.Tensor, gt_imgs: torch.Tensor):
        # img shape (B, C, H, W)

        l1_loss = F.l1_loss(imgs, gt_imgs)
        ssim_loss = 1 - ssim(imgs, gt_imgs, data_range=1, size_average=True)
        loss = self.l1_loss_ratio * l1_loss + self.ssim_loss_ratio * ssim_loss

        return {"l1_loss": l1_loss, "ssim_loss": ssim_loss, "loss": loss}

    def render(self, gs_parameters: torch.Tensor, render_size: int = None):
        assert gs_parameters.ndim == 3

        img_h, img_w = self.img_size, self.img_size

        if render_size is not None:
            img_h, img_w = render_size, render_size

        assert img_h == img_w, "Current gaussian renderer expects square outputs."
        render_scale = float(img_h) / self.img_size

        out_imgs = []
        for i in range(gs_parameters.shape[0]):
            gs_out = generate_2D_gaussian_splatting_step(
                sr_size=torch.tensor([img_h, img_w]),
                gs_parameters=gs_parameters[i],
                scale=render_scale,
                sample_coords=None,
                scale_modify=torch.tensor([render_scale, render_scale]),
                default_step_size=1.2,
                cuda_rendering=True,
                mode="scale_modify",
                if_dmax=True,
                dmax_mode="fix",
                dmax=0.3,
            )

            gs_out = gs_out.unsqueeze(0)
            gs_out = gs_out[:, :, :img_h, :img_w]
            out_imgs.append(gs_out)

        return torch.cat(out_imgs, dim=0)

    @staticmethod
    def get_N_reference_points(h, w, device="cuda"):
        # step_y = 1/(h+1)
        # step_x = 1/(w+1)
        step_y = 1 / h
        step_x = 1 / w
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                step_y / 2, 1 - step_y / 2, h, dtype=torch.float32, device=device
            ),
            torch.linspace(
                step_x / 2, 1 - step_x / 2, w, dtype=torch.float32, device=device
            ),
        )
        reference_points = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
        reference_points = reference_points[None, :, None]
        return reference_points

    def forward_shared(self, img_features: torch.Tensor):
        b, seq_len, h = img_features.shape
        freq_cis = self.freq_cis.unsqueeze(0).repeat(b, 1, 1, 1).to(img_features.device)
        queries = (
            self.gaussian_embedding.unsqueeze(0).repeat(b, 1, 1).to(img_features.device)
        )

        gaussian_features = self.decoder_decoder(
            img_features, queries, freqs_cis=freq_cis
        )
        gaussian_features = self.gaussian_proj(self.gaussian_norm(gaussian_features))

        gaussian_features = rearrange(
            gaussian_features,
            "b l (n c) -> b (l n) c",
            n=self.num_gaussian_per_patch,
            c=self.gaussian_channel,
        )

        sigma = self.mlp_sigma(gaussian_features)
        rho = self.mlp_rho(gaussian_features)
        alpha = self.mlp_alpha(gaussian_features)
        rgb = self.mlp_rgb(gaussian_features)
        mean = self.mlp_mean(gaussian_features)

        gaussian_h = gaussian_w = self.grid_size * self.num_gaussian_sqrt
        mean = (
            mean
            / torch.tensor([gaussian_h, gaussian_w], device=mean.device)[None, None]
        )
        reference_offset = self.get_N_reference_points(
            gaussian_h, gaussian_w, device=mean.device
        )
        pos = reference_offset + mean

        gs_params = torch.cat([sigma, rho, alpha, rgb, pos], dim=-1)

        recon_imgs = self.render(gs_parameters=gs_params)

        return recon_imgs

    def forward(self, pixels: torch.Tensor, img_features: torch.Tensor):
        recon_imgs = self.forward_shared(img_features=img_features)

        loss = self.calc_loss(imgs=recon_imgs, gt_imgs=pixels)

        return loss


def GaussianAE_B(
    img_size: int = 224,
    patch_size: int = 14,
    in_channels: int = 3,
    gaussian_channel: int = 192,
    num_gaussian_per_patch: int = 256,
    l1_loss_ratio: float = 1.0,
    ssim_loss_ratio: float = 0.1,
    **kwargs,
):
    return GaussianAutoEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        gaussian_channel=gaussian_channel,
        num_gaussian_per_patch=num_gaussian_per_patch,
        l1_loss_ratio=l1_loss_ratio,
        ssim_loss_ratio=ssim_loss_ratio,
        transformer_config=TransformerArgs(n_layer=12, n_head=12, dim=768, **kwargs),
    )
