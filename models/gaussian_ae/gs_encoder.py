import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.transformer import (
    TransformerArgs,
    Decoder_Decoder,
    precompute_freqs_cis_2d,
)
from einops import rearrange
from fused_ssim import fused_ssim
from utils.gaussian_splatting import generate_2D_gaussian_splatting_step


class GaussianAutoEncoder(nn.Module):
    def __init__(
        self,
        transformer_config: TransformerArgs,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        gaussian_channels: int = 60,
        num_gaussian_per_patch: int = 96,
        l1_loss_ratio: float = 1.0,
        ssim_loss_ratio: float = 0.1,
    ):
        super().__init__()

        self.l1_loss_ratio = l1_loss_ratio
        self.ssim_loss_ratio = ssim_loss_ratio
        self.gaussian_channels = gaussian_channels
        self.num_gaussian_per_patch = num_gaussian_per_patch
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
        self.gaussian_embedding = nn.Parameter(torch.zeros(img_seq_len, gaussian_dim))

        self.decoder_decoder = Decoder_Decoder(config=transformer_config)

        self.gaussian_norm = nn.RMSNorm(gaussian_dim)
        self.gaussian_proj = nn.Linear(
            gaussian_dim, self.gaussian_channels * self.num_gaussian_per_patch
        )

        self.xy_proj = nn.Sequential(
            nn.Linear(self.gaussian_channels, self.gaussian_channels),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels, self.gaussian_channels * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels * 4, 2),
        )

        self.sigma_proj = nn.Sequential(
            nn.Linear(self.gaussian_channels, self.gaussian_channels),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels, self.gaussian_channels * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels * 4, 2),
        )

        self.rho_proj = nn.Sequential(
            nn.Linear(self.gaussian_channels, self.gaussian_channels),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels, self.gaussian_channels * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels * 4, 1),
        )

        self.alpha_proj = nn.Sequential(
            nn.Linear(self.gaussian_channels, self.gaussian_channels),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels, self.gaussian_channels * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels * 4, 1),
        )

        self.feat_proj = nn.Sequential(
            nn.Linear(self.gaussian_channels, self.gaussian_channels),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels, self.gaussian_channels * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels * 4, 3),
        )

        self.freq_cis = precompute_freqs_cis_2d(
            grid_size=self.grid_size,
            n_elem=transformer_config.dim // transformer_config.n_head,
            base=transformer_config.rope_base,
        )

    def calc_loss(self, imgs: torch.Tensor, gt_imgs: torch.Tensor):
        # img shape (B, C, H, W)

        l1_loss = F.l1_loss(imgs, gt_imgs)
        ssim_loss = 1 - fused_ssim(imgs, gt_imgs)
        loss = self.l1_loss_ratio * l1_loss + self.ssim_loss_ratio * ssim_loss

        return {"l1_loss": l1_loss, "ssim_loss": ssim_loss, "loss": loss}

    def render(self, gs_parameters: torch.Tensor, render_size: int = None):
        img_h, img_w = self.img_size, self.img_size

        if render_size is not None:
            img_h, img_w = render_size, render_size

        assert img_h == img_w, "Current gaussian renderer expects square outputs."
        render_scale = float(img_h) / self.img_size

        if gs_parameters.ndim == 2:
            return generate_2D_gaussian_splatting_step(
                sr_size=(img_h, img_w),
                gs_parameters=gs_parameters.float(),
                scale=render_scale,
                scale_modify=(render_scale, render_scale),
                cuda_rendering=True,
                if_dmax=True,
            )

        out_imgs = []
        for i in range(gs_parameters.shape[0]):
            out_imgs.append(
                generate_2D_gaussian_splatting_step(
                    sr_size=(img_h, img_w),
                    gs_parameters=gs_parameters[i].float(),
                    scale=render_scale,
                    scale_modify=(render_scale, render_scale),
                    cuda_rendering=True,
                    if_dmax=True,
                )
            )

        return torch.cat(out_imgs, dim=0)

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
            c=self.gaussian_channels,
        )

        sigma = self.sigma_proj(gaussian_features)
        rho = self.rho_proj(gaussian_features)
        alpha = self.alpha_proj(gaussian_features)
        feat = self.feat_proj(gaussian_features)
        xy = torch.sigmoid(self.xy_proj(gaussian_features))

        gs_parameters = torch.cat([sigma, rho, alpha, feat, xy], dim=-1)

        recon_imgs = self.render(gs_parameters=gs_parameters)

        return recon_imgs

    def forward(self, pixels: torch.Tensor, img_features: torch.Tensor):
        recon_imgs = self.forward_shared(img_features=img_features)

        loss = self.calc_loss(imgs=recon_imgs, gt_imgs=pixels)

        return loss


def GaussianAE_B(
    img_size: int = 224,
    patch_size: int = 14,
    in_channels: int = 3,
    gaussian_channels: int = 48,
    num_gaussian_per_patch: int = 64,
    l1_loss_ratio: float = 1.0,
    ssim_loss_ratio: float = 0.1,
    **kwargs,
):
    return GaussianAutoEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        gaussian_channels=gaussian_channels,
        num_gaussian_per_patch=num_gaussian_per_patch,
        l1_loss_ratio=l1_loss_ratio,
        ssim_loss_ratio=ssim_loss_ratio,
        transformer_config=TransformerArgs(n_layer=12, n_head=12, dim=768, **kwargs),
    )
