import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.transformer import (
    TransformerArgs,
    Decoder_Decoder,
    precompute_freqs_cis_2d,
)
from time import perf_counter
from einops import rearrange
from fused_ssim import fused_ssim
from gs_plat import (
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
    rasterize_gaussians_sum,
)


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

        self.inverse_scale_proj = nn.Sequential(
            nn.Linear(self.gaussian_channels, self.gaussian_channels),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels, self.gaussian_channels * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels * 4, 2),
        )

        self.rot_proj = nn.Sequential(
            nn.Linear(self.gaussian_channels, self.gaussian_channels),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels, self.gaussian_channels * 4),
            nn.ReLU(),
            nn.Linear(self.gaussian_channels * 4, 1),
        )

        # RGB value
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

    def _get_scale(self, scale: torch.Tensor, upsample_ratio: float = None):
        scale = 1 / scale

        if upsample_ratio is not None:
            scale = upsample_ratio * scale

        return scale

    def render_inner(
        self,
        img_h,
        img_w,
        tile_bounds,
        xy: torch.Tensor,
        inverse_scale: torch.Tensor,
        rot: torch.Tensor,
        feat: torch.Tensor,
        upsample_ratio=None,
    ):
        scale = self._get_scale(inverse_scale, upsample_ratio)

        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp

        # enable tiles
        enable_topk_norm = True
        tmp = (
            xy,
            radii,
            conics,
            num_tiles_hit,
            feat,
            img_h,
            img_w,
            self.block_h,
            self.block_w,
            enable_topk_norm,
        )
        out_image = rasterize_gaussians_sum(*tmp)

        out_image = (
            out_image.view(-1, img_h, img_w, self.in_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return out_image

    def render(
        self,
        xy: torch.Tensor,
        inverse_scale: torch.Tensor,
        rot: torch.Tensor,
        feat: torch.Tensor,
        render_size: int = None,
    ):
        img_h, img_w = self.img_size, self.img_size

        if render_size is not None:
            img_h, img_w = render_size, render_size

        tile_bounds = (
            (img_w + self.block_w - 1) // self.block_w,
            (img_h + self.block_h - 1) // self.block_h,
            1,
        )
        upsample_ratio = float(img_h) / self.img_size

        if xy.ndim == 2:
            return self.render_inner(
                img_h=img_h,
                img_w=img_w,
                tile_bounds=tile_bounds,
                xy=xy,
                inverse_scale=inverse_scale,
                rot=rot,
                feat=feat,
                upsample_ratio=upsample_ratio,
            )

        out_imgs = []
        for i in range(xy.shape[0]):
            out_imgs.append(
                self.render_inner(
                    img_h=img_h,
                    img_w=img_w,
                    tile_bounds=tile_bounds,
                    xy=xy[i],
                    inverse_scale=inverse_scale[i],
                    rot=rot[i],
                    feat=feat[i],
                    upsample_ratio=upsample_ratio,
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
            img_features, queries, freq_cis=freq_cis
        )
        gaussian_features = self.gaussian_proj(self.gaussian_norm(gaussian_features))

        gaussian_features = rearrange(
            gaussian_features,
            "b l (n c) -> b (l n) c",
            n=self.num_gaussian_per_patch,
            c=self.gaussian_channels,
        )

        xy = self.xy_proj(gaussian_features)
        inverse_scale = self.inverse_scale_proj(gaussian_features)
        rot = self.rot_proj(gaussian_features)
        feat = self.feat_proj(gaussian_features)

        recon_imgs = self.render(xy=xy, inverse_scale=inverse_scale, rot=rot, feat=feat)

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
