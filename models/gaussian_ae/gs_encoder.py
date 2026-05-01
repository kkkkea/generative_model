import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.transformer import (
    Cross_Self_Decoder,
    precompute_freqs_cis_2d,
)
from models.encoder.dinov2 import Dinov2withNorm
from models.encoder.dinov3 import Dinov3withNorm
from transformers import AutoImageProcessor

from einops import rearrange
from fused_ssim import fused_ssim
from utils.gs_cuda_tiled.gswrapper import gaussiansplatting_render
from utils.gradient import compute_gmap_batch


def batch_psnr(pred, target, max_val=1.0, eps=1e-8):
    pred = pred.float()
    target = target.float()

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))  # (B,)
    psnr = 10 * torch.log10(max_val**2 / (mse + eps))
    return psnr  # (B,)


def calc_loss(imgs, gt_imgs):
    l1_loss = F.l1_loss(imgs, gt_imgs)
    ssim = fused_ssim(imgs, gt_imgs)
    psnr = batch_psnr(imgs, gt_imgs).mean()

    return l1_loss, psnr, ssim


def get_grid(
    h, w, x_lim=(0.0, 1.0), y_lim=(0.0, 1.0), device=None, dtype=torch.float32
):
    x = torch.linspace(x_lim[0], x_lim[1], steps=w + 1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(y_lim[0], y_lim[1], steps=h + 1, device=device, dtype=dtype)[:-1]

    x = x + 0.5 / w
    y = y + 0.5 / h

    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return grid


class GaussianAutoEncoder(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        encoder_type: str = "dinov3",
        img_size: int = 256,
        num_gaussian: int = 16384,
        group_size: int = 64,
        init_random_ratio: float = 0.3,
        init_scale: float = 1.0,
        max_xy_shift: float = 4.0,
        inv_scale_delta_range: float = 0.5,
        rotation_delta_range: float = 1.0,
        color_delta_scale: float = 1.0,
        l1_loss_ratio: float = 1.0,
        ssim_loss_ratio: float = 0.1,
        n_head: int = 32,
        ffn_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        norm_eps: float = 1e-5,
        cross_layer: int = 6,
        self_layer: int = 8,
    ):
        super().__init__()

        self.l1_loss_ratio = l1_loss_ratio
        self.ssim_loss_ratio = ssim_loss_ratio
        self.num_gaussian = num_gaussian
        self.group_size = group_size
        self.init_random_ratio = init_random_ratio
        self.init_scale = init_scale
        self.img_size = img_size
        self.max_xy_shift = max_xy_shift
        self.inv_scale_delta_range = inv_scale_delta_range
        self.rotation_delta_range = rotation_delta_range
        self.color_delta_scale = color_delta_scale

        if self.num_gaussian % self.group_size != 0:
            raise ValueError(
                f"num_gaussian={self.num_gaussian} must be divisible by "
                f"group_size={self.group_size}."
            )
        self.gs_token_count = self.num_gaussian // self.group_size

        proc = AutoImageProcessor.from_pretrained(encoder_path)
        if encoder_type == "dinov2":
            self.encoder = Dinov2withNorm(dinov2_path=encoder_path)
        elif encoder_type == "dinov3":
            self.encoder = Dinov3withNorm(dinov3_path=encoder_path)
        else:
            raise ValueError("encoder type not supported")

        self.register_buffer(
            "encoder_mean",
            torch.tensor(proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "encoder_std",
            torch.tensor(proc.image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.hidden_dim = self.encoder.hidden_size
        self.feature_grid_size = self.img_size // self.encoder.patch_size
        feature_token_count = self.feature_grid_size**2
        if self.gs_token_count != feature_token_count:
            raise ValueError(
                f"num_gaussian / group_size must match the encoder feature grid: "
                f"{self.gs_token_count} != {feature_token_count}."
            )

        self.inner_gs_pos = nn.Parameter(torch.zeros(1, 1, self.group_size, 8))

        self.gsembed_layer = nn.Sequential(
            nn.Linear(self.group_size * 8, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.to_gs = nn.Linear(self.hidden_dim, self.group_size * 8)

        self.decoder_decoder = Cross_Self_Decoder(
            dim=self.hidden_dim,
            n_head=n_head,
            ffn_dropout_p=ffn_dropout_p,
            attn_dropout_p=attn_dropout_p,
            resid_dropout_p=resid_dropout_p,
            norm_eps=norm_eps,
            cross_layer=cross_layer,
            self_layer=self_layer,
        )

        self.register_buffer(
            "freq_cis",
            precompute_freqs_cis_2d(
                grid_size=self.feature_grid_size,
                n_elem=self.hidden_dim // n_head,
                base=10000,
            ),
        )

    @staticmethod
    def _format_param_count(num_params: int) -> str:
        if num_params >= 1_000_000_000:
            return f"{num_params / 1_000_000_000:.2f}B"
        if num_params >= 1_000_000:
            return f"{num_params / 1_000_000:.2f}M"
        if num_params >= 1_000:
            return f"{num_params / 1_000:.2f}K"
        return str(num_params)

    def print_parameter_summary(self):
        lines = []
        total_params = 0
        trainable_params = 0

        for module_name, module in self.named_children():
            if module_name == "gaussian_embedding":
                continue

            module_total = sum(param.numel() for param in module.parameters())
            module_trainable = sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            total_params += module_total
            trainable_params += module_trainable
            lines.append((module_name, module_total, module_trainable))

        name_width = max(len("module"), max(len(name) for name, _, _ in lines))
        total_width = max(
            len("total params"),
            max(len(self._format_param_count(total)) for _, total, _ in lines),
        )
        trainable_width = max(
            len("trainable"),
            max(len(self._format_param_count(trainable)) for _, _, trainable in lines),
        )

        header = (
            f"{'module':<{name_width}}  "
            f"{'total params':>{total_width}}  "
            f"{'trainable':>{trainable_width}}"
        )
        print(header)
        print("-" * len(header))
        for name, total, trainable in lines:
            print(
                f"{name:<{name_width}}  "
                f"{self._format_param_count(total):>{total_width}}  "
                f"{self._format_param_count(trainable):>{trainable_width}}"
            )
        print("-" * len(header))
        print(
            f"{'total':<{name_width}}  "
            f"{self._format_param_count(total_params):>{total_width}}  "
            f"{self._format_param_count(trainable_params):>{trainable_width}}"
        )

    def calc_loss(self, imgs: torch.Tensor, gt_imgs: torch.Tensor):
        # img shape (B, C, H, W)

        l1_loss = F.l1_loss(imgs, gt_imgs)
        ssim_loss = 1 - fused_ssim(imgs, gt_imgs)
        loss = self.l1_loss_ratio * l1_loss + self.ssim_loss_ratio * ssim_loss

        return {"l1_loss": l1_loss, "ssim_loss": ssim_loss, "loss": loss}

    def sample_pos(self, prob):
        _, N = prob.shape

        if self.num_gaussian > N:
            raise ValueError(
                f"num_gaussian={self.num_gaussian} cannot be larger than "
                f"the number of pixels={N} when sampling without replacement."
            )

        prob = prob.float()
        return torch.topk(prob, k=self.num_gaussian, dim=1, largest=True).indices

    def get_init_gaussians(self, imgs: torch.Tensor):
        B, C, H, W = imgs.shape
        if C != 3:
            raise ValueError(f"Expected RGB images with 3 channels, got {C}.")

        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        else:
            imgs = imgs.float()
            if imgs.detach().amax() > 2:
                imgs = imgs / 255.0

        g_map, _ = compute_gmap_batch(images=imgs)

        pos = self.sample_pos(prob=g_map)

        pixel_xy = get_grid(h=H, w=W, device=imgs.device, dtype=imgs.dtype).reshape(
            -1, 2
        )
        xy = pixel_xy[pos].contiguous()

        inverse_scale = torch.full(
            (B, self.num_gaussian, 2),
            float(1 / self.init_scale),
            device=imgs.device,
            dtype=imgs.dtype,
        )
        rotation = torch.zeros(
            B, self.num_gaussian, 1, device=imgs.device, dtype=imgs.dtype
        )

        sample_grid = (xy * 2.0 - 1.0).view(B, self.num_gaussian, 1, 2)
        colors = F.grid_sample(imgs, sample_grid, align_corners=False)
        colors = colors.squeeze(-1).permute(0, 2, 1).contiguous()

        return xy, inverse_scale, rotation, colors

    def render(
        self,
        xy: torch.Tensor,
        inverse_scale: torch.Tensor,
        rot: torch.Tensor,
        colors: torch.Tensor,
        upsample_ratio: float = 1.0,
    ):
        assert xy.ndim == 3
        b, _, _ = xy.shape

        render_size = int(round(self.img_size * upsample_ratio))
        render_hw = (render_size, render_size)

        scale = upsample_ratio / inverse_scale

        out_imgs = []
        for i in range(b):
            gs_out = gaussiansplatting_render(
                means2d=xy[i],
                scales2d=scale[i],
                rotation=math.pi * rot[i],
                colors=colors[i],
                image_size=render_hw,
            )

            out_imgs.append(gs_out)

        return torch.cat(out_imgs, dim=0)

    def patchify_gaussian(self, gs_parameters: torch.Tensor) -> torch.Tensor:
        # gs shape: [B, num_gaussian, 8]
        B = gs_parameters.shape[0]
        gs_parameters = gs_parameters.reshape(B, -1, self.group_size, 8)
        gs_parameters = gs_parameters + self.inner_gs_pos

        return gs_parameters.reshape(B, -1, self.group_size * 8)

    def unpatchify_gaussian(self, gs: torch.Tensor) -> torch.Tensor:
        B = gs.shape[0]

        gs = gs.reshape(B, -1, 8)

        return gs

    def normalize_for_encoder(self, pixel_values: torch.Tensor) -> torch.Tensor:
        mean = self.encoder_mean.to(
            device=pixel_values.device, dtype=pixel_values.dtype
        )
        std = self.encoder_std.to(device=pixel_values.device, dtype=pixel_values.dtype)
        return (pixel_values - mean) / std

    def apply_gs_delta(
        self,
        init_xy: torch.Tensor,
        init_inv_scale: torch.Tensor,
        init_rot: torch.Tensor,
        init_colors: torch.Tensor,
        gs_delta: torch.Tensor,
    ):
        raw_xy = gs_delta[..., 0:2]
        raw_inv_scale = gs_delta[..., 2:4]
        raw_rot = gs_delta[..., 4:5]
        raw_colors = gs_delta[..., 5:8]

        eps = 1e-4
        max_shift = self.max_xy_shift / self.img_size
        xy = init_xy + max_shift * torch.tanh(raw_xy)
        xy = xy.clamp(0.0, 1.0)

        inv_scale_delta = self.inv_scale_delta_range * torch.tanh(raw_inv_scale)
        inverse_scale = init_inv_scale * torch.exp(inv_scale_delta)

        rot = init_rot + self.rotation_delta_range * torch.tanh(raw_rot)
        rot = rot.clamp(-1.0, 1.0)

        _inverse_sigmoid_colors = torch.logit(init_colors.clamp(eps, 1.0 - eps))
        colors = torch.sigmoid(
            _inverse_sigmoid_colors + self.color_delta_scale * raw_colors
        )

        return xy, inverse_scale, rot, colors

    def forward_inference(
        self,
        pixel_values: torch.Tensor,
        upsample_ratio: float = 1.0,
    ):
        init_xy, init_inv_scale, init_rot, init_colors = self.get_init_gaussians(
            imgs=pixel_values
        )

        # init gaussian
        init_gs_params = torch.cat(
            [init_xy, init_inv_scale, init_rot, init_colors], dim=-1
        )
        gs_tokens = self.patchify_gaussian(gs_parameters=init_gs_params)
        gs_tokens = self.gsembed_layer(gs_tokens)

        imgs = self.normalize_for_encoder(pixel_values)
        with torch.no_grad():
            img_features = self.encoder(imgs)

        b, seq_len, h = img_features.shape
        freq_cis = self.freq_cis.unsqueeze(0).expand(b, -1, -1, -1)

        gs_features = self.decoder_decoder(img_features, gs_tokens, freqs_cis=freq_cis)
        gs_features = self.to_gs(gs_features)
        gs_delta = self.unpatchify_gaussian(gs=gs_features)

        xy, inverse_scale, rot, colors = self.apply_gs_delta(
            init_xy=init_xy,
            init_inv_scale=init_inv_scale,
            init_rot=init_rot,
            init_colors=init_colors,
            gs_delta=gs_delta,
        )

        return self.render(
            xy=xy,
            inverse_scale=inverse_scale,
            rot=rot,
            colors=colors,
            upsample_ratio=upsample_ratio,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        gt_pixels: torch.Tensor = None,
        upsample_ratio: float = 1.0,
    ):
        if gt_pixels is None:
            gt_pixels = pixels

        recon_imgs = self.forward_inference(
            pixel_values=pixels,
            upsample_ratio=upsample_ratio,
        )

        loss = self.calc_loss(imgs=recon_imgs, gt_imgs=gt_pixels)

        return loss


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    gs_model = GaussianAutoEncoder(
        encoder_path="/home/kkkkea/.cache/dinov3", init_scale=1
    )
    gs_model.print_parameter_summary()

    img = Image.open("cat.png").convert("RGB")
    transforms = transforms.Compose([transforms.ToTensor()])
    img = transforms(img).unsqueeze(0)

    xy, inv_scale, rot, colors = gs_model.get_init_gaussians(imgs=img)

    print(xy.shape)
    print(inv_scale.shape)
    print(rot.shape)
    print(colors.shape)

    recon = gs_model.render(
        xy=xy.to("cuda"),
        inverse_scale=inv_scale.to("cuda"),
        rot=rot.to("cuda"),
        colors=colors.to("cuda"),
        upsample_ratio=1.0,
    )

    print(recon.shape)

    plt.imshow(recon[0].cpu().permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.show()
