import torch
import torch.nn.functional as F


def compute_gmap_batch(images, gamma=1.0, eps=1e-8):
    """
    images: Tensor, shape [B, C, H, W], range usually [0, 1]
    return:
        gmap: [B, H*W]，每张图归一化后平方
        g_norm: [B, H, W]，未平方前的梯度图
    """

    # gamma correction
    x = images.clamp(min=0.0) ** (1.0 / gamma)

    B, C, H, W = x.shape

    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    # 对每个 channel 独立卷积
    weight_y = sobel_y.repeat(C, 1, 1, 1)
    weight_x = sobel_x.repeat(C, 1, 1, 1)

    gy = F.conv2d(x, weight_y, padding=1, groups=C)  # [B, C, H, W]
    gx = F.conv2d(x, weight_x, padding=1, groups=C)

    # 对 channel 做 L2 norm，等价于你原来的 norm(np.stack(...), axis=0)
    gy = torch.linalg.vector_norm(gy, ord=2, dim=1)  # [B, H, W]
    gx = torch.linalg.vector_norm(gx, ord=2, dim=1)  # [B, H, W]

    g_norm = torch.sqrt(gy**2 + gx**2)  # [B, H, W]

    # 每张图单独归一化
    max_val = g_norm.flatten(1).amax(dim=1).view(B, 1, 1)
    g_norm = g_norm / (max_val + eps)

    # flatten 后平方
    gmap = g_norm.flatten(1) ** 2  # [B, H*W]

    return gmap, g_norm
