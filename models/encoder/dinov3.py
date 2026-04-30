import torch
import torch.nn as nn
from transformers import DINOv3ViTModel, DINOv3ViTImageProcessor


class Dinov3withNorm(nn.Module):
    def __init__(self, dinov3_path: str, normalize: bool = True):
        super().__init__()

        try:
            self.encoder = DINOv3ViTModel.from_pretrained(
                dinov3_path, local_files_only=True
            )
        except (OSError, ValueError, AttributeError):
            self.encoder = DINOv3ViTModel.from_pretrained(
                dinov3_path, local_files_only=False
            )

        self.encoder.requires_grad_(False)

        if normalize:
            self.encoder.norm.elementwise_affine = False
            self.encoder.norm.weight = None
            self.encoder.norm.bias = None

        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def dinov3_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)

        return x.last_hidden_state[:, 5:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov3_forward(x)


def dinov3_test():
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import torch.nn.functional as F

    model_name = "/home/kkkkea/.cache/dinov3"
    device = "cuda"

    processor = DINOv3ViTImageProcessor.from_pretrained(model_name)
    model = Dinov3withNorm(dinov3_path=model_name).to(device)
    print(processor)

    img = Image.open("cat.png").convert("RGB")
    inputs = processor(
        images=img, return_tensors="pt", size={"height": 256, "width": 256}
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    print(outputs.shape)

    features_flat = outputs.squeeze(0).cpu().numpy()
    # features_flat = features_flat.reshape(16, 16, 16, 16, 3)
    # features_flat = torch.einsum("hwpqc->chpwq", features_flat)
    # features_flat = features_flat.reshape(3, 256, 256)
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat)  # 变为 [N, 3]
    pca_features = (pca_features - pca_features.min()) / (
        pca_features.max() - pca_features.min()
    )

    # 6. 还原为二维网格
    # 计算当前的网格尺寸 (取决于输入分辨率)
    h_feat = inputs["pixel_values"].shape[2] // 16
    w_feat = inputs["pixel_values"].shape[3] // 16
    pca_img = pca_features.reshape(h_feat, w_feat, 3)

    vis_tensor = torch.from_numpy(pca_img).permute(2, 0, 1).unsqueeze(0)
    vis_up = F.interpolate(
        vis_tensor, size=(256, 256), mode="bilinear", align_corners=False
    )
    pca_img = vis_up[0].permute(1, 2, 0).numpy()

    # 7. 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("DINOv3 PCA Visualization")
    plt.imshow(pca_img)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    dinov3_test()
