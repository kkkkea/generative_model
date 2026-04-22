from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DIV2KDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        target_size: int = 224,
        val_limit: int = 32,
        transform: Optional[Callable] = None,
        return_path: bool = False,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ):
        """
        Args:
            root: DIV2K 根目录，例如:
                  root/
                    DIV2K_train_HR/
                    DIV2K_valid_HR/
            split: "train" 或 "valid"
            transform: 你自定义的图像变换
            return_path: 是否额外返回图片路径
            extensions: 支持的图片后缀
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_path = return_path
        self.extensions = tuple(e.lower() for e in extensions)

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop((target_size, target_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

        if split == "train":
            self.img_dir = self.root / "DIV2K_train_HR"
        elif split == "valid":
            self.img_dir = self.root / "DIV2K_valid_HR"
        else:
            raise ValueError(f"split must be 'train' or 'valid', got {split}")

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        self.image_paths = sorted(
            [
                p
                for p in self.img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.extensions
            ]
        )

        if self.split == "valid":
            self.image_paths = self.image_paths[:val_limit]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return {
                "image": image,
                "path": str(img_path),
            }

        return image, None, None
