from typing import Callable, Optional
import warnings

from torch.utils.data.dataset import Dataset
from datasets import load_dataset, load_from_disk
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class Aesthetic4KDataset(Dataset):
    def __init__(
        self,
        split="train",
        lr_size: int = 256,
        downsample_ratio: int = 8,
        local_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        val_limit: Optional[int] = None,
        skip_small_images: bool = True,
        max_image_pixels: Optional[int] = Image.MAX_IMAGE_PIXELS,
    ):
        self.split = split
        self.lr_size = lr_size
        self.gt_size = self.lr_size * downsample_ratio
        self.max_image_pixels = max_image_pixels

        if local_path is not None:
            self.ds = load_from_disk(local_path)[split]
        else:
            self.ds = load_dataset("zhang0jhon/Aesthetic-4K", split=split)

        if skip_small_images or max_image_pixels is not None:
            self.ds = self.ds.filter(
                self._is_valid_image_size,
                desc=(
                    f"Filtering images smaller than {self.gt_size}x{self.gt_size} "
                    "or too large"
                ),
            )

        if val_limit is not None and split != "train":
            self.ds = self.ds.select(range(min(val_limit, len(self.ds))))

        if transform is None:
            crop = (
                transforms.RandomCrop((self.gt_size, self.gt_size))
                if split == "train"
                else transforms.CenterCrop((self.gt_size, self.gt_size))
            )
            self.transform = transforms.Compose([crop, transforms.ToTensor()])
        else:
            self.transform = transform

        self.downsample = transforms.Resize(
            (self.lr_size, self.lr_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _is_valid_image_size(self, item) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                image = item["image"]
                width, height = image.size
        except Image.DecompressionBombError:
            return False

        if width < self.gt_size or height < self.gt_size:
            return False
        if self.max_image_pixels is not None and width * height > self.max_image_pixels:
            return False
        return True

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"].convert("RGB")

        gt = self.transform(img) if self.transform is not None else img
        lr = self.downsample(gt)

        meta = {}

        return lr, gt, meta


if __name__ == "__main__":
    dataset = Aesthetic4KDataset()

    lr, gt, meta = dataset[0]

    print(lr.shape)
    print(gt.shape)
    print(meta)
