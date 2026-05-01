from typing import Callable, Optional
from io import BytesIO
import warnings

from torch.utils.data.dataset import Dataset
from datasets import Image as HFImage
from datasets import load_dataset, load_from_disk
from PIL import Image as PILImage
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
        max_image_pixels: Optional[int] = PILImage.MAX_IMAGE_PIXELS,
    ):
        self.split = split
        self.lr_size = lr_size
        self.gt_size = self.lr_size * downsample_ratio
        self.max_image_pixels = max_image_pixels

        if local_path is not None:
            self.ds = load_from_disk(local_path)[split]
        else:
            self.ds = load_dataset("zhang0jhon/Aesthetic-4K", split=split)

        self.ds = self.ds.cast_column("image", HFImage(decode=False))

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
                warnings.simplefilter("ignore", PILImage.DecompressionBombWarning)
                width, height = self._get_image_size(item["image"])
        except (PILImage.DecompressionBombError, OSError, ValueError):
            return False

        if width < self.gt_size or height < self.gt_size:
            return False
        if self.max_image_pixels is not None and width * height > self.max_image_pixels:
            return False
        return True

    @staticmethod
    def _get_image_size(image_item) -> tuple[int, int]:
        image = Aesthetic4KDataset._open_image(image_item)
        try:
            return image.size
        finally:
            image.close()

    @staticmethod
    def _open_image(image_item):
        if isinstance(image_item, dict):
            if image_item.get("bytes") is not None:
                return PILImage.open(BytesIO(image_item["bytes"]))
            if image_item.get("path") is not None:
                return PILImage.open(image_item["path"])
        return image_item

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = self._open_image(item["image"]).convert("RGB")

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
