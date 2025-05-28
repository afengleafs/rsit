import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


class MNIST(Dataset):
    """
    Tiny wrapper around torchvision.datasets.MNIST that
    * optionally maps images from [0,1] to [-1,1]
    * returns (image, label) where image is a float32 tensor
    """

    def __init__(self,
                 root: str | os.PathLike = "./mnist",
                 train: bool = True,
                 download: bool = True,
                 to_minus1_1: bool = True) -> None:
        self.to_minus1_1 = to_minus1_1

        # ----- build transform pipeline -----
        transforms_list = [T.ToTensor()]          # PIL -> float32 tensor in [0,1]
        if to_minus1_1:
            transforms_list.append(T.Lambda(lambda x: x * 2.0 - 1.0))  # -> [-1,1]

        transform = T.Compose(transforms_list)

        self.ds = torchvision.datasets.MNIST(
            root=Path(root),
            train=train,
            download=download,
            transform=transform
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # torchvision already returns (tensor, label) after transform
        return self.ds[idx]


# ----------------------------------------------------------------------
# Quick demo / sanity‑check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ds = MNIST(train=True, to_minus1_1=True)
    img, label = ds[0]  # img ∈ [-1,1] shape=[1,28,28]

    print("Tensor shape:", img.shape)
    print("Label:", label)
    print("Value range: [{:.3f}, {:.3f}]".format(img.min().item(), img.max().item()))

    # Convert back to [0,1] for display
    img_vis = (img + 1) / 2 if ds.to_minus1_1 else img
    plt.imshow(img_vis.squeeze(0), cmap="gray")
    plt.title(f"MNIST sample (label={label})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Save example
    plt.imsave("test.png", img_vis.squeeze(0).numpy(), cmap="gray")
    print("Saved: test.png")
