"""
Data utilities for AdaRpGAN.

Supports:
  - CIFAR-10 (32×32, auto-download via torchvision)
  - CelebA   (auto-download via torchvision, center-crop to img_size)
  - Image folder (any directory of PNG/JPG images — for FFHQ, CelebA-HQ, etc.)

Usage
-----
    from utils.data import get_cifar10_loaders, get_celeba_loader, get_image_folder_loader
"""

from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


# ──────────────────────────────────────────────
# CIFAR-10 (original)
# ──────────────────────────────────────────────

def get_cifar10_loaders(
    batch_size: int    = 64,
    data_root:  str    = "./data",
    num_workers: int   = 4,
    pin_memory: bool   = True,
    drop_last:  bool   = True,
    augment:    bool   = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for CIFAR-10.

    Images are normalised to [-1, 1].
    Training set uses horizontal flip + random crop augmentation by default.
    """

    # Normalise to [-1, 1]  (mean=0.5, std=0.5 per channel)
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), norm])

    val_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_ds = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=val_tf)

    kw = dict(num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)

    print(f"[Data] CIFAR-10 loaded: {len(train_ds):,} train / {len(val_ds):,} val")
    print(f"       batch_size={batch_size}, augment={augment}")

    return train_loader, val_loader


# ──────────────────────────────────────────────
# CelebA (auto-download via torchvision)
# ──────────────────────────────────────────────

def get_celeba_loader(
    data_root:   str  = "./data",
    img_size:    int  = 64,
    batch_size:  int  = 32,
    num_workers: int  = 4,
    pin_memory:  bool = True,
    drop_last:   bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for CelebA (train split).

    Images are center-cropped to 178×178 (face region), resized to
    img_size × img_size, and normalised to [-1, 1].
    """
    tf = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    ds = datasets.CelebA(root=data_root, split="train", download=True, transform=tf)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
    )

    print(f"[Data] CelebA loaded: {len(ds):,} images, img_size={img_size}")
    print(f"       batch_size={batch_size}")

    return loader


# ──────────────────────────────────────────────
# Generic image folder (for FFHQ, CelebA-HQ, etc.)
# ──────────────────────────────────────────────

class FlatImageFolder(Dataset):
    """Reads all PNG/JPG images from a flat directory (no class subfolders)."""

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, root: str, transform=None):
        self.root      = root
        self.transform = transform
        self.paths     = sorted([
            os.path.join(root, f) for f in os.listdir(root)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ])
        assert len(self.paths) > 0, f"No images found in {root}"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_image_folder_loader(
    data_dir:    str,
    img_size:    int  = 256,
    batch_size:  int  = 16,
    num_workers: int  = 4,
    pin_memory:  bool = True,
    drop_last:   bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for a flat directory of images.
    Useful for FFHQ-256, CelebA-HQ-256, etc.
    """
    tf = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    ds = FlatImageFolder(data_dir, transform=tf)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
    )

    print(f"[Data] Image folder loaded: {len(ds):,} images from {data_dir}")
    print(f"       img_size={img_size}, batch_size={batch_size}")

    return loader


def get_infinite_loader(loader: DataLoader):
    """Wrap a DataLoader in an infinite generator (useful for fixed-step training)."""
    while True:
        yield from loader
