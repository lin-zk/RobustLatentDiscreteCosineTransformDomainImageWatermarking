# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

import os
from glob import glob
from PIL.Image import open
from torch import load
from torch.utils.data import Dataset

class CustomImageFolder(Dataset):
    """纯图片数据集（无标签）加载器"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.jpg")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.bmp")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.tiff")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.gif")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.PNG")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.JPEG")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.JPG")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.BMP")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.TIFF")))
        self.filenames.extend(glob(os.path.join(data_dir, "*.GIF")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = open(filename)
        if self.transform:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.filenames)

class PrecomputedVaeDataset(Dataset):
    """预推理的VAE数据集加载器（逐条加载版）"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sample_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.pt') or f.endswith('.pth')
        ])
        if not self.sample_files:
            raise RuntimeError("No .pt or .pth files found in {}".format(data_dir))

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.sample_files[idx])
        item = load(sample_path, weights_only=True)
        orig = item["orig"][0]
        latent = item["latent"][0]
        recon = item["recon"][0]
        return orig, latent, recon