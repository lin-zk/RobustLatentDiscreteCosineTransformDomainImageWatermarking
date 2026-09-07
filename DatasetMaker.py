# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from model import Diffusion

def is_image_file(filename):
    IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

def preprocess_image(img_path, image_resolution):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_resolution),
        transforms.CenterCrop(image_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(img)

def check_and_make(dataset_dir, save_dir, vae, image_resolution, device):
    os.makedirs(save_dir, exist_ok=True)
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: {split_dir} not found, skipping.")
            continue
        img_files = [f for f in os.listdir(split_dir) if is_image_file(f)]
        split_save_dir = os.path.join(save_dir, split)
        os.makedirs(split_save_dir, exist_ok=True)
        for fname in tqdm(img_files, desc=f"Processing {split}"):
            img_path = os.path.join(split_dir, fname)
            img_tensor = preprocess_image(img_path, image_resolution).unsqueeze(0).to(device)
            with torch.no_grad():
                latent = Diffusion.encode_image(vae, img_tensor)
                recon = Diffusion.decode_latent(vae, latent)
            sample = {
                "orig": img_tensor.cpu(),
                "latent": latent.cpu(),
                "recon": recon.cpu()
            }
            base_name = os.path.splitext(fname)[0]
            torch.save(sample, os.path.join(split_save_dir, f"{base_name}.pt"))
             # 主动释放显存和内存
            del img_tensor, latent, recon, sample

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script for preprocessing image dataset.")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        required=True, 
        help="Root directory of the original image dataset."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        required=True, 
        help="Directory to save the inferred tensors."
    )
    parser.add_argument(
        "--diffusion_dir", 
        type=str, 
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Directory containing the VAE model."
    )
    parser.add_argument(
        "--image_resolution", 
        type=int, 
        default=512,
        help="Resolution to resize images to."
    )
    parser.add_argument(
        "--cuda", 
        type=int, 
        default=0,
        help="CUDA device index to use."
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.diffusion_dir, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    check_and_make(args.dataset_dir, args.save_dir, vae, args.image_resolution, device)
    print("Dataset preprocessing finished.")