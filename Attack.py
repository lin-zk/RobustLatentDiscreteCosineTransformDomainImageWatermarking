# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Reference : https://github.com/xuandongzhao/watermarkattacker
"""
@inproceedings{zhao2024invisible,
  title={Invisible Image Watermarks Are Provably Removable Using Generative AI},
  author={Zhao, Xuandong and Zhang, Kexun and Su, Zihao and Vasan, Saastha and Grishchenko, Ilya and Kruegel, Christopher and Vigna, Giovanni and Wang, Yu-Xiang and Li, Lei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
"""
# @Thanks to : Copilot for assistance

import os
from pathlib import Path
import argparse
from attack.regen_pipe import ReSDPipeline
from attack.wmattacker import *


def apply_attacks(input_dir, output_dir, cuda, diffusion_dir):
	# 加载扩散模型
	device = f"cuda:{cuda}" if cuda >= 0 else "cpu"
	pipe = ReSDPipeline.from_pretrained(
		diffusion_dir,
		torch_dtype=torch.float16, revision="fp16"
	)
	pipe.set_progress_bar_config(disable=True)
	pipe.to(device)
	print('Finished loading model')
	attacks = {
		
		'VAE_B_images': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
		'VAE_C_images': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
		'Diffusion_images': DiffWMAttacker(pipe, batch_size=5, noise_step=60, captions={}),

		"Brightness_images": BrightnessAttacker(brightness=0.5),
		"Contrast_images": ContrastAttacker(contrast=0.5),
		"JPEG_images": JPEGAttacker(quality=50),
		"G-Blur_images": GaussianBlurAttacker(kernel_size=5, sigma=1),
		"G-Noise_images": GaussianNoiseAttacker(std=0.05),
		'Scale_images': ScaleAttacker(scale=0.5),
		# "BM3D_images": BM3DAttacker(),  # 默认就是std = 0.1 了，耗时很长不使用了
		
	}
	for subfolder in attacks.keys():
		Path(os.path.join(output_dir, subfolder)).mkdir(parents=True, exist_ok=True)
	files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    # 支持的图片扩展名
	img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    # 过滤非图片文件
	img_files = [f for f in files if f.lower().endswith(img_exts)]
    # 对每种攻击批量处理
	for subfolder, attacker in attacks.items():
		input_paths = [os.path.join(input_dir, f) for f in img_files]
		output_paths = [os.path.join(output_dir, subfolder, f) for f in img_files]
		try:
			attacker.attack(input_paths, output_paths)
			print(f"Successfully processed attack: {subfolder}")
		except Exception as e:
			print(f"Error processing {subfolder}: {e}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Apply attacks to images.")
	parser.add_argument(
		"--input_dir", 
		type=str, 
		help="Path to the input directory containing images."
	)
	parser.add_argument(
		"--cuda", 
		type=int, 
		default=0, 
		help="CUDA device index to use."
	)
	parser.add_argument(
		"--diffusion_dir", 
		type=str, 
		default="stabilityai/stable-diffusion-2-1", 
		help="Path to the diffusion model directory."
	)
	args = parser.parse_args()
	
	apply_attacks(args.input_dir, args.input_dir, args.cuda, args.diffusion_dir)