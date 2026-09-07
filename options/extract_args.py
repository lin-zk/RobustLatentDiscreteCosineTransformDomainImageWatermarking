# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

import argparse
import os
import sys
from datetime import datetime

class ExtractOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(description="Options for extracting watermark from images.")
		self.initialized = False

	def initialize(self):
		self.parser.add_argument(
			"--seed", 
			type=int, 
			default=333333, 
			help="Random seed for initialization."
		)
		self.parser.add_argument(
			"--num_workers",
			type=int,
			default=2,
			help="Number of workers for data loading.",
		)
		self.parser.add_argument(
			"--acc_threshold",
			type=float,
			default=65.625,
			help="Threshold for extraction bitwise accuracy.",
		)
		self.parser.add_argument(
			"--data_dir", 
			type=str, 
			required=True, 
			help="Directory with image dataset."
		)
		self.parser.add_argument(
			"--checkpoint_dir",
			type=str,
			required=True,
			help="Directory to load checkpoint from.",
		)
		self.parser.add_argument(
			"--diffusion_dir",
			type=str,
			default="stabilityai/stable-diffusion-xl-base-1.0",
			help="Path for diffusion root dir.",
		)
		self.parser.add_argument(
			"--fingerprint_length",
			type=int,
			default=64,
			help="Number of bits in the fingerprint.",
		)
		self.parser.add_argument(
			"--image_resolution",
			type=int,
			default=512,
			help="Height and width of square images.",
		)
		self.parser.add_argument(
			"--image_channels",
			type=int,
			default=3,
			help="Channels of images.",
		)
		self.parser.add_argument(
			"--batch_size", 
			type=int, 
			default=16, 
			help="Batch size."
		)
		self.parser.add_argument(
			"--cuda", 
			type=int, 
			default=0,
			help="CUDA device index to use."
		)

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		# 当前主程序的文件名
		self.opt.main_code = os.path.basename(sys.argv[0])

		self.opt.output_dir = self.opt.data_dir  # 输入目录和输出目录相同
		self.opt.time = datetime.now().strftime('%Y年%m月%d日%H时%M分%S秒')
		self.opt.expr_dir = os.path.join(self.opt.output_dir)  # 创建输出目录

		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')
		os.makedirs(self.opt.expr_dir, exist_ok=True)
		file_name = os.path.join(self.opt.expr_dir, 'extract_opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))  # 写入本次参数
			opt_file.write('-------------- End ----------------\n')
		return self.opt