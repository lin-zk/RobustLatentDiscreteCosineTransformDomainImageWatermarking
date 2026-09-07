# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

import argparse
from datetime import datetime
import os
import sys

class TrainOptions():
	def __init__(self, stage):
		self.parser = argparse.ArgumentParser(description="Options for training models.")
		self.initialized = False
		self.stage = stage

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
			"--data_dir", 
			type=str, 
			required=True, 
			help="Directory with image dataset."
		)
		self.parser.add_argument(
			"--diffusion_dir",
			type=str,
			default="stabilityai/stable-diffusion-xl-base-1.0",
			help="Path for diffusion root dir.",
		)
		self.parser.add_argument(
			"--output_dir", 
			type=str, 
			required=True, 
			help="Directory to save results to."
		)
		self.parser.add_argument(
			"--checkpoint_dir",
			type=str,
			default=None,
			help="Directory to load checkpoint from.",
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
			"--epochs", 
			type=int, 
			default=20, 
			help="Number of training epochs (Only Used at Stage2)."
		)
		self.parser.add_argument(
			"--batch_size", 
			type=int, 
			default=64, 
			help="Batch size."
		)
		self.parser.add_argument(
			"--lr", 
			type=float, 
			default=1e-4, 
			help="Learning rate."
		)
		self.parser.add_argument(
			"--cuda", 
			type=int, 
			default=0,
			help="CUDA device index to use."
		)
		self.parser.add_argument(
			"--noise_std",
			type=float,
			default=0.40,
			help="Standard deviation of Gaussian noise added to the latent.",
		)
		self.parser.add_argument(
			"--dropout_p",
			type=float,
			default=0.20,
			help="Dropout probability for the decoder.",
		)
		self.parser.add_argument(
			"--l2_loss_await",
			help="Train without L2 loss for the first x iterations",
			type=int,
			default=500,
		)
		self.parser.add_argument(
			"--l2_loss_weight",
			type=float,
			default=100,
			help="L2 loss weight for image fidelity.",
		)
		self.parser.add_argument(
			"--l2_loss_ramp",
			type=int,
			default=3000,
			help="Linearly increase L2 loss weight over x iterations.",
		)
		self.parser.add_argument(
			"--BCE_loss_weight",
			type=float,
			default=10,
			help="BCE loss weight for fingerprint reconstruction.",
		)
		self.parser.add_argument(
			"--energyblance_loss_weight",
			type=float,
			default=10000,
			help="Energy loss weight for fingerprint reconstruction.",
	    )
		
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

        # 当前主程序的文件名
		self.opt.main_code = os.path.basename(sys.argv[0])

		self.opt.time = datetime.now().strftime('%Y%m%d-%H%M%S')
		self.opt.expr_dir = os.path.join(self.opt.output_dir, f"Stage{self.stage}", self.opt.time)  # 创建实验目录

		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')
		os.makedirs(self.opt.expr_dir, exist_ok=True)
		file_name = os.path.join(self.opt.expr_dir, 'train_opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))  # 写入本次参数
			opt_file.write('-------------- End ----------------\n')
		return self.opt