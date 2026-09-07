# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms

def process_images(folder1, folder2, output_folder, IMAGE_RESOLUTION=512):
	transform = transforms.Compose([
		transforms.Resize(IMAGE_RESOLUTION),
		transforms.CenterCrop(IMAGE_RESOLUTION),
		transforms.Lambda(lambda img: img.convert("RGB"))
	])

	different_folder = os.path.join(output_folder, "Different")
	os.makedirs(different_folder, exist_ok=True)

	# 获取两个文件夹中的文件列表并过滤非图片文件
	image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".GIF", ".TIFF"}
	files1 = {f for f in os.listdir(folder1) if os.path.splitext(f)[1].lower() in image_extensions}
	files2 = {f for f in os.listdir(folder2) if os.path.splitext(f)[1].lower() in image_extensions}

	# 创建水印图片文件夹文件名的映射（最后一个_后的部分作为键）
	files2_mapping = {os.path.splitext(f.rsplit("_", 1)[-1])[0]: f for f in files2}

	# 找到匹配的文件
	common_files = [(f1, files2_mapping[os.path.splitext(f1)[0]]) for f1 in files1 if os.path.splitext(f1)[0] in files2_mapping]

	# 过滤非图片文件
	image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
	common_files = [(f1, f2) for f1, f2 in common_files if os.path.splitext(f1)[1].lower() in image_extensions and os.path.splitext(f2)[1].lower() in image_extensions]

	for file1_name, file2_name in tqdm(common_files, desc="Processing images"):
		try:
			# 打开图片
			img1_path = os.path.join(folder1, file1_name)
			img2_path = os.path.join(folder2, file2_name)
			img1 = Image.open(img1_path)
			img2 = Image.open(img2_path)

			# 应用变换
			img1 = transform(img1)
			img2 = transform(img2)

			# 转换图片为 numpy 数组
			arr1 = np.array(img1, dtype=np.int16)
			arr2 = np.array(img2, dtype=np.int16)

			# 计算差异
			diff = np.clip(abs(arr1 - arr2) * 10.0, 0, 255).astype(np.uint8)

			# 保存差异图片
			diff_image = Image.fromarray(diff, "RGB")

			# 拼接图片
			combined_width = img1.width + diff_image.width + img2.width
			combined_image = Image.new("RGB", (combined_width, img1.height))
			combined_image.paste(img1, (0, 0))
			combined_image.paste(diff_image, (img1.width, 0))
			combined_image.paste(img2, (img1.width + diff_image.width, 0))

			# 保存拼接后的图片
			combined_image.save(os.path.join(different_folder, file1_name))
		except Exception as e:
			print(f"Error processing {file1_name} and {file2_name}: {e}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Process images and find differences.")
	parser.add_argument(
		"--input_o", 
		type=str, 
		help="Path to the original images(Folder)."
	)
	parser.add_argument(
		"--input_w", 
		type=str, 
		help="Path to the watermarked images(Folder)."
	)
	parser.add_argument(
		"--image_resolution", 
		type=int, 
		default=512, 
		help="Resolution to resize and crop images."
	)
	args = parser.parse_args()

	process_images(args.input_o, args.input_w, args.input_w, args.image_resolution)
