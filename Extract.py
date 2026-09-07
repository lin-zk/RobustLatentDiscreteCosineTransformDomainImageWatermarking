# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

### 导入基本包
import os
from os.path import join
from time import time
from tqdm import tqdm
### 导入torch包
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
### 导入自定义包
from diffusers import AutoencoderKL
from model import StegaStamp, Diffusion
from options.extract_args import ExtractOptions
from data.dataloader import CustomImageFolder

args =  ExtractOptions().parse()  # 解析参数
if args.cuda == -1:
	device = torch.device("cpu")  # 使用CPU
else:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
	if args.cuda >= torch.cuda.device_count():
		raise RuntimeError(f"Invalid CUDA device index: {args.cuda}.")
	device = torch.device(f"cuda:{args.cuda}")  # 使用指定的GPU
	torch.cuda.set_device(device)  # 设置当前GPU
os.environ["PYTHONHASHSEED"] = str(args.seed)  # 设置Python的随机种子
torch.manual_seed(args.seed)  # 设置Torch的随机种子
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(args.seed)  # 设置GPU的随机种子

OUTPUT_PATH = args.expr_dir
if not os.path.exists(OUTPUT_PATH):
	os.makedirs(OUTPUT_PATH)

IMAGE_RESOLUTION = args.image_resolution
IMAGE_CHANNELS = args.image_channels
SECRET_SIZE = args.fingerprint_length

def load_data():
	transform = transforms.Compose(
		[
			transforms.Resize(IMAGE_RESOLUTION),
			transforms.CenterCrop(IMAGE_RESOLUTION),
			transforms.Lambda(lambda img: img.convert("RGB")),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		]
	)

	time_start = time()
	print(f"Loading image folder {args.data_dir} ...")
	imageset = CustomImageFolder(args.data_dir, transform=transform)
	print(f"Finished. Loading took {time() - time_start:.2f}s")
	return imageset

def main():
	print("loading VAE model...")
	vae = AutoencoderKL.from_pretrained(args.diffusion_dir, subfolder="vae")  # 载入vae
	vae.to(device), vae.eval()
	for param in vae.parameters():
		param.requires_grad = False  # 冻结参数
	print("VAE loaded.")

	# 加载数据集
	imageset = load_data()
	dataLoader = DataLoader(imageset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	# 创建模型
	decoder = StegaStamp.StegaStampDecoder(SECRET_SIZE).to(device)  # 实例化提取器

	# 加载模型
	print(f"Loading model from {args.checkpoint_dir}...")
	decoder.load_state_dict(torch.load(join(args.checkpoint_dir, "decoder.ckpt"))["state_dict"])
	decoder.eval()
	for param in decoder.parameters():
		param.requires_grad = False
	print("Model loaded successfully.")

	# 初始化累积变量
	total_acc = 0.0
	total_sr = 0.0
	num_images = 0

	# 保存水印图像并计算指标
	with torch.no_grad():
		print("Extracting...")
		results = []  # 用于保存每张图片的结果
		for _, (images, filenames) in enumerate(tqdm(dataLoader, desc="Extracting", unit="batch")):

			fingerprinted_images = images.to(device)
			fingerprinted_image_latent = Diffusion.encode_image(vae, fingerprinted_images)
			# 提取水印
			decoder_output = decoder(fingerprinted_image_latent)
			fingerprints_predicted = (decoder_output > 0).float()

			# 保存水印并计算指标
			for i, _ in enumerate(fingerprinted_images):
				original_filename = filenames[i]
				original_basename = os.path.basename(original_filename)
				name, _ = os.path.splitext(original_basename)

				# 从文件名中提取水印十六进制串
				watermark_hex = name.split("_")[-1] 
				watermark_bin = ''.join(f"{int(c, 16):04b}" for c in watermark_hex) 
				watermark_tensor = torch.tensor([int(b) for b in watermark_bin], dtype=torch.float, device=device) 

				# 计算比特准确率
				acc_value = 100.0 * (1.0 - torch.mean(torch.abs(watermark_tensor - fingerprints_predicted[i])))

				# 判断是否成功
				if acc_value >= args.acc_threshold:
					success = "Success"
					total_sr += 1
				else:
					success = "Fail"

				# 累积指标
				total_acc += acc_value.item()
				num_images += 1

				# 转换预测的水印为十六进制字符串
				fingerprints_predicted_bin = ''.join(map(str, fingerprints_predicted[i].cpu().numpy().astype(int)))
				fingerprints_predicted_hex = ''.join(f"{int(fingerprints_predicted_bin[j:j+4], 2):X}" for j in range(0, len(fingerprints_predicted_bin), 4))

				# 保存结果
				results.append(f"{name}: {fingerprints_predicted_hex} Watermark: {watermark_hex} Acc.: {acc_value:.1f}% {success}")

		# 计算平均值
		avg_acc = total_acc / num_images
		avg_sr = total_sr / num_images

		# 打印结果
		print(f"Total Images Processed: {num_images}")
		print(f"Average Bitwise Accuracy: {avg_acc:.1f}%")
		print(f"Average Success Rate: {100.0 * avg_sr:.1f}%")

		# 保存结果到文本文件
		results_path = os.path.join(OUTPUT_PATH, "extracting_metrics.txt")
		with open(results_path, "w") as f:
			f.write("Extracting Metrics:\n")
			f.write(f"Total Images Processed: {num_images}\n")
			f.write(f"Average Bitwise Accuracy: {avg_acc:.1f}%\n")
			f.write(f"Average Success Rate: {100.0 * avg_sr:.1f}%\n\n")
			f.write("Details:\n")
			f.write("\n".join(results))

	print("Extracting completed.")

if __name__ == "__main__":
    main()
    print("Extracting completed.")
