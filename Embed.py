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
import torch_dct
### 导入自定义包
from diffusers import AutoencoderKL
from model import StegaStamp, Diffusion
from options.embed_args import EmbedOptions
from data.dataloader import CustomImageFolder
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import peak_signal_noise_ratio as peak_signal_noise_ratio_tensor
from torchmetrics.functional import structural_similarity_index_measure as structural_similarity_tensor
import lpips

args = EmbedOptions().parse()  # 解析参数
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

def get_fingerprints(fingerprint_length):
	if args.fingerprint_mode == "fix":
		z = torch.zeros((1, fingerprint_length), dtype=torch.float).random_(0, 2)  # 已经设置过随机种子
	elif args.fingerprint_mode == "certain":
		if args.fingerprint_content is None:
			raise ValueError("Fingerprint content must be provided for 'certain' mode.")
		if args.fingerprint_content.endswith(".txt"):
			if os.path.exists(args.fingerprint_content):
				with open(args.fingerprint_content, "r") as f:
					content = f.read().strip()
					if not all(c in "0123456789ABCDEF" for c in content.upper()):
						raise ValueError("File contains invalid characters. Only hexadecimal (0-9, A-F) is allowed.")
					binary_content = ''.join(f"{int(c, 16):04b}" for c in content.upper())
					z = torch.tensor([int(bit) for bit in binary_content]).float().unsqueeze(0)
			else:
				raise ValueError(f"File {args.fingerprint_content} does not exist.")
		else:
			content = args.fingerprint_content.strip()
			if not all(c in "0123456789ABCDEF" for c in content.upper()):
				raise ValueError("Provided fingerprint content contains invalid characters. Only hexadecimal (0-9, A-F) is allowed.")
			binary_content = ''.join(f"{int(c, 16):04b}" for c in content.upper())
			z = torch.tensor([int(bit) for bit in binary_content]).float().unsqueeze(0)
	return z

def generate_random_fingerprints(fingerprint_length, batch_size, offset):
	#更变随机种子
	current_seed = torch.initial_seed()
	new_seed = current_seed + offset
	torch.manual_seed(new_seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(new_seed)
	#生成随机水印
	z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
	return z

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
	print("loading diffusion model...")
	vae = AutoencoderKL.from_pretrained(args.diffusion_dir, subfolder="vae")  # 载入vae
	vae.to(device)
	vae.eval()
	for param in vae.parameters():
		param.requires_grad = False  # 冻结参数
	print("Diffusion model loaded.")

	# 加载数据集
	imageset = load_data()
	dataLoader = DataLoader(imageset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	# 创建模型
	encoder = StegaStamp.StegaStampEncoder(SECRET_SIZE).to(device)  # 实例化频域嵌入器

	# 加载模型
	print(f"Loading model from {args.checkpoint_dir}...")
	encoder.load_state_dict(torch.load(join(args.checkpoint_dir, "encoder.ckpt"))["state_dict"])
	encoder.eval()
	for param in encoder.parameters():
		param.requires_grad = False  # 冻结参数
	print("Model loaded successfully.")

	if args.fingerprint_mode == "fix" or args.fingerprint_mode == "certain":
		fingerprints = get_fingerprints(SECRET_SIZE)  # 预先生成水印

	# 初始化 LPIPS 模型（使用 VGG16）
	lpips_loss = lpips.LPIPS(net='vgg').to(device)

	# 初始化累积变量
	total_psnr = 0.0
	total_ssim = 0.0
	total_lpips = 0.0
	total_psnr_tensor = 0.0
	total_ssim_tensor = 0.0
	num_images = 0

	# 保存水印图像并计算指标
	with torch.no_grad():
		print("Embedding...")
		for batch_idx, (images, filenames) in enumerate(tqdm(dataLoader, desc="Embedding", unit="batch")):
			batch_size = min(args.batch_size, images.size(0))
			if args.fingerprint_mode == "random":
				fingerprints = generate_random_fingerprints(SECRET_SIZE, batch_size, batch_idx)  # 生成随机水印
			else:
				fingerprints = fingerprints.repeat(batch_size, 1)

			clean_images = images.to(device)
			clean_latent = Diffusion.encode_image(vae, clean_images)
			fingerprints = fingerprints.to(device)

			# DCT分支 残差
			clean_latent_dct = torch_dct.dct_2d(clean_latent, norm="ortho")
			fingerprinted_latent_dct_residual = encoder(fingerprints, clean_latent_dct)
			fingerprinted_latent_residual = torch_dct.idct_2d(fingerprinted_latent_dct_residual, norm="ortho")
			fingerprinted_latent = fingerprinted_latent_residual + clean_latent

			# 输出水印图片
			fingerprinted_images_residual = Diffusion.decode_latent(vae, fingerprinted_latent) - Diffusion.decode_latent(vae, clean_latent)
			fingerprinted_images = fingerprinted_images_residual + clean_images

			# 保存水印图像并计算指标
			for i, fingerprinted_image in enumerate(fingerprinted_images):
				original_filename = filenames[i]
				original_basename = os.path.basename(original_filename)
				name, _ = os.path.splitext(original_basename)
				fingerprint_str = ''.join(f"{int(''.join(map(str, fingerprints[i].cpu().numpy().astype(int)[j:j+4])), 2):X}" for j in range(0, len(fingerprints[i]), 4))
				output_filename = f"{name}_{fingerprint_str}.png"
				output_path = os.path.join(OUTPUT_PATH, output_filename)
				transforms.ToPILImage()(((fingerprinted_image.cpu().clamp(-1, 1) + 1) * 127.5).byte()).save(output_path)

				# 转换 clean_images 和 fingerprinted_images 到 [0, 255] 且为整型
				clean_image_np = ((clean_images[i].cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy().transpose(1, 2, 0)  # HWC 格式
				fingerprinted_image_np = ((fingerprinted_image.cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy().transpose(1, 2, 0)  # HWC 格式

				# 计算 PSNR
				psnr_value = psnr(clean_image_np, fingerprinted_image_np, data_range=255)

				# 计算 SSIM
				ssim_value = ssim(clean_image_np, fingerprinted_image_np, data_range=255, channel_axis=-1)

				# 计算 LPIPS
				clean_image_tensor = ((clean_images[i].unsqueeze(0).clamp(-1, 1) + 1) * 0.5).to(device)  # 转换到 [0, 1]
				fingerprinted_image_tensor = ((fingerprinted_image.unsqueeze(0).clamp(-1, 1) + 1) * 0.5).to(device)  # 转换到 [0, 1]
				lpips_value = lpips_loss(clean_image_tensor, fingerprinted_image_tensor).item()

				psnr_tensor_value = peak_signal_noise_ratio_tensor(clean_image_tensor, fingerprinted_image_tensor, data_range=1.0).item()
				ssim_tensor_value = structural_similarity_tensor(clean_image_tensor, fingerprinted_image_tensor, data_range=1.0).item()

				# 累积指标
				total_psnr += psnr_value
				total_ssim += ssim_value
				total_lpips += lpips_value
				total_psnr_tensor += psnr_tensor_value
				total_ssim_tensor += ssim_tensor_value
				num_images += 1

	# 计算平均值
	avg_psnr = total_psnr / num_images
	avg_ssim = total_ssim / num_images
	avg_lpips = total_lpips / num_images
	avg_psnr_tensor = total_psnr_tensor / num_images
	avg_ssim_tensor = total_ssim_tensor / num_images

	# 打印结果
	print(f"Total Images Processed: {num_images}")
	print(f"Average PSNR: {avg_psnr:.2f}")
	print(f"Average SSIM: {avg_ssim:.2f}")
	print(f"Average LPIPS: {avg_lpips:.2f}")
	print(f"Average PSNR (Tensor): {avg_psnr_tensor:.2f}")
	print(f"Average SSIM (Tensor): {avg_ssim_tensor:.2f}")

	# 保存结果到文本文件
	results_path = os.path.join(OUTPUT_PATH, "embedding_metrics.txt")
	with open(results_path, "w") as f:
		f.write("Embedding Metrics:\n")
		f.write(f"Total Images Processed: {num_images}\n")
		f.write(f"Average PSNR: {avg_psnr:.2f}\n")
		f.write(f"Average SSIM: {avg_ssim:.2f}\n")
		f.write(f"Average LPIPS: {avg_lpips:.2f}\n")
		f.write(f"Average PSNR (Tensor): {avg_psnr_tensor:.2f}\n")
		f.write(f"Average SSIM (Tensor): {avg_ssim_tensor:.2f}\n")

if __name__ == "__main__":
    main()
    print("Embedding completed.")
