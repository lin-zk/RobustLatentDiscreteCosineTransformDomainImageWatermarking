# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

### 导入基本包
import os
from os.path import join
from time import time
from datetime import datetime
from tqdm import tqdm
### 导入torch包
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_dct
### 导入指标计算包
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torchmetrics.functional import peak_signal_noise_ratio as peak_signal_noise_ratio_tensor
from torchmetrics.functional import structural_similarity_index_measure as structural_similarity_tensor
import lpips as LPIPS_PKG
### 导入自定义包
from diffusers import AutoencoderKL
from model import StegaStamp, Diffusion
from options.train_args import TrainOptions
from data.dataloader import PrecomputedVaeDataset

args = TrainOptions(2).parse()  # 解析参数
if args.cuda == -1:
    device = torch.device("cpu")  # 使用CPU
else:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
    if args.cuda >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid CUDA device index: {args.cuda}.")
    device = torch.device(f"cuda:{args.cuda}")  # 使用指定的GPU
    torch.cuda.set_device(device)  # 设置当前GPU
# 检验是否1阶段训练完成
flag_ckpt = torch.load(join(args.checkpoint_dir, "flag.ckpt"), map_location=device)
if flag_ckpt["stage"] == 1 and not flag_ckpt["train_finished"]:
    raise RuntimeError("Stage 1 training is not finished. Please complete stage 1 training before starting stage 2.")
os.environ["PYTHONHASHSEED"] = str(args.seed)  # 设置Python的随机种子
torch.manual_seed(args.seed)  # 设置Torch的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)  # 设置GPU的随机种子

LOGS_PATH = os.path.join(args.expr_dir, "logs/")
CHECKPOINTS_PATH = os.path.join(args.expr_dir, "checkpoints/")
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)

IMAGE_RESOLUTION = args.image_resolution
IMAGE_CHANNELS = args.image_channels
SECRET_SIZE = args.fingerprint_length

writer = SummaryWriter(LOGS_PATH)

def generate_random_fingerprints(fingerprint_length, batch_size, offset):
    #更变随机种子
    new_seed = args.seed + offset
    torch.manual_seed(new_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(new_seed)
    os.environ["PYTHONHASHSEED"] = str(new_seed)  # 设置Python的随机种子
    #生成随机水印
    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
    return z

def load_data():
    time_start = time()
    print(f"Loading image folder {args.data_dir} ...")
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    dataset_train = PrecomputedVaeDataset(train_dir)
    dataset_val = PrecomputedVaeDataset(val_dir)
    print(f"Finished. Loading took {time() - time_start:.2f}s")
    return dataset_train, dataset_val

def main():
    # 初始化 LPIPS 分数模型（使用 VGG16）
    lpips_loss = LPIPS_PKG.LPIPS(net='vgg').to(device)

    # 载入扩散模型
    print("loading diffusion model...")
    vae = AutoencoderKL.from_pretrained(args.diffusion_dir, subfolder="vae")  # 载入vae
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False  # 冻结参数
    print("Diffusion model loaded.")

    # 加载数据集
    dataset_train, dataset_val = load_data()
    dataLoader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataLoader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=args.num_workers)

    # 创建模型
    encoder = StegaStamp.StegaStampEncoder(SECRET_SIZE).to(device)  # 实例化频域嵌入器
    decoder = StegaStamp.StegaStampDecoder(SECRET_SIZE, dropout_p=args.dropout_p).to(device)  # 实例化提取器

    print(f"Loading model from {args.checkpoint_dir}...")
    encoder.load_state_dict(torch.load(join(args.checkpoint_dir, "encoder.ckpt"), map_location=device)["state_dict"])
    decoder.load_state_dict(torch.load(join(args.checkpoint_dir, "decoder.ckpt"), map_location=device)["state_dict"])
    print("Model loaded successfully.")

    optimizer = AdamW(
        params=list(decoder.parameters()) + list(encoder.parameters()),
        weight_decay=0.1,
        lr=args.lr
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1000, 
        verbose=True,
        threshold=1e-6,
        min_lr=1e-20
    )

    # 初始化基本参数
    total_validation_time = 0

    # 载入各种参数
    epoch_last = flag_ckpt["epoch"]
    global_step = flag_ckpt["global_step"]
    acc_ema = flag_ckpt["acc_ema"].to(device)
    total_training_time = flag_ckpt["total_training_time"]
    if flag_ckpt["stage"] == 2:
        total_validation_time = flag_ckpt["total_validation_time"]
        optim_ckpt = torch.load(join(args.checkpoint_dir, "optim.ckpt"), map_location=device)
        optimizer.load_state_dict(optim_ckpt["state_dict"])
        scheduler.load_state_dict(optim_ckpt["scheduler_state_dict"])

    # 训练
    for epoch in range(1, args.epochs + 1):
        train_start_time = time()  # 记录当前训练开始时间
        encoder.train(), decoder.train()
        for (clean_images, clean_latent, clean_recon) in tqdm(dataLoader_train, desc="Training {}".format(epoch), unit="batch"):
            clean_images = clean_images.to(device)
            batch_size = min(args.batch_size, clean_images.size(0))
            fingerprints = generate_random_fingerprints(SECRET_SIZE, batch_size, global_step)  # 生成随机水印

            # 训练核心代码
            clean_latent = clean_latent.to(device)
            clean_recon = clean_recon.to(device)
            fingerprints = fingerprints.to(device)

            ## DCT
            clean_latent_dct = torch_dct.dct_2d(clean_latent, norm="ortho")
            ### DCT嵌入
            fingerprinted_latent_dct_residual = encoder(fingerprints, clean_latent_dct)
            fingerprinted_latent_residual = torch_dct.idct_2d(fingerprinted_latent_dct_residual, norm="ortho")
            fingerprinted_latent = fingerprinted_latent_residual + clean_latent

            # 输出水印图片
            fingerprinted_images_residual = Diffusion.decode_latent(vae, fingerprinted_latent) - clean_recon

            # 无噪提取
            fingerprinted_images = fingerprinted_images_residual + clean_images
            fingerprinted_latent_for_decoder = Diffusion.encode_image(vae, fingerprinted_images)
            decoder_output = decoder(fingerprinted_latent_for_decoder)
            if args.noise_std:
                noise_std = args.noise_std
                fingerprinted_latent_noised = fingerprinted_latent + torch.randn_like(fingerprinted_latent) * noise_std
                decoder_output_noised = decoder(fingerprinted_latent_noised)
            
            # 计算损失
            MSE = nn.MSELoss()
            l2_loss = MSE(fingerprinted_images, clean_images)

            BCE = nn.BCEWithLogitsLoss()
            BCE_loss = BCE(decoder_output.view(-1), fingerprints.view(-1))
            if args.noise_std:
                BCE_loss_noised = BCE(decoder_output_noised.view(-1), fingerprints.view(-1))
            if args.noise_std:
                loss = args.l2_loss_weight * l2_loss + args.BCE_loss_weight * (BCE_loss + BCE_loss_noised) / 2
            else:
                loss = args.l2_loss_weight * l2_loss + args.BCE_loss_weight * BCE_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch > 1:
                scheduler.step(loss)

            # 计算水印提取正确率
            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )
            acc_ema = acc_ema * 0.999 + bitwise_accuracy * 0.001

            if global_step % 50 == 0:
                # 记录训练信息
                print(f"Total training time: {total_training_time:.2f}s")
                print("Step {}".format(global_step), end=" ")
                print("Epoch {}".format(epoch))
                print("Loss {}".format(loss))
                print("Bitwise accuracy {:.1f}%".format(bitwise_accuracy * 100.0), end=" ")
                print("EMA Bitwise accuracy {:.1f}%".format(acc_ema * 100.0), end=" ")
                print("BCE loss {}".format(BCE_loss))
                print("L2 loss {}".format(l2_loss))
            writer.add_scalar("Train/metrics/bitwise accuracy", (bitwise_accuracy * 100.0), global_step)
            writer.add_scalar("Train/metrics/EMA bitwise accuracy", (acc_ema * 100.0), global_step)
            writer.add_scalar("Train/loss/total loss", loss, global_step)
            writer.add_scalar("Train/loss/BCE loss", BCE_loss, global_step)
            writer.add_scalar("Train/loss/l2 loss", l2_loss, global_step)
            writer.add_scalar("Train/weight/l2 loss weight", args.l2_loss_weight, global_step)
            writer.add_scalar("Train/weight/BCE loss weight", args.BCE_loss_weight, global_step)
            writer.add_scalar("Train/other/learning rate", optimizer.param_groups[0]['lr'], global_step)
            del clean_images, clean_latent, clean_recon    
            global_step += 1
        training_time = time() - train_start_time
        total_training_time += training_time
        writer.add_scalar("Train/total training time", total_training_time, epoch + epoch_last)

        # 验证与保存模型
        encoder.eval(), decoder.eval()
        with torch.no_grad():
            print("Verifying...")
            loss = l2_loss = BCE_loss = bitwise_accuracy = psnr = ssim = lpips = psnr_tensor = ssim_tensor = 0.0
            random_batch_index = torch.randint(0, len(dataLoader_val), (1,), device=device).item()  # 选取随机batch
            val_start_time = time()  # 记录验证开始时间
            for batch_idx, (clean_images, clean_latent, clean_recon) in enumerate(tqdm(dataLoader_val, desc="Verifying {}".format(epoch), unit="batch")):
                clean_images = clean_images.to(device)
                batch_size = min(16, clean_images.size(0))
                fingerprints = generate_random_fingerprints(SECRET_SIZE, batch_size, global_step + batch_idx)  # 生成随机水印

                clean_latent = clean_latent.to(device)
                clean_recon = clean_recon.to(device)
                fingerprints = fingerprints.to(device)

                # DCT
                clean_latent_dct = torch_dct.dct_2d(clean_latent, norm="ortho")
                fingerprinted_latent_dct_normalized_residual = encoder(fingerprints, clean_latent_dct)
                # 处理到潜空间尺度
                fingerprinted_latent_residual = torch_dct.idct_2d(fingerprinted_latent_dct_normalized_residual, norm="ortho")
                fingerprinted_latent = fingerprinted_latent_residual + clean_latent

                # 输出水印图片
                fingerprinted_images_residual = Diffusion.decode_latent(vae, fingerprinted_latent) - clean_recon
                fingerprinted_images = fingerprinted_images_residual + clean_images
                # VAE编码水印图片
                fingerprinted_latent_for_decoder = Diffusion.encode_image(vae, fingerprinted_images)
                ### 提取水印
                decoder_output = decoder(fingerprinted_latent_for_decoder)

                # 记录选取的随机batch的结果变量用于生成图片
                if batch_idx == random_batch_index:
                    clean_images_for_image = clean_images ## 原始图像
                    fingerprinted_image_for_image = fingerprinted_images ## 水印后潜在表示

                MSE = nn.MSELoss()
                l2_loss_current = MSE(fingerprinted_images, clean_images)
                BCE_loss_current = BCE(decoder_output.view(-1), fingerprints.view(-1))
                loss_current = args.l2_loss_weight * l2_loss_current + args.BCE_loss_weight * BCE_loss_current

                # 计算指标
                psnr_current = ssim_current = lpips_current = psnr_tensor_current = ssim_tensor_current = 0.0
                for i, fingerprinted_image in enumerate(fingerprinted_images):
                    # 转换 clean_images 和 fingerprinted_images 到 [0, 255] 且为整型
                    clean_image_np = ((clean_images[i].cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy().transpose(1, 2, 0)  # HWC 格式
                    fingerprinted_image_np = ((fingerprinted_image.cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy().transpose(1, 2, 0)  # HWC 格式

                    # 计算 PSNR
                    psnr_current += peak_signal_noise_ratio(clean_image_np, fingerprinted_image_np, data_range=255)

                    # 计算 SSIM
                    ssim_current += structural_similarity(clean_image_np, fingerprinted_image_np, data_range=255, channel_axis=-1)

                    # 计算 LPIPS
                    clean_image_tensor = ((clean_images[i].unsqueeze(0).clamp(-1, 1) + 1) * 0.5).to(device)  # 转换到 [0, 1]
                    fingerprinted_image_tensor = ((fingerprinted_image.unsqueeze(0).clamp(-1, 1) + 1) * 0.5).to(device)  # 转换到 [0, 1]
                    lpips_current += lpips_loss(clean_image_tensor, fingerprinted_image_tensor).item()

                    # tensor 域 [0,1] PSNR/SSIM
                    psnr_tensor_current += peak_signal_noise_ratio_tensor(clean_image_tensor, fingerprinted_image_tensor, data_range=1.0).item()
                    ssim_tensor_current += structural_similarity_tensor(clean_image_tensor, fingerprinted_image_tensor, data_range=1.0).item()
                
                # 平均指标
                psnr_current /= batch_size
                ssim_current /= batch_size
                lpips_current /= batch_size
                psnr_tensor_current /= batch_size
                ssim_tensor_current /= batch_size

                fingerprints_predicted_current = (decoder_output > 0).float()
                bitwise_accuracy_current = 1.0 - torch.mean(
                    torch.abs(fingerprints - fingerprints_predicted_current)
                )

                # 加到batch损失中
                batch_num = len(dataLoader_val)
                l2_loss += l2_loss_current / batch_num
                BCE_loss += BCE_loss_current / batch_num
                loss += loss_current / batch_num
                bitwise_accuracy += bitwise_accuracy_current / batch_num
                psnr += psnr_current / batch_num
                ssim += ssim_current / batch_num
                lpips += lpips_current / batch_num
                psnr_tensor += psnr_tensor_current / batch_num
                ssim_tensor += ssim_tensor_current / batch_num
                
                del clean_images, clean_latent, clean_recon
            # 记录验证时间
            validation_time = time() - val_start_time
            total_validation_time += validation_time
            writer.add_scalar("Val/total validation time", total_validation_time, epoch)
            # 记录验证信息
            print("Validation results:")
            print("Epoch {}".format(epoch + epoch_last))
            print("Loss {}".format(loss))
            print("Bitwise accuracy {:.1f}%".format(bitwise_accuracy * 100.0), end=" ")
            print("BCE loss {}".format(BCE_loss))
            print("L2 loss {}".format(l2_loss))
            writer.add_scalar("Val/metrics/bitwise accuracy", (bitwise_accuracy * 100.0), epoch + epoch_last)
            writer.add_scalar("Val/metrics/psnr", psnr, epoch + epoch_last)
            writer.add_scalar("Val/metrics/ssim", ssim, epoch + epoch_last)
            writer.add_scalar("Val/metrics/lpips", lpips, epoch + epoch_last)
            writer.add_scalar("Val/metrics/psnr tensor", psnr_tensor, epoch + epoch_last)
            writer.add_scalar("Val/metrics/ssim tensor", ssim_tensor, epoch + epoch_last)
            writer.add_scalar("Val/loss/total loss", loss, epoch + epoch_last)
            writer.add_scalar("Val/loss/BCE loss", BCE_loss, epoch + epoch_last)
            writer.add_scalar("Val/loss/l2 loss", l2_loss, epoch + epoch_last)
            writer.add_scalar("Val/weight/l2 loss weight", args.l2_loss_weight, epoch + epoch_last)
            writer.add_scalar("Val/weight/BCE loss weight", args.BCE_loss_weight, epoch + epoch_last)
            # 为tensorboard保存图像(4张或更少)
            show_num = min(4, clean_images_for_image.size(0))
            clean_images_for_image = clean_images_for_image[:show_num]
            fingerprinted_image_for_image = fingerprinted_image_for_image[:show_num]

            clean_images_for_image = (clean_images_for_image.clamp(-1.0, 1.0) + 1.0) / 2.0
            fingerprinted_image_for_image = (fingerprinted_image_for_image.clamp(-1.0, 1.0) + 1.0) / 2.0

            writer.add_image(
                "Val/clean images",
                make_grid(clean_images_for_image, normalize=False, scale_each=True),
                epoch + epoch_last,
            )
            writer.add_image(
                "Val/fingerprinted image",
                make_grid(fingerprinted_image_for_image, normalize=False, scale_each=True),
                epoch + epoch_last,
            )
            writer.add_image(
                "Val/residual image",
                make_grid((fingerprinted_image_for_image - clean_images_for_image) * 10.0, normalize=False, scale_each=True),
                epoch + epoch_last,
            )

            # 保存模型
            # 创建子文件夹
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subfolder_name = f"{epoch + epoch_last}_{bitwise_accuracy:.4f}_{loss:.4f}_{timestamp}"
            save_path = join(CHECKPOINTS_PATH, subfolder_name)
            os.makedirs(save_path, exist_ok=True)

            # 保存模型为ckpt格式，分开存好debug
            torch.save(
                {
                    "epoch": epoch + epoch_last,
                    "global_step": global_step,
                    "total_training_time" : total_training_time,
                    "total_validation_time" : total_validation_time,
                    "acc_ema": acc_ema,
                    "stage": 2
                },
                join(save_path, "flag.ckpt"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                },
                join(save_path, "optim.ckpt"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": encoder.state_dict(),
                },
                join(save_path, "encoder.ckpt"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": decoder.state_dict()
                },
                join(save_path, "decoder.ckpt"),
            )

if __name__ == "__main__":
    main()
    print("Training completed.")