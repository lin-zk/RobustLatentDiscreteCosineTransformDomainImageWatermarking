# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Reference : https://github.com/ningyu1991/ArtificialGANFingerprints
"""
@inproceedings{yu2021artificial,
  author={Yu, Ning and Skripniuk, Vladislav and Abdelnabi, Sahar and Fritz, Mario},
  title={Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}"""
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
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import torch_dct
### 导入模型训练相关包
from diffusers import AutoencoderKL
from model import StegaStamp
from options.train_args import TrainOptions
from data.dataloader import PrecomputedVaeDataset

args = TrainOptions(1).parse()  # 解析参数
if args.cuda == -1:
    device = torch.device("cpu")  # 使用CPU
else:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
    if args.cuda >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid CUDA device index: {args.cuda}.")
    device = torch.device(f"cuda:{args.cuda}")  # 使用指定的GPU
    torch.cuda.set_device(device)  # 设置当前GPU
# 检验是否1阶段或训练完成
if args.checkpoint_dir is not None:
    flag_ckpt = torch.load(join(args.checkpoint_dir, "flag.ckpt"), map_location=device)
    if flag_ckpt["stage"] != 1:
        raise RuntimeError("This script is for stage 1 training only. Please use the appropriate script for stage 2.")
    elif flag_ckpt["train_finished"]:
        raise RuntimeError("Stage 1 training is already finished. Please use the appropriate script for stage 2.")
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
    dataset_train = PrecomputedVaeDataset(train_dir)
    print(f"Finished. Loading took {time() - time_start:.2f}s")
    return dataset_train

def main():
    # 载入扩散模型
    print("loading diffusion model...")
    vae = AutoencoderKL.from_pretrained(args.diffusion_dir, subfolder="vae")  # 载入vae
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False  # 冻结参数
    print("Diffusion model loaded.")

    # 加载数据集
    dataset_train = load_data()
    dataLoader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 创建模型
    encoder = StegaStamp.StegaStampEncoder(SECRET_SIZE).to(device)  # 实例化嵌入器
    decoder = StegaStamp.StegaStampDecoder(SECRET_SIZE).to(device)  # 实例化提取器

    # 如果命令行参数中提供了模型路径，则载入参数继续训练
    if args.checkpoint_dir is not None:
        print("Loading last training model...")
        print(f"Loading model from {args.checkpoint_dir}...")
        encoder.load_state_dict(torch.load(join(args.checkpoint_dir, "encoder.ckpt"), map_location=device)["state_dict"])
        decoder.load_state_dict(torch.load(join(args.checkpoint_dir, "decoder.ckpt"), map_location=device)["state_dict"])
        print("Model loaded successfully.")

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=0.1,
    )
    
	# 初始化基本参数
    epoch_last = 0
    global_step = 0
    steps_since_l2_loss_activated = -1
    energyblance_loss_weight = l2_loss_weight = 0
    total_training_time = 0
    acc_ema = 0.5
    train_finished = False
    epoch = 1
    epoch_now = float('inf')

    # 载入各种参数
    if args.checkpoint_dir is not None:
        epoch_last = flag_ckpt["epoch"]
        global_step = flag_ckpt["global_step"]
        acc_ema = flag_ckpt["acc_ema"]
        steps_since_l2_loss_activated = flag_ckpt["steps_since_l2_loss_activated"]
        l2_loss_weight = flag_ckpt["l2_loss_weight"]
        total_training_time = flag_ckpt["total_training_time"]
        optimizer.load_state_dict(torch.load(join(args.checkpoint_dir, "optim.ckpt"), map_location=device)["state_dict"])

    # 训练
    BCE_loss_weight = args.BCE_loss_weight
    while not train_finished:
        encoder.train(), decoder.train()
        train_start_time = time()  # 记录当前训练开始时间
        for (clean_images, clean_latent, clean_recon) in tqdm(dataLoader_train, desc="Training {}".format(epoch), unit="batch"):
            clean_images = clean_images.to(device)
            batch_size = min(dataLoader_train.batch_size, clean_images.size(0))
            fingerprints = generate_random_fingerprints(SECRET_SIZE, batch_size, global_step)  # 生成随机水印

            # 调整损失权重
            if steps_since_l2_loss_activated >= 0:
                if steps_since_l2_loss_activated < args.l2_loss_await:
                    steps_since_l2_loss_activated += 1
                if steps_since_l2_loss_activated >= args.l2_loss_await:
                    if acc_ema > 0.985:
                        l2_loss_weight = min(
                            args.l2_loss_weight * (steps_since_l2_loss_activated - args.l2_loss_await) / args.l2_loss_ramp,
                            args.l2_loss_weight,
                        )
                        energyblance_loss_weight = args.energyblance_loss_weight * ((l2_loss_weight / args.l2_loss_weight) ** 2)
                        steps_since_l2_loss_activated += 1

            # 训练核心代码
            clean_latent = clean_latent.to(device)
            clean_recon = clean_recon.to(device)
            fingerprints = fingerprints.to(device)

            ## DCT
            clean_latent_dct = torch_dct.dct_2d(clean_latent, norm="ortho")
            fingerprinted_latent_dct_residual = encoder(fingerprints, clean_latent_dct)
            fingerprinted_latent_residual = torch_dct.idct_2d(fingerprinted_latent_dct_residual, norm="ortho")
            fingerprinted_latent = fingerprinted_latent_residual + clean_latent

            ### 提取水印
            decoder_output = decoder(fingerprinted_latent)
            if args.noise_std:
                noise_std = args.noise_std
                fingerprinted_latent_noised = fingerprinted_latent + torch.randn_like(fingerprinted_latent) * noise_std
                decoder_output_noised = decoder(fingerprinted_latent_noised)

            # 计算损失
            energy_map = fingerprinted_latent_residual.pow(2).mean(dim=1, keepdim=True)
            l2_loss = energy_map.mean()
            
            BCE = nn.BCEWithLogitsLoss()
            BCE_loss = BCE(decoder_output.view(-1), fingerprints.view(-1))
            if args.noise_std:
                BCE_loss_noised = BCE(decoder_output_noised.view(-1), fingerprints.view(-1))

            # 能量均衡正则
            if args.energyblance_loss_weight:
                energy_var = energy_map.var()
            else:
                energy_var = 0.0

            if args.noise_std:
                loss = l2_loss_weight * l2_loss + energyblance_loss_weight * energy_var + BCE_loss_weight * (BCE_loss + BCE_loss_noised) / 2
            else:
                loss = l2_loss_weight * l2_loss +  energyblance_loss_weight * energy_var + BCE_loss_weight * BCE_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算水印提取正确率
            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )
            acc_ema = acc_ema * 0.99 + bitwise_accuracy * 0.01
            # 操作l2权重启动控制标志
            if steps_since_l2_loss_activated == -1:
                if acc_ema > 0.90:
                    steps_since_l2_loss_activated = 0

            if l2_loss_weight >= args.l2_loss_weight - 1e-6:
                if acc_ema > 0.990:
                    if epoch_now == float('inf') : epoch_now = epoch
                elif acc_ema > 0.985:
                    optimizer.param_groups[0]['lr'] = args.lr ** 1.5
                else: pass

            if global_step % 50 == 0 or train_finished:
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
                writer.add_scalar("Train/loss/total loss", loss, global_step)
                writer.add_scalar("Train/loss/BCE loss", BCE_loss, global_step)
                writer.add_scalar("Train/loss/l2 loss", l2_loss, global_step)
                writer.add_scalar("Train/loss/energy variance", energy_var, global_step)
                writer.add_scalar("Train/weight/l2 loss weight", l2_loss_weight, global_step)
                writer.add_scalar("Train/weight/BCE loss weight", BCE_loss_weight, global_step)
                writer.add_scalar("Train/weight/energy variance weight", energyblance_loss_weight, global_step)
                writer.add_scalar("Train/other/learning rate", optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar("Train/metrics/EMA bitwise accuracy", (acc_ema * 100.0), global_step)
            global_step += 1
            del clean_images, clean_latent, clean_recon
        training_time = time() - train_start_time
        total_training_time += training_time
        writer.add_scalar("Train/total training time", total_training_time, epoch + epoch_last)

        if (l2_loss_weight >= args.l2_loss_weight - 1e-6) and (acc_ema > 0.990) and (epoch - epoch_now > 1):
            train_finished = True
                
        if epoch % 100 == 0 or train_finished:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subfolder_name = f"{epoch}_{acc_ema:.4f}_{timestamp}"
            save_path = join(CHECKPOINTS_PATH, subfolder_name)
            os.makedirs(save_path, exist_ok=True)

            # 保存模型为ckpt格式，分开存好debug
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "steps_since_l2_loss_activated" : steps_since_l2_loss_activated,
                    "l2_loss_weight" : l2_loss_weight,
                    "total_training_time" : total_training_time,
                    "train_finished": train_finished,
                    "acc_ema": acc_ema,
                    "stage": 1
                },
                join(save_path, "flag.ckpt"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": optimizer.state_dict()
                },
                join(save_path, "optim.ckpt"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": encoder.state_dict()
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
        epoch += 1

if __name__ == "__main__":
    main()
    print("Training completed.")