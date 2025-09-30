# 鲁棒“潜在-DCT域”图像水印

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[中文](README.md) | [English](README_en.md)

本仓库是我们论文的官方实现，**提出了一种在VAE潜在空间中利用频域嵌入的鲁棒图像水印新方法**。相关代码将在后续上传。

## 📋 目录
<table>
<tr>
<td>

- [概述](#概述)  
- [功能特点](#功能特点)  
- [配置环境](#配置环境)  
- [使用说明](#使用说明)  
- [项目结构](#项目结构)  

</td>
<td>

- [性能评估](#性能评估)  
- [致谢](#致谢)  
- [许可证](#许可证)  
- [联系方式](#联系方式)  
<!-- - [引用](#引用) -->

</td>
</tr>
</table>

## 🔍 概述

![后续将上传框架架构图](Image/Overview.png)

我们提出了一种在变分自编码器（VAE）潜在空间中运行的频域水印框架。我们的方法通过两阶段训练过程，在保持高视觉质量的同时，实现了对各种攻击的卓越鲁棒性。

## ✨ 功能特点

- 🎯 **高鲁棒性**：抵抗包括JPEG压缩、噪声、模糊和基于扩散等攻击
- 🚀 **高效处理**：使用预训练的VAE以及DCT变换实现快速嵌入和提取
- 🔑 **灵活密钥**：支持三种密钥生成模式（随机、固定、指定）
- 📊 **全面指标**：PSNR、SSIM、LPIPS用于质量评估
- 🛡️ **攻击模拟**：内置示例攻击代码用于鲁棒性评估
- 📈 **阈值计算**：自动计算水印检测的阈值

## 🛠️ 配置环境

### 先决条件
- Python 3.8+
- PyTorch 2.4+
- 支持CUDA的GPU（推荐）

### 设置环境
```bash
git clone https://github.com/lin-zk/HiRaLD
cd HiRaLD
conda create --name envPy38_HiRaLD python=3.8
conda activate envPy38_HiRaLD
pip install -r requirements.txt
```

## 🚀 使用说明

![后续将上传训练架构图](Image/TrainArch.png)

### 预训练模型权重

我们将在后续公开我们训练的模型权重文件，包括多个版本，分别在DiffusionDB以及ImageNet数据集上训练，你也可以通过下面的1、2、3步自行训练。

### 1. 数据集预处理
你可以选择自己的数据集，并通过dataset_dir传入参数，数据集需要由```train```、```val```、```test```三个子文件夹构成。
```bash
python DatasetMaker.py \
    --dataset_dir /你的/数据集/路径/ \
    --save_dir /你/保存/处理后/数据集/的路径
```

### 2. 阶段1训练（水印编码-解码器联合训练）
```bash
python TrainStage1.py \
    --data_dir /你/保存的/处理后/数据集/路径 \
    --output_dir /你的/实验/文件夹
```
结果保存于 save_dir/Stage1，支持tensorboard实时查看训练过程

### 3. 阶段2训练（跨域微调）
```bash
python TrainStage2.py \
    --data_dir /你/保存的/处理后/数据集/路径 \
    --output_dir /你的/实验/文件夹 \
    --checkpoint_dir 第一阶段/训练/最后一个/保存/模型文件夹/的路径（Stage1/实验时间/checkpoints中的文件）
```
结果保存于 save_dir/Stage2，支持tensorboard实时查看训练过程

### 4. 嵌入水印
```bash
python Embed.py \
    --data_dir /待水印/图片/路径 \
    --output_dir /水印图片/保存/路径 \
    --checkpoint_dir 第二阶段/训练/最优/保存/模型文件夹/的路径（Stage2/实验时间/checkpoints中的文件） \
    --fingerprint_type 你需要的水印模式（random、fix、certain） \
    <--fingerprint_content 如果你选择了"certain"，那么你需要给出水印内容，你需要用十六进制写在一个txt文档中，并通过此参数传入其路径，或者直接通过此参数传入十六进制的字符串>
```
结果保存于output_dir，水印图片文件名为“原图片名_十六进制水印字符串.png”，此外还输出embedding_metrics.txt用于计入嵌入质量（PSNR、SSIM、LPIPS）

### 5. 提取水印
```bash
python Extract.py \
    --checkpoint_dir 第二阶段/训练/最优/保存/模型文件夹/的路径（stage2/实验时间/checkpoints中的文件） \
    --data_dir /水印/图片/路径
```
结果保存于data_dir/extracting_metrics.txt，记录了平均提取Acc.以及Esr.以及各图片的详细提取信息

### 攻击模拟
应用多种攻击测试水印鲁棒性：
```bash
python Attack.py --input_dir /水印/图片/路径
```
攻击结果将保存在input_dir的各个子文件夹中（按攻击种类分类）

### 批量提取
使用提供的脚本进行自动化多水印路径提取：
```bash
bash ExtractAfterAttack.sh python Extract.py \
    --checkpoint_dir 第二阶段/训练/最优/保存/模型文件夹/的路径（stage2/实验时间/checkpoints中的文件） \
    --data_dir /水印/图片/路径
```
结果保存于data_dir/各攻击子文件夹/extracting_metrics.txt

### 工具脚本

#### 计算检测阈值
```bash
python CalculateAccThreshold.py
```

#### 差异可视化
```bash
python Different.py \
    --input_o /待水印/图片/路径 \
    --input_w /水印/图片/路径
```
结果保存于input_w/Different，内容是由原图、差分图、水印图横向拼接的图片

## 📁 项目结构

```
HiRaLD/
├── requirements.txt         # 依赖项清单
├── DatasetMaker.py          # 第1步：使用VAE预推理创建数据集
├── TrainStage1.py           # 第2步：水印编码-解码预训练
├── TrainStage2.py           # 第3步：正式训练，适应VAE
├── Embed.py                 # 水印嵌入，支持灵活的密钥模式
├── Extract.py               # 水印提取与验证
├── Attack.py                # 攻击模拟
├── Different.py             # 差异可视化工具
├── CalculateAccThreshold.py # 检测阈值计算
├── ExtractAfterAttack.sh    # 批量水印提取脚本
├── attack/                  
│   ├── regen_pipe.py        
│   └── wmattacker.py        
├── data/                    
│   └── dataloader.py        
├── model/                   
│   ├── StegaStamp.py        
│   └── Diffusion.py         
└── options/                 # 配置文件
    ├── train_args.py        # 训练参数
    ├── embed_args.py        # 嵌入参数
    └── extract_args.py      # 提取参数
```

## 📊 性能评估

![后续将上传性能测试结果图](Image/Performance.png)

### 支持的攻击
- **VAE攻击**：bmshj2018-factorized、bmshj2018-hyperprior、mbt2018-mean、mbt2018、cheng2020-anchor
- **扩散攻击**：基于Stable Diffusion的重生成
- **传统攻击**：JPEG压缩、高斯噪声/模糊、亮度/对比度调整
- **几何攻击**：缩放变换、旋转变换（需额外增强）、裁剪变换（需额外增强）

### 指标
- **视觉质量**：PSNR、SSIM、LPIPS
- **鲁棒性**：比特准确率（Acc.）、提取成功率（Esr.）

## 🙏 致谢

此工作基于以下优秀的开源项目：

- **StegaStamp**：用于基础编码器-解码器架构 [[Yu et al., ICCV 2021]](https://github.com/ningyu1991/ArtificialGANFingerprints)
- **Watermark Attacker**：用于攻击实现 [[Zhao et al., NeurIPS 2024]](https://github.com/xuandongzhao/watermarkattacker)
- **Diffusers**：用于VAE和扩散模型实现
- **GitHub Copilot**：用于开发协助

## 📝 许可证

此项目基于MIT许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 📧 联系方式

如有问题，请联系：
- **作者**：Lin_zk
- **邮箱**：1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
- **QQ**：1751740699

<!-- ## 📄 引用

如果您在研究中使用了此代码，请引用我们的论文：

```bibtex
@{,
  title={},
  author={},
  journal={},
  year={}
} 
``` -->

---

⭐ **如果您觉得此仓库有帮助，请点亮Star！**
