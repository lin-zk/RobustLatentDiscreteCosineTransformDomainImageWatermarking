# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

#!/bin/bash

# 获取原始命令
base_cmd="${@:1}"

# 批量处理的子文件夹
folders=(
    ""
    "VAE_B_images"
	"VAE_C_images"
    'Diffusion_images'
    "Brightness_images"
    "Contrast_images"
    "JPEG_images"
    "G-Noise_images"
    "G-Blur_images"
	"Scale_images"
	# "BM3D_images"
    # 如有更多子文件夹，继续添加
)

for folder in "${folders[@]}"
do
    # 匹配 --data_dir 后的路径，并在其后加 /$folder
    cmd=$(echo "$base_cmd" | sed -E "s|(--data_dir[ ]+[^ ]+)|\1\/$folder|")
    echo "Running: $cmd"
    eval $cmd
done