# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

# 编码器：图像→潜在表示 ##
def encode_image(vae, image_tensor):
    latent = vae.encode(image_tensor).latent_dist.sample()
    return latent

# 解码器：潜在表示→图像 ##
def decode_latent(vae, latent):
    decoded_image = vae.decode(latent).sample
    return decoded_image