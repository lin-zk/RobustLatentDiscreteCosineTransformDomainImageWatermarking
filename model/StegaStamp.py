# -*- coding: utf-8 -*-
# @Author  : Lin_zk
# @E-mail  : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Citation: https://github.com/ningyu1991/ArtificialGANFingerprints
"""
@inproceedings{yu2021artificial,
  author={Yu, Ning and Skripniuk, Vladislav and Abdelnabi, Sahar and Fritz, Mario},
  title={Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}"""
# @Thanks to : Copilot for assistance

import math
import torch
from torch import nn
from torch.nn.functional import leaky_relu

class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        fingerprint_size,
        resolution=64,
        channels=4,
    ):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.channels = channels
        self.secret_dense = nn.Linear(self.fingerprint_size, 16 * 16 * channels)

        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        self.fingerprint_upsample = nn.Upsample(scale_factor=(2**(log_resolution-4), 2**(log_resolution-4)))
        self.conv1 = nn.Conv2d(2 * channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * channels, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, channels, 1)

    def forward(self, fingerprint, image):
        fingerprint = leaky_relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view((-1, self.channels, 16, 16))
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)
        conv1 = leaky_relu(self.conv1(inputs))
        conv2 = leaky_relu(self.conv2(conv1))
        conv3 = leaky_relu(self.conv3(conv2))
        conv4 = leaky_relu(self.conv4(conv3))
        conv5 = leaky_relu(self.conv5(conv4))
        up6 = leaky_relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = leaky_relu(self.conv6(merge6))
        up7 = leaky_relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = leaky_relu(self.conv7(merge7))
        up8 = leaky_relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = leaky_relu(self.conv8(merge8))
        up9 = leaky_relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = leaky_relu(self.conv9(merge9))
        conv10 = leaky_relu(self.conv10(conv9))
        residual = self.residual(conv10)
        return residual


class StegaStampDecoder(nn.Module):
    def __init__(self, fingerprint_size, resolution=64, channels=4, dropout_p=0.00):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution
        self.channels = channels
        self.decoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, (3, 3), 2, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.LeakyReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image):
        x = self.decoder(image)
        x = x.reshape(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)