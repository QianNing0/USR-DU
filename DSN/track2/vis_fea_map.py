#原文链接：https: // blog.csdn.net / weixin_40500230 / article / details / 93845890
import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np


# if var.size(1) > 1:
#     convert = var.new(1, 3, 1, 1)
#     convert[0, 0, 0, 0] = 65.738
#     convert[0, 1, 0, 0] = 129.057
#     convert[0, 2, 0, 0] = 25.064
#     var.mul_(convert).div_(256)
#     var = var.sum(dim=1, keepdim=True)
# vis_fea_map.draw_features(1, 1, abs(var).cpu().numpy(),"***.png")
def draw_features(x, savename):
    img = x[0, 0, :, :]
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map

    cv2.imwrite(savename,img)


def draw_features_per(width,height,x,savename):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        # print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()

