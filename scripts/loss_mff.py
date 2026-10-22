import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import exp
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF
from model.loss_ssim import ssim

# =================================================================
# 基础组件：Sobelxy (修改为设备无关，自动适配输入device)
# =================================================================
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        
        # 移除 .cuda() 硬编码，改用 register_buffer 自动随模型迁移设备
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)

    def forward(self, x):
        # 确保 weight 和 x 在同一个 device
        if self.weightx.device != x.device:
            self.weightx = self.weightx.to(x.device)
            self.weighty = self.weighty.to(x.device)
            
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

# =================================================================
# 改进版 L_Intensity: 基于区域能量 (Region Energy)
# =================================================================
class L_Intensity_FocusAware(nn.Module):
    def __init__(self):
        super(L_Intensity_FocusAware, self).__init__()
        self.sobel = Sobelxy()
        # 7x7 平均池化，用于平滑噪声，提取宏观清晰度
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]

        # 1. 计算梯度
        gradA = self.sobel(image_A_Y)
        gradB = self.sobel(image_B_Y)

        # 2. 关键改进：计算区域能量 (Region Energy)
        # 取绝对值后再池化，避免正负抵消
        energy_A = self.avg_pool(gradA)
        energy_B = self.avg_pool(gradB)

        # 3. 生成基于区域的 Mask (伪标签)
        # 这样生成的 Mask 是大块连续的，不会有细碎噪点
        mask_target = (energy_A > energy_B).float()

        # 4. 构建伪真值 (Pseudo GT)
        intensity_joint = mask_target * image_A_Y + (1 - mask_target) * image_B_Y

        # 5. 计算 L1 Loss
        Loss_intensity = F.l1_loss(image_fused[:, :1, :, :], intensity_joint)
        return Loss_intensity

# =================================================================
# 新增：L_Binarization (二值化约束)
# =================================================================
class L_Binarization(nn.Module):
    def __init__(self):
        super(L_Binarization, self).__init__()

    def forward(self, mask):
        # 约束 mask 接近 0 或 1
        # f(x) = x * (1 - x) 在 x=0.5 时最大，在 x=0/1 时为 0
        return torch.mean(mask * (1 - mask))

# =================================================================
# 其它组件 (保持不变或微调)
# =================================================================
class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        
        gradient_A = self.sobelconv(image_A_Y)
        gradient_B = self.sobelconv(image_B_Y)
        gradient_fused = self.sobelconv(image_fused_Y)
        
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        # 假设你外部已经定义了 ssim 函数，这里引用
        # from your_ssim_module import ssim 

    def forward(self, image_A, image_B, image_fused):
        # 简单的平均 SSIM
        weight_A = 0.5
        weight_B = 0.5
        # 确保 ssim 函数存在
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM

# =================================================================
# 主 Loss 类：整合所有 Loss
# =================================================================
class fusion_loss_mff(nn.Module):
    def __init__(self):
        super(fusion_loss_mff, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity_FocusAware() # 已经替换为基于能量的版本
        self.L_SSIM = L_SSIM()
        self.L_Bin = L_Binarization() # 新增二值化 Loss

    # 修改接口：增加 mask 参数 (默认为 None)
    def forward(self, image_A, image_B, image_fused, mask=None):
        # 1. 强度/重建 Loss (基于伪标签)
        loss_l1 = self.L_Inten(image_A, image_B, image_fused)
        
        # 2. 梯度 Loss
        loss_gradient = self.L_Grad(image_A, image_B, image_fused)
        
        # 3. SSIM Loss
        loss_SSIM_val = 1 - self.L_SSIM(image_A, image_B, image_fused)
        
        # 4. 二值化 Loss (如果有 mask 输入)
        loss_bin = 0.0
        if mask is not None:
            loss_bin = self.L_Bin(mask)

        # 5. 总 Loss 组合 (你可以在这里调整内部权重，或者返回出去在外面加权)
        # 这里返回分项，方便外面 main loop 打印和加权
        return loss_l1, loss_gradient, loss_SSIM_val, loss_bin