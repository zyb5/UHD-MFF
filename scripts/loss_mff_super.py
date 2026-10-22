import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss_ssim import ssim  # 确保你引入了正确的 ssim

# =================================================================
# 基础组件：Sobelxy (保持不变，用于提取梯度特征)
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
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)

    def forward(self, x):
        if self.weightx.device != x.device:
            self.weightx = self.weightx.to(x.device)
            self.weighty = self.weighty.to(x.device)
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

# =================================================================
# 1. 有监督 L_Intensity (L1 Pixel Loss)
# =================================================================
class L_Intensity_Supervised(nn.Module):
    def __init__(self):
        super(L_Intensity_Supervised, self).__init__()

    def forward(self, image_fused, image_GT):
        """
        直接计算融合图像与真值图像的 L1 距离
        """
        # 如果你希望在 RGB 三通道上都计算 Loss，请去掉 [:, :1, :, :]
        # 如果依然只关注 Y 通道 (亮度)，保持原样
        loss_intensity = F.l1_loss(image_fused, image_GT) 
        return loss_intensity

# =================================================================
# 2. 有监督 L_Grad (Gradient Loss)
# =================================================================
class L_Grad_Supervised(nn.Module):
    def __init__(self):
        super(L_Grad_Supervised, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_fused, image_GT):
        """
        计算融合图像梯度与真值图像梯度的 L1 距离
        通常梯度计算在亮度通道(Y)上效果最好
        """
        # 提取第一通道 (通常是 Y 通道或者灰度)
        image_fused_Y = image_fused[:, :1, :, :]
        image_GT_Y = image_GT[:, :1, :, :]
        
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_GT = self.sobelconv(image_GT_Y)
        
        loss_gradient = F.l1_loss(gradient_fused, gradient_GT)
        return loss_gradient

# =================================================================
# 3. 有监督 L_SSIM
# =================================================================
class L_SSIM_Supervised(nn.Module):
    def __init__(self):
        super(L_SSIM_Supervised, self).__init__()

    def forward(self, image_fused, image_GT):
        """
        直接最大化 Fused 与 GT 之间的 SSIM
        Loss = 1 - SSIM(Fused, GT)
        """
        return 1 - ssim(image_fused, image_GT)

# =================================================================
# 4. 有监督 Mask Loss (新增建议)
# =================================================================
class L_Mask_Supervised(nn.Module):
    def __init__(self):
        super(L_Mask_Supervised, self).__init__()
        self.bce = nn.BCELoss() # 或者 nn.MSELoss()

    def forward(self, mask_pred, mask_GT):
        """
        如果你有 Mask 的真值 (决策图)，这是最强的监督信号。
        mask_pred: 网络输出的权重图 (0~1)
        mask_GT: 理想的决策图 (0 或 1)
        """
        return self.bce(mask_pred, mask_GT)

# =================================================================
# 主 Loss 类：整合
# =================================================================
class fusion_loss_mff_supervised(nn.Module):
    def __init__(self):
        super(fusion_loss_mff_supervised, self).__init__()
        self.L_Inten = L_Intensity_Supervised()
        self.L_Grad = L_Grad_Supervised()
        self.L_SSIM = L_SSIM_Supervised()
        
        # 如果你也想监督 Mask (通常有监督训练会同时监督图像和 Mask)
        self.L_Mask = L_Mask_Supervised() 

    def forward(self, image_fused, image_GT, mask_pred=None, mask_GT=None):
        """
        参数:
        image_fused: 网络输出的融合图像
        image_GT:    Ground Truth (全清晰图像)
        mask_pred:   网络输出的 Mask (可选)
        mask_GT:     Mask 的真值 (可选)
        """

        # 1. 强度 Loss
        loss_l1 = self.L_Inten(image_fused, image_GT)
        
        # 2. 梯度 Loss
        loss_gradient = self.L_Grad(image_fused, image_GT)
        
        # 3. SSIM Loss
        loss_ssim_val = self.L_SSIM(image_fused, image_GT)
        
        # 4. Mask Loss (如果提供了 Mask 真值)
        loss_mask = 0.0
        if mask_pred is not None and mask_GT is not None:
            loss_mask = self.L_Mask(mask_pred, mask_GT)
        
        # 如果没有 Mask GT，但想保留原来的二值化约束，可以用回原来的 L_Binarization
        # 但通常有监督场景下，直接监督 Mask 效果最好。

        return loss_l1, loss_gradient, loss_ssim_val, loss_mask