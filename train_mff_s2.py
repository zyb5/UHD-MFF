
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import kornia
import sys
from data.o_fusion_dataset import DistillDataSet
from data.o_fusion_dataset import RandomCropPair
import datetime
import os
from torchvision.utils import save_image
import transforms as T
from model.context_encoder import Generator_for_info
from scripts.loss_mff import fusion_loss_mff
from itertools import chain
from model.MF_WP import WeightMapPredictor
from model.MF_MP_NEW import Generator
from model.LUT_WP import OptimizableLUT
from model.calculate import MFlut_SpeedOptimized, apply_fusion_2d_as_residual, load_partial_weights,save_generator_context,save_lut

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout  # 屏幕输出
        self.log = open(filename, "a", encoding='utf-8')  # 文件输出 (追加模式)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保每条 print 都能立即写入文件，防止断电丢失
        
    def flush(self):
        # 这是一个空方法，为了兼容 sys.stdout 的 flush 调用
        self.terminal.flush()
        self.log.flush()


def fine_tune_lut(lut1_model, lut2_model, Wmp1, train_loader, val_loader, device, epochs, learning_rate):

    best_val_loss = 1e5
    lut1_model.eval()
    Wmp1.eval()
    lut2_model.train()
    
    optimizer = optim.AdamW(lut2_model.parameters(), lr=learning_rate) 
    loss_fuc = fusion_loss_mff().to(device)
    print("Start Stage 2 Training (Residual Refinement)...")
    for epoch in range(epochs):

        train_loss = 0
        train_loss_gradient = 0
        train_loss_ssim = 0
        train_loss_l1 = 0
        train_loss_sparse = 0 # 新增稀疏 Loss 记录

        for step, data in enumerate(train_loader):
            I_A, I_B, fuse, _ = data
            I_A = I_A.to(device)
            I_B = I_B.to(device)

            optimizer.zero_grad() # 梯度清零
            
            with torch.no_grad(): # 显式告诉 PyTorch 这里不传梯度，省显存
                B, C, H, W = I_A.shape
                scale_factor = 0.25 
                small_H, small_W = int(H * scale_factor), int(W * scale_factor)
                I_A_small = F.interpolate(I_A, size=(small_H, small_W), mode='bilinear', align_corners=False)
                I_B_small = F.interpolate(I_B, size=(small_H, small_W), mode='bilinear', align_corners=False)

                lut1 = lut1_model()
                out1_small = MFlut_SpeedOptimized(I_A_small * 255., I_B_small * 255., lut1, Wmp1)
                
                mask_small = F.softmax(out1_small, dim=1)
                # mask_big 是主分支提供的粗糙 Mask (0~1)
                mask_big = F.interpolate(mask_small, size=(H, W), mode='bilinear', align_corners=False)
                # 取出前景通道
                mask_coarse = mask_big[:, :1, :, :]
                
            lut2 = lut2_model()
            residual_mask = apply_fusion_2d_as_residual(I_A * 255., I_B * 255., lut2)
            
            mask_final = mask_coarse + residual_mask
            mask_final = torch.clamp(mask_final, 0, 1)
            fuse_res = mask_final * I_A + (1 - mask_final) * I_B

            loss_l1, loss_gradient, loss_ssim, _ = loss_fuc(I_A, I_B, fuse_res, mask_final)
            
            loss_sparse = torch.mean(torch.abs(residual_mask)) * 0.05
           
            total_loss = loss_l1 * 20. + loss_gradient * 0.7 + loss_ssim * 0.8 + loss_sparse

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_loss_gradient += loss_gradient.item()
            train_loss_l1 += loss_l1.item()
            train_loss_ssim += loss_ssim.item()
            train_loss_sparse += loss_sparse.item()

        tb_writer.add_scalar("train_total_loss", train_loss/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_gradient", train_loss_gradient/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_l1", train_loss_l1 / len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_ssim", train_loss_ssim / len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_sparse", train_loss_sparse / len(train_loader), epoch)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.6f} - gradient: {train_loss_gradient / len(train_loader):.6f}" 
              f"- l1: {train_loss_l1 / len(train_loader):.6f} - ssim: {train_loss_ssim / len(train_loader):.6f} -sparse:{train_loss_sparse / len(train_loader):.6f} ")

        if (epoch + 1) % 10 == 0:
            val_loss, val_loss_l1, val_loss_gradient, val_loss_ssim, val_loss_sparse = validate_lut(lut1_model, lut2_model, Wmp1, val_loader, device)

            tb_writer.add_scalar("val_total_loss", val_loss/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_gradient", val_loss_gradient/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_l1", val_loss_l1 / len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_ssim", val_loss_ssim / len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_sparse", val_loss_sparse / len(val_loader), epoch)

            if val_loss < best_val_loss :
                best_val_loss = val_loss
                
                filename2 = f"stage2_epoch{epoch}_valloss{val_loss:.6f}_lut2.npy"
                full_path2 = os.path.join(filefold_path, filename2)

                save_lut(lut2_model, full_path2)

            print(f"Validation - Epoch {epoch} - Loss: {val_loss / len(val_loader):.6f} - gradient: {val_loss_gradient / len(val_loader):.6f} - l1: {val_loss_l1 / len(val_loader):.6f} - loss_ssim: {val_loss_ssim / len(val_loader):.6f} -bin:{val_loss_sparse / len(train_loader):.6f} ")


def validate_lut(lut1_model, lut2_model, Wmp1, val_loader, device):

    train_loss = 0
    train_loss_gradient = 0
    train_loss_ssim = 0
    train_loss_l1 = 0
    train_loss_sparse = 0

    loss_fuction = fusion_loss_mff()
    Wmp1.eval()

    save_folder = f"val_fused"  
    save_folder = os.path.join(filefold_path, save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with torch.no_grad():
        lut1 = lut1_model()
        for step, data in enumerate(val_loader):
            I_A, I_B, fuse, task = data
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            
            B, C, H, W = I_A.shape
            scale_factor = 0.25 # 1/4
            small_H, small_W = int(H * scale_factor), int(W * scale_factor)
            
            I_A_small = F.interpolate(I_A, size=(small_H, small_W), mode='bilinear')
            I_B_small = F.interpolate(I_B, size=(small_H, small_W), mode='bilinear')
            out1_small = MFlut_SpeedOptimized(I_A_small * 255., I_B_small * 255., lut1, Wmp1)
            mask_small = F.softmax(out1_small, dim=1)
            mask_big = F.interpolate(mask_small, size=(H, W), mode='bilinear')
            mask_coarse = mask_big[:, :1, :, :]

            residual_mask = apply_fusion_2d_as_residual(I_A * 255., I_B * 255., lut2)

            mask_final = torch.clamp(mask_coarse + residual_mask, 0, 1)
            fuse_res = mask_final * I_A + (1 - mask_final) * I_B

            loss_l1, loss_gradient, loss_ssim, _ = loss_fuction(I_A, I_B, fuse_res, mask_final)

            loss_sparse = torch.mean(torch.abs(residual_mask)) * 0.05
            total_loss = loss_l1 * 20. + loss_gradient * 0.7 + loss_ssim * 0.8 + loss_sparse

            train_loss += total_loss.item()
            train_loss_gradient += loss_gradient.item()
            train_loss_l1 += loss_l1.item() 
            train_loss_ssim += loss_ssim.item()
            train_loss_sparse += loss_sparse.item()

    return train_loss, train_loss_l1, train_loss_gradient, train_loss_ssim, train_loss_sparse


if __name__ == "__main__":

    if os.path.exists("./finetune_lut_mff_v8_lr_s2") is False:                             
        os.makedirs("./finetune_lut_mff_v8_lr_s2")
        
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    filefold_path = "./finetune_lut_mff_v8_lr_s2/finetune_lut_{}".format(file_name)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)
    
    txt_log_file = os.path.join(filefold_path, "train_output.txt")
    print(f"Log file will be saved to: {txt_log_file}")
    sys.stdout = DualLogger(txt_log_file)
    
    print(f"Experimental Dir: {filefold_path}")
    print(f"Time: {file_name}")
    print("-" * 30)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lut_filepath = "finetune_lut_mff_v8_lr/finetune_lut_20260226-101145/fine_tuned_epoch799_valloss766.512117_lut1.npy" 
    # lut_filepath = "finetune_lut_mff_v8_4k_enhance/finetune_lut_20251218-103157/fine_tuned_epoch799_valloss230.319876_lut1.npy"  
    lut_tensor1 = torch.tensor(np.load(lut_filepath).astype(np.float32), device=DEVICE)
    lut1 = OptimizableLUT(lut_tensor1)  

    # lut_filepath2 = "finetune_lut_mff_v8_4k_enhance_s2/finetune_lut_20251218-220848/stage2_epoch799_valloss34.089382_lut2.npy"   ##  out_lut_2d_cbcr_.npy
    lut_filepath2 = "out_lut_2d_cbcr_.npy"
    lut_tensor2 = torch.tensor(np.load(lut_filepath2).astype(np.float32), device=DEVICE)
    lut2 = OptimizableLUT(lut_tensor2)

    wmp_file = "finetune_lut_mff_v8_lr/finetune_lut_20260226-101145/generator_context_epoch799_valloss766.512117_1.pth"
    # wmp_file = "finetune_lut_mff_v8_4k_enhance/finetune_lut_20251218-103157/generator_context_epoch799_valloss230.319876_1.pth"
    wmp = Generator_for_info(6).to(DEVICE)
    pretrained_dict = torch.load(wmp_file, map_location=DEVICE) 
    load_partial_weights(wmp, pretrained_dict)

    batch_size = 8
    # visible_path = "/data1/zyb/dataset/UHD-LL/train/multifocus-4k-selected-1500/near/"   
    # infrared_path = "/data1/zyb/dataset/UHD-LL/train/multifocus-4k-selected-1500/far/"  
    # fuse_train = "../dataset/UHD-LL/train/normal/"
    # test_visible_path = "/data1/zyb/dataset/UHD-LL/test/multifocus-4k-selected-300/near/"
    # test_infrared_path = "/data1/zyb/dataset/UHD-LL/test/multifocus-4k-selected-300/far/"
    # fuse_test = "/data1/zyb/dataset/UHD-LL/test/normal_selected-300/"
    
    visible_path = "/data1/zyb/dataset/RealMFF-MFF/A/" #"/data1/zyb/dataset/UHD-LL/train/multifocus-4k-selected-1500/near/"   
    infrared_path = "/data1/zyb/dataset/RealMFF-MFF/B/"#"/data1/zyb/dataset/UHD-LL/train/multifocus-4k-selected-1500/far/"  
    fuse_train = "/data1/zyb/dataset/RealMFF-MFF/A/"#"../dataset/UHD-LL/train/normal/"
    test_visible_path = "/data1/zyb/dataset/MFI-WHU/source_1/" #"/data1/zyb/dataset/UHD-LL/test/multifocus-4k-selected-300/near/"
    test_infrared_path = "/data1/zyb/dataset/MFI-WHU/source_2/" #"/data1/zyb/dataset/UHD-LL/test/multifocus-4k-selected-300/far/"
    fuse_test = "/data1/zyb/dataset/MFI-WHU/source_1/"#"/data1/zyb/dataset/UHD-LL/test/normal_selected-300/"

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    data_transform = {
        "train": RandomCropPair(size=(128, 128)),
        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    train_dataset = DistillDataSet(visible_path=visible_path,
                                  infrared_path=infrared_path,
                                  other_fuse_path=fuse_train,
                                  phase="train",
                                  transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_dataset = DistillDataSet(visible_path=test_visible_path,
                                infrared_path=test_infrared_path,
                                other_fuse_path=fuse_test,
                                phase="val",
                                transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    fine_tune_lut(lut1, lut2, wmp, train_loader, val_loader, DEVICE, epochs=800, learning_rate=5e-6)