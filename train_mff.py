
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.o_fusion_dataset import DistillDataSet
from data.o_fusion_dataset import RandomCropPair
import datetime
import os
import transforms as T
from model.context_encoder import Generator_for_info
from scripts.loss_mff import fusion_loss_mff
from itertools import chain
from model.LUT_WP import OptimizableLUT
from model.calculate import MFlut_SpeedOptimized, load_partial_weights,save_generator_context,save_lut

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


def fine_tune_lut(lut1_model, Wmp1, train_loader, val_loader, device, epochs, learning_rate):

    best_val_loss = 1e5
    lut1_model.train()
    Wmp1.train()

    optimizer = optim.AdamW(chain(lut1_model.parameters(), Wmp1.parameters()), lr=learning_rate)   
    loss_fuc = fusion_loss_mff().to(device)
    
    for epoch in range(epochs):
        lut1_model.train()
        Wmp1.train()
        temper = max(0.2, 0.7 * (0.97 ** epoch))
        train_loss = 0
        train_loss_gradient = 0
        train_loss_ssim = 0
        train_loss_l1 = 0
        train_loss_bin = 0

        for step, data in enumerate(train_loader):
            I_A, I_B, fuse, _ = data
            I_A = I_A.to(device)
            I_B = I_B.to(device)

            optimizer.zero_grad() 
            
            B, C, H, W = I_A.shape
            scale_factor = 0.25 # 1/4
            small_H, small_W = int(H * scale_factor), int(W * scale_factor)
            I_A_small = F.interpolate(I_A, size=(small_H, small_W), mode='bilinear', align_corners=False)
            I_B_small = F.interpolate(I_B, size=(small_H, small_W), mode='bilinear', align_corners=False)

            lut1 = lut1_model()
            out1_small = MFlut_SpeedOptimized(I_A_small * 255., I_B_small * 255., lut1, Wmp1)
            mask_small = F.softmax(out1_small, dim=1)
            mask_full = F.interpolate(mask_small, size=(H, W), mode='bilinear', align_corners=False)

            fuse_res = mask_full[:,0:1,:,:] * I_A + mask_full[:,1:2,:,:] * I_B

            loss_l1, loss_gradient, loss_ssim, loss_bin = loss_fuc(I_A, I_B, fuse_res, mask_full)
           
            total_loss = loss_l1 * 20. + loss_gradient * 0.7 + loss_ssim * 0.8 + loss_bin * 30.

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_loss_gradient += loss_gradient.item()
            train_loss_l1 += loss_l1.item()
            train_loss_ssim += loss_ssim.item()
            train_loss_bin += loss_bin.item()

        tb_writer.add_scalar("train_total_loss", train_loss/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_gradient", train_loss_gradient/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_l1", train_loss_l1 / len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_ssim", train_loss_ssim / len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_bin", train_loss_bin / len(train_loader), epoch)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.6f} - gradient: {train_loss_gradient / len(train_loader):.6f}" 
              f"- l1: {train_loss_l1 / len(train_loader):.6f} - ssim: {train_loss_ssim / len(train_loader):.6f} -bin:{train_loss_bin / len(train_loader):.6f} ")

        if (epoch + 1) % 10 == 0:
            val_loss, val_loss_l1, val_loss_gradient, val_loss_ssim, val_loss_bin = validate_lut(lut1_model, Wmp1, val_loader, device, temper)

            tb_writer.add_scalar("val_total_loss", val_loss/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_gradient", val_loss_gradient/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_l1", val_loss_l1 / len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_ssim", val_loss_ssim / len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_bin", val_loss_bin / len(val_loader), epoch)

            if val_loss < best_val_loss :
                best_val_loss = val_loss
                filename1 = f"fine_tuned_epoch{epoch}_valloss{val_loss:.6f}_lut1.npy"

                full_path1 = os.path.join(filefold_path, filename1)

                save_lut(lut1_model, full_path1)

                context_filename1 = f"generator_context_epoch{epoch}_valloss{val_loss:.6f}_1.pth"
                generator_context_save_path1 = os.path.join(filefold_path, context_filename1)
                save_generator_context(Wmp1, save_path=generator_context_save_path1)
            
            print(f"Validation - Epoch {epoch} - Loss: {val_loss / len(val_loader):.6f} - gradient: {val_loss_gradient / len(val_loader):.6f} - l1: {val_loss_l1 / len(val_loader):.6f} - loss_ssim: {val_loss_ssim / len(val_loader):.6f} -bin:{train_loss_bin / len(train_loader):.6f} ")


def validate_lut(lut1_model, Wmp1, val_loader, device, t):

    train_loss = 0
    train_loss_gradient = 0
    train_loss_ssim = 0
    train_loss_l1 = 0
    train_loss_bin = 0

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
            
            I_A_small = F.interpolate(I_A, size=(small_H, small_W), mode='bilinear', align_corners=False)
            I_B_small = F.interpolate(I_B, size=(small_H, small_W), mode='bilinear', align_corners=False)

            out1_small = MFlut_SpeedOptimized(I_A_small * 255., I_B_small * 255., lut1, Wmp1)
            mask_small = F.softmax(out1_small, dim=1)
            mask_full = F.interpolate(mask_small, size=(H, W), mode='bilinear', align_corners=False)

            fuse_res = mask_full[:,0:1,:,:] * I_A + mask_full[:,1:2,:,:] * I_B

            loss_l1, loss_gradient, loss_ssim, loss_bin = loss_fuction(I_A, I_B, fuse_res, mask_full)

            total_loss = loss_l1 * 20. + loss_gradient * 0.7 + loss_ssim * 0.8 + loss_bin * 30

            train_loss += total_loss.item()
            train_loss_gradient += loss_gradient.item()
            train_loss_l1 += loss_l1.item() 
            train_loss_ssim += loss_ssim.item()
            train_loss_bin += loss_bin.item()

    return train_loss, train_loss_l1, train_loss_gradient, train_loss_ssim, train_loss_bin

if __name__ == "__main__":  # finetune_lut_mff_v8_4k_15

    if os.path.exists("./finetune_lut_mff_v8_lr") is False:                             
        os.makedirs("./finetune_lut_mff_v8_lr")
        
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    filefold_path = "./finetune_lut_mff_v8_lr/finetune_lut_{}".format(file_name)
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

    # lut_filepath = "finetune_lut_mff_v8_4k_enhance/finetune_lut_20251217-212349/fine_tuned_epoch799_valloss824.697906_lut1.npy"   # pretrained_weights/LUT_expanded.npy
    lut_filepath = "pretrained_weights/LUT_expanded.npy"
    lut_tensor1 = torch.tensor(np.load(lut_filepath).astype(np.float32), device=DEVICE)

    lut1 = OptimizableLUT(lut_tensor1)  

    # wmp_file = "finetune_lut_mff_v8_4k_enhance/finetune_lut_20251217-212349/generator_context_epoch799_valloss824.697906_1.pth"    #  finetune_allinone/finetune_lut_20250829-110817/generator_context_epoch59_valloss809.538791_1.pth
    wmp_file = "finetune_allinone/finetune_lut_20250829-110817/generator_context_epoch59_valloss809.538791_1.pth"
    wmp = Generator_for_info(6).to(DEVICE)
    pretrained_dict = torch.load(wmp_file, map_location=DEVICE)
    load_partial_weights(wmp, pretrained_dict)

    batch_size = 8
    visible_path = "/data1/zyb/dataset/RealMFF-MFF/A/" #"/data1/zyb/dataset/UHD-LL/train/multifocus-4k-selected-1500/near/"   
    infrared_path = "/data1/zyb/dataset/RealMFF-MFF/B/"#"/data1/zyb/dataset/UHD-LL/train/multifocus-4k-selected-1500/far/"  
    fuse_train = "/data1/zyb/dataset/RealMFF-MFF/B/"#"../dataset/UHD-LL/train/normal/"
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

    fine_tune_lut(lut1, wmp, train_loader, val_loader, DEVICE, epochs=800, learning_rate=1e-5)