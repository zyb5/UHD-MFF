
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import math
from torchvision.transforms import ToPILImage
import time

from data.simple_dataset import RandomCropPair, SimpleDataSet
import transforms as T
from model.context_encoder import Generator_for_info
from model.calculate import (
    apply_fusion_2d_as_residual, MFlut_lowfreq, load_lookup_table,
    MFlut_SpeedOptimized, FastGuidedFilter)

# 尝试导入 thop 用于计算 FLOPs
try:
    from thop import profile
except ImportError:
    profile = None
    
    
def estimate_energy(macs, input_shape, lut_sampling_points=4):
    """
    估算 Energy Cost (pJ)。
    参考标准: Horowitz, "1.1 Computing's energy problem (and what we can do about it)", ISSCC 2014.
    
    参数:
        macs: CNN 部分的 MACs (由 thop 计算)
        input_shape: 输入图像尺寸 (B, C, H, W)
        lut_sampling_points: LUT 每个像素的采样数 (MuLUT通常为4, 简单查表为1)
    """
    B, C, H, W = input_shape
    num_pixels = H * W
    
    # --- 1. 定义能耗系数 (pJ) ---
    E_MULT = 3.7        # FP32 Multiply
    E_ADD = 0.9         # FP32 Add
    E_SRAM = 5.0        # SRAM Cache Access (LUT 查找)
    E_DRAM = 640.0      # DRAM Access (输入/输出显存读写)
    
    # --- 2. 计算 CNN 部分能耗 ---
    # 1 MAC = 1 Mult + 1 Add
    energy_cnn = macs * (E_MULT + E_ADD)
    
    # --- 3. 计算 LUT 部分能耗 (SRAM) ---
    # 假设 LUT 足够小常驻 SRAM。每个像素查 N 个点。
    # 这里计算两部分：Context 分支(小图) 和 Residual 分支(大图)
    # 注意：这里我们只能基于 input_shape 估算。
    # 如果你的逻辑是 "小图跑CNN+LUT1" 和 "大图跑LUT2"，需要分别加和。
    # 下面逻辑假设 input_shape 是全分辨率输入:
    
    # 估算: 假设 LUT 操作覆盖全图 (保守估计)
    total_lut_lookups = num_pixels * lut_sampling_points
    energy_lut = total_lut_lookups * E_SRAM
    
    # --- 4. 计算 DRAM 能耗 (IO Overhead) ---
    # 读取输入 (C通道) + 写入输出 (C通道)
    # float32 = 4 bytes, 但通常 DRAM Access 按 64-bit 或 block 算，
    # 这里按论文通用做法: 读写次数 * 640pJ
    # 读: H*W*C, 写: H*W*C
    total_dram_access = (num_pixels * C * 2) 
    energy_dram = total_dram_access * E_DRAM
    
    total_energy_pJ = energy_cnn + energy_lut + energy_dram
    
    return total_energy_pJ, energy_cnn, energy_lut, energy_dram


def main():
    lut1_path = "/data1/zyb/UHD-MFF_OPENSOURCE/ckpts/lut1.npy"
    lut2_path = "/data1/zyb/UHD-MFF_OPENSOURCE/ckpts/lut2.npy"
    context_file1 = "/data1/zyb/UHD-MFF_OPENSOURCE/ckpts/generator_context.pth"

    near_focus_dir = "/data1/zyb/UHD-MFF_OPENSOURCE/UHD-MFF/UHD-MFF-Syn/near_focus"
    far_focus_dir = "/data1/zyb/UHD-MFF_OPENSOURCE/UHD-MFF/UHD-MFF-Syn/far_focus"
    save_dir = "mff_res/test_syn/"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lut1 = load_lookup_table(lut1_path).to(device)
    lut2 = load_lookup_table(lut2_path).to(device)

    get_context1 = Generator_for_info(6)
    get_context1.load_state_dict(torch.load(context_file1))
    get_context1 = get_context1.to(device).eval()

    # ==========================================
    # Added: Parameters & FLOPs Calculation
    # ==========================================
    print("-" * 30)
    print("Model Complexity Statistics:")
    
    # 1. 计算参数量 (Parameters)
    cnn_params = sum(p.numel() for p in get_context1.parameters())
    
    # 计算 LUT 的参数量 (元素个数)
    lut1_params = lut1.numel()
    lut2_params = lut2.numel()
    
    total_params = cnn_params + lut1_params + lut2_params
    
    print(f"CNN Parameters: {cnn_params / 1e6:.4f} M")
    print(f"LUT1 (5D) Parameters: {lut1_params / 1e6:.4f} M")
    print(f"LUT2 (5D) Parameters: {lut2_params / 1e6:.4f} M")
    print(f"Total System Parameters: {total_params / 1e6:.4f} M (百万)")

    # 2. 计算 FLOPs (使用固定尺寸 256x256 进行基准测试)
    # 注意: FLOPs 取决于输入尺寸。这里假设 Generator 接收 6 通道输入 (2张图concat)
    if profile:
        dummy_h, dummy_w = 3840, 2160
        dummy_input = torch.randn(1, 6, dummy_h, dummy_w).to(device)
        flops, _ = profile(get_context1, inputs=(dummy_input,), verbose=False)
        print(f"FLOPs (Input {dummy_h}x{dummy_w}): {flops / 1e9:.4f} G (十亿)")
        
        scale = 0.25
        small_h, small_w = int(dummy_h * scale), int(dummy_w * scale)
        dummy_input_small = torch.randn(1, 6, small_h, small_w).to(device)
        
        print(f"Calculating CNN FLOPs on small scale ({small_h}x{small_w})...")
        macs, _ = profile(get_context1, inputs=(dummy_input_small,), verbose=False)
        
        # --- 计算 Energy Cost ---
        # 我们需要手动聚合两部分的开销：
        # Part A: 小图上的 CNN + LUT1
        # Part B: 大图上的 LUT2 (Residual)
        
        # 1. CNN Energy (小图)
        E_MULT = 3.7; E_ADD = 0.9
        e_cnn = macs * (E_MULT + E_ADD)
        
        # 2. LUT Energy (SRAM)
        # 假设 MFlut_SpeedOptimized 在小图采样 N=4 个点 (MuLUT标准)
        # 假设 apply_fusion_2d_as_residual 在大图采样 N=4 个点
        E_SRAM = 5.0
        lut_sampling_points = 4 # <--- 根据你的LUT核大小修改 (例如 2x2=4)
        
        ops_lut1 = (small_h * small_w) * lut_sampling_points # 小图查表次数
        ops_lut2 = (dummy_h * dummy_w) * lut_sampling_points # 大图查表次数
        e_sram = (ops_lut1 + ops_lut2) * E_SRAM
        
        # 3. DRAM Energy (Input/Output)
        # 读 2张 4K图, 写 1张 4K图 (RGB)
        # 通道数 C=3
        E_DRAM = 640.0
        num_pixels_4k = dummy_h * dummy_w
        # 读: I_A, I_B (4K) -> 2 * H*W*3
        # 写: Result (4K)   -> 1 * H*W*3
        total_io_access = (num_pixels_4k * 3 * 3) 
        e_dram = total_io_access * E_DRAM
        
        total_energy = e_cnn + e_sram + e_dram
        
        print(f"\n[Energy Cost Analysis for 4K Image]")
        print(f"CNN MACs (Small Scale): {macs / 1e9:.4f} G")
        print(f"Energy Breakdown:")
        print(f"  - Compute (CNN): {e_cnn / 1e9:.2f} mJ")
        print(f"  - Memory (SRAM/LUT): {e_sram / 1e9:.2f} mJ")
        print(f"  - Memory (DRAM/IO):  {e_dram / 1e9:.2f} mJ")
        print(f"Total Energy Cost: {total_energy / 1e9:.4f} mJ")
        
    else:
        print("Warning: 'thop' library not found. Skipping FLOPs calculation.")
        print("You can install it via: pip install thop")
    print("-" * 30)
    # ==========================================

    guided_filter = FastGuidedFilter(r=8, eps=1e-4).to(device)
    
    data_transform = {
        "val": T.Compose([T.ToTensor()])
    }
    
    val_dataset = SimpleDataSet(visible_path=near_focus_dir,
                                infrared_path=far_focus_dir,
                                phase="val",
                                transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=val_dataset.collate_fn)

    times = []
    warmup_skip = 5
    print("starting (With Guided Filter)...")

    RES_THRESHOLD = 1024

    # ==========================================
    # Added: Reset Peak Memory Stats
    # ==========================================
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    # ==========================================

    with torch.no_grad():
        for step, data in enumerate(val_loader):
            I_A, I_B, task = data
            I_A = I_A.to(device)
            I_B = I_B.to(device)

            B, C, H, W = I_A.shape
            if max(H, W) < RES_THRESHOLD:
                I_A_small = I_A
                I_B_small = I_B
            else:
                scale_factor = 0.25 # 1/4
                small_H, small_W = int(H * scale_factor), int(W * scale_factor)
                I_A_small = F.interpolate(I_A, size=(small_H, small_W), mode='bilinear', align_corners=False)
                I_B_small = F.interpolate(I_B, size=(small_H, small_W), mode='bilinear', align_corners=False)
            
            torch.cuda.synchronize()
            start_time = time.time()

            out1_small = MFlut_SpeedOptimized(I_A_small * 255., I_B_small * 255., lut1, get_context1)
            mask_small = F.softmax(out1_small, dim=1)
            mask_big = F.interpolate(mask_small, size=(H, W), mode='bilinear', align_corners=False)
            
            mask_coarse = mask_big[:, 0:1, :, :]

            residual_mask = apply_fusion_2d_as_residual(I_A * 255., I_B * 255., lut2)

            mask_final = mask_coarse + residual_mask
            mask_final = torch.clamp(mask_final, 0, 1)
            fuse_res = mask_final * I_A + (1 - mask_final) * I_B

            torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)

            if not os.path.splitext(task[0])[1]:  
                task_with_extension = task[0] + ".png"  
            else:
                task_with_extension = task[0]
            save_path = os.path.join(save_dir, task_with_extension)

            fusion_result = fuse_res.squeeze(0).clamp(0, 1).cpu()
            fusion_result_image = ToPILImage()(fusion_result)
            fusion_result_image.save(save_path)                      

    # ==========================================
    # Added: Print Peak Memory Usage
    # ==========================================
    print("-" * 30)
    if len(times) > warmup_skip:
        times_after_warmup = times[warmup_skip:]
        avg_time = np.mean(times_after_warmup)
        std_time = np.std(times_after_warmup)
        print(f"处理完成！跳过前 {warmup_skip} 张图像后，平均时间: {avg_time:.6f} 秒，时间标准差: {std_time:.6f} 秒")
    else:
        print(f"图像数量不足以跳过前 {warmup_skip} 张！总图像数: {len(times)}")

    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert Bytes to MB
        print(f"Peak GPU Memory Usage: {max_memory:.2f} MB")
    print("-" * 30)
    # ==========================================

if __name__ == "__main__":
    main()

