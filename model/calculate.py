import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
import cv2
import kornia.filters as KF
import kornia


def gumbel_softmax_sample(logits, temperature=0.5, hard=True):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    y = F.softmax((logits + gumbel_noise) / temperature, dim=1)
    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
        # Straight-through: use hard mask in forward, soft grad in backward
        y = (y_hard - y).detach() + y
    return y

def rgb_to_ycbcr(img):
    return torch.stack(
        (0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
               128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
               128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
               dim=1)


def ycbcr_to_rgb(img):
    return torch.stack(
         (img[:, 0, :, :] + (img[:, 2, :, :] - 0.5) * 1.402,
                img[:, 0, :, :] - (img[:, 1, :, :] - 0.5) * 0.344136 - (img[:, 2, :, :] - 0.5) * 0.714136,
                img[:, 0, :, :] + (img[:, 1, :, :] - 0.5) * 1.772),
                dim=1)


def init_weights_recommended(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def load_partial_weights(model, pretrained_state_dict):
    model_dict = model.state_dict()
    matched_dict = {}

    for k, v in pretrained_state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                matched_dict[k] = v  
            else:
                print(f"Skipping layer (shape mismatch): {k}, model: {model_dict[k].shape}, pretrained: {v.shape}")
        else:
            print(f"Skipping layer (not found in model): {k}")
    
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    print(f"?? Loaded {len(matched_dict)} / {len(pretrained_state_dict)} layers.")


def fuse_cbcr(output, I_A, I_B):
    
    ycbcr_A = rgb_to_ycbcr(I_A)
    ycbcr_B = rgb_to_ycbcr(I_B)
    
    cb_A = ycbcr_A[:, 1, :, :]  # [B, H, W]
    cr_A = ycbcr_A[:, 2, :, :]
    
    cb_B = ycbcr_B[:, 1, :, :]
    cr_B = ycbcr_B[:, 2, :, :]

    # 
    x1_cb = cb_A
    x2_cb = cb_B
    delta1_cb = torch.abs(x1_cb - 0.5)
    delta2_cb = torch.abs(x2_cb - 0.5)

    denominator_cb = delta1_cb + delta2_cb
    cbf = torch.where(
        denominator_cb == 0,
        torch.full_like(x1_cb, 0.5),  # 
        (x1_cb * delta1_cb + x2_cb * delta2_cb) / denominator_cb
    )

    # 
    x1_cr = cr_A
    x2_cr = cr_B
    delta1_cr = torch.abs(x1_cr - 0.5)
    delta2_cr = torch.abs(x2_cr - 0.5)

    denominator_cr = delta1_cr + delta2_cr
    crf = torch.where(
        denominator_cr == 0,
        torch.full_like(x1_cr, 0.5),
        (x1_cr * delta1_cr + x2_cr * delta2_cr) / denominator_cr
    )

    fused_ycbcr = torch.cat([output, cbf.unsqueeze(1), crf.unsqueeze(1)], dim=1)  # [B, 3, H, W]

    fused_rgb = ycbcr_to_rgb(fused_ycbcr).clamp(0.0, 1.0)

    return fused_rgb


def save_generator_context(generator_context, save_path="generator_context.pth"):

    torch.save(generator_context.state_dict(), save_path)
    print(f"Generator_for_info weights saved to {save_path}")
    
def save_weight_predictor(weight_predictor, save_path="weight_predictor.pth"):

    torch.save(weight_predictor.state_dict(), save_path)
    print(f"weight_predictor weights saved to {save_path}")


def save_lut(lut_module, path):

    lut_weights = lut_module().detach().cpu().numpy()
    np.save(path, lut_weights)
    print(f"Fine-tuned LUT saved to {path}")


def FDlut_with_contextenc(visible_img, infrared_img, lut, get_context):        # [0, 255]  lut1, lut2, lut3, Generator_context1, Generator_context2, Generator_context3

    image_cat = torch.cat((visible_img, infrared_img), dim=1)  # [0, 255]
    context = get_context(image_cat)

    context_scaled = (context * 255. / 16.0).squeeze(1)         # [0, 16]
    infrared_scaled = infrared_img / 16.0                # [0, 16]
    ycbcr_vis = rgb_to_ycbcr(visible_img/255.)           # [0, 1]
    ycbcr_vis_scaled = ycbcr_vis * 255.0/16.0                  # [0, 16]

    y_vi_scaled = ycbcr_vis_scaled[:, 0, :, :]  # [b, 1, h, w]   # [0, 16]
    cb_cr = ycbcr_vis[:, 1:, :, :]              # [0, 1]
    

    if infrared_img.shape[1] == 1:
        ir_scaled = infrared_scaled[:, 0, :, :] / 16.0 
    else:
        ir_gray = rgb_to_ycbcr(infrared_img/255.)
        ir_scaled = ir_gray * 255.0/16.0
        ir_scaled = ir_scaled[:, 0, :, :]  

    # 
    ir_floor = torch.floor(ir_scaled).long()
    ir_ceil = torch.clamp(ir_floor + 1, 0, lut.shape[3] - 1)
    ir_alpha = ir_scaled - ir_floor

    y_vi_floor = torch.floor(y_vi_scaled).long()
    y_vi_ceil = torch.clamp(y_vi_floor + 1, 0, lut.shape[0] - 1)
    y_vi_alpha = y_vi_scaled - y_vi_floor

    c_floor = torch.floor(context_scaled).long()
    c_ceil = torch.clamp(c_floor + 1, 0, lut.shape[2] - 1)
    c_alpha = context_scaled - c_floor

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        visible_img.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        visible_img.device)

    grad_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)  
    grad_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # [b, 1, h, w]
    min_val = gradient.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_val = gradient.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized = (gradient - min_val) / (max_val - min_val + 1e-8)  

    gradient_scaled = (gradient_normalized * 255.)
    gradient_scaled = (gradient_scaled / 16.0).squeeze(1)

    g_floor = torch.floor(gradient_scaled).long()
    g_ceil = torch.clamp(g_floor + 1, 0, lut.shape[1] - 1)
    g_alpha = gradient_scaled - g_floor

    ir_alpha = ir_alpha.unsqueeze(-1)
    y_vi_alpha = y_vi_alpha.unsqueeze(-1)
    g_alpha = g_alpha.unsqueeze(-1)
    c_alpha = c_alpha.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[y_vi_floor, g_floor, c_floor, ir_floor], lut[y_vi_floor, g_floor, c_floor, ir_ceil], ir_alpha),
                    lerp(lut[y_vi_floor, g_floor, c_ceil, ir_floor], lut[y_vi_floor, g_floor, c_ceil, ir_ceil], ir_alpha),
                    c_alpha,
                ),
                lerp(
                    lerp(lut[y_vi_floor, g_ceil, c_floor, ir_floor], lut[y_vi_floor, g_ceil, c_floor, ir_ceil], ir_alpha),
                    lerp(lut[y_vi_floor, g_ceil, c_ceil, ir_floor], lut[y_vi_floor, g_ceil, c_ceil, ir_ceil], ir_alpha),
                    c_alpha,
                ),
                g_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[y_vi_ceil, g_floor, c_floor, ir_floor], lut[y_vi_ceil, g_floor, c_floor, ir_ceil], ir_alpha),
                    lerp(lut[y_vi_ceil, g_floor, c_ceil, ir_floor], lut[y_vi_ceil, g_floor, c_ceil, ir_ceil], ir_alpha),
                    c_alpha,
                ),
                lerp(
                    lerp(lut[y_vi_ceil, g_ceil, c_floor, ir_floor], lut[y_vi_ceil, g_ceil, c_floor, ir_ceil], ir_alpha),
                    lerp(lut[y_vi_ceil, g_ceil, c_ceil, ir_floor], lut[y_vi_ceil, g_ceil, c_ceil, ir_ceil], ir_alpha),
                    c_alpha,
                ),
                g_alpha,
            ),
            y_vi_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)

    return fusion_y


# def MFlut_(model_x, model_y, lut):       # 强度 强度 梯度 梯度

#     # image_cat = torch.cat((model_x, model_y), dim=1)  # [0, 255]
#     # context = get_context(image_cat)

#     # context_scaled = (context * 255. / 16.0).squeeze(1)         # [0, 16]
#     # infrared_scaled = model_y / 16.0                # [0, 16]
#     ycbcr_vis = rgb_to_ycbcr(model_x/255.)           # [0, 1]
#     ycbcr_vis_scaled = ycbcr_vis * 255.0/16.0                  # [0, 16]

#     y_vi_scaled = ycbcr_vis_scaled[:, 0, :, :]  # [b, 1, h, w]   # [0, 16]
#     cb_cr = ycbcr_vis[:, 1:, :, :]              # [0, 1]
    

#     if model_y.shape[1] == 1:
#         ir_scaled = model_y[:, 0, :, :] / 16.0 
#     else:
#         ir_gray = rgb_to_ycbcr(model_y/255.)
#         ir_scaled = ir_gray * 255.0/16.0
#         ir_scaled = ir_scaled[:, 0, :, :]  

#     ir_floor = torch.floor(ir_scaled).long()
#     ir_ceil = torch.clamp(ir_floor + 1, 0, lut.shape[3] - 1)
#     ir_alpha = ir_scaled - ir_floor

#     y_vi_floor = torch.floor(y_vi_scaled).long()
#     y_vi_ceil = torch.clamp(y_vi_floor + 1, 0, lut.shape[0] - 1)
#     y_vi_alpha = y_vi_scaled - y_vi_floor

#     # c_floor = torch.floor(context_scaled).long()
#     # c_ceil = torch.clamp(c_floor + 1, 0, lut.shape[2] - 1)
#     # c_alpha = context_scaled - c_floor

#     sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
#         model_x.device)
#     sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
#         model_x.device)

#     grad_near_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)  
#     grad_near_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

#     gradient_near = torch.sqrt(grad_near_x ** 2 + grad_near_y ** 2)  # [b, 1, h, w]
#     min_near = gradient_near.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
#     max_near = gradient_near.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

#     gradient_normalized_near = (gradient_near - min_near) / (max_near - min_near + 1e-8)  

#     gradient_scaled_near = (gradient_normalized_near * 255.)
#     gradient_scaled_near = (gradient_scaled_near / 16.0).squeeze(1)

#     g_near_floor = torch.floor(gradient_scaled_near).long()
#     g_near_ceil = torch.clamp(g_near_floor + 1, 0, lut.shape[1] - 1)
#     g_near_alpha = gradient_scaled_near - g_near_floor


#     grad_far_x = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_x, padding=1)  
#     grad_far_y = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_y, padding=1)

#     gradient_far = torch.sqrt(grad_far_x ** 2 + grad_far_y ** 2)  # [b, 1, h, w]
#     min_far = gradient_far.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
#     max_far = gradient_far.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

#     gradient_normalized_far = (gradient_far - min_far) / (max_far - min_far + 1e-8)  

#     gradient_scaled_far = (gradient_normalized_far * 255.)
#     gradient_scaled_far = (gradient_scaled_far / 16.0).squeeze(1)

#     g_far_floor = torch.floor(gradient_scaled_far).long()
#     g_far_ceil = torch.clamp(g_far_floor + 1, 0, lut.shape[1] - 1)
#     g_far_alpha = gradient_scaled_far - g_far_floor


#     ir_alpha = ir_alpha.unsqueeze(-1)
#     y_vi_alpha = y_vi_alpha.unsqueeze(-1)
#     g_near_alpha = g_near_alpha.unsqueeze(-1)
#     g_far_alpha = g_far_alpha.unsqueeze(-1)


#     def lerp(v1, v2, alpha):
#         out = v1 * (1 - alpha) + v2 * alpha
#         return out

#     fusion_result = (
#         lerp(
#             lerp(
#                 lerp(
#                     lerp(lut[y_vi_floor, g_near_floor, g_far_floor, ir_floor], lut[y_vi_floor, g_near_floor, g_far_floor, ir_ceil], ir_alpha),
#                     lerp(lut[y_vi_floor, g_near_floor, g_far_ceil, ir_floor], lut[y_vi_floor, g_near_floor, g_far_ceil, ir_ceil], ir_alpha),
#                     g_far_alpha,
#                 ),
#                 lerp(
#                     lerp(lut[y_vi_floor, g_near_ceil, g_far_floor, ir_floor], lut[y_vi_floor, g_near_ceil, g_far_floor, ir_ceil], ir_alpha),
#                     lerp(lut[y_vi_floor, g_near_ceil, g_far_ceil, ir_floor], lut[y_vi_floor, g_near_ceil, g_far_ceil, ir_ceil], ir_alpha),
#                     g_far_alpha,
#                 ),
#                 g_near_alpha,
#             ),
#             lerp(
#                 lerp(
#                     lerp(lut[y_vi_ceil, g_near_floor, g_far_floor, ir_floor], lut[y_vi_ceil, g_near_floor, g_far_floor, ir_ceil], ir_alpha),
#                     lerp(lut[y_vi_ceil, g_near_floor, g_far_ceil, ir_floor], lut[y_vi_ceil, g_near_floor, g_far_ceil, ir_ceil], ir_alpha),
#                     g_far_alpha,
#                 ),
#                 lerp(
#                     lerp(lut[y_vi_ceil, g_near_ceil, g_far_floor, ir_floor], lut[y_vi_ceil, g_near_ceil, g_far_floor, ir_ceil], ir_alpha),
#                     lerp(lut[y_vi_ceil, g_near_ceil, g_far_ceil, ir_floor], lut[y_vi_ceil, g_near_ceil, g_far_ceil, ir_ceil], ir_alpha),
#                     g_far_alpha,
#                 ),
#                 g_near_alpha,
#             ),
#             y_vi_alpha,
#         )
#     )

#     fusion_y = fusion_result.permute(0, 3, 1, 2)
#     return fusion_y



def MFlut_2(model_x, model_y, lut, get_context):       # 强度 梯度 梯度 场景信息

    image_cat = torch.cat((model_x, model_y), dim=1)  # [0, 255]
    
    context = get_context(image_cat)

    context_scaled = (context * 255. / 16.0).squeeze(1)         # [0, 16]

    ycbcr_vis = rgb_to_ycbcr(model_x/255.)           # [0, 1]
    ycbcr_vis_scaled = ycbcr_vis * 255.0/16.0                  # [0, 16]

    y_vi_scaled = ycbcr_vis_scaled[:, 0, :, :]  # [b, 1, h, w]   # [0, 16]
    
    

    if model_y.shape[1] == 1:
        ir_scaled = model_y[:, 0, :, :] / 16.0 
    else:
        ir_gray = rgb_to_ycbcr(model_y/255.)
        ir_scaled = ir_gray * 255.0/16.0
        ir_scaled = ir_scaled[:, 0, :, :]  

    # ir_floor = torch.floor(ir_scaled).long()
    # ir_ceil = torch.clamp(ir_floor + 1, 0, lut.shape[3] - 1)
    # ir_alpha = ir_scaled - ir_floor

    y_vi_floor = torch.floor(y_vi_scaled).long()
    y_vi_ceil = torch.clamp(y_vi_floor + 1, 0, lut.shape[0] - 1)
    y_vi_alpha = y_vi_scaled - y_vi_floor

    c_floor = torch.floor(context_scaled).long()
    c_ceil = torch.clamp(c_floor + 1, 0, lut.shape[2] - 1)
    c_alpha = context_scaled - c_floor

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        model_x.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        model_x.device)

    grad_near_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)  
    grad_near_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

    gradient_near = torch.sqrt(grad_near_x ** 2 + grad_near_y ** 2)  # [b, 1, h, w]
    min_near = gradient_near.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_near = gradient_near.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized_near = (gradient_near - min_near) / (max_near - min_near + 1e-8)  

    gradient_scaled_near = (gradient_normalized_near * 255.)
    gradient_scaled_near = (gradient_scaled_near / 16.0).squeeze(1)

    g_near_floor = torch.floor(gradient_scaled_near).long()
    g_near_ceil = torch.clamp(g_near_floor + 1, 0, lut.shape[1] - 1)
    g_near_alpha = gradient_scaled_near - g_near_floor


    grad_far_x = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_x, padding=1)  
    grad_far_y = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_y, padding=1)

    gradient_far = torch.sqrt(grad_far_x ** 2 + grad_far_y ** 2)  # [b, 1, h, w]
    min_far = gradient_far.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_far = gradient_far.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized_far = (gradient_far - min_far) / (max_far - min_far + 1e-8)  

    gradient_scaled_far = (gradient_normalized_far * 255.)
    gradient_scaled_far = (gradient_scaled_far / 16.0).squeeze(1)

    g_far_floor = torch.floor(gradient_scaled_far).long()
    g_far_ceil = torch.clamp(g_far_floor + 1, 0, lut.shape[1] - 1)
    g_far_alpha = gradient_scaled_far - g_far_floor


    # ir_alpha = ir_alpha.unsqueeze(-1)
    y_vi_alpha = y_vi_alpha.unsqueeze(-1)
    g_near_alpha = g_near_alpha.unsqueeze(-1)
    g_far_alpha = g_far_alpha.unsqueeze(-1)
    c_alpha = c_alpha.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[y_vi_floor, g_near_floor, g_far_floor, c_floor], lut[y_vi_floor, g_near_floor, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[y_vi_floor, g_near_floor, g_far_ceil, c_floor], lut[y_vi_floor, g_near_floor, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                lerp(
                    lerp(lut[y_vi_floor, g_near_ceil, g_far_floor, c_floor], lut[y_vi_floor, g_near_ceil, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[y_vi_floor, g_near_ceil, g_far_ceil, c_floor], lut[y_vi_floor, g_near_ceil, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                g_near_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[y_vi_ceil, g_near_floor, g_far_floor, c_floor], lut[y_vi_ceil, g_near_floor, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[y_vi_ceil, g_near_floor, g_far_ceil, c_floor], lut[y_vi_ceil, g_near_floor, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                lerp(
                    lerp(lut[y_vi_ceil, g_near_ceil, g_far_floor, c_floor], lut[y_vi_ceil, g_near_ceil, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[y_vi_ceil, g_near_ceil, g_far_ceil, c_floor], lut[y_vi_ceil, g_near_ceil, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                g_near_alpha,
            ),
            y_vi_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y


def MFlut_2_new(model_x, model_y, lut, get_context):       # laplacian 梯度 梯度 场景信息

    image_cat = torch.cat((model_x, model_y), dim=1)  # [0, 255]

    context = get_context(image_cat)

    context_scaled = (context * 255. / 16.0).squeeze(1)         # [0, 16]
    infrared_scaled = model_y / 16.0                # [0, 16]
    ycbcr_vis = rgb_to_ycbcr(model_x/255.)           # [0, 1]
    ycbcr_vis_scaled = ycbcr_vis * 255.0/16.0                  # [0, 16]

    y_vi_scaled = ycbcr_vis[:, 0, :, :]  # [b, 1, h, w]   # [0, 1]
    y_vi_scaled = y_vi_scaled.unsqueeze(1)
    cb_cr = ycbcr_vis[:, 1:, :, :]              # [0, 1]

    lap = KF.laplacian(y_vi_scaled, kernel_size=3)
    lap = lap.squeeze(1) 
    lap_norm = (lap - lap.min()) / (lap.max() - lap.min()+ 1e-5)     # 归一化 0-1 
    lap_scaled = lap_norm * 16.0

    if model_y.shape[1] == 1:
        ir_scaled = model_y[:, 0, :, :] / 16.0 
    else:
        ir_gray = rgb_to_ycbcr(model_y/255.)
        ir_scaled = ir_gray * 255.0/16.0
        ir_scaled = ir_scaled[:, 0, :, :]  
    
    lap_floor = torch.floor(lap_scaled).long()
    lap_ceil = torch.clamp(lap_floor + 1, 0, lut.shape[0] - 1)
    lap_alpha = lap_scaled - lap_floor

    c_floor = torch.floor(context_scaled).long()
    c_ceil = torch.clamp(c_floor + 1, 0, lut.shape[2] - 1)
    c_alpha = context_scaled - c_floor

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        model_x.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        model_x.device)

    grad_near_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)  
    grad_near_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

    gradient_near = torch.sqrt(grad_near_x ** 2 + grad_near_y ** 2)  # [b, 1, h, w]
    min_near = gradient_near.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_near = gradient_near.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized_near = (gradient_near - min_near) / (max_near - min_near + 1e-8)  

    gradient_scaled_near = (gradient_normalized_near * 255.)
    gradient_scaled_near = (gradient_scaled_near / 16.0).squeeze(1)

    g_near_floor = torch.floor(gradient_scaled_near).long()
    g_near_ceil = torch.clamp(g_near_floor + 1, 0, lut.shape[1] - 1)
    g_near_alpha = gradient_scaled_near - g_near_floor


    grad_far_x = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_x, padding=1)  
    grad_far_y = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_y, padding=1)

    gradient_far = torch.sqrt(grad_far_x ** 2 + grad_far_y ** 2)  # [b, 1, h, w]
    min_far = gradient_far.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_far = gradient_far.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized_far = (gradient_far - min_far) / (max_far - min_far + 1e-8)  

    gradient_scaled_far = (gradient_normalized_far * 255.)
    gradient_scaled_far = (gradient_scaled_far / 16.0).squeeze(1)

    g_far_floor = torch.floor(gradient_scaled_far).long()
    g_far_ceil = torch.clamp(g_far_floor + 1, 0, lut.shape[1] - 1)
    g_far_alpha = gradient_scaled_far - g_far_floor

    lap_alpha = lap_alpha.unsqueeze(-1)
    g_near_alpha = g_near_alpha.unsqueeze(-1)
    g_far_alpha = g_far_alpha.unsqueeze(-1)
    c_alpha = c_alpha.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[lap_floor, g_near_floor, g_far_floor, c_floor], lut[lap_floor, g_near_floor, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_floor, g_near_floor, g_far_ceil, c_floor], lut[lap_floor, g_near_floor, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                lerp(
                    lerp(lut[lap_floor, g_near_ceil, g_far_floor, c_floor], lut[lap_floor, g_near_ceil, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_floor, g_near_ceil, g_far_ceil, c_floor], lut[lap_floor, g_near_ceil, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                g_near_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[lap_ceil, g_near_floor, g_far_floor, c_floor], lut[lap_ceil, g_near_floor, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_ceil, g_near_floor, g_far_ceil, c_floor], lut[lap_ceil, g_near_floor, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                lerp(
                    lerp(lut[lap_ceil, g_near_ceil, g_far_floor, c_floor], lut[lap_ceil, g_near_ceil, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_ceil, g_near_ceil, g_far_ceil, c_floor], lut[lap_ceil, g_near_ceil, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                g_near_alpha,
            ),
            lap_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y


def MFlut_SpeedOptimized(image_A, image_B, lut, get_context=None):       # laplacian 梯度 梯度 场景信息

    if image_A.shape[1] == 3:
        Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
        Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]
    else:
        Y_A = image_A
        Y_B = image_B
    
    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_A.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_A.device).view(1, 1, 3, 3).float()

    grad_A_x = F.conv2d(Y_A, sobel_x, padding=1)
    grad_A_y = F.conv2d(Y_A, sobel_y, padding=1)
    grad_B_x = F.conv2d(Y_B, sobel_x, padding=1)
    grad_B_y = F.conv2d(Y_B, sobel_y, padding=1)

    G_A = torch.abs(grad_A_x) + torch.abs(grad_A_y)
    G_B = torch.abs(grad_B_x) + torch.abs(grad_B_y)

    diff_high = torch.tanh((G_A - G_B) * 10.0) 

    idx_1 = (diff_high + 1.0) * 0.5 * 255/16.0
        
    idx_1 = idx_1.squeeze(1)
    idx_1_floor = torch.floor(idx_1).long()
    idx_1_ceil = torch.clamp(idx_1_floor + 1, 0, lut.shape[0] - 1)
    idx_1_alpha = idx_1 - idx_1_floor
    idx_1_alpha = idx_1_alpha.unsqueeze(-1)
    
    E_A = F.avg_pool2d(G_A, kernel_size=15, stride=1, padding=7)
    E_B = F.avg_pool2d(G_B, kernel_size=15, stride=1, padding=7)
    diff_low = torch.tanh((E_A - E_B) * 10.0) 
    idx_2 = (diff_low + 1.0) * 0.5 * 255/ 16.0
    
    idx_2 = idx_2.squeeze(1)
    idx_2_floor = torch.floor(idx_2).long()
    idx_2_ceil = torch.clamp(idx_2_floor + 1, 0, lut.shape[1] - 1)
    idx_2_alpha = idx_2 - idx_2_floor
    idx_2_alpha = idx_2_alpha.unsqueeze(-1)

    max_g = torch.max(G_A, G_B)
    idx_3 = torch.tanh(max_g) * 255/16.0
    
    idx_3 = idx_3.squeeze(1)
    idx_3_floor = torch.floor(idx_3).long()
    idx_3_ceil = torch.clamp(idx_3_floor + 1, 0, lut.shape[2] - 1)
    idx_3_alpha = idx_3 - idx_3_floor
    idx_3_alpha = idx_3_alpha.unsqueeze(-1)
    
    ctx = get_context(torch.cat([image_A, image_B], dim=1))
    idx_4 = torch.clamp(ctx, 0, 1) * 255/16.0

    idx_4 = idx_4.squeeze(1)
    idx_4_floor = torch.floor(idx_4).long()
    idx_4_ceil = torch.clamp(idx_4_floor + 1, 0, lut.shape[3] - 1)
    idx_4_alpha = idx_4 - idx_4_floor
    idx_4_alpha = idx_4_alpha.unsqueeze(-1)

    # print("index4_min: ",idx_4_floor.min())
    # print("index4_max: ",idx_4_ceil.max())

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[idx_1_floor, idx_2_floor, idx_3_floor, idx_4_floor], lut[idx_1_floor, idx_2_floor, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_floor, idx_2_floor, idx_3_ceil, idx_4_floor], lut[idx_1_floor, idx_2_floor, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                lerp(
                    lerp(lut[idx_1_floor, idx_2_ceil, idx_3_floor, idx_4_floor], lut[idx_1_floor, idx_2_ceil, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_floor, idx_2_ceil, idx_3_ceil, idx_4_floor], lut[idx_1_floor, idx_2_ceil, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                idx_2_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[idx_1_ceil, idx_2_floor, idx_3_floor, idx_4_floor], lut[idx_1_ceil, idx_2_floor, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_ceil, idx_2_floor, idx_3_ceil, idx_4_floor], lut[idx_1_ceil, idx_2_floor, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                lerp(
                    lerp(lut[idx_1_ceil, idx_2_ceil, idx_3_floor, idx_4_floor], lut[idx_1_ceil, idx_2_ceil, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_ceil, idx_2_ceil, idx_3_ceil, idx_4_floor], lut[idx_1_ceil, idx_2_ceil, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                idx_2_alpha,
            ),
            idx_1_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y


def MFlut_SpeedOptimized_5d(image_A, image_B, img_depth, lut, get_context=None):       # laplacian 梯度 梯度 场景信息

    if image_A.shape[1] == 3:
        Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
        Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]
        Y_depth = 0.299 * img_depth[:, 0:1, :, :] + 0.587 * img_depth[:, 1:2, :, :] + 0.114 * img_depth[:, 2:3, :, :]
    else:
        Y_A = image_A
        Y_B = image_B
        Y_depth = img_depth
    
    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0
    Y_depth = Y_depth / 255.0
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_A.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_A.device).view(1, 1, 3, 3).float()

    grad_A_x = F.conv2d(Y_A, sobel_x, padding=1)
    grad_A_y = F.conv2d(Y_A, sobel_y, padding=1)
    grad_B_x = F.conv2d(Y_B, sobel_x, padding=1)
    grad_B_y = F.conv2d(Y_B, sobel_y, padding=1)

    G_A = torch.abs(grad_A_x) + torch.abs(grad_A_y)
    G_B = torch.abs(grad_B_x) + torch.abs(grad_B_y)

    diff_high = torch.tanh((G_A - G_B) * 10.0) 

    idx_1 = (diff_high + 1.0) * 0.5 * 255/16.0
        
    idx_1 = idx_1.squeeze(1)
    idx_1_floor = torch.floor(idx_1).long()
    idx_1_ceil = torch.clamp(idx_1_floor + 1, 0, lut.shape[0] - 1)
    idx_1_alpha = idx_1 - idx_1_floor
    idx_1_alpha = idx_1_alpha.unsqueeze(-1)
    
    E_A = F.avg_pool2d(G_A, kernel_size=15, stride=1, padding=7)
    E_B = F.avg_pool2d(G_B, kernel_size=15, stride=1, padding=7)
    diff_low = torch.tanh((E_A - E_B) * 10.0) 
    idx_2 = (diff_low + 1.0) * 0.5 * 255/ 16.0
    
    idx_2 = idx_2.squeeze(1)
    idx_2_floor = torch.floor(idx_2).long()
    idx_2_ceil = torch.clamp(idx_2_floor + 1, 0, lut.shape[1] - 1)
    idx_2_alpha = idx_2 - idx_2_floor
    idx_2_alpha = idx_2_alpha.unsqueeze(-1)

    max_g = torch.max(G_A, G_B)
    idx_3 = torch.tanh(max_g) * 255/16.0
    
    idx_3 = idx_3.squeeze(1)
    idx_3_floor = torch.floor(idx_3).long()
    idx_3_ceil = torch.clamp(idx_3_floor + 1, 0, lut.shape[2] - 1)
    idx_3_alpha = idx_3 - idx_3_floor
    idx_3_alpha = idx_3_alpha.unsqueeze(-1)
    
    ctx = get_context(torch.cat([image_A, image_B], dim=1))
    idx_4 = torch.clamp(ctx, 0, 1) * 255/16.0

    idx_4 = idx_4.squeeze(1)
    idx_4_floor = torch.floor(idx_4).long()
    idx_4_ceil = torch.clamp(idx_4_floor + 1, 0, lut.shape[3] - 1)
    idx_4_alpha = idx_4 - idx_4_floor
    idx_4_alpha = idx_4_alpha.unsqueeze(-1)
    
    
    idx_5 = torch.clamp(Y_depth, 0, 1) * 255 / 16.0

    idx_5 = idx_5.squeeze(1)
    idx_5_floor = torch.floor(idx_5).long()
    idx_5_ceil = torch.clamp(idx_5_floor + 1, 0, lut.shape[4] - 1)
    idx_5_floor = torch.clamp(idx_5_floor, 0, lut.shape[4] - 1)
    idx_5_alpha = (idx_5 - idx_5_floor.float()).unsqueeze(-1)


    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    def interp4_at_depth(idx_5_fixed):

        return lerp(
            lerp(
                lerp(
                    lerp(
                        lut[idx_1_floor, idx_2_floor, idx_3_floor, idx_4_floor, idx_5_fixed],
                        lut[idx_1_floor, idx_2_floor, idx_3_floor, idx_4_ceil,  idx_5_fixed],
                        idx_4_alpha
                    ),
                    lerp(
                        lut[idx_1_floor, idx_2_floor, idx_3_ceil, idx_4_floor, idx_5_fixed],
                        lut[idx_1_floor, idx_2_floor, idx_3_ceil, idx_4_ceil,  idx_5_fixed],
                        idx_4_alpha
                    ),
                    idx_3_alpha
                ),
                lerp(
                    lerp(
                        lut[idx_1_floor, idx_2_ceil, idx_3_floor, idx_4_floor, idx_5_fixed],
                        lut[idx_1_floor, idx_2_ceil, idx_3_floor, idx_4_ceil,  idx_5_fixed],
                        idx_4_alpha
                    ),
                    lerp(
                        lut[idx_1_floor, idx_2_ceil, idx_3_ceil, idx_4_floor, idx_5_fixed],
                        lut[idx_1_floor, idx_2_ceil, idx_3_ceil, idx_4_ceil,  idx_5_fixed],
                        idx_4_alpha
                    ),
                    idx_3_alpha
                ),
                idx_2_alpha
            ),
            lerp(
                lerp(
                    lerp(
                        lut[idx_1_ceil, idx_2_floor, idx_3_floor, idx_4_floor, idx_5_fixed],
                        lut[idx_1_ceil, idx_2_floor, idx_3_floor, idx_4_ceil,  idx_5_fixed],
                        idx_4_alpha
                    ),
                    lerp(
                        lut[idx_1_ceil, idx_2_floor, idx_3_ceil, idx_4_floor, idx_5_fixed],
                        lut[idx_1_ceil, idx_2_floor, idx_3_ceil, idx_4_ceil,  idx_5_fixed],
                        idx_4_alpha
                    ),
                    idx_3_alpha
                ),
                lerp(
                    lerp(
                        lut[idx_1_ceil, idx_2_ceil, idx_3_floor, idx_4_floor, idx_5_fixed],
                        lut[idx_1_ceil, idx_2_ceil, idx_3_floor, idx_4_ceil,  idx_5_fixed],
                        idx_4_alpha
                    ),
                    lerp(
                        lut[idx_1_ceil, idx_2_ceil, idx_3_ceil, idx_4_floor, idx_5_fixed],
                        lut[idx_1_ceil, idx_2_ceil, idx_3_ceil, idx_4_ceil,  idx_5_fixed],
                        idx_4_alpha
                    ),
                    idx_3_alpha
                ),
                idx_2_alpha
            ),
            idx_1_alpha
        )

    fusion_depth_floor = interp4_at_depth(idx_5_floor)
    fusion_depth_ceil = interp4_at_depth(idx_5_ceil)

    # 再沿 depth 维度做第 5 次线性插值
    fusion_result = lerp(
        fusion_depth_floor,
        fusion_depth_ceil,
        idx_5_alpha
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)

    return fusion_y


def MFlut_SpeedOptimized_bin8(image_A, image_B, lut, get_context=None):       # laplacian 梯度 梯度 场景信息

    if image_A.shape[1] == 3:
        Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
        Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]
    else:
        Y_A = image_A
        Y_B = image_B
    
    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_A.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_A.device).view(1, 1, 3, 3).float()

    grad_A_x = F.conv2d(Y_A, sobel_x, padding=1)
    grad_A_y = F.conv2d(Y_A, sobel_y, padding=1)
    grad_B_x = F.conv2d(Y_B, sobel_x, padding=1)
    grad_B_y = F.conv2d(Y_B, sobel_y, padding=1)

    G_A = torch.abs(grad_A_x) + torch.abs(grad_A_y)
    G_B = torch.abs(grad_B_x) + torch.abs(grad_B_y)

    diff_high = torch.tanh((G_A - G_B) * 10.0) 

    idx_1 = (diff_high + 1.0) * 0.5 * 255/32.0
        
    idx_1 = idx_1.squeeze(1)
    idx_1_floor = torch.floor(idx_1).long()
    idx_1_ceil = torch.clamp(idx_1_floor + 1, 0, lut.shape[0] - 1)
    idx_1_alpha = idx_1 - idx_1_floor
    idx_1_alpha = idx_1_alpha.unsqueeze(-1)
    
    E_A = F.avg_pool2d(G_A, kernel_size=15, stride=1, padding=7)
    E_B = F.avg_pool2d(G_B, kernel_size=15, stride=1, padding=7)
    diff_low = torch.tanh((E_A - E_B) * 10.0) 
    idx_2 = (diff_low + 1.0) * 0.5 * 255/ 32.0
    
    idx_2 = idx_2.squeeze(1)
    idx_2_floor = torch.floor(idx_2).long()
    idx_2_ceil = torch.clamp(idx_2_floor + 1, 0, lut.shape[1] - 1)
    idx_2_alpha = idx_2 - idx_2_floor
    idx_2_alpha = idx_2_alpha.unsqueeze(-1)

    max_g = torch.max(G_A, G_B)
    idx_3 = torch.tanh(max_g) * 255/32.0
    
    idx_3 = idx_3.squeeze(1)
    idx_3_floor = torch.floor(idx_3).long()
    idx_3_ceil = torch.clamp(idx_3_floor + 1, 0, lut.shape[2] - 1)
    idx_3_alpha = idx_3 - idx_3_floor
    idx_3_alpha = idx_3_alpha.unsqueeze(-1)
    
    ctx = get_context(torch.cat([image_A, image_B], dim=1))
    idx_4 = torch.clamp(ctx, 0, 1) * 255/32.0

    idx_4 = idx_4.squeeze(1)
    idx_4_floor = torch.floor(idx_4).long()
    idx_4_ceil = torch.clamp(idx_4_floor + 1, 0, lut.shape[3] - 1)
    idx_4_alpha = idx_4 - idx_4_floor
    idx_4_alpha = idx_4_alpha.unsqueeze(-1)

    # print("index4_min: ",idx_4_floor.min())
    # print("index4_max: ",idx_4_ceil.max())

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[idx_1_floor, idx_2_floor, idx_3_floor, idx_4_floor], lut[idx_1_floor, idx_2_floor, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_floor, idx_2_floor, idx_3_ceil, idx_4_floor], lut[idx_1_floor, idx_2_floor, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                lerp(
                    lerp(lut[idx_1_floor, idx_2_ceil, idx_3_floor, idx_4_floor], lut[idx_1_floor, idx_2_ceil, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_floor, idx_2_ceil, idx_3_ceil, idx_4_floor], lut[idx_1_floor, idx_2_ceil, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                idx_2_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[idx_1_ceil, idx_2_floor, idx_3_floor, idx_4_floor], lut[idx_1_ceil, idx_2_floor, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_ceil, idx_2_floor, idx_3_ceil, idx_4_floor], lut[idx_1_ceil, idx_2_floor, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                lerp(
                    lerp(lut[idx_1_ceil, idx_2_ceil, idx_3_floor, idx_4_floor], lut[idx_1_ceil, idx_2_ceil, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_ceil, idx_2_ceil, idx_3_ceil, idx_4_floor], lut[idx_1_ceil, idx_2_ceil, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                idx_2_alpha,
            ),
            idx_1_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y


def MFlut_SpeedOptimized_bin32(image_A, image_B, lut, get_context=None):       # laplacian 梯度 梯度 场景信息

    if image_A.shape[1] == 3:
        Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
        Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]
    else:
        Y_A = image_A
        Y_B = image_B
    
    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_A.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_A.device).view(1, 1, 3, 3).float()

    grad_A_x = F.conv2d(Y_A, sobel_x, padding=1)
    grad_A_y = F.conv2d(Y_A, sobel_y, padding=1)
    grad_B_x = F.conv2d(Y_B, sobel_x, padding=1)
    grad_B_y = F.conv2d(Y_B, sobel_y, padding=1)

    G_A = torch.abs(grad_A_x) + torch.abs(grad_A_y)
    G_B = torch.abs(grad_B_x) + torch.abs(grad_B_y)

    diff_high = torch.tanh((G_A - G_B) * 10.0) 

    idx_1 = (diff_high + 1.0) * 0.5 * 255/8.0
        
    idx_1 = idx_1.squeeze(1)
    idx_1_floor = torch.floor(idx_1).long()
    idx_1_ceil = torch.clamp(idx_1_floor + 1, 0, lut.shape[0] - 1)
    idx_1_alpha = idx_1 - idx_1_floor
    idx_1_alpha = idx_1_alpha.unsqueeze(-1)
    
    E_A = F.avg_pool2d(G_A, kernel_size=15, stride=1, padding=7)
    E_B = F.avg_pool2d(G_B, kernel_size=15, stride=1, padding=7)
    diff_low = torch.tanh((E_A - E_B) * 10.0) 
    idx_2 = (diff_low + 1.0) * 0.5 * 255/ 8.0
    
    idx_2 = idx_2.squeeze(1)
    idx_2_floor = torch.floor(idx_2).long()
    idx_2_ceil = torch.clamp(idx_2_floor + 1, 0, lut.shape[1] - 1)
    idx_2_alpha = idx_2 - idx_2_floor
    idx_2_alpha = idx_2_alpha.unsqueeze(-1)

    max_g = torch.max(G_A, G_B)
    idx_3 = torch.tanh(max_g) * 255/8.0
    
    idx_3 = idx_3.squeeze(1)
    idx_3_floor = torch.floor(idx_3).long()
    idx_3_ceil = torch.clamp(idx_3_floor + 1, 0, lut.shape[2] - 1)
    idx_3_alpha = idx_3 - idx_3_floor
    idx_3_alpha = idx_3_alpha.unsqueeze(-1)
    
    ctx = get_context(torch.cat([image_A, image_B], dim=1))
    idx_4 = torch.clamp(ctx, 0, 1) * 255/8.0

    idx_4 = idx_4.squeeze(1)
    idx_4_floor = torch.floor(idx_4).long()
    idx_4_ceil = torch.clamp(idx_4_floor + 1, 0, lut.shape[3] - 1)
    idx_4_alpha = idx_4 - idx_4_floor
    idx_4_alpha = idx_4_alpha.unsqueeze(-1)

    # print("index4_min: ",idx_4_floor.min())
    # print("index4_max: ",idx_4_ceil.max())

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[idx_1_floor, idx_2_floor, idx_3_floor, idx_4_floor], lut[idx_1_floor, idx_2_floor, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_floor, idx_2_floor, idx_3_ceil, idx_4_floor], lut[idx_1_floor, idx_2_floor, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                lerp(
                    lerp(lut[idx_1_floor, idx_2_ceil, idx_3_floor, idx_4_floor], lut[idx_1_floor, idx_2_ceil, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_floor, idx_2_ceil, idx_3_ceil, idx_4_floor], lut[idx_1_floor, idx_2_ceil, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                idx_2_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[idx_1_ceil, idx_2_floor, idx_3_floor, idx_4_floor], lut[idx_1_ceil, idx_2_floor, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_ceil, idx_2_floor, idx_3_ceil, idx_4_floor], lut[idx_1_ceil, idx_2_floor, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                lerp(
                    lerp(lut[idx_1_ceil, idx_2_ceil, idx_3_floor, idx_4_floor], lut[idx_1_ceil, idx_2_ceil, idx_3_floor, idx_4_ceil], idx_4_alpha),
                    lerp(lut[idx_1_ceil, idx_2_ceil, idx_3_ceil, idx_4_floor], lut[idx_1_ceil, idx_2_ceil, idx_3_ceil, idx_4_ceil], idx_4_alpha),
                    idx_3_alpha,
                ),
                idx_2_alpha,
            ),
            idx_1_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y



def MFlut_2_new(model_x, model_y, lut, get_context):       # laplacian 梯度 梯度 场景信息

    image_cat = torch.cat((model_x, model_y), dim=1)  # [0, 255]

    context = get_context(image_cat)

    context_scaled = (context * 255. / 16.0).squeeze(1)         # [0, 16]
    infrared_scaled = model_y / 16.0                # [0, 16]
    ycbcr_vis = rgb_to_ycbcr(model_x/255.)           # [0, 1]
    ycbcr_vis_scaled = ycbcr_vis * 255.0/16.0                  # [0, 16]

    y_vi_scaled = ycbcr_vis[:, 0, :, :]  # [b, 1, h, w]   # [0, 1]
    y_vi_scaled = y_vi_scaled.unsqueeze(1)
    cb_cr = ycbcr_vis[:, 1:, :, :]              # [0, 1]

    lap = KF.laplacian(y_vi_scaled, kernel_size=3)
    lap = lap.squeeze(1) 
    lap_norm = (lap - lap.min()) / (lap.max() - lap.min()+ 1e-5)     # 归一化 0-1 
    lap_scaled = lap_norm * 16.0

    if model_y.shape[1] == 1:
        ir_scaled = model_y[:, 0, :, :] / 16.0 
    else:
        ir_gray = rgb_to_ycbcr(model_y/255.)
        ir_scaled = ir_gray * 255.0/16.0
        ir_scaled = ir_scaled[:, 0, :, :]  
    
    lap_floor = torch.floor(lap_scaled).long()
    lap_ceil = torch.clamp(lap_floor + 1, 0, lut.shape[0] - 1)
    lap_alpha = lap_scaled - lap_floor

    c_floor = torch.floor(context_scaled).long()
    c_ceil = torch.clamp(c_floor + 1, 0, lut.shape[2] - 1)
    c_alpha = context_scaled - c_floor

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        model_x.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        model_x.device)

    grad_near_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)  
    grad_near_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

    gradient_near = torch.sqrt(grad_near_x ** 2 + grad_near_y ** 2)  # [b, 1, h, w]
    min_near = gradient_near.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_near = gradient_near.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized_near = (gradient_near - min_near) / (max_near - min_near + 1e-8)  

    gradient_scaled_near = (gradient_normalized_near * 255.)
    gradient_scaled_near = (gradient_scaled_near / 16.0).squeeze(1)

    g_near_floor = torch.floor(gradient_scaled_near).long()
    g_near_ceil = torch.clamp(g_near_floor + 1, 0, lut.shape[1] - 1)
    g_near_alpha = gradient_scaled_near - g_near_floor


    grad_far_x = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_x, padding=1)  
    grad_far_y = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_y, padding=1)

    gradient_far = torch.sqrt(grad_far_x ** 2 + grad_far_y ** 2)  # [b, 1, h, w]
    min_far = gradient_far.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_far = gradient_far.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized_far = (gradient_far - min_far) / (max_far - min_far + 1e-8)  

    gradient_scaled_far = (gradient_normalized_far * 255.)
    gradient_scaled_far = (gradient_scaled_far / 16.0).squeeze(1)

    g_far_floor = torch.floor(gradient_scaled_far).long()
    g_far_ceil = torch.clamp(g_far_floor + 1, 0, lut.shape[1] - 1)
    g_far_alpha = gradient_scaled_far - g_far_floor

    lap_alpha = lap_alpha.unsqueeze(-1)
    g_near_alpha = g_near_alpha.unsqueeze(-1)
    g_far_alpha = g_far_alpha.unsqueeze(-1)
    c_alpha = c_alpha.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[lap_floor, g_near_floor, g_far_floor, c_floor], lut[lap_floor, g_near_floor, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_floor, g_near_floor, g_far_ceil, c_floor], lut[lap_floor, g_near_floor, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                lerp(
                    lerp(lut[lap_floor, g_near_ceil, g_far_floor, c_floor], lut[lap_floor, g_near_ceil, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_floor, g_near_ceil, g_far_ceil, c_floor], lut[lap_floor, g_near_ceil, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                g_near_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[lap_ceil, g_near_floor, g_far_floor, c_floor], lut[lap_ceil, g_near_floor, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_ceil, g_near_floor, g_far_ceil, c_floor], lut[lap_ceil, g_near_floor, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                lerp(
                    lerp(lut[lap_ceil, g_near_ceil, g_far_floor, c_floor], lut[lap_ceil, g_near_ceil, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_ceil, g_near_ceil, g_far_ceil, c_floor], lut[lap_ceil, g_near_ceil, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                g_near_alpha,
            ),
            lap_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y


def MFlut_wox1(image_A, image_B, lut, get_context=None):       # laplacian 梯度 梯度 场景信息

    if image_A.shape[1] == 3:
        Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
        Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]
    else:
        Y_A = image_A
        Y_B = image_B
    
    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_A.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_A.device).view(1, 1, 3, 3).float()

    grad_A_x = F.conv2d(Y_A, sobel_x, padding=1)
    grad_A_y = F.conv2d(Y_A, sobel_y, padding=1)
    grad_B_x = F.conv2d(Y_B, sobel_x, padding=1)
    grad_B_y = F.conv2d(Y_B, sobel_y, padding=1)

    G_A = torch.abs(grad_A_x) + torch.abs(grad_A_y)
    G_B = torch.abs(grad_B_x) + torch.abs(grad_B_y)
    
    E_A = F.avg_pool2d(G_A, kernel_size=15, stride=1, padding=7)
    E_B = F.avg_pool2d(G_B, kernel_size=15, stride=1, padding=7)
    diff_low = torch.tanh((E_A - E_B) * 10.0) 
    idx_2 = (diff_low + 1.0) * 0.5 * 255/ 16.0
    
    idx_2 = idx_2.squeeze(1)
    idx_2_floor = torch.floor(idx_2).long()
    idx_2_ceil = torch.clamp(idx_2_floor + 1, 0, lut.shape[1] - 1)
    idx_2_alpha = idx_2 - idx_2_floor
    idx_2_alpha = idx_2_alpha.unsqueeze(-1)

    max_g = torch.max(G_A, G_B)
    idx_3 = torch.tanh(max_g) * 255/16.0
    
    idx_3 = idx_3.squeeze(1)
    idx_3_floor = torch.floor(idx_3).long()
    idx_3_ceil = torch.clamp(idx_3_floor + 1, 0, lut.shape[2] - 1)
    idx_3_alpha = idx_3 - idx_3_floor
    idx_3_alpha = idx_3_alpha.unsqueeze(-1)
    
    ctx = get_context(torch.cat([image_A, image_B], dim=1))
    idx_4 = torch.clamp(ctx, 0, 1) * 255/16.0

    idx_4 = idx_4.squeeze(1)
    idx_4_floor = torch.floor(idx_4).long()
    idx_4_ceil = torch.clamp(idx_4_floor + 1, 0, lut.shape[3] - 1)
    idx_4_alpha = idx_4 - idx_4_floor
    idx_4_alpha = idx_4_alpha.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
    lerp(
        lerp(
            lerp(lut[idx_2_floor, idx_3_floor, idx_4_floor], lut[idx_2_floor, idx_3_floor, idx_4_ceil], idx_4_alpha),
            lerp(lut[idx_2_floor, idx_3_ceil, idx_4_floor], lut[idx_2_floor, idx_3_ceil, idx_4_ceil], idx_4_alpha),
            idx_3_alpha, 
        ),
        lerp(
            lerp(lut[idx_2_ceil, idx_3_floor, idx_4_floor], lut[idx_2_ceil, idx_3_floor, idx_4_ceil], idx_4_alpha),
            lerp(lut[idx_2_ceil, idx_3_ceil, idx_4_floor], lut[idx_2_ceil, idx_3_ceil, idx_4_ceil], idx_4_alpha),
            idx_3_alpha, 
        ),
        idx_2_alpha, 
    )
)

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y


def MFlut_wox2(image_A, image_B, lut, get_context=None):       # laplacian 梯度 梯度 场景信息

    if image_A.shape[1] == 3:
        Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
        Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]
    else:
        Y_A = image_A
        Y_B = image_B
    
    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_A.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_A.device).view(1, 1, 3, 3).float()

    grad_A_x = F.conv2d(Y_A, sobel_x, padding=1)
    grad_A_y = F.conv2d(Y_A, sobel_y, padding=1)
    grad_B_x = F.conv2d(Y_B, sobel_x, padding=1)
    grad_B_y = F.conv2d(Y_B, sobel_y, padding=1)

    G_A = torch.abs(grad_A_x) + torch.abs(grad_A_y)
    G_B = torch.abs(grad_B_x) + torch.abs(grad_B_y)

    diff_high = torch.tanh((G_A - G_B) * 10.0) 

    idx_1 = (diff_high + 1.0) * 0.5 * 255/16.0
        
    idx_1 = idx_1.squeeze(1)
    idx_1_floor = torch.floor(idx_1).long()
    idx_1_ceil = torch.clamp(idx_1_floor + 1, 0, lut.shape[0] - 1)
    idx_1_alpha = idx_1 - idx_1_floor
    idx_1_alpha = idx_1_alpha.unsqueeze(-1)

    max_g = torch.max(G_A, G_B)
    idx_3 = torch.tanh(max_g) * 255/16.0
    
    idx_3 = idx_3.squeeze(1)
    idx_3_floor = torch.floor(idx_3).long()
    idx_3_ceil = torch.clamp(idx_3_floor + 1, 0, lut.shape[2] - 1)
    idx_3_alpha = idx_3 - idx_3_floor
    idx_3_alpha = idx_3_alpha.unsqueeze(-1)
    
    ctx = get_context(torch.cat([image_A, image_B], dim=1))
    idx_4 = torch.clamp(ctx, 0, 1) * 255/16.0

    idx_4 = idx_4.squeeze(1)
    idx_4_floor = torch.floor(idx_4).long()
    idx_4_ceil = torch.clamp(idx_4_floor + 1, 0, lut.shape[3] - 1)
    idx_4_alpha = idx_4 - idx_4_floor
    idx_4_alpha = idx_4_alpha.unsqueeze(-1)


    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
    lerp(
        lerp(
            lerp(lut[idx_1_floor, idx_3_floor, idx_4_floor], lut[idx_1_floor, idx_3_floor, idx_4_ceil], idx_4_alpha),
            lerp(lut[idx_1_floor, idx_3_ceil, idx_4_floor], lut[idx_1_floor, idx_3_ceil, idx_4_ceil], idx_4_alpha),
            idx_3_alpha, 
        ),
        lerp(
            lerp(lut[idx_1_ceil, idx_3_floor, idx_4_floor], lut[idx_1_ceil, idx_3_floor, idx_4_ceil], idx_4_alpha),
            lerp(lut[idx_1_ceil, idx_3_ceil, idx_4_floor], lut[idx_1_ceil, idx_3_ceil, idx_4_ceil], idx_4_alpha),
            idx_3_alpha, 
        ),
        idx_1_alpha, 
    )
)

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y


def MFlut_wox3(image_A, image_B, lut, get_context=None):       # laplacian 梯度 梯度 场景信息

    if image_A.shape[1] == 3:
        Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
        Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]
    else:
        Y_A = image_A
        Y_B = image_B
    
    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_A.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_A.device).view(1, 1, 3, 3).float()

    grad_A_x = F.conv2d(Y_A, sobel_x, padding=1)
    grad_A_y = F.conv2d(Y_A, sobel_y, padding=1)
    grad_B_x = F.conv2d(Y_B, sobel_x, padding=1)
    grad_B_y = F.conv2d(Y_B, sobel_y, padding=1)

    G_A = torch.abs(grad_A_x) + torch.abs(grad_A_y)
    G_B = torch.abs(grad_B_x) + torch.abs(grad_B_y)

    diff_high = torch.tanh((G_A - G_B) * 10.0) 

    idx_1 = (diff_high + 1.0) * 0.5 * 255/16.0
        
    idx_1 = idx_1.squeeze(1)
    idx_1_floor = torch.floor(idx_1).long()
    idx_1_ceil = torch.clamp(idx_1_floor + 1, 0, lut.shape[0] - 1)
    idx_1_alpha = idx_1 - idx_1_floor
    idx_1_alpha = idx_1_alpha.unsqueeze(-1)
    
    E_A = F.avg_pool2d(G_A, kernel_size=15, stride=1, padding=7)
    E_B = F.avg_pool2d(G_B, kernel_size=15, stride=1, padding=7)
    diff_low = torch.tanh((E_A - E_B) * 10.0) 
    idx_2 = (diff_low + 1.0) * 0.5 * 255/ 16.0
    
    idx_2 = idx_2.squeeze(1)
    idx_2_floor = torch.floor(idx_2).long()
    idx_2_ceil = torch.clamp(idx_2_floor + 1, 0, lut.shape[1] - 1)
    idx_2_alpha = idx_2 - idx_2_floor
    idx_2_alpha = idx_2_alpha.unsqueeze(-1)
    
    ctx = get_context(torch.cat([image_A, image_B], dim=1))
    idx_4 = torch.clamp(ctx, 0, 1) * 255/16.0

    idx_4 = idx_4.squeeze(1)
    idx_4_floor = torch.floor(idx_4).long()
    idx_4_ceil = torch.clamp(idx_4_floor + 1, 0, lut.shape[3] - 1)
    idx_4_alpha = idx_4 - idx_4_floor
    idx_4_alpha = idx_4_alpha.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
    lerp(
        lerp(
            lerp(lut[idx_1_floor, idx_2_floor, idx_4_floor], lut[idx_1_floor, idx_2_floor, idx_4_ceil], idx_4_alpha),
            lerp(lut[idx_1_floor, idx_2_ceil, idx_4_floor], lut[idx_1_floor, idx_2_ceil, idx_4_ceil], idx_4_alpha),
            idx_2_alpha, 
        ),
        lerp(
            lerp(lut[idx_1_ceil, idx_2_floor, idx_4_floor], lut[idx_1_ceil, idx_2_floor, idx_4_ceil], idx_4_alpha),
            lerp(lut[idx_1_ceil, idx_2_ceil, idx_4_floor], lut[idx_1_ceil, idx_2_ceil, idx_4_ceil], idx_4_alpha),
            idx_2_alpha, 
        ),
        idx_1_alpha, 
    )
)

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y


def MFlut_wox4(image_A, image_B, lut, get_context=None):       # laplacian 梯度 梯度 场景信息

    if image_A.shape[1] == 3:
        Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
        Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]
    else:
        Y_A = image_A
        Y_B = image_B
    
    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image_A.device).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image_A.device).view(1, 1, 3, 3).float()

    grad_A_x = F.conv2d(Y_A, sobel_x, padding=1)
    grad_A_y = F.conv2d(Y_A, sobel_y, padding=1)
    grad_B_x = F.conv2d(Y_B, sobel_x, padding=1)
    grad_B_y = F.conv2d(Y_B, sobel_y, padding=1)

    G_A = torch.abs(grad_A_x) + torch.abs(grad_A_y)
    G_B = torch.abs(grad_B_x) + torch.abs(grad_B_y)

    diff_high = torch.tanh((G_A - G_B) * 10.0) 

    idx_1 = (diff_high + 1.0) * 0.5 * 255/16.0
        
    idx_1 = idx_1.squeeze(1)
    idx_1_floor = torch.floor(idx_1).long()
    idx_1_ceil = torch.clamp(idx_1_floor + 1, 0, lut.shape[0] - 1)
    idx_1_alpha = idx_1 - idx_1_floor
    idx_1_alpha = idx_1_alpha.unsqueeze(-1)
    
    E_A = F.avg_pool2d(G_A, kernel_size=15, stride=1, padding=7)
    E_B = F.avg_pool2d(G_B, kernel_size=15, stride=1, padding=7)
    diff_low = torch.tanh((E_A - E_B) * 10.0) 
    idx_2 = (diff_low + 1.0) * 0.5 * 255/ 16.0
    
    idx_2 = idx_2.squeeze(1)
    idx_2_floor = torch.floor(idx_2).long()
    idx_2_ceil = torch.clamp(idx_2_floor + 1, 0, lut.shape[1] - 1)
    idx_2_alpha = idx_2 - idx_2_floor
    idx_2_alpha = idx_2_alpha.unsqueeze(-1)

    max_g = torch.max(G_A, G_B)
    idx_3 = torch.tanh(max_g) * 255/16.0
    
    idx_3 = idx_3.squeeze(1)
    idx_3_floor = torch.floor(idx_3).long()
    idx_3_ceil = torch.clamp(idx_3_floor + 1, 0, lut.shape[2] - 1)
    idx_3_alpha = idx_3 - idx_3_floor
    idx_3_alpha = idx_3_alpha.unsqueeze(-1)
    

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
    lerp(
        lerp(
            lerp(lut[idx_1_floor, idx_2_floor, idx_3_floor], lut[idx_1_floor, idx_2_floor, idx_3_ceil], idx_3_alpha),
            lerp(lut[idx_1_floor, idx_2_ceil, idx_3_floor], lut[idx_1_floor, idx_2_ceil, idx_3_ceil], idx_3_alpha),
            idx_2_alpha, 
        ),
        lerp(
            lerp(lut[idx_1_ceil, idx_2_floor, idx_3_floor], lut[idx_1_ceil, idx_2_floor, idx_3_ceil], idx_3_alpha),
            lerp(lut[idx_1_ceil, idx_2_ceil, idx_3_floor], lut[idx_1_ceil, idx_2_ceil, idx_3_ceil], idx_3_alpha),
            idx_2_alpha, 
        ),
        idx_1_alpha, 
    )
)

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y



def MFlut_lowf(model_x, model_y, lut, get_context):       # laplacian 梯度 梯度 场景信息
    
    image_cat = torch.cat((model_x, model_y), dim=1)  # [0, 255]
    context = get_context(image_cat)

    context_scaled = (context * 255. / 16.0).squeeze(1)         # [0, 16]
    infrared_scaled = model_y / 16.0                # [0, 16]
    ycbcr_vis = rgb_to_ycbcr(model_x/255.)           # [0, 1]
    ycbcr_vis_scaled = ycbcr_vis * 255.0/16.0                  # [0, 16]

    y_vi_scaled = ycbcr_vis[:, 0, :, :]  # [b, 1, h, w]   # [0, 1]
    y_vi_scaled = y_vi_scaled.unsqueeze(1)
    cb_cr = ycbcr_vis[:, 1:, :, :]              # [0, 1]

    lap = KF.laplacian(y_vi_scaled, kernel_size=3)
    lap = lap.squeeze(1) 
    lap_norm = (lap - lap.min()) / (lap.max() - lap.min()+ 1e-5)     # 归一化 0-1 
    lap_scaled = lap_norm * 16.0

    if model_y.shape[1] == 1:
        ir_scaled = model_y[:, 0, :, :] / 16.0 
    else:
        ir_gray = rgb_to_ycbcr(model_y/255.)
        ir_scaled = ir_gray * 255.0/16.0
        ir_scaled = ir_scaled[:, 0, :, :]  

    # ir_floor = torch.floor(ir_scaled).long()
    # ir_ceil = torch.clamp(ir_floor + 1, 0, lut.shape[3] - 1)
    # ir_alpha = ir_scaled - ir_floor

    # y_vi_floor = torch.floor(y_vi_scaled).long()
    # y_vi_ceil = torch.clamp(y_vi_floor + 1, 0, lut.shape[0] - 1)
    # y_vi_alpha = y_vi_scaled - y_vi_floor
    
    lap_floor = torch.floor(lap_scaled).long()
    lap_ceil = torch.clamp(lap_floor + 1, 0, lut.shape[0] - 1)
    lap_alpha = lap_scaled - lap_floor

    c_floor = torch.floor(context_scaled).long()
    c_ceil = torch.clamp(c_floor + 1, 0, lut.shape[2] - 1)
    c_alpha = context_scaled - c_floor

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        model_x.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        model_x.device)

    grad_near_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)  
    grad_near_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

    gradient_near = torch.sqrt(grad_near_x ** 2 + grad_near_y ** 2)  # [b, 1, h, w]
    min_near = gradient_near.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_near = gradient_near.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized_near = (gradient_near - min_near) / (max_near - min_near + 1e-8)  

    gradient_scaled_near = (gradient_normalized_near * 255.)
    gradient_scaled_near = (gradient_scaled_near / 16.0).squeeze(1)

    g_near_floor = torch.floor(gradient_scaled_near).long()
    g_near_ceil = torch.clamp(g_near_floor + 1, 0, lut.shape[1] - 1)
    g_near_alpha = gradient_scaled_near - g_near_floor


    grad_far_x = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_x, padding=1)  
    grad_far_y = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_y, padding=1)

    gradient_far = torch.sqrt(grad_far_x ** 2 + grad_far_y ** 2)  # [b, 1, h, w]
    min_far = gradient_far.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
    max_far = gradient_far.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

    gradient_normalized_far = (gradient_far - min_far) / (max_far - min_far + 1e-8)  

    gradient_scaled_far = (gradient_normalized_far * 255.)
    gradient_scaled_far = (gradient_scaled_far / 16.0).squeeze(1)

    g_far_floor = torch.floor(gradient_scaled_far).long()
    g_far_ceil = torch.clamp(g_far_floor + 1, 0, lut.shape[1] - 1)
    g_far_alpha = gradient_scaled_far - g_far_floor


    # ir_alpha = ir_alpha.unsqueeze(-1)
    lap_alpha = lap_alpha.unsqueeze(-1)
    g_near_alpha = g_near_alpha.unsqueeze(-1)
    g_far_alpha = g_far_alpha.unsqueeze(-1)
    c_alpha = c_alpha.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[lap_floor, g_near_floor, g_far_floor, c_floor], lut[lap_floor, g_near_floor, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_floor, g_near_floor, g_far_ceil, c_floor], lut[lap_floor, g_near_floor, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                lerp(
                    lerp(lut[lap_floor, g_near_ceil, g_far_floor, c_floor], lut[lap_floor, g_near_ceil, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_floor, g_near_ceil, g_far_ceil, c_floor], lut[lap_floor, g_near_ceil, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                g_near_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[lap_ceil, g_near_floor, g_far_floor, c_floor], lut[lap_ceil, g_near_floor, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_ceil, g_near_floor, g_far_ceil, c_floor], lut[lap_ceil, g_near_floor, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                lerp(
                    lerp(lut[lap_ceil, g_near_ceil, g_far_floor, c_floor], lut[lap_ceil, g_near_ceil, g_far_floor, c_ceil], c_alpha),
                    lerp(lut[lap_ceil, g_near_ceil, g_far_ceil, c_floor], lut[lap_ceil, g_near_ceil, g_far_ceil, c_ceil], c_alpha),
                    g_far_alpha,
                ),
                g_near_alpha,
            ),
            lap_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)
    return fusion_y



def MFlut_lowfreq(model_x, model_y, lut):

    B, C, H, W = model_x.shape

    # =========================
    # 维度 0：低频模糊度比例（替换原 Laplacian）
    # =========================
    def gaussian_blur(x, sigma=7):
        k = int(4 * sigma + 1)
        return kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma))

    lowA = gaussian_blur(model_x, sigma=7)
    lowB = gaussian_blur(model_y, sigma=7)

    patch = 15

    meanA = F.avg_pool2d(lowA, patch, stride=1, padding=patch//2)
    meanA2 = F.avg_pool2d(lowA*lowA, patch, stride=1, padding=patch//2)
    varA = meanA2 - meanA*meanA

    meanB = F.avg_pool2d(lowB, patch, stride=1, padding=patch//2)
    meanB2 = F.avg_pool2d(lowB*lowB, patch, stride=1, padding=patch//2)
    varB = meanB2 - meanB*meanB

    blur_ratio = varA / (varA + varB + 1e-6)      # [0,1]
    blur_min = blur_ratio.amin(dim=[2,3], keepdim=True)
    blur_max = blur_ratio.amax(dim=[2,3], keepdim=True)

    blur_norm = (blur_ratio - blur_min) / (blur_max - blur_min + 1e-6)    
    
    blur_scaled = (blur_norm * 255./16.).squeeze(1)

    blur_floor = torch.floor(blur_scaled).long()
    blur_ceil = torch.clamp(blur_floor + 1, 0, lut.shape[0] - 1)
    blur_alpha = blur_scaled - blur_floor
    blur_alpha = blur_alpha.unsqueeze(-1)

    # =========================
    # 维度 1：低频梯度差（替换原 gradA）
    # =========================
    sobel = kornia.filters.Sobel()

    gradA = sobel(model_x).norm(dim=1, keepdim=True)
    gradB = sobel(model_y).norm(dim=1, keepdim=True)

    gradA_lf = gaussian_blur(gradA, sigma=3)
    gradB_lf = gaussian_blur(gradB, sigma=3)

    grad_contrast = (gradA_lf - gradB_lf).abs()  # [0,~]
    g_min = grad_contrast.amin(dim=[2,3], keepdim=True)
    g_max = grad_contrast.amax(dim=[2,3], keepdim=True)

    grad_norm = (grad_contrast - g_min) / (g_max - g_min + 1e-6)
    grad_scaled = (grad_norm *255/ 16).squeeze(1)

    g_floor = torch.floor(grad_scaled).long()
    g_ceil = torch.clamp(g_floor + 1, 0, lut.shape[1] - 1)
    g_alpha = (grad_scaled - g_floor).unsqueeze(-1)

    # =========================
    # 维度 2：低频亮度差（替换原 gradB）
    # =========================
    lowA2 = gaussian_blur(model_x, sigma=5)
    lowB2 = gaussian_blur(model_y, sigma=5)

    lf_diff = (lowA2 - lowB2).abs()
    d_min = lf_diff.amin(dim=[2,3], keepdim=True)
    d_max = lf_diff.amax(dim=[2,3], keepdim=True)

    diff_norm = (lf_diff - d_min) / (d_max - d_min + 1e-6)
    diff_scaled = (diff_norm *255./ 16.).squeeze(1)

    d_floor = torch.floor(diff_scaled).long()
    d_ceil = torch.clamp(d_floor + 1, 0, lut.shape[2] - 1)
    d_alpha = (diff_scaled - d_floor).unsqueeze(-1)

    # =========================
    # 维度 3：上下文低频（替换原 context）
    # =========================
    mean_A = torch.nn.functional.avg_pool2d(model_x, kernel_size=15, stride=1, padding=7)
    mean_B = torch.nn.functional.avg_pool2d(model_y, kernel_size=15, stride=1, padding=7)
    dim4 = mean_A - mean_B      # 范围可能是 [-255, 255]
    dim4_norm = (dim4 - dim4.min()) / (dim4.max() - dim4.min() + 1e-8)
    dim4_scaled = (dim4_norm * 255/16.0).squeeze(1)
    
    dim4_floor = torch.floor(dim4_scaled).long()
    dim4_ceil = torch.clamp(dim4_floor + 1, 0, lut.shape[3] - 1)
    dim4_alpha = dim4_scaled - dim4_floor
    dim4_alpha = dim4_alpha.unsqueeze(-1)  # 与其它维度保持一致

    # =========================
    # LUT 插值
    # =========================

    def lerp(v1, v2, a):
        return v1 * (1 - a) + v2 * a

    out = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[blur_floor, g_floor, d_floor, dim4_floor], lut[blur_floor, g_floor, d_floor, dim4_ceil], dim4_alpha),
                    lerp(lut[blur_floor, g_floor, d_ceil, dim4_floor], lut[blur_floor, g_floor, d_ceil, dim4_ceil], dim4_alpha),
                    d_alpha),
                lerp(
                    lerp(lut[blur_floor, g_ceil, d_floor, dim4_floor], lut[blur_floor, g_ceil, d_floor, dim4_ceil], dim4_alpha),
                    lerp(lut[blur_floor, g_ceil, d_ceil, dim4_floor], lut[blur_floor, g_ceil, d_ceil, dim4_ceil], dim4_alpha),
                    d_alpha),
                g_alpha),
            lerp(
                lerp(
                    lerp(lut[blur_ceil, g_floor, d_floor, dim4_floor], lut[blur_ceil, g_floor, d_floor, dim4_ceil], dim4_alpha),
                    lerp(lut[blur_ceil, g_floor, d_ceil, dim4_floor], lut[blur_ceil, g_floor, d_ceil, dim4_ceil], dim4_alpha),
                    d_alpha),
                lerp(
                    lerp(lut[blur_ceil, g_ceil, d_floor, dim4_floor], lut[blur_ceil, g_ceil, d_floor, dim4_ceil], dim4_alpha),
                    lerp(lut[blur_ceil, g_ceil, d_ceil, dim4_floor], lut[blur_ceil, g_ceil, d_ceil, dim4_ceil], dim4_alpha),
                    d_alpha),
                g_alpha),
            blur_alpha)
    )

    output = out.permute(0, 3, 1, 2)    # [B, 2, H, W]
    return output


class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # 使用 AvgPool2d 实现 Box Filter，速度极快
        self.box_filter = nn.AvgPool2d(kernel_size=2*r+1, stride=1, padding=r)

    def forward(self, guide, input_mask):
        # guide: [B, 1, H, W] 引导图 (通常是灰度图)
        # input_mask: [B, 1, H, W] 待修正的 Mask
        
        # 1. 计算各种均值
        mean_I = self.box_filter(guide)
        mean_p = self.box_filter(input_mask)
        mean_Ip = self.box_filter(guide * input_mask)
        mean_II = self.box_filter(guide * guide)

        # 2. 计算协方差和方差
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        # 3. 计算线性系数 a, b
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        # 4. 对系数进行平滑
        mean_a = self.box_filter(a)
        mean_b = self.box_filter(b)

        # 5. 输出精细化 Mask
        q = mean_a * guide + mean_b
        return q


def load_lookup_table(filepath):
    try:
        lut = np.load(filepath).astype(np.float32)
        lut = torch.tensor(lut, device="cuda")  # 
        return lut
    except Exception as e:
        print(f" : {e}")
        return 0
    

def apply_fusion_2d_with_interpolation(visible_img, infrared_img, lut, get_context):        # [0, 255]

    image_cat = torch.cat((visible_img, infrared_img), dim=1)  # [0, 255]
    context = get_context(image_cat)

    context_scaled1 = (context[:,0,:,:]*255./8.).squeeze(1)      
    context_scaled2 = (context[:,1,:,:]*255./8.).squeeze(1)   

    c_floor1 = torch.floor(context_scaled1).long()
    c_ceil1 = torch.clamp(c_floor1 + 1, 0, lut.shape[1] - 1)
    c_alpha1 = context_scaled1 - c_floor1

    c_floor2 = torch.floor(context_scaled2).long()
    c_ceil2 = torch.clamp(c_floor2 + 1, 0, lut.shape[0] - 1)
    c_alpha2 = context_scaled2 - c_floor2

    c_alpha1 = c_alpha1.unsqueeze(-1)
    c_alpha2 = c_alpha2.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = lerp(
        lerp(lut[c_floor1, c_floor2], lut[c_floor1, c_ceil2], c_alpha2),
        lerp(lut[c_ceil1, c_floor2], lut[c_ceil1, c_ceil2], c_alpha2),
        c_alpha1
    )

    fusion_r = fusion_result.permute(0, 3, 1, 2)
    
    return fusion_r


# def apply_fusion_2d_as_residual(model_x, model_y, lut):       # 2d lut做补偿，主要查找维度为两张图的梯度信息  (32,32,1)

#     # image_cat = torch.cat((model_x, model_y), dim=1)  # [0, 255]
#     # context = get_context(image_cat)

#     # context_scaled = (context * 255. / 16.0).squeeze(1)         # [0, 16]
#     # infrared_scaled = model_y / 16.0                # [0, 16]
#     ycbcr_vis = rgb_to_ycbcr(model_x/255.)           # [0, 1]
#     # ycbcr_vis_scaled = ycbcr_vis * 255.0/16.0                  # [0, 16]

#     # y_vi_scaled = ycbcr_vis_scaled[:, 0, :, :]  # [b, 1, h, w]   # [0, 16]
#     # cb_cr = ycbcr_vis[:, 1:, :, :]              # [0, 1]
    
#     if model_y.shape[1] == 1:
#         ir_scaled = model_y[:, 0, :, :] / 16.0 
#     else:
#         ir_gray = rgb_to_ycbcr(model_y/255.)
#         ir_scaled = ir_gray * 255.0/16.0
#         ir_scaled = ir_scaled[:, 0, :, :]  


#     sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
#         model_x.device)
#     sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
#         model_x.device)

#     grad_near_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)  
#     grad_near_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

#     gradient_near = torch.sqrt(grad_near_x ** 2 + grad_near_y ** 2)  # [b, 1, h, w]
#     min_near = gradient_near.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
#     max_near = gradient_near.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

#     gradient_normalized_near = (gradient_near - min_near) / (max_near - min_near + 1e-8)  

#     gradient_scaled_near = (gradient_normalized_near * 255.)
#     gradient_scaled_near = (gradient_scaled_near / 16.0).squeeze(1)

#     g_near_floor = torch.floor(gradient_scaled_near).long()
#     g_near_ceil = torch.clamp(g_near_floor + 1, 0, lut.shape[1] - 1)
#     g_near_alpha = gradient_scaled_near - g_near_floor

#     grad_far_x = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_x, padding=1)  
#     grad_far_y = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_y, padding=1)

#     gradient_far = torch.sqrt(grad_far_x ** 2 + grad_far_y ** 2)  # [b, 1, h, w]
#     min_far = gradient_far.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
#     max_far = gradient_far.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

#     gradient_normalized_far = (gradient_far - min_far) / (max_far - min_far + 1e-8)  

#     gradient_scaled_far = (gradient_normalized_far * 255.)
#     gradient_scaled_far = (gradient_scaled_far / 16.0).squeeze(1)

#     g_far_floor = torch.floor(gradient_scaled_far).long()
#     g_far_ceil = torch.clamp(g_far_floor + 1, 0, lut.shape[1] - 1)
#     g_far_alpha = gradient_scaled_far - g_far_floor

#     g_near_alpha = g_near_alpha.unsqueeze(-1)
#     g_far_alpha = g_far_alpha.unsqueeze(-1)

#     def lerp(v1, v2, alpha):
#         out = v1 * (1 - alpha) + v2 * alpha
#         return out

#     fusion_result = lerp(
#         lerp(lut[g_near_floor, g_far_floor], lut[g_near_floor, g_far_ceil], g_far_alpha),
#         lerp(lut[g_near_ceil, g_far_floor], lut[g_near_ceil, g_far_ceil], g_far_alpha),
#         g_near_alpha
#     )

#     fusion_r = fusion_result.permute(0, 3, 1, 2)
#     return fusion_r


# def apply_fusion_2d_as_residual(model_x, model_y, lut):       # 2d lut做补偿，主要查找维度为两张图的梯度信息  (32,32,1)

#     # image_cat = torch.cat((model_x, model_y), dim=1)  # [0, 255]
#     # context = get_context(image_cat)

#     # context_scaled = (context * 255. / 16.0).squeeze(1)         # [0, 16]
#     # infrared_scaled = model_y / 16.0                # [0, 16]
#     ycbcr_vis = rgb_to_ycbcr(model_x/255.)           # [0, 1]
#     # ycbcr_vis_scaled = ycbcr_vis * 255.0/16.0                  # [0, 16]

#     # y_vi_scaled = ycbcr_vis_scaled[:, 0, :, :]  # [b, 1, h, w]   # [0, 16]
#     # cb_cr = ycbcr_vis[:, 1:, :, :]              # [0, 1]
    
#     # if model_y.shape[1] == 1:
#     #     ir_scaled = model_y[:, 0, :, :] / 32.0 
#     # else:
#     #     ir_gray = rgb_to_ycbcr(model_y/255.)
#     #     ir_scaled = ir_gray * 255.0/32.0
#     #     ir_scaled = ir_scaled[:, 0, :, :]  


#     ir_gray = rgb_to_ycbcr(model_y/255.)
#     ir_scaled = ir_gray * 255.0/32.0
#     ir_scaled = ir_scaled[:, 0, :, :] 


#     sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
#         model_x.device)
#     sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
#         model_x.device)

#     grad_near_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)  
#     grad_near_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

#     gradient_near = torch.sqrt(grad_near_x ** 2 + grad_near_y ** 2)  # [b, 1, h, w]
#     min_near = gradient_near.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
#     max_near = gradient_near.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

#     gradient_normalized_near = (gradient_near - min_near) / (max_near - min_near + 1e-8)  

#     gradient_scaled_near = (gradient_normalized_near * 255.)
#     gradient_scaled_near = (gradient_scaled_near / 32.0).squeeze(1)

#     g_near_floor = torch.floor(gradient_scaled_near).long()
#     g_near_ceil = torch.clamp(g_near_floor + 1, 0, lut.shape[1] - 1)
#     g_near_alpha = gradient_scaled_near - g_near_floor

#     grad_far_x = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_x, padding=1)  
#     grad_far_y = torch.nn.functional.conv2d(ir_gray[:, :1, :, :], sobel_y, padding=1)

#     gradient_far = torch.sqrt(grad_far_x ** 2 + grad_far_y ** 2)  # [b, 1, h, w]
#     min_far = gradient_far.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # Min for [h, w]
#     max_far = gradient_far.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # Max for [h, w]

#     gradient_normalized_far = (gradient_far - min_far) / (max_far - min_far + 1e-8)  

#     gradient_scaled_far = (gradient_normalized_far * 255.)
#     gradient_scaled_far = (gradient_scaled_far / 32.0).squeeze(1)

#     g_far_floor = torch.floor(gradient_scaled_far).long()
#     g_far_ceil = torch.clamp(g_far_floor + 1, 0, lut.shape[1] - 1)
#     g_far_alpha = gradient_scaled_far - g_far_floor

#     g_near_alpha = g_near_alpha.unsqueeze(-1)
#     g_far_alpha = g_far_alpha.unsqueeze(-1)

#     def lerp(v1, v2, alpha):
#         out = v1 * (1 - alpha) + v2 * alpha
#         return out

#     fusion_result = lerp(
#         lerp(lut[g_near_floor, g_far_floor], lut[g_near_floor, g_far_ceil], g_far_alpha),
#         lerp(lut[g_near_ceil, g_far_floor], lut[g_near_ceil, g_far_ceil], g_far_alpha),
#         g_near_alpha
#     )

#     fusion_r = fusion_result.permute(0, 3, 1, 2)
#     return fusion_r



# def apply_fusion_2d_as_residual(image_A, image_B, lut):
#     """
#     2D Residual LUT 查找
#     Input:
#         image_A, image_B: [B, C, H, W] (范围 0-255)
#         lut: [32, 32, 1] (需要 permute 为 [1, 1, 32, 32] 给 grid_sample)
#     Output:
#         residual_mask: [B, 1, H, W] (范围 [-1, 1] 或根据 LUT 值决定)
#     """

#     Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
#     Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]

#     Y_A = Y_A / 255.0
#     Y_B = Y_B / 255.0

#     # =========================================================
#     # 2. 特征提取: 像素级梯度 (Sobel) 
#     # =========================================================
#     # 动态创建 Sobel 核 (建议在类初始化时 register_buffer，这里为了函数独立性写在内部)
#     device = image_A.device
#     sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).view(1, 1, 3, 3).float()
#     sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).view(1, 1, 3, 3).float()

#     # 计算梯度 (Padding=1 保持尺寸不变)
#     gA_x = F.conv2d(Y_A, sobel_x, padding=1)
#     gA_y = F.conv2d(Y_A, sobel_y, padding=1)
#     gB_x = F.conv2d(Y_B, sobel_x, padding=1)
#     gB_y = F.conv2d(Y_B, sobel_y, padding=1)

#     # 梯度幅值
#     G_A = torch.abs(gA_x) + torch.abs(gA_y)
#     G_B = torch.abs(gB_x) + torch.abs(gB_y)
    
#     coord_x = torch.tanh((G_A - G_B) * 10.0) 
    
#     max_g = torch.max(G_A, G_B)
#     coord_y = torch.tanh(max_g * 5.0) * 2.0 - 1.0 
    
#     # 堆叠坐标: [B, H, W, 2] -> 最后一维是 (x, y)
#     grid = torch.stack((coord_x, coord_y), dim=3)

#     # =========================================================
#     # 4. LUT 查找 (Grid Sample) 
#     # =========================================================
    
#     # 调整 LUT 形状以适配 grid_sample
#     # 输入 lut 是 [32, 32, 1] -> 需要变成 [1, 1, 32, 32] (Batch, Channel, H, W)
#     # 这里的 H对应coord_y, W对应coord_x
#     lut_in = lut.permute(2, 0, 1).unsqueeze(0) 
    
#     # 扩展 Batch 维度: [B, 1, 32, 32]
#     B = image_A.shape[0]
#     lut_expanded = lut_in.expand(B, -1, -1, -1)
    
#     # 极速采样 (O(1) 复杂度)
#     # align_corners=True 是通用的对齐标准
#     # padding_mode='border' 保证坐标超出 -1/1 时取边界值，不会黑屏
#     residual_mask = F.grid_sample(lut_expanded, grid, mode='bilinear', padding_mode='border', align_corners=True)

#     return residual_mask



def apply_fusion_2d_as_residual(image_A, image_B, lut):
    """
    2D Residual LUT 查找
    Input:
        image_A, image_B: [B, C, H, W] (范围 0-255)
        lut: [32, 32, 1] (需要 permute 为 [1, 1, 32, 32] 给 grid_sample)
    Output:
        residual_mask: [B, 1, H, W] (范围 [-1, 1] 或根据 LUT 值决定)
    """

    Y_A = 0.299 * image_A[:, 0:1, :, :] + 0.587 * image_A[:, 1:2, :, :] + 0.114 * image_A[:, 2:3, :, :]
    Y_B = 0.299 * image_B[:, 0:1, :, :] + 0.587 * image_B[:, 1:2, :, :] + 0.114 * image_B[:, 2:3, :, :]

    Y_A = Y_A / 255.0
    Y_B = Y_B / 255.0

    # =========================================================
    # 2. 特征提取: 像素级梯度 (Sobel) 
    # =========================================================
    # 动态创建 Sobel 核 (建议在类初始化时 register_buffer，这里为了函数独立性写在内部)
    device = image_A.device
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                              device=image_A.device).view(1, 1, 3, 3).float()

    # 计算 Laplacian 响应 (保留符号，因为我们需要比大小)
    # 注意：这里取绝对值，因为我们关心的是“变化剧烈程度”，不管方向
    lap_A = torch.abs(F.conv2d(Y_A, lap_kernel, padding=1))
    lap_B = torch.abs(F.conv2d(Y_B, lap_kernel, padding=1))

    # =========================================================
    # 维度构建
    # =========================================================
    
    # --- Dim 1: 二阶锐度差异 (Laplacian Difference) ---
    # 这比 Sobel 更关注细微的线条（如文字、发丝）
    coord_x = torch.tanh((lap_A - lap_B) * 10.0) # [B, 1, H, W]
    
    # --- Dim 2 (Y轴): 纹理强度 ---
    max_g = torch.max(lap_A, lap_B)
    coord_y = torch.tanh(max_g * 5.0) * 2.0 - 1.0 # [B, 1, H, W]
    
    # 【核心修改点】
    # 1. 先在通道维 (dim=1) 拼接 -> [B, 2, H, W]
    grid_tensor = torch.cat((coord_x, coord_y), dim=1) 
    
    # 2. 调整维度顺序 -> [B, H, W, 2]
    # grid_sample 要求的格式是 (Batch, Height, Width, 2)
    grid = grid_tensor.permute(0, 2, 3, 1)

    # =========================================================
    # 4. LUT 查找 (Grid Sample) 
    # =========================================================
    
    # 调整 LUT 形状以适配 grid_sample
    # 输入 lut 是 [32, 32, 1] -> 需要变成 [1, 1, 32, 32] (Batch, Channel, H, W)
    # 这里的 H对应coord_y, W对应coord_x
    if isinstance(lut, torch.nn.Module):
        if hasattr(lut, 'lut'):
            lut_tensor = lut.lut # 提取名为 'lut' 的 Parameter
        else:
            lut_tensor = lut()   # 或者尝试调用 forward() 获取
    else:
        lut_tensor = lut # 如果已经是 tensor，直接用

    # 现在 lut_tensor 肯定是 [32, 32, 1] 的 Tensor 了
    # Permute: [32, 32, 1] -> [1, 1, 32, 32]
    lut_in = lut_tensor.permute(2, 0, 1).unsqueeze(0) 
    
    B = image_A.shape[0]
    lut_expanded = lut_in.expand(B, -1, -1, -1)
    
    residual_mask = F.grid_sample(lut_expanded, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return residual_mask



