from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from glob import glob
import transforms as T
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F

# class SimpleDataSet(Dataset):
#     def __init__(self, visible_path, visible_gt_path, infrared_path, phase="train", transform=None):
#         self.phase = phase
#         self.visible_files = sorted(glob(os.path.join(visible_path, "*")))
#         self.visible_gt_files = sorted(glob(os.path.join(visible_gt_path, "*")))
#         self.infrared_files = sorted(glob(os.path.join(infrared_path, "*")))
#         self.transform = transform
#
#     def __len__(self):
#         l = len(self.infrared_files)
#         return l
#
#     def __getitem__(self, item):
#         image_A_path = self.visible_files[item]
#         image_A_gt_path = self.visible_gt_files[item]
#         image_B_path = self.infrared_files[item]
#         image_A = Image.open(image_A_path).convert(mode='RGB')
#         image_A_gt = Image.open(image_A_gt_path).convert(mode='RGB')
#         image_B = Image.open(image_B_path).convert(mode='L')   ##########
#
#         image_A = self.transform(image_A)
#         image_A_gt = self.transform(image_A_gt)
#         image_B = self.transform(image_B)
#
#         name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]
#
#         return image_A, image_A_gt, image_B, name
#
#     @staticmethod
#     def collate_fn(batch):
#         images_A, image_A_gt, images_B, name = zip(*batch)
#         images_A = torch.stack(images_A, dim=0)
#         image_A_gt = torch.stack(image_A_gt, dim=0)
#         images_B = torch.stack(images_B, dim=0)
#         return images_A, image_A_gt, images_B, name

'''
class RandomCropPair:
    def __init__(self, size):
        self.size = size  # 裁剪尺寸 (h, w)

    def __call__(self, vis_img, ir_img):
        # 获取随机裁剪参数
        i, j, h, w = RandomCrop.get_params(vis_img, output_size=self.size)
        # 对可见光和红外图像使用相同的裁剪参数
        vis_img = F.crop(vis_img, i, j, h, w)
        ir_img = F.crop(ir_img, i, j, h, w)

        vis_img = F.to_tensor(vis_img)
        ir_img = F.to_tensor(ir_img)
        return vis_img, ir_img

class SimpleDataSet(Dataset):
    def __init__(self, visible_path, infrared_path, GT_path, phase="train", transform=None):
        self.phase = phase
        self.visible_files = sorted(glob(os.path.join(visible_path, "*.*")))
        self.infrared_files = sorted(glob(os.path.join(infrared_path, "*.*")))
        self.GT_files = sorted(glob(os.path.join(GT_path, "*.*")))
        self.transform = transform

    def __len__(self):
        l = len(self.infrared_files)
        return l

    def __getitem__(self, item):
        image_A_path = self.visible_files[item]
        image_B_path = self.infrared_files[item]
        image_GT_path = self.GT_files[item]
        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='RGB')   ##########
        image_GT = Image.open(image_GT_path).convert(mode='RGB')   ##########

        # image_A = self.transform(image_A)
        # image_B = self.transform(image_B)

        if self.transform is not None:
            if isinstance(self.transform, RandomCropPair):
                image_A, image_B, image_GT = self.transform(image_A, image_B, image_GT)
            else:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)
                image_GT = self.transform(image_GT)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, image_GT, name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, image_GT, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        image_GT = torch.stack(image_GT, dim=0)
        return images_A, images_B, image_GT, name
'''


class RandomCropPair:
    def __init__(self, size):
        self.size = size  # 

    def __call__(self, vis_img, ir_img):
        # 
        i, j, h, w = RandomCrop.get_params(vis_img, output_size=self.size)
        # 
        vis_img = F.crop(vis_img, i, j, h, w)
        ir_img = F.crop(ir_img, i, j, h, w)

        vis_img = F.to_tensor(vis_img)
        ir_img = F.to_tensor(ir_img)
        return vis_img, ir_img

class SimpleDataSet(Dataset):
    def __init__(self, visible_path, infrared_path, phase="train", transform=None):
        self.phase = phase
        self.visible_files = sorted(glob(os.path.join(visible_path, "*.*")))
        self.infrared_files = sorted(glob(os.path.join(infrared_path, "*.*")))
        self.transform = transform

    def __len__(self):
        l = len(self.infrared_files)
        return l

    def __getitem__(self, item):
        image_A_path = self.visible_files[item]
        image_B_path = self.infrared_files[item]
    
        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='RGB')   ##########

        # image_A = self.transform(image_A)
        # image_B = self.transform(image_B)

        if self.transform is not None:
            if isinstance(self.transform, RandomCropPair):
                image_A, image_B = self.transform(image_A, image_B)
            else:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        return images_A, images_B, name





