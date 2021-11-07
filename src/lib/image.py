# %%
import os
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
# import albumentations as albu

def img_scaler(img:np.array) -> np.array:
    """ 0~255の範囲にスケーリングする
    Args:
        img (np.array): 入力画像
    Returns:
        np.array: スケーリング画像
    Note:
        画像である必要はないが、array全体でスケーリングされる点に注意。
    """
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    return img


img = np.random.rand(2,2,3)*100
print(img)
# %%
img_scaler(img)
# %%
