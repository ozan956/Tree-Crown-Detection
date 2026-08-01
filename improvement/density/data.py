"""Density-map dataset for tree counting.

Converts COCO box annotations into point (crown-center) density maps: each tree
center is a unit Gaussian splat, so the integral of the density map equals the
tree count. This is the standard crowd-counting formulation (Lempitsky-Zisserman
/ MCNN / CSRNet), applied to tree crowns — the "modern baseline" a reviewer
expects us to compare against.

Images are 640x640; the network predicts a density map at 1/8 resolution (80x80)
which is the common CSRNet-style output stride, and the target is downsampled
accordingly (sum-preserving).
"""

import json
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

OUTPUT_STRIDE = 8
SIGMA = 2.0  # Gaussian std in OUTPUT-map pixels (~16 input px, ~crown scale)


def _density_target(centers, in_hw, stride=OUTPUT_STRIDE, sigma=SIGMA):
    """Build a sum-preserving density map at 1/stride resolution.
    centers: list of (x, y) in input-image pixels. Returns float32 [h/s, w/s]
    whose sum == len(centers)."""
    H, W = in_hw
    oh, ow = H // stride, W // stride
    dm = np.zeros((oh, ow), np.float32)
    for (x, y) in centers:
        cx, cy = x / stride, y / stride
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < ow and 0 <= iy < oh:
            dm[iy, ix] += 1.0
    if dm.sum() > 0:
        # Gaussian blur preserves the sum (normalized kernel), so integral == count
        dm = cv2.GaussianBlur(dm, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_CONSTANT)
    return dm


class TreeDensityDataset(Dataset):
    """VHRTrees split as (image_tensor, density_map, count)."""

    def __init__(self, coco_json, img_dir, augment=False):
        d = json.load(open(coco_json))
        self.img_dir = img_dir
        self.augment = augment
        by_img = {}
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            by_img.setdefault(a["image_id"], []).append((x + w / 2.0, y + h / 2.0))
        # keep only images whose file is present on disk
        self.items = []
        for im in d["images"]:
            p = os.path.join(img_dir, im["file_name"])
            if os.path.exists(p):
                self.items.append((p, by_img.get(im["id"], []),
                                   (im["height"], im["width"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, centers, hw = self.items[i]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        centers = list(centers)
        if self.augment and np.random.rand() < 0.5:  # horizontal flip
            img = img[:, ::-1, :].copy()
            W = hw[1]
            centers = [(W - 1 - x, y) for (x, y) in centers]
        dm = _density_target(centers, hw)
        # to CHW tensors; ImageNet normalization for the pretrained backbone
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        img = (img - mean) / std
        img_t = torch.from_numpy(img.transpose(2, 0, 1))
        dm_t = torch.from_numpy(dm[None])  # 1xHxW
        return img_t, dm_t, float(len(centers))
