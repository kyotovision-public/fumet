# Copyright Genki Kinoshita, 2024. All rights reserved.
#
# This file is part of a software package that is based on the Monodepth2
# software, originally developed by Niantic. and distributed under the
# Monodepth2 license. The terms of the Monodepth2 license apply to any usage
# of this file in conjunction with the original Monodepth2 software.

from __future__ import annotations

import torch
import torch.utils.data as data
from PIL import Image  # using pillow-simd for increased speed
from torchvision.transforms import v2


def load_pil(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class BaseEval(data.Dataset):
    def __init__(
        self,
        data_path,
        filenames,
        height,
        width,
        adj_frame_idxs,
        num_scales,
        img_ext=".jpg",
    ):
        super().__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        self.adj_frame_idxs = adj_frame_idxs

        self.img_ext = img_ext

        self.pil_loader = load_pil
        self.to_tensor = v2.Compose([v2.ToImageTensor(), v2.ConvertDtype(torch.float32)])

        self.img_interp = v2.InterpolationMode.LANCZOS
        self.resize_func_dict = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize_func_dict[i] = v2.Resize((self.height // s, self.width // s), interpolation=self.img_interp, antialias=True)

    def preprocess(self, input_dict, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for key in list(input_dict):
            if key[0] == "color":
                name, f_idx, _ = key
                for scale in range(self.num_scales):
                    input_dict[(name, f_idx, scale)] = self.resize_func_dict[scale](input_dict[(name, f_idx, scale - 1)])

        for key in list(input_dict):
            f = input_dict[key]
            if key[0] == "color":
                name, f_idx, scale = key
                input_dict[(name, f_idx, scale)] = self.to_tensor(f)
                input_dict[(name + "_aug", f_idx, scale)] = self.to_tensor(color_aug(f))

    def load_intrinsics(self, folder, frame_idx):
        return self.K.copy()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_color(self, folder, frame_idx, side, do_flip):
        raise NotImplementedError

    def get_colors(self, folder, frame_idx, do_flip):
        raise NotImplementedError

    def index_to_folder_and_frame_idx(self, idx: int):
        raise NotImplementedError
