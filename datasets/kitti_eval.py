# Copyright Genki Kinoshita, 2024. All rights reserved.
#
# This file is part of a software package that is based on the Monodepth2
# software, originally developed by Niantic. and distributed under the
# Monodepth2 license. The terms of the Monodepth2 license apply to any usage
# of this file in conjunction with the original Monodepth2 software.

from __future__ import annotations

import os

import numpy as np
import torch

from .base_eval import BaseEval


class KITTIEval(BaseEval):
    """Superclass for different types of KITTI dataset loaders"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.array([[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def __getitem__(self, idx):
        scene_name, frame_idx, side = self.index_to_folder_and_frame_idx(idx)

        input_dict = {("color", i, -1): self.get_color(scene_name, frame_idx + i, side) for i in self.adj_frame_idxs}

        self.preprocess(input_dict, lambda x: x)

        for i in self.adj_frame_idxs:
            del input_dict[("color", i, -1)]
            del input_dict[("color_aug", i, -1)]
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.load_intrinsics(scene_name, frame_idx)

            K[0, :] *= self.width // (2**scale)
            K[1, :] *= self.height // (2**scale)
            inv_K = np.linalg.pinv(K)

            input_dict[("K", scale)] = torch.from_numpy(K)
            input_dict[("inv_K", scale)] = torch.from_numpy(inv_K)

        return input_dict

    def get_color(self, folder, frame_idx, side):
        color = self.pil_loader(self.get_image_path(folder, frame_idx, side))
        return color

    def get_image_path(self, folder, frame_idx, side):
        f_str = "{:010d}{}".format(frame_idx, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f"image_0{self.side_map[side]}", "data", f_str)
        return image_path

    def index_to_folder_and_frame_idx(self, idx):
        line = self.filenames[idx].split()
        folder = line[0]

        if len(line) == 3:
            frame_idx = int(line[1])
            side = line[2]
        else:
            frame_idx = 0
            side = None
        return folder, frame_idx, side
