# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Modifications made by Genki Kinoshita, 2024.
# These modifications were conducted mainly for creating a base dataset class and inheriting it.
#
# This modified version is also licensed under the terms of the Monodepth2
# licence, as outlined in the LICENSE file.

from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import PIL.Image as pil
import skimage.transform
import torch
from torchvision.transforms import v2

from kitti_utils import generate_depth_map

from .base import Base, color_aug_func


class KITTI(Base):
    """KITTI dataset which loads the original velodyne depth maps for ground truth"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

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

    def get_image_path(self, folder, frame_idx, side):
        f_str = "{:010d}{}".format(frame_idx, self.img_ext)
        # image_path = os.path.join(self.data_path, folder, f"image_{self.width}x{self.height}_0{self.side_map[side]}", f_str)
        image_path = os.path.join(self.data_path, folder, f"image_0{self.side_map[side]}", "data", f_str)
        return image_path

    def get_segm_path(self, folder, frame_idx, side):
        f_str = "{:010d}{}".format(frame_idx, self.segm_ext)
        segm_path = os.path.join(self.data_path, folder, f"{self.segm_dirname}_{self.width}x{self.height}_0{self.side_map[side]}", f_str)
        return segm_path

    def get_depth(self, folder, frame_idx, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(self.data_path, folder, "velodyne_points/data/{:010d}.bin".format(int(frame_idx)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode="constant")

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_idx = int(line[1])

        velo_filename = os.path.join(self.data_path, scene_name, "velodyne_points/data/{:010d}.bin".format(int(frame_idx)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_idx, side, do_flip):
        color = self.pil_loader(self.get_image_path(folder, frame_idx, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_segms_heights_tensor(self, folder, frame_idx, side, do_flip):
        segms, heights = self.tensor_segms_heights_loader(self.get_segm_path(folder, frame_idx, side))
        if do_flip:
            return torch.flip(segms, dims=(2,)), heights
        return segms, heights

    def __getitem__(self, idx):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <adj_frame_idx>, <scale>)          for raw colour images,
            ("color_aug", <adj_frame_idx>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)             for camera intrinsics,
            "depth_gt"                                   for ground truth depth maps.

        <adj_frame_idx> is an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'idx',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        scene_name, frame_idx, side = self.index_to_folder_and_frame_idx(idx)
        input_dict = {("color", i, -1): self.get_color(scene_name, frame_idx + i, side, do_flip) for i in self.adj_frame_idxs}

        segms_road, heights = self.get_segms_heights_tensor(scene_name, frame_idx, side, do_flip)
        road = segms_road[-1]
        segms = segms_road[:-1]
        input_dict["road"] = road
        input_dict["segms"] = segms
        input_dict["heights"] = heights
        input_dict["n_inst"] = heights.shape[0]

        if do_color_aug:
            aug_params = v2.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = partial(color_aug_func, aug_params=aug_params)
        else:
            color_aug = lambda x: x
        self.preprocess(input_dict, color_aug)

        for i in self.adj_frame_idxs:
            del input_dict[("color", i, -1)]
            del input_dict[("color_aug", i, -1)]
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.load_intrinsics(scene_name, frame_idx)

            K[0, :] *= self.width // (2**scale)
            K[1, :] *= self.height // (2**scale)
            if do_flip:
                K[0, 2] = self.width // (2**scale) - K[0, 2]
            inv_K = np.linalg.pinv(K)

            input_dict[("K", scale)] = torch.from_numpy(K)
            input_dict[("inv_K", scale)] = torch.from_numpy(inv_K)

        if self.load_depth:
            depth_gt = self.get_depth(scene_name, frame_idx, side, do_flip)
            input_dict["depth_gt"] = np.expand_dims(depth_gt, 0)
            input_dict["depth_gt"] = torch.from_numpy(input_dict["depth_gt"].astype(np.float32))
        return input_dict
