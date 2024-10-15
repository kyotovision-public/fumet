from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import PIL.Image as pil
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from .base import Base, color_aug_func


class Cityscapes(Base):
    """Superclass for different types of KITTI dataset loaders"""

    RAW_WIDTH = 1024
    RAW_HEIGHT = 384
    SMALL_TH = 200

    city2cam_height = {
        "aachen": 1.22,
        "bochum": 1.18,
        "bremen": 1.22,
        "cologne": 1.22,
        "darmstadt": 1.22,
        "dusseldorf": 1.22,
        "erfurt": 1.22,
        "hamburg": 1.18,
        "hanover": 1.18,
        "jena": 1.22,
        "krefeld": 1.18,
        "monchengladbach": 1.18,
        "strasbourg": 1.18,
        "stuttgart": 1.22,
        "tubingen": 1.22,
        "ulm": 1.22,
        "weimar": 1.22,
        "zurich": 1.22,
    }
    cam_settings: list[float] = list(set(city2cam_height.values()))

    def __init__(
        self,
        data_path: str,
        filenames: list[str],
        height: int,
        width: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        adj_frame_idxs: list[int],
        num_scales: int,
        segm_dirname: str | None = None,
        is_train=False,
        img_ext=".jpg",
        segm_ext=".npz",
    ):
        super().__init__(data_path, filenames, height, width, adj_frame_idxs, num_scales, segm_dirname, is_train, img_ext, segm_ext)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.cano_K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    def align_img_size(self, input_dict, city: str, frame_name: str) -> None:
        orig_fx, orig_fy, orig_cx, orig_cy = self.load_intrinsic_params(city, frame_name)
        zoom_x = self.fx / orig_fx
        zoom_y = self.fy / orig_fy

        resized_h = int(zoom_y * self.RAW_HEIGHT)
        resized_w = int(zoom_x * self.RAW_WIDTH)

        resized_cx = orig_cx * zoom_x
        resized_cy = orig_cy * zoom_y

        top = int(resized_cy - self.cy)
        left = int(resized_cx - self.cx)
        for key in list(input_dict):
            if key[0] == "color":
                img = input_dict[key]
                img = F.resize(img, (resized_h, resized_w), interpolation=self.img_interp, antialias=True)
                input_dict[key] = F.crop(img, top, left, self.height, self.width)

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

        city, frame_name = self.index_to_folder_and_frame_idx(idx)
        input_dict = self.get_colors(city, frame_name, do_flip)

        if do_color_aug:
            aug_params = v2.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = partial(color_aug_func, aug_params=aug_params)
        else:
            color_aug = lambda x: x

        self.align_img_size(input_dict, city, frame_name)
        self.preprocess(input_dict, color_aug)

        for i in self.adj_frame_idxs:
            del input_dict[("color", i, -1)]
            del input_dict[("color_aug", i, -1)]
        # adjusting intrinsics to match each scale in the pyramid
        segms_road, heights = self.get_segms_heights_tensor(city, frame_name, do_flip)
        road = segms_road[-1]
        segms = segms_road[:-1]
        input_dict["road"] = road
        input_dict["segms"] = segms
        input_dict["heights"] = heights
        input_dict["n_inst"] = heights.shape[0]

        input_dict["gt_cam_height"] = self.get_cam_height(city)

        for scale in range(self.num_scales):
            K = self.cano_K.copy()
            K[0, :] /= 2**scale
            K[1, :] /= 2**scale
            if do_flip:
                K[0, 2] = self.width / (2**scale) - K[0, 2]
            inv_K = np.linalg.pinv(K)

            input_dict[("K", scale)] = torch.from_numpy(K)
            input_dict[("inv_K", scale)] = torch.from_numpy(inv_K)

        return input_dict

    def index_to_folder_and_frame_idx(self, idx):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            ulm ulm_000064_000012
        """
        city, frame_name = self.filenames[idx].split()
        return city, frame_name

    def get_cam_height(self, city: str):
        return self.city2cam_height[city]

    def check_depth(self):
        return False

    def load_intrinsic_params(self, city: float, frame_name: float) -> tuple[float, float, float, float]:
        # adapted from SfMLearner
        camera_file = os.path.join(self.data_path, city, f"{frame_name}_cam.txt")
        camera = np.loadtxt(camera_file, delimiter=",")
        fx = camera[0]
        fy = camera[4]
        cx = camera[2]
        cy = camera[5]
        return fx, fy, cx, cy

    def get_colors(self, city, frame_name, do_flip):
        color = np.array(self.pil_loader(self.get_image_path(city, frame_name)))
        w = color.shape[1] // 3
        input_dict = {}
        input_dict[("color", -1, -1)] = pil.fromarray(color[:, :w])
        input_dict[("color", 0, -1)] = pil.fromarray(color[:, w : 2 * w])
        input_dict[("color", 1, -1)] = pil.fromarray(color[:, 2 * w :])

        if do_flip:
            for key in input_dict:
                input_dict[key] = input_dict[key].transpose(pil.FLIP_LEFT_RIGHT)
        return input_dict

    def get_segms_heights_tensor(self, city, frame_name, do_flip):
        segms, heights = self.tensor_segms_heights_loader(self.get_segm_path(city, frame_name))
        large_inst_bools = segms[:-1].sum((1, 2)) > self.SMALL_TH
        segms = torch.cat((segms[:-1][large_inst_bools], segms[-1:]), dim=0)
        heights = heights[large_inst_bools]
        if do_flip:
            return torch.flip(segms, dims=(2,)), heights
        return segms, heights

    def get_image_path(self, city, frame_name):
        return os.path.join(self.data_path, city, f"{frame_name}{self.img_ext}")

    def get_segm_path(self, city, frame_name):
        return os.path.join(self.data_path, city, f"{frame_name}{self.segm_ext}")
