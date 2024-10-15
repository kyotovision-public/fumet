from __future__ import annotations

import json
import os

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from .base_eval import BaseEval


class CityscapesEval(BaseEval):
    """Superclass for different types of KITTI dataset loaders"""

    RAW_WIDTH = 2048
    RAW_HEIGHT = 1024 * 0.75

    def __init__(
        self,
        data_path: str,
        filenames: list[str],
        height: int,
        width: int,
        adj_frame_idxs: list[int],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        num_scales: int,
        img_ext=".jpg",
    ):
        super().__init__(data_path, filenames, height, width, adj_frame_idxs, num_scales, img_ext)
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
        input_dict["orig_K"] = torch.tensor((orig_fx, orig_fy, orig_cx, orig_cy), dtype=torch.float32)

    def __getitem__(self, idx):
        city, frame_name = self.index_to_folder_and_frame_idx(idx)
        input_dict = self.get_color(city, frame_name)

        self.align_img_size(input_dict, city, frame_name)
        self.preprocess(input_dict, lambda x: x)

        for i in self.adj_frame_idxs:
            del input_dict[("color", i, -1)]
            del input_dict[("color_aug", i, -1)]
        # adjusting intrinsics to match each scale in the pyramid

        for scale in range(self.num_scales):
            K = self.cano_K.copy()
            K[0, :] /= 2**scale
            K[1, :] /= 2**scale
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

    def load_intrinsic_params(self, city: float, frame_name: float) -> tuple[float, float, float, float]:
        # adapted from SfMLearner
        split = "test"

        camera_file = os.path.join(self.data_path, "camera", split, city, frame_name + "_camera.json")
        if not os.path.exists(camera_file):
            a, b, c = frame_name.split("_")
            c = str(19).zfill(6)
            frame_name = "_".join([a, b, c])
            camera_file = os.path.join(self.data_path, "camera", split, city, frame_name + "_camera.json")
        with open(camera_file, "r") as f:
            camera = json.load(f)
        fx = camera["intrinsic"]["fx"]
        fy = camera["intrinsic"]["fy"]
        cx = camera["intrinsic"]["u0"]
        cy = camera["intrinsic"]["v0"]
        return fx, fy, cx, cy

    def get_color(self, city, frame_name):
        color = self.pil_loader(self.get_image_path(city, frame_name))
        # crop down to cityscapes size
        w, h = color.size
        crop_h = h * 3 // 4
        color = color.crop((0, 0, w, crop_h))
        return {("color", 0, -1): color}

    def get_image_path(self, city, frame_name):
        image_path = os.path.join(self.data_path, "leftImg8bit_sequence", "test", city, frame_name + "_leftImg8bit.png")
        return image_path
