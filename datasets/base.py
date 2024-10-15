from __future__ import annotations

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image  # using pillow-simd for increased speed
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

ROAD_ID = 100


def load_pil(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def color_aug_func(img, aug_params):
    for fn_idx in aug_params[0]:
        factor = aug_params[fn_idx + 1]
        if fn_idx == 0:
            img = F.adjust_brightness(img, factor)
        elif fn_idx == 1:
            img = F.adjust_contrast(img, factor)
        elif fn_idx == 2:
            img = F.adjust_saturation(img, factor)
        elif fn_idx == 3:
            img = F.adjust_hue(img, factor)
    return img


def expand_segms(segms: np.ndarray) -> tuple[np.ndarray, bool]:
    h, w = segms.shape
    ids = np.unique(segms)[1:]  # remove background (id 0)
    does_road_exist = len(ids) > 0 and ids[-1] == ROAD_ID
    if not does_road_exist:  # if road does not exist, we assume there is no car
        return np.zeros((1, h, w), dtype=bool), does_road_exist

    ids = ids[:-1]  # remove ROAD_ID
    expanded_segms = np.zeros((len(ids) + 1, h, w), dtype=bool)  # road segment is on the last channel
    for ch, id_ in enumerate(ids):
        expanded_segms[ch, segms == id_] = True
    expanded_segms[-1, segms == ROAD_ID] = True
    return expanded_segms, does_road_exist


def load_segms_heights_as_tensor(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    npz = np.load(path)
    segms, does_road_exist = expand_segms(npz["segms"].astype(np.uint8))
    segms = torch.from_numpy(segms)
    heights = torch.from_numpy(npz["heights"].astype(np.float32)) if does_road_exist else torch.empty((0,), dtype=torch.float32)
    return segms, heights


class Base(data.Dataset):
    def __init__(
        self,
        data_path,
        filenames,
        height,
        width,
        adj_frame_idxs,
        num_scales,
        segm_dirname: str | None = None,
        is_train=False,
        img_ext=".jpg",
        segm_ext=".npz",
        K: np.ndarray = np.array([[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
    ):
        super().__init__()
        self.cam_settings = None  # valid only when using cityscapes

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.segm_dirname = segm_dirname  # used in kitti dataset
        self.K = K

        self.adj_frame_idxs = adj_frame_idxs  # default: [0, -1, 1]

        self.is_train = is_train
        self.img_ext = img_ext
        self.segm_ext = segm_ext

        self.pil_loader = load_pil
        self.tensor_segms_heights_loader = load_segms_heights_as_tensor
        self.to_tensor = v2.Compose([v2.ToImageTensor(), v2.ConvertDtype(torch.float32)])

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            v2.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.img_interp = v2.InterpolationMode.LANCZOS
        self.resize_func_dict = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize_func_dict[i] = v2.Resize((self.height // s, self.width // s), interpolation=self.img_interp, antialias=True)
        self.load_depth = self.check_depth()

    def preprocess(self, input_dict, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # if not using list(), "dict changed size during iteration" error is thrown.
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

    def get_cam_height(self, city):
        raise NotImplementedError

    def get_road_segms_heights_tensor(self, folder, frame_idx, side, do_flip):
        raise NotImplementedError

    def get_color(self, folder, frame_idx, side, do_flip):
        raise NotImplementedError

    def get_colors(self, folder, frame_idx, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_idx, side, do_flip):
        raise NotImplementedError

    def index_to_folder_and_frame_idx(self, idx: int):
        raise NotImplementedError
