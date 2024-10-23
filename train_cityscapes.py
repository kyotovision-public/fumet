# Copyright Genki Kinoshita, 2024. All rights reserved.
#
# This file is part of a software package that is based on the Monodepth2
# software, originally developed by Niantic. and distributed under the
# Monodepth2 license. The terms of the Monodepth2 license apply to any usage
# of this file in conjunction with the original Monodepth2 software.

import os
import random

import numpy as np
import torch

from options import Options
from trainer_cityscapes import TrainerCityscapes


def seed_all(seed):
    if not seed:
        seed = 1
    print(f"Using seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    options = Options()
    opts = options.parse()
    seed_all(opts.random_seed)
    trainer = TrainerCityscapes(opts)
    trainer.train()
