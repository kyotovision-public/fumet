# Copyright Genki Kinoshita, 2024. All rights reserved.
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the license/LICENSE_Monodepth2 file.
#
# Modifications made by Genki Kinoshita, 2024.
# These modifications include:
#   - Added seed_all function
#
# This modified version is also licensed under the terms of the Monodepth2
# licence, as outlined in the license/LICENSE_Monodepth2 file.

import os
import random

import numpy as np
import torch

from options import Options
from trainer_kitti import TrainerKITTI


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
    trainer = TrainerKITTI(opts)
    trainer.train()
