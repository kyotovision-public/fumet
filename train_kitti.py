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
