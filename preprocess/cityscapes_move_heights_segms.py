# Copyright Genki Kinoshita, 2024. All rights reserved.
#
# This file is part of a software package that is based on the Monodepth2
# software, originally developed by Niantic. and distributed under the
# Monodepth2 license. The terms of the Monodepth2 license apply to any usage
# of this file in conjunction with the original Monodepth2 software.

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def main():
    src_root_dir = Path(__file__).parents[1] / "cityscapes_data_heights_segms"
    tgt_root_dir = Path(__file__).parents[1] / "cityscapes_data"

    for city_dir in src_root_dir.iterdir():
        if not city_dir.is_dir():
            continue
        with ProcessPoolExecutor(max_workers=10) as executor:
            for src_path in city_dir.glob("*.npz"):
                tgt_path = tgt_root_dir / city_dir.name / src_path.name
                executor.submit(src_path.rename, tgt_path)


if __name__ == "__main__":
    main()
