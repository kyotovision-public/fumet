from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def main():
    src_root_dir = Path(__file__).parents[1] / "kitti_data_heights_segms"
    tgt_root_dir = Path(__file__).parents[1] / "kitti_data"

    for date_dir in src_root_dir.iterdir():
        if not date_dir.is_dir():
            continue
        with ProcessPoolExecutor(max_workers=10) as executor:
            for drive_dir in date_dir.iterdir():
                if not drive_dir.is_dir():
                    continue
                for src_segm_dir in drive_dir.glob("heights_segms*"):
                    tgt_segm_dir = tgt_root_dir / src_segm_dir.relative_to(src_root_dir)
                    executor.submit(src_segm_dir.rename, tgt_segm_dir)


if __name__ == "__main__":
    main()
