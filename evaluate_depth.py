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

import warnings

warnings.filterwarnings("ignore")


import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import networks
from eval_utils import batch_post_process_disparity, compute_errors
from layers import disp_to_depth
from options import Options


def seed_all(seed):
    if not seed:
        seed = 1
    print(f"Using seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def evaluate_depth(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    log_path = (
        Path(opt.root_log_dir)
        / opt.dataset
        / f"{opt.width}x{opt.height}"
        / f"{opt.model_name}{'_' if opt.model_name and opt.ckpt_timestamp else ''}{opt.ckpt_timestamp}"
    )
    save_dir: Path = log_path / "result"
    print(f"All results are saved at {save_dir}")
    models_dir = log_path / "models"
    if opt.epoch_for_eval is not None:
        weights_dir = models_dir / f"weights_{opt.epoch_for_eval}"
    else:
        weights_dir = models_dir / "best_weights"
    save_dir.mkdir(parents=False, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(os.path.join(splits_dir, opt.eval_split, "test_files.txt")) as f:
        filenames = f.readlines()

    print("-> Loading model from ", weights_dir)
    encoder_path = weights_dir / "encoder.pth"
    depth_decoder_path = weights_dir / "depth.pth"

    if opt.arch == "HR-Depth":
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, opt.scales)
    elif opt.arch == "Lite-Mono":
        encoder = networks.LiteMono(model="lite-mono", drop_path_rate=0.2, width=opt.width, height=opt.height)
        depth_decoder = networks.LiteMonoDecoder(encoder.num_ch_enc, opt.scales)
    elif opt.arch == "VADepth":
        encoder = networks.VANEncoder("small", False)
        depth_decoder = networks.VANDecoder(encoder.num_ch_enc, opt.scales)
    elif opt.arch == "monodepth2":
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, opt.scales)

    # LOADING PRETRAINED MODEL
    print("Loading pretrained encoder")
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("Loading pretrained decoder")
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    is_cityscapes = opt.eval_split == "cityscapes"
    if is_cityscapes:
        dataset = datasets.CityscapesEval(
            opt.data_path,
            filenames,
            feed_height,
            feed_width,
            [0],
            opt.fx,
            opt.fy,
            opt.cx,
            opt.cy,
            4,
        )
    else:
        img_ext = ".png" if opt.png else ".jpg"
        dataset = datasets.KITTIEval(
            opt.data_path,
            filenames,
            feed_height,
            feed_width,
            [0],
            4,
            img_ext=img_ext,
        )
    dataloader = DataLoader(
        dataset,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pred_disps = []
    orig_intrinsics = []

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for data in tqdm(dataloader, dynamic_ncols=True):
            input_color = data[("color", 0, 0)].to(device)

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output_dict = depth_decoder(encoder(input_color))

            pred_disp, _ = disp_to_depth(output_dict[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
            if is_cityscapes:
                orig_intrinsics.append(data["orig_K"])

    pred_disps = np.concatenate(pred_disps)
    if is_cityscapes:
        orig_intrinsics = torch.cat(orig_intrinsics).numpy()

    if is_cityscapes:
        gt_depth_idxs = list(range(len(filenames)))
    else:
        gt_path: Path = Path(splits_dir) / opt.eval_split / "gt_depths.npz"
        gt_depths = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)["data"]

    print("-> Evaluating")

    errors = []
    errors_scaling = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        if is_cityscapes:
            gt_depth = np.load(os.path.join(splits_dir, opt.eval_split, "gt_depths", str(gt_depth_idxs[i]).zfill(3) + "_depth.npy"))
            gt_height, gt_width = gt_depth.shape[:2]
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

            orig_fx, orig_fy, orig_cx, orig_cy = orig_intrinsics[i]
            zoom_x = opt.fx / orig_fx
            zoom_y = opt.fy / orig_fy

            resized_cx = orig_cx * zoom_x
            resized_cy = orig_cy * zoom_y

            resized_left = resized_cx - opt.cx
            resized_right = resized_left + feed_width
            resized_top = resized_cy - opt.cy
            resized_bottom = resized_top + feed_height
            unresized_left = int(resized_left / zoom_x)
            unresized_right = int(resized_right / zoom_x)
            unresized_top = int(resized_top / zoom_y)
            unresized_bottom = int(resized_bottom / zoom_y)
            gt_depth = gt_depth[unresized_top:unresized_bottom, unresized_left:unresized_right]
            gt_height, gt_width = gt_depth.shape[:2]
        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        # pred_disp = cv2.resize(pred_disp, (gt_width, gt_height), interpolation=cv2.INTER_LINEAR_EXACT)
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "cityscapes":
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        elif opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        scaled_pred_depth = pred_depth * ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        scaled_pred_depth[scaled_pred_depth < MIN_DEPTH] = MIN_DEPTH
        scaled_pred_depth[scaled_pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))
        errors_scaling.append(compute_errors(gt_depth, scaled_pred_depth))

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(f"Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    mean_errors = np.array(errors).mean(0)
    mean_errors_scaling = np.array(errors_scaling).mean(0)

    print("|             ", end="")
    print("| " + ("{:>8} | " * 8).format("abse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("| w/o scaling ", end="")
    print(("|{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "|")
    print("| w/  scaling ", end="")
    print(("|{: 8.3f}  " * 8).format(*mean_errors_scaling.tolist()) + "|")

    output_filename = f"result{'_postprocess' if opt.post_process else ''}_{opt.eval_split}_{opt.epoch_for_eval}.txt"
    with open(save_dir / output_filename, "w") as f:
        f.write(f"Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}\n\n")
        f.write("\n| " + ("{:>8} | " * 8).format("abse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        f.write("\n")
        f.write(("|{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "|")
        f.write("\n")
        f.write(("|{: 8.3f}  " * 8).format(*mean_errors_scaling.tolist()) + "|")
        f.write("\n")

    print("\n-> Done!")


if __name__ == "__main__":
    options = Options().parse()
    seed_all(options.random_seed)
    evaluate_depth(options)
