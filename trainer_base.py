# Copyright Genki Kinoshita, 2024. All rights reserved.
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the license/LICENSE_Monodepth2 file.
#
# Modifications made by Genki Kinoshita, 2024.
# These modifications were conducted to incorporate FUMET training framework into the one of Monodepth2.
#
# This modified version is also licensed under the terms of the Monodepth2
# licence, as outlined in the license/LICENSE_Monodepth2 file.

import warnings

warnings.filterwarnings("ignore")
import json
import os
import pickle
import random
import time
from argparse import Namespace
from datetime import datetime
from math import log
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datasets
import networks
from layers import SSIM, BackprojectDepth, Project3D, compute_depth_errors, disp_to_depth, transformation_from_parameters
from utils import (
    generate_homo_pix_grid,
    normalize_image,
    readlines,
    sec_to_hm_str,
)

segms_heights_str_set = ("segms", "heights")


def collate_fn(dict_batch):
    # batch: [{...} x batch_size]
    ret_dct = {key: default_collate([d[key] for d in dict_batch]) for key in dict_batch[0] if key not in segms_heights_str_set}
    ret_dct["segms_flat"] = torch.cat([d["segms"] for d in dict_batch], dim=0)
    ret_dct["heights_flat"] = torch.cat([d["heights"] for d in dict_batch], dim=0)
    ret_dct["n_inst"] = torch.tensor(
        [d["n_inst"] for d in dict_batch], dtype=torch.int32
    )  # to use `repeat_interleave()`, dtype needs to be int32 or int64
    return ret_dct


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainerBase:
    def __init__(self, opts):
        self.opt = opts
        can_resume = self.opt.resume and self.opt.ckpt_timestamp
        if can_resume:
            self.log_path = os.path.join(
                self.opt.root_log_dir,
                self.opt.dataset,
                f"{self.opt.width}x{self.opt.height}",
                f"{self.opt.model_name}{'_' if self.opt.model_name else ''}{self.opt.ckpt_timestamp}",
            )
            if not os.path.exists(self.log_path):
                raise FileNotFoundError(f"{self.log_path} does not exist.")
        else:
            self.log_path = os.path.join(
                self.opt.root_log_dir,
                self.opt.dataset,
                f"{opts.width}x{opts.height}",
                f"{opts.model_name}{'_' if opts.model_name else ''}{datetime.now().strftime('%m-%d-%H:%M')}",
            )

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.adj_frame_idxs)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.adj_frame_idxs[0] == 0, "adj_frame_idxs must start with 0"

        if self.opt.arch == "HR-Depth":
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained" and not can_resume)
            self.models["depth"] = networks.HRDepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        elif self.opt.arch == "Lite-Mono":
            self.models["encoder"] = networks.LiteMono(model="lite-mono", drop_path_rate=0.2, width=self.opt.width, height=self.opt.height)
            self.models["depth"] = networks.LiteMonoDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        elif self.opt.arch == "VADepth":
            self.models["encoder"] = networks.VANEncoder("small", self.opt.weights_init == "pretrained" and not can_resume)
            self.models["depth"] = networks.VANDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        elif self.opt.arch == "monodepth2":
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained" and not can_resume)
            self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.opt.arch == "Lite-Mono":
            self.models_pose["pose_encoder"] = networks.LiteMonoResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained" and not can_resume, num_input_images=self.num_pose_frames
            )
            self.models_pose["pose"] = networks.LiteMonoPoseDecoder(
                self.models_pose["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2
            )
        else:
            self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained" and not can_resume, num_input_images=self.num_pose_frames
            )
            self.models_pose["pose"] = networks.PoseDecoder(
                self.models_pose["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2
            )
        self.models_pose["pose_encoder"].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())
        self.models_pose["pose"].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        self.epoch = 0
        if can_resume:
            if self.opt.last_epoch_for_resume is None:
                weights_dir, self.epoch = self.search_last_epoch()
            else:
                weights_dir = Path(self.log_path) / "models" / f"weights_{self.opt.last_epoch_for_resume}"
                self.epoch = self.opt.last_epoch_for_resume
            self.epoch += 1
            self.load_model(weights_dir)
            self.load_cam_heights(weights_dir, False)

        if self.opt.arch == "Lite-Mono":
            self.model_optimizer = optim.AdamW(self.parameters_to_train, 0.0001, weight_decay=1e-2)
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, 0.0001, weight_decay=1e-2)
            if not can_resume and self.opt.weights_init == "pretrained":
                self.load_model_litemono()
            self.model_lr_scheduler = ChainedScheduler(
                self.model_optimizer,
                T_0=30,
                T_mul=1,
                eta_min=5e-6,
                last_epoch=-1,
                max_lr=0.0001,
                warmup_steps=0,
                gamma=0.9,
            )
            self.model_pose_lr_scheduler = ChainedScheduler(
                self.model_pose_optimizer,
                T_0=30,
                T_mul=1,
                eta_min=1e-5,
                last_epoch=-1,
                max_lr=0.0001,
                warmup_steps=0,
                gamma=0.9,
            )
            # self.model_lr_scheduler.last_epoch = self.epoch - 1
            # self.model_pose_lr_scheduler.last_epoch = self.epoch - 1
        else:
            self.parameters_to_train = self.parameters_to_train + self.parameters_to_train_pose
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, self.opt.gamma)
            self.model_lr_scheduler.last_epoch = self.epoch - 1

        print("Training model named:  ", self.opt.model_name)
        if self.opt.dry_run:
            print("\n=====================================================================================\n")
            print("          This is dry-run mode, so no data will be saved               ")
            print("\n=====================================================================================\n")
        else:
            print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:  ", self.device)

        # data
        dataset_dict: dict[str, datasets.Base] = {
            "kitti": datasets.KITTI,
            "cityscapes": datasets.Cityscapes,
        }
        self.dataset = dataset_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        img_ext = ".png" if self.opt.png else ".jpg"
        num_train_samples = len(train_filenames)
        self.n_iter = num_train_samples // self.opt.batch_size
        self.num_total_steps = self.n_iter * self.opt.num_epochs
        self.min_road_area = int(self.opt.width * self.opt.height * 0.01)

        if self.opt.dataset == "cityscapes":
            if self.opt.fx is None or self.opt.fy is None or self.opt.cx is None or self.opt.cy is None:
                raise ValueError("--fx and --fy is not specified though you are using CityscapesDataset.")
            train_dataset = self.dataset(
                self.opt.data_path,
                train_filenames,
                self.opt.height,
                self.opt.width,
                self.opt.fx,
                self.opt.fy,
                self.opt.cx,
                self.opt.cy,
                self.opt.adj_frame_idxs,
                self.num_scales,
                segm_dirname=self.opt.segm_dirname,
                is_train=True,
                img_ext=img_ext,
            )
            val_dataset = self.dataset(
                self.opt.data_path,
                val_filenames,
                self.opt.height,
                self.opt.width,
                self.opt.fx,
                self.opt.fy,
                self.opt.cx,
                self.opt.cy,
                self.opt.adj_frame_idxs,
                self.num_scales,
                is_train=False,
                img_ext=img_ext,
                segm_dirname=self.opt.segm_dirname,
            )
        else:
            if self.opt.fx and self.opt.fy:
                K = np.array(
                    [
                        [self.opt.fx, 0, 0.5, 0],
                        [0, self.opt.fy, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
                train_dataset = self.dataset(
                    self.opt.data_path,
                    train_filenames,
                    self.opt.height,
                    self.opt.width,
                    self.opt.adj_frame_idxs,
                    self.num_scales,
                    segm_dirname=self.opt.segm_dirname,
                    is_train=True,
                    img_ext=img_ext,
                    K=K,
                )
            else:
                train_dataset = self.dataset(
                    self.opt.data_path,
                    train_filenames,
                    self.opt.height,
                    self.opt.width,
                    self.opt.adj_frame_idxs,
                    self.num_scales,
                    segm_dirname=self.opt.segm_dirname,
                    is_train=True,
                    img_ext=img_ext,
                )
            val_dataset = self.dataset(
                self.opt.data_path,
                val_filenames,
                self.opt.height,
                self.opt.width,
                self.opt.adj_frame_idxs,
                self.num_scales,
                is_train=False,
                img_ext=img_ext,
                segm_dirname=self.opt.segm_dirname,
            )
        g = torch.Generator()
        g.manual_seed(self.opt.random_seed)
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        if not self.opt.dry_run:
            self.writers = {}
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2**scale)
            w = self.opt.width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["de/abse", "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        self.standard_metric = self.opt.standard_metric
        if not (self.standard_metric in self.depth_metric_names or self.standard_metric == "loss"):
            raise KeyError(f"{self.standard_metric} is not in {self.depth_metric_names + ['loss']}")

        print("Using split:  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))

        self.homo_pix_grid = generate_homo_pix_grid(self.opt.height, self.opt.width)

        if not self.opt.dry_run:
            self.save_opts()

        self.scale_init_dict = {scale: torch.full((len(self.train_loader.dataset),), torch.nan) for scale in self.opt.scales}
        self.whole_cam_heights_dict: dict[int, torch.Tensor] = self.scale_init_dict.copy()

    def set_train(self):
        """Convert all models to training mode"""
        for m in self.models.values():
            m.train()
        for m in self.models_pose.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()
        for m in self.models_pose.values():
            m.eval()

    def train_epoch(self):
        """Run a single epoch of training and validation"""
        print("Training")
        self.set_train()

        for iteration, batch_input_dict in tqdm(enumerate(self.train_loader), dynamic_ncols=True):
            before_op_time = time.time()

            batch_input_dict = {key: ipt.to(self.device) for key, ipt in batch_input_dict.items()}
            batch_output_dict, loss_dict = self.process_batch(batch_input_dict)

            self.model_optimizer.zero_grad(set_to_none=True)
            if self.opt.arch == "Lite-Mono":
                self.model_pose_optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            self.model_optimizer.step()
            if self.opt.arch == "Lite-Mono":
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = iteration % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step >= 2000 and self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(iteration, duration, loss_dict["loss"].cpu().data)

                if "depth_gt" in batch_input_dict:
                    self.compute_depth_losses(batch_input_dict, batch_output_dict, loss_dict)
                if not self.opt.dry_run:
                    self.log_train(batch_input_dict, batch_output_dict, loss_dict)
            del loss_dict
            self.step += 1
        self.model_lr_scheduler.step()
        if self.opt.arch == "Lite-Mono":
            self.model_pose_lr_scheduler.step()

    def set_gradual_loss_weights(self):
        keep_epoch: int = self.opt.keep_metric_loss_weight_epoch
        self.fine_metric_rate = (
            log(self.epoch // keep_epoch + 1) / log(self.opt.gradual_limit_epoch // keep_epoch + 1)
            if self.epoch // keep_epoch < self.opt.gradual_limit_epoch // keep_epoch
            else 1.0
        )
        self.rough_metric_rate = (
            -log(self.epoch // keep_epoch + 1) / log(self.opt.gradual_limit_epoch // keep_epoch + 1) + 1
            if self.epoch // keep_epoch < self.opt.gradual_limit_epoch // keep_epoch
            else self.opt.rough_metric_weight_min
        )
        if self.opt.initial_cam_height is not None and self.opt.fix_fine_metric_rate:
            self.fine_metric_rate = 1

    def process_batch(self, input_dict, mode="train"):
        """Pass a minibatch through the network and generate images and losses"""

        features = self.models["encoder"](input_dict["color_aug", 0, 0])
        output_dict = self.models["depth"](features)
        output_dict.update(self.predict_poses(input_dict))
        self.generate_images_pred(input_dict, output_dict)
        loss_dict = self.compute_losses(input_dict, output_dict, mode)

        return output_dict, loss_dict

    def predict_poses(self, input_dict):
        """Predict poses between input frames for monocular sequences."""
        output_dict = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            pose_feats = {f_i: input_dict["color_aug", f_i, 0] for f_i in self.opt.adj_frame_idxs}
            for f_i in self.opt.adj_frame_idxs[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models_pose["pose"](pose_inputs)
                output_dict[("axisangle", 0, f_i)] = axisangle
                output_dict[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                output_dict[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat([input_dict[("color_aug", i, 0)] for i in self.opt.adj_frame_idxs], 1)
            pose_inputs = [self.models_pose["pose_encoder"](pose_inputs)]
            axisangle, translation = self.models_pose["pose"](pose_inputs)
            for i, f_i in enumerate(self.opt.adj_frame_idxs[1:]):
                output_dict[("axisangle", 0, f_i)] = axisangle
                output_dict[("translation", 0, f_i)] = translation
                output_dict[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])
        return output_dict

    def val_epoch(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        avg_loss_dict = {}
        with torch.no_grad():
            for batch_input_dict in self.val_loader:
                batch_input_dict = {key: ipt.to(self.device) for key, ipt in batch_input_dict.items()}
                batch_output_dict, loss_dict = self.process_batch(batch_input_dict, mode="val")
                if "depth_gt" in batch_input_dict:
                    self.compute_depth_losses(batch_input_dict, batch_output_dict, loss_dict)
                for loss_name in loss_dict:
                    if loss_name in avg_loss_dict:
                        avg_loss_dict[loss_name] += loss_dict[loss_name]
                    else:
                        avg_loss_dict[loss_name] = loss_dict[loss_name]
                del batch_input_dict, batch_output_dict, loss_dict
            n_iter = len(self.val_loader)
            for loss_name in avg_loss_dict:
                avg_loss_dict[loss_name] /= n_iter
            if not self.opt.dry_run:
                self.log_val(avg_loss_dict)
            return avg_loss_dict

    def generate_images_pred(self, input_dict, output_dict):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        source_scale = 0
        for scale in self.opt.scales:
            disp = output_dict[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            cam_points = self.backproject_depth[source_scale](depth, input_dict[("inv_K", source_scale)])
            output_dict[("depth", scale)] = depth

            h = self.opt.height
            w = self.opt.width
            output_dict[("cam_pts", scale)] = cam_points[:, :-1, :].view(-1, 3, h, w).permute(0, 2, 3, 1)  # [bs, h, w, 3]
            for adj_frame_idx in self.opt.adj_frame_idxs[1:]:
                T = output_dict[("cam_T_cam", 0, adj_frame_idx)]
                pix_coords = self.project_3d[source_scale](cam_points, input_dict[("K", source_scale)], T)
                output_dict[("sample", adj_frame_idx, scale)] = pix_coords
                output_dict[("color", adj_frame_idx, scale)] = F.grid_sample(
                    input_dict[("color", adj_frame_idx, source_scale)],
                    output_dict[("sample", adj_frame_idx, scale)],
                    padding_mode="border",
                    align_corners=True,
                )

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_rough_metric_loss(
        self,
        depth_repeat: torch.Tensor,
        segms_flat: torch.Tensor,
        obj_pix_heights: torch.Tensor,
        obj_heights_flat: torch.Tensor,
        fy: float,
        n_inst_appear_frames: int,
    ) -> torch.Tensor:
        pred_heights = obj_pix_heights[:, None, None] * depth_repeat / fy  # [sum(batch_n_inst), h, w]
        loss = torch.abs((obj_heights_flat[:, None, None] - pred_heights) * segms_flat).sum(dim=(1, 2)) / segms_flat.sum(dim=(1, 2))
        assert not loss.mean().isnan()
        return loss.mean() / n_inst_appear_frames

    def compute_depth_losses(self, input_dict, output_dict, loss_dict):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        batch_depth_pred = output_dict[("depth", 0)].detach()
        batch_depth_pred = torch.clamp(F.interpolate(batch_depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)

        batch_depth_gt = input_dict["depth_gt"]
        batch_mask = batch_depth_gt > 0

        # garg/eigen crop
        batch_crop_mask = torch.zeros_like(batch_mask)
        batch_crop_mask[:, :, 153:371, 44:1197] = 1
        batch_mask = batch_mask * batch_crop_mask

        batch_depth_gt = batch_depth_gt[batch_mask]
        batch_depth_pred = batch_depth_pred[batch_mask]
        ## turnoff median scaling
        # batch_depth_pred *= torch.median(batch_depth_gt) / torch.median(batch_depth_pred)

        batch_depth_pred = torch.clamp(batch_depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(batch_depth_gt, batch_depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            loss_dict[metric] = depth_errors[i].cpu().numpy()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(
            print_string.format(
                self.epoch,
                batch_idx,
                samples_per_sec,
                loss,
                sec_to_hm_str(time_sofar),
                sec_to_hm_str(training_time_left),
            )
        )

    def log_train(self, input_dict, output_dict, loss_dict):
        """Write an event to the tensorboard events file"""
        writer = self.writers["train"]
        for name, loss in loss_dict.items():
            writer.add_scalar(name, loss, self.step)

        if self.opt.log_image:
            for j in range(min(4, self.opt.batch_size)):  # write a maximum of four images
                for scale in self.opt.scales:
                    for frame_id in self.opt.adj_frame_idxs:
                        writer.add_image(f"color_{frame_id}_{scale}/{j}", input_dict[("color", frame_id, scale)][j].data, self.step)
                        if scale == 0 and frame_id != 0:
                            writer.add_image(
                                f"color_pred_{frame_id}_{scale}/{j}",
                                output_dict[("color", frame_id, scale)][j].data,
                                self.step,
                            )

                    writer.add_image(f"disp_{scale}/{j}", normalize_image(output_dict[("disp", scale)][j]), self.step)
                    if not self.opt.disable_automasking:
                        writer.add_image(
                            f"automask_{scale}/{j}",
                            output_dict[f"identity_selection/{scale}"][j][None, ...],
                            self.step,
                        )

    def log_val(self, loss_dict):
        """Write an event to the tensorboard events file"""
        for name, loss in loss_dict.items():
            self.writers["val"].add_scalar(name, loss, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            os.chmod(models_dir, 0o775)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, "opt.json"), "w") as f:
            json.dump(to_save, f, indent=2)

    def load_opts(self):
        json_path = os.path.join(self.log_path, "models", "opt.json")
        try:
            with open(json_path, "r") as f:
                self.opt = Namespace()
                for k, v in json.load(f):
                    setattr(self.opt, k, v)
        except FileNotFoundError:
            print(f"FileNotFoundError: option json file path {self.log_path} does not exist.")

    def save_model(self, is_best=False):
        """Save model weights to disk"""
        if is_best:
            save_best_folder = os.path.join(self.log_path, "models", "best_weights")
            if not os.path.exists(save_best_folder):
                os.makedirs(save_best_folder)
                os.chmod(save_best_folder, 0o775)
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            os.chmod(save_folder, 0o775)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == "encoder":
                # save the sizes - these are needed at prediction time
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
            torch.save(to_save, save_path)
            if is_best:
                torch.save(to_save, os.path.join(save_best_folder, f"{model_name}.pth"))

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            torch.save(to_save, save_path)
            if is_best:
                torch.save(to_save, os.path.join(save_best_folder, f"{model_name}.pth"))

        if self.does_prev_cam_height_dict_exist:
            cam_height_expect_path = os.path.join(save_folder, "cam_height_expect.pkl")
            with open(cam_height_expect_path, "wb") as f:
                pickle.dump(self.prev_cam_height_dict, f, pickle.HIGHEST_PROTOCOL)

        if self.opt.arch == "Lite-Mono":
            optimizer_save_path = os.path.join(save_folder, "adamw.pth")
            optimizer_save_path_pose = os.path.join(save_folder, "adamw_pose.pth")
            torch.save(self.model_optimizer.state_dict(), optimizer_save_path)
            torch.save(self.model_pose_optimizer.state_dict(), optimizer_save_path_pose)
            if is_best:
                torch.save(self.model_optimizer.state_dict(), os.path.join(save_best_folder, "adamw.pth"))
                torch.save(self.model_pose_optimizer.state_dict(), os.path.join(save_best_folder, "adamw_pose.pth"))
        else:
            optimizer_save_path = os.path.join(save_folder, "adam.pth")
            torch.save(self.model_optimizer.state_dict(), optimizer_save_path)
            if is_best:
                torch.save(self.model_optimizer.state_dict(), os.path.join(save_best_folder, "adam.pth"))

    def search_last_epoch(self) -> tuple[Path, int]:
        root_weights_dir = Path(self.log_path) / "models"
        last_epoch = -1
        for weights_dir in root_weights_dir.glob("weights_*"):
            epoch = int(weights_dir.name[8:])
            if epoch > last_epoch:
                last_epoch = epoch
        return root_weights_dir / f"weights_{last_epoch}", last_epoch

    def load_cam_heights(self, weights_dir: Path, alert_if_not_exist: bool):
        # load prev_cam_height
        cam_height_expect_path = weights_dir / "cam_height_expect.pkl"
        if cam_height_expect_path.exists():
            print("Loading prev_cam_height_dict")
            with open(cam_height_expect_path, "rb") as f:
                self.prev_cam_height_dict = pickle.load(f)
                self.does_prev_cam_height_dict_exist = True
        elif alert_if_not_exist:
            raise FileNotFoundError(f"\n{cam_height_expect_path} does not exists.\n")
        else:
            print(f"\n{cam_height_expect_path} does not exists.\n")

    def load_model(self, weights_dir: Path):
        """Load model(s) from disk"""

        print(f"loading model from folder {weights_dir}")

        for model_name in self.opt.models_to_load:
            print(f"Loading {model_name} weights...")
            path = weights_dir / f"{model_name}.pth"
            pose_or_depth = self._pose_or_depth(model_name)
            if pose_or_depth == "depth":
                model_dict = self.models[model_name].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[model_name].load_state_dict(model_dict)
            else:
                model_dict = self.models_pose[model_name].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[model_name].load_state_dict(model_dict)

        if self.opt.arch == "Lite-Mono":
            optimizer_load_path = weights_dir / "adamw.pth"
            optimizer_load_path_pose = weights_dir / "adamw_pose.pth"
        else:
            optimizer_load_path = weights_dir / "adam.pth"
        if optimizer_load_path.exists():
            print("Loading optimizer weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find optimizer weights so the optimizer is randomly initialized")
        if self.opt.arch == "Lite-Mono":
            if optimizer_load_path_pose.exists():
                print("Loading optimizer weights for pose")
                optimizer_dict_pose = torch.load(optimizer_load_path_pose)
                self.model_pose_optimizer.load_state_dict(optimizer_dict_pose)
            else:
                print("Cannot find optimizer weights for pose so the optimizer is randomly initialized")

    def _pose_or_depth(self, model_name: str) -> str:
        if model_name in ["encoder", "depth"]:
            return "depth"
        return "pose"

    def load_model_litemono(self):
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(self.opt.litemono_pretrain_path)["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith("norm"))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print("Loaded pretrained Lite-Mono encoder from", self.opt.litemono_pretrain_path)

    def train(self):
        raise NotImplementedError("train() is not implemented")

    def compute_losses(self, input_dict, output_dict, mode):
        raise NotImplementedError("compute_losses() is not implemented")

    def compute_cam_heights_and_road_normals(self, *args):
        raise NotImplementedError("compute_cam_heights_and_road_normals() is not implemented")

    def log_cam_height(self):
        raise NotImplementedError("log_cam_height() is not implemented")
