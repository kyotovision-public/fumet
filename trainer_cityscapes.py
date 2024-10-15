import warnings

warnings.filterwarnings("ignore")
import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from layers import get_smooth_loss
from trainer_base import TrainerBase
from utils import (
    calc_obj_pix_height_over_dist_to_horizon_with_flat,
    cam_pts2cam_height_with_cross_prod,
    cam_pts2normal,
    compute_scaled_cam_heights,
    masks_to_pix_heights,
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


class TrainerCityscapes(TrainerBase):
    def __init__(self, opts):
        self.does_prev_cam_height_dict_exist = False
        super().__init__(opts)
        self.scale_init_dict: dict[float, dict[int, list[float]]] = {
            cam_setting: {scale: [] for scale in self.opt.scales} for cam_setting in self.dataset.cam_settings
        }

    def train(self):
        """Run the entire training pipeline"""
        self.step = self.epoch * len(self.train_loader)
        self.start_time = time.time()
        lower_is_better = self.standard_metric[:2] == "de" or self.standard_metric == "loss"
        th_best = 1000000 if lower_is_better else -1

        print(f"========= Training has started from {self.epoch} epoch. ========= ")

        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.n_inst_frames = 0
            self.whole_cam_heights_dict = deepcopy(self.scale_init_dict)

            self.train_epoch()
            val_loss_dict = self.val_epoch()
            val_loss = val_loss_dict[self.standard_metric]
            if self.opt.dry_run:
                print("\n=====================================================================================")
                print("          This is dry-run mode               ")
                print("=====================================================================================\n")

            epoch = self.epoch
            if self.opt.first_update_epoch is not None:
                if self.epoch == self.opt.first_update_epoch:
                    self.prev_cam_height_dict: dict[float, dict[int, float]] = {
                        cam_setting: {scale: 0.0 for scale in self.opt.scales} for cam_setting in self.dataset.cam_settings
                    }
                    self.does_prev_cam_height_dict_exist = True
                if self.opt.first_update_epoch == 0:
                    epoch += 1
            if self.does_prev_cam_height_dict_exist:
                new_cam_height_dict = {
                    cam_setting: {
                        scale: torch.quantile(torch.tensor(cam_heights, dtype=torch.float32), q=0.5) if len(cam_heights) > 0 else 0
                        for scale, cam_heights in dct.items()
                    }
                    for cam_setting, dct in self.whole_cam_heights_dict.items()
                }
                print(f"pseudo camera height has been updated at the end of the epoch{self.epoch}")
                print("new_cam_height_dict: ", end="")
                print(new_cam_height_dict)
                self.prev_cam_height_dict = {
                    cam_setting: {
                        scale: (prev_cam_height * (epoch - 1) * epoch / 2 + epoch * new_cam_height_dict[cam_setting][scale])
                        / (epoch * (epoch + 1) / 2)
                        if new_cam_height_dict[cam_setting][scale] > 0
                        else prev_cam_height
                        for scale, prev_cam_height in prev_dct.items()
                    }
                    for cam_setting, prev_dct in self.prev_cam_height_dict.items()
                }
                if not self.opt.dry_run:
                    self.log_cam_height()
            if not self.opt.dry_run:
                if (lower_is_better and val_loss < th_best) or (not lower_is_better and val_loss > th_best):
                    self.save_model(is_best=True)
                    th_best = val_loss
                else:
                    self.save_model(is_best=False)

    def compute_losses(self, input_dict, output_dict, mode):
        """Compute the reprojection and smoothness losses for a minibatch"""

        self.set_gradual_loss_weights()

        source_scale = 0
        loss_dict = {}
        total_loss = 0

        batch_road: torch.Tensor = input_dict["road"]
        batch_road_appear_bools = batch_road.sum((1, 2)) > self.min_road_area
        batch_road_wo_no_road = batch_road[batch_road_appear_bools]
        inv_Ks = input_dict[("inv_K", source_scale)][:, :3, :3][batch_road_appear_bools]  # [bs, 3, 3]
        batch_gt_cam_height_wo_no_road = input_dict["gt_cam_height"][batch_road_appear_bools]

        h = self.opt.height
        w = self.opt.width
        img_area = h * w
        batch_n_inst: torch.Tensor = input_dict["n_inst"]

        n_inst_appear_frames = (batch_n_inst > 0).sum()
        if n_inst_appear_frames > 0:
            segms_flat: torch.Tensor = input_dict["segms_flat"]
            heights_flat: torch.Tensor = input_dict["heights_flat"]

            flat_road_appear_bools = batch_road_appear_bools.repeat_interleave(batch_n_inst, dim=0)

            # HACK: if no road is detected, regard that there is no car
            segms_flat = segms_flat[flat_road_appear_bools]
            batch_n_inst = batch_n_inst[batch_road_appear_bools]
            heights_flat = heights_flat[flat_road_appear_bools]
            gt_cam_heights_repeat: torch.Tensor = batch_gt_cam_height_wo_no_road.repeat_interleave(batch_n_inst)

            obj_pix_heights = masks_to_pix_heights(segms_flat)

            if self.opt.remove_outliers and batch_road_wo_no_road.shape[0] > 0 and self.does_prev_cam_height_dict_exist:
                outlier_rel_th = self.opt.outlier_relative_error_th

                bs_wo_no_road = batch_road_wo_no_road.shape[0]
                batch_cam_pts_wo_no_road: torch.Tensor = output_dict[("cam_pts", source_scale)].detach()[batch_road_appear_bools]

                batch_road_normals = torch.zeros((bs_wo_no_road, 3, 1), device=self.device)
                for batch_wo_no_road_idx in range(bs_wo_no_road):
                    batch_road_normals[batch_wo_no_road_idx] = cam_pts2normal(
                        batch_cam_pts_wo_no_road[batch_wo_no_road_idx], batch_road_wo_no_road[batch_wo_no_road_idx]
                    )
                horizons = (inv_Ks.transpose(1, 2) @ batch_road_normals).squeeze()  # [bs_wo_no_road, 3]
                obj_pix_height_over_dist_to_horizon = calc_obj_pix_height_over_dist_to_horizon_with_flat(
                    self.homo_pix_grid, segms_flat, horizons, batch_n_inst
                )
                cam_heights = torch.tensor(
                    [self.prev_cam_height_dict[cam_setting.item()][0] for cam_setting in gt_cam_heights_repeat],
                    dtype=torch.float32,
                    device=self.device,
                )
                approx_heights = obj_pix_height_over_dist_to_horizon * cam_heights
                relative_err = (approx_heights - heights_flat).abs() / heights_flat
                inlier_bools = relative_err < outlier_rel_th
                segms_flat = segms_flat[inlier_bools]
                obj_pix_heights = obj_pix_heights[inlier_bools]
                heights_flat = heights_flat[inlier_bools]
                batch_n_inst = torch.tensor([chunk.sum() for chunk in torch.split(inlier_bools, batch_n_inst.tolist())], device=self.device)
                n_inst_appear_frames = (batch_n_inst > 0).sum()

        if mode == "train":
            self.n_inst_frames += n_inst_appear_frames

        fy: float = input_dict[("K", source_scale)][0, 1, 1]
        batch_target: torch.Tensor = input_dict[("color", 0, source_scale)]
        for scale in self.opt.scales:
            loss = 0.0
            reprojection_losses = []
            batch_disp: torch.Tensor = output_dict[("disp", scale)]
            batch_color: torch.Tensor = input_dict[("color", 0, scale)]
            batch_cam_pts_wo_no_road: torch.Tensor = output_dict[("cam_pts", scale)][batch_road_appear_bools]  # [bs, h, w, 3]

            for adj_frame_idx in self.opt.adj_frame_idxs[1:]:
                batch_pred = output_dict[("color", adj_frame_idx, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(batch_pred, batch_target))

            reprojection_losses = torch.cat(reprojection_losses, 1)  # [bs, 2, h, w]

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for adj_frame_idx in self.opt.adj_frame_idxs[1:]:
                    batch_pred = input_dict[("color", adj_frame_idx, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(batch_pred, batch_target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)  # [bs, 2, h, w]

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 0.00001
                # combined_loss = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)  # [bs, 4, h, w]
                combined_loss = torch.cat((reprojection_loss, identity_reprojection_loss), dim=1)  # [bs, 4, h, w]
            else:
                combined_loss = reprojection_loss

            if combined_loss.shape[1] == 1:
                to_optimize = combined_loss
            else:
                to_optimize, idxs = torch.min(combined_loss, dim=1)  # idxs have already been detached
                if self.opt.disable_road_masking:
                    # consider images where over 75% pixels are moving as moving scene.
                    # if the scene is moving, we do not mask out the road
                    automasks = (idxs < 2).float().sum((1, 2))  # [bs,]
                    batch_moving_bools = (1 - automasks / img_area) < 0.75
                    batch_road_moving_scene = (batch_moving_bools[:, None, None] * batch_road) > 0  # convert uint8 into bool for deprecation
                    to_optimize[batch_road_moving_scene] = reprojection_loss.amin(1)[batch_road_moving_scene]

            final_reprojection_loss = to_optimize.mean()
            loss_dict[f"loss/reprojection_{scale}"] = final_reprojection_loss.item()
            loss += final_reprojection_loss

            mean_disp = batch_disp.mean(2, True).mean(3, True)
            norm_disp = batch_disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, batch_color)

            loss_dict[f"loss/smoothness_{scale}"] = smooth_loss.item()
            loss += self.opt.disparity_smoothness * smooth_loss / (2**scale)

            if batch_road_wo_no_road.shape[0] > 0:
                fine_metric_loss, unscaled_cam_heights, road_normal_neg_wo_no_road = self.compute_cam_heights_and_road_normals(
                    batch_cam_pts_wo_no_road,
                    batch_road_wo_no_road,
                    batch_gt_cam_height=batch_gt_cam_height_wo_no_road,
                )
                if self.does_prev_cam_height_dict_exist:
                    loss_dict[f"loss/fine_metric_{scale}"] = fine_metric_loss.item()
                    loss += self.opt.fine_metric_weight * self.fine_metric_rate * fine_metric_loss
                if n_inst_appear_frames > 0:
                    depth_repeat: torch.Tensor = (
                        output_dict[("depth", scale)].squeeze(1)[batch_road_appear_bools].repeat_interleave(batch_n_inst, dim=0)
                    )  # [sum(batch_n_insts), h, w]
                    if self.opt.rough_metric_weight > 0:
                        rough_metric_loss = self.compute_rough_metric_loss(
                            depth_repeat,
                            segms_flat,
                            obj_pix_heights,
                            heights_flat,
                            fy,
                            n_inst_appear_frames,
                        )
                        loss_dict[f"loss/rough_metric_{scale}"] = rough_metric_loss.item()
                        loss += self.opt.rough_metric_weight * self.rough_metric_rate * rough_metric_loss
                    if mode == "train":
                        scaled_cam_heights = compute_scaled_cam_heights(
                            segms_flat,
                            batch_n_inst,
                            road_normal_neg_wo_no_road,
                            batch_cam_pts_wo_no_road.detach(),
                            unscaled_cam_heights,
                            heights_flat,
                            remove_nan=False,
                        )
                        for i, cam_setting in enumerate(batch_gt_cam_height_wo_no_road):
                            if not scaled_cam_heights[i].isnan():
                                cam_setting: float = cam_setting.item()
                                self.whole_cam_heights_dict[cam_setting][scale].append(scaled_cam_heights[i].item())
            else:
                loss_dict[f"loss/rough_metric_{scale}"] = 0.0
                if self.does_prev_cam_height_dict_exist:
                    loss_dict[f"loss/fine_metric_{scale}"] = 0.0

            total_loss += loss
            loss_dict[f"loss/{scale}"] = loss.item()

        total_loss /= self.num_scales
        loss_dict["loss"] = total_loss
        return loss_dict

    def compute_cam_heights_and_road_normals(
        self,
        batch_cam_pts: torch.Tensor,  # [bs, h, w, 3]
        batch_road: torch.Tensor,
        requires_loss: bool = True,
        batch_gt_cam_height: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        bs = batch_cam_pts.shape[0]
        frame_unscaled_cam_heights = torch.zeros(bs, device=self.device)
        if requires_loss:
            cam_height_loss = torch.tensor(0.0, device=self.device)
        batch_road_normal_neg = torch.zeros((bs, 3), device=self.device)
        batch_cam_height, batch_img_normal = cam_pts2cam_height_with_cross_prod(batch_cam_pts)
        batch_road[:, -1, :] = 0
        batch_road[:, :, 0] = 0
        batch_road[:, :, -1] = 0
        for batch_idx in range(bs):
            road_cam_heights = batch_cam_height[batch_idx, batch_road[batch_idx] == 1]  # [?,]
            road_normals = batch_img_normal[batch_idx, batch_road[batch_idx] == 1]  # [?, 3]
            sum_normals = road_normals.detach().sum(0)
            batch_road_normal_neg[batch_idx] = -sum_normals / torch.norm(sum_normals)  # positive value
            frame_unscaled_cam_heights[batch_idx] = road_cam_heights.detach().quantile(0.5)
            if requires_loss and self.does_prev_cam_height_dict_exist and batch_gt_cam_height is not None:
                cam_height_loss = (
                    cam_height_loss + torch.abs(self.prev_cam_height_dict[batch_gt_cam_height[batch_idx].item()][0] - road_cam_heights).mean()
                )
        if requires_loss:
            return cam_height_loss / bs, frame_unscaled_cam_heights, batch_road_normal_neg
        return frame_unscaled_cam_heights, batch_road_normal_neg

    def log_cam_height(self):
        for writer in self.writers.values():
            for cam_setting in self.dataset.cam_settings:
                for scale in self.opt.scales:
                    writer.add_scalar(f"cam_height_expect_{cam_setting}_{scale}", self.prev_cam_height_dict[cam_setting][scale], self.step)
            writer.add_scalar("n_inst_frames", self.n_inst_frames, self.step)
