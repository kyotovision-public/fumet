# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Modifications made by Genki Kinoshita, 2024.
# These modifications include:
#   - Added some arguments to the parser
#
# This modified version is also licensed under the terms of the Monodepth2
# licence, as outlined in the LICENSE file.

import argparse
import os

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-n", "--dry_run", action="store_true")

        # PATHS
        self.parser.add_argument("--data_path", type=str, help="Path to the training data", default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--root_log_dir", type=str, help="Path to the log directory", default=os.path.join(file_dir, "logs"))
        self.parser.add_argument("--litemono_pretrain_path", type=str, help="Path to the Lite-Mono pretrained model")

        # TRAINING options
        self.parser.add_argument(
            "--initial_cam_height", type=float, help="Initial camera height. So far, this option is valid only for KITTI dataset."
        )
        self.parser.add_argument(
            "--fix_fine_metric_rate",
            action="store_true",
            help="If `--initial_cam_height` is set, the weight of fine metric loss will be fixed through epochs. If not set, this option will be ignored.",
        )
        self.parser.add_argument("--disable_road_masking", action="store_true", help="Disable automasking only in the road region")
        self.parser.add_argument(
            "--remove_outliers", action="store_true", help="Remove outlier cars when computing scale factor and rough metric loss"
        )
        self.parser.add_argument("--outlier_relative_error_th", type=float, default=0.2, help="Threshold for relative error of outliers")

        self.parser.add_argument(
            "--first_update_epoch", type=int, help="Update camera height from this epoch. If not set, update will not be done", default=1
        )
        self.parser.add_argument("--gamma", type=float, help="gamma for step_lr_scheduler", default=0.5)
        self.parser.add_argument("--random_seed", help="random seed", type=int, default=1)
        self.parser.add_argument(
            "--standard_metric",
            type=str,
            help="Metric as an indicator of best model",
            choices=["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3", "loss"],
            default="loss",
        )
        self.parser.add_argument(
            "--segm_dirname",
            type=str,
            help="Prefix of segm directory name. This option is used only for kitti dataset",
            default="heights_segms_with_road",
        )
        self.parser.add_argument("--resume", help="Whether resuming training", action="store_true")
        self.parser.add_argument(
            "--ckpt_timestamp",
            nargs="?",
            const="",
            type=str,
            help="Timestamp of checkpoint. This option is used both for training and evaluation",
        )
        self.parser.add_argument("--last_epoch_for_resume", type=int, help="The last epoch number for resuming training")
        self.parser.add_argument("--model_name", type=str, help="Name of the folder to save the model in", default="")
        self.parser.add_argument(
            "--split",
            type=str,
            help="Which training split to use",
            choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "cityscapes", "cityscapes_preprocessed"],
            default="eigen_zhou",
        )
        self.parser.add_argument("--num_layers", type=int, help="Number of resnet layers", default=18, choices=[18, 34, 50, 101, 152])
        self.parser.add_argument(
            "--dataset",
            type=str,
            help="Dataset name to train on",
            default="kitti",
            choices=["kitti", "cityscapes"],
        )
        self.parser.add_argument("--png", help="If set, train from raw KITTI png files (instead of jpegs)", action="store_true")
        self.parser.add_argument("--height", type=int, help="Input image height", default=192)
        self.parser.add_argument("--width", type=int, help="Input image width", default=640)
        self.parser.add_argument("--fx", type=float, help="focal length in x")
        self.parser.add_argument("--fy", type=float, help="focal length in y")
        self.parser.add_argument("--cx", type=float, help="principal point in x")
        self.parser.add_argument("--cy", type=float, help="principal point in y")
        self.parser.add_argument("--disparity_smoothness", type=float, help="disparity smoothness weight", default=1e-3)
        self.parser.add_argument("--keep_metric_loss_weight_epoch", type=int, default=1, help="Epoch to keep the weight of fine/rough metric loss")
        self.parser.add_argument("--fine_metric_weight", type=float, help="Weight of fine metric loss", default=0.01)
        self.parser.add_argument("--rough_metric_weight", type=float, help="Weight of rough metric loss", default=1.0)
        self.parser.add_argument("--rough_metric_weight_min", type=float, help="Lower bound of the weight of rough metric loss", default=0.005)
        self.parser.add_argument(
            "--gradual_limit_epoch",
            type=int,
            help="Upper bound of epoch for gradual change of the weight of fine/rough metric loss",
            default=20,
        )
        self.parser.add_argument("--scales", nargs="+", type=int, help="resolution scales used in the loss", default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth", type=float, help="minimum depth", default=0.1)
        self.parser.add_argument("--max_depth", type=float, help="maximum depth", default=100.0)
        self.parser.add_argument(
            "--adj_frame_idxs",
            nargs="+",
            type=int,
            help="Indices of adjacent frames centered on the frame of current interest",
            default=[0, -1, 1],
        )

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=8)
        self.parser.add_argument("--learning_rate", type=float, help="learning rate", default=5e-5)
        self.parser.add_argument("--num_epochs", type=int, help="number of epochs", default=40)
        self.parser.add_argument("--scheduler_step_size", type=int, help="step size of the scheduler", default=15)

        # ARCHITECTURE options
        self.parser.add_argument(
            "--arch", type=str, choices=["monodepth2", "HR-Depth", "Lite-Mono", "VADepth"], default="monodepth2", help="model architecture"
        )

        # ABLATION options
        self.parser.add_argument("--avg_reprojection", help="If set, uses average reprojection loss", action="store_true")
        self.parser.add_argument("--disable_automasking", help="If set, doesn't do auto-masking", action="store_true")
        self.parser.add_argument("--no_ssim", help="If set, disables ssim in the loss", action="store_true")
        self.parser.add_argument(
            "--weights_init",
            type=str,
            help="pretrained or scratch",
            default="pretrained",
            choices=["pretrained", "scratch"],
        )
        self.parser.add_argument(
            "--pose_model_input",
            type=str,
            help="How many images the pose network gets",
            default="pairs",
            choices=["pairs", "all"],
        )

        # SYSTEM options
        self.parser.add_argument("--no_cuda", help="If set disables CUDA", action="store_true")
        self.parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=6)

        # LOADING options
        self.parser.add_argument(
            "--models_to_load",
            nargs="+",
            type=str,
            help="models to load",
            default=["encoder", "depth", "pose_encoder", "pose"],
        )

        # LOGGING options
        self.parser.add_argument("--log_frequency", type=int, help="Number of batches between each tensorboard log", default=500)
        self.parser.add_argument("--log_image", action="store_true", help="Whether saving disparities, automasks and images in log")
        ## save_frequency is invalid in trainer_with_segm.py
        self.parser.add_argument("--save_frequency", type=int, help="Number of epochs between each save", default=1)

        # EVALUATION options
        self.parser.add_argument("--pred_depth_scale_factor", help="If set, multiply predictions by this number", type=float, default=1)
        self.parser.add_argument(
            "--eval_split",
            type=str,
            default="eigen",
            choices=["eigen", "eigen_benchmark", "cityscapes"],
            help="Which split to run eval on",
        )
        self.parser.add_argument(
            "--post_process",
            help="If set, will perform the flipping post processing " "from the original monodepth paper",
            action="store_true",
        )
        self.parser.add_argument("--epoch_for_eval", help="the number epochs for using evaluation", type=int)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
