# FUMET

This is an official implementation of FUMET, which trains monocular depth estimators and makes them learn metric scale only from dashcam videos. FUMET is introduced in 
> **Camera Height Doesn't Change: Unsupervised Training for Metric Monocular Road-Scene Depth Estimation**
> 
> [Genki Kinoshita](https://GenkiK.github.io) and [Ko Nishino](https://vision.ist.i.kyoto-u.ac.jp/people/)
>
> [Project Page](https://vision.ist.i.kyoto-u.ac.jp/research/fumet/)  [ECCV 2024 (arXiv PDF)](https://arxiv.org/abs/2312.04530)

<p align="center">
  <img src="assets/teaser.gif" alt="Qualitative results on KITTI" width="1000"/>
</p>

**NOTE: All data, codes, and weights will be publicly available soon.**

## Setup

### (1) Build with [Apptainer (formerly Singularity)](https://apptainer.org/documentation/)
We built the environment with SingularityCE 3.10.4. You have to install Apptainer/Singularity in advance. Please refer to its official documentation.</br>
You can build and run the environment with the following command: 
```
singularity build  --fakeroot fumet.sif fumet.def
singularity run --nv fumet.sif
```

### (2) Build on local
We tested our code on Ubuntu18.04 with CUDA11.7.1 and cuDNN8. You have to install `pyenv` or install Python 3.10.4 locally in advance. Here is the command to build with `pyenv`.
```
pyenv install 3.10.4
pyenv local 3.10.4
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Dataset preparation for training
So far, we only provide the training dataset with 640x192 resolution for KITTI and 512x192 resolution for Cityscapes.

### KITTI
Please refer to [the documentation of Monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) to download and unzip KITTI dataset. We expect that you have converted the png images to jpeg as in Monodepth2.

After downloading KITTI raw dataset, you have to download `.npz` data, which consist of car masks and their estimated heights, from [HERE](...).

Then, unzip the data and move under `kitti_data` with the following command:
```
cd /path/to/fumet
unzip kitti_data_heights_segms.zip -d kitti_data_heights_segms
python3 preprocess/kitti_move_heights_segms.py
rmdir kitti_data_heights_segms
```

### Cityscapes
Please follow [SfMLearner](https://github.com/tinghuiz/SfMLearner#preparing-training-data) and download `leftImg8bit_sequence_trainvaltest.zip` and `camera_trainvaltest.zip` from the [Cityscapes official page](https://www.cityscapes-dataset.com/downloads/), then preprocess data with SfMLearner's [prepare_train_data.py](https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py) script. As [ManyDepth](https://github.com/nianticlabs/manydepth), we used the following command:
```
python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir <path to downloaded cityscapes data> \
    --dataset_name cityscapes \
    --dump_root /path/to/fumet/cityscapes_data \
    --seq_length 3 \
    --num_threads 8
```
We assume that the data will be stored in `/path/to/fumet/cityscapes_data`.

You also have to download `.npz` data including car masks and their estimated heights from [HERE](...).
After downloading, please unzip the data and move under `cityscapes_data` with the following command:
```
cd /path/to/fumet
unzip cityscapes_data_heights_segms.zip -d cityscapes_data_heights_segms
python3 preprocess/cityscapes_move_heights_segms.py
rmdir cityscapes_data_heights_segms
```

## Training 

### KITTI
Once you finish dataset preparation, you can train the model by running the following command:
```
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd /path/to/fumet
DEVICE=<CUDA index>

CUDA_VISIBLE_DEVICES=$DEVICE python -OO train_kitti.py \
    --model_name monodepth2_R50 \
    --height 192 \
    --width 640 \
    --remove_outliers \
    --disable_road_masking \
    --num_epochs=60 \
    --num_layers=50 \
    --num_workers=6
```
In the default settings, model weights and tensorboard logs will be saved in /path/to/fumet/logs/kitti/640x192/{model_name}_Month-Date-HH:MM. You can change the log directory by setting `--root_log_dir`.

If you want to change the model architecture, you should set `--arch`. So far, we support [Monodepth2](https://github.com/nianticlabs/monodepth2)(default), [VADepth](https://github.com/xjixzz/vadepth-net), [HR-Depth](https://github.com/shawLyu/HR-Depth), and [Lite-Mono](https://github.com/noahzn/Lite-Mono).

For Lite-Mono, please download the weights pretrained on ImageNet from [HERE](https://surfdrive.surf.nl/files/index.php/s/InMMGd5ZP2fXuia).

```
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd /path/to/fumet
DEVICE=<CUDA index>

# VADepth
CUDA_VISIBLE_DEVICES=$DEVICE python -OO train_kitti.py \
    --model_name VADepth \
    --height 192 \
    --width 640 \
    --remove_outliers \
    --disable_road_masking \
    --num_epochs 60 \
    --num_workers 6 \
    --arch VADepth

# HR-Depth
CUDA_VISIBLE_DEVICES=$DEVICE python -OO train_kitti.py \
    --model_name HR-Depth \
    --height 192 \
    --width 640 \
    --remove_outliers \
    --disable_road_masking \
    --num_epochs 60 \
    --num_workers 6 \
    --arch HR-Depth \
    --num_layers=18

# Lite-Mono
LITEMONO_PRETRAIN_PATH=/path/to/donwloaded/lite-mono/pretrained/weights
CUDA_VISIBLE_DEVICES=$DEVICE python -OO train_kitti.py \
    --model_name Lite-Mono \
    --height 192 \
    --width 640 \
    --remove_outliers \
    --disable_road_masking \
    --num_epochs 60 \
    --num_workers 6 \
    --arch Lite-Mono \
    --num_layers=18 \
    --litemono_pretrain_path $LITEMONO_PRETRAIN_PATH \
    --scales 0 1 2
```

### Cityscapes
You can train the model on Cityscapes with the following command:
```
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd /path/to/fumet
DEVICE=<CUDA index>

NUM_EPOCHS=35
NUM_WORKERS=6
GRADUAL_LIMIT_EPOCH=12 # 39810(KITTI) / 69731(Cityscapes) * 20(default value on KITTI)
SCHEDULER_STEP_SIZE=8 # 39810 / 69731 * 15

CUDA_VISIBLE_DEVICES=$DEVICE python -OO train_cityscapes.py \
    --model_name monodepth2_R50 \
    --dataset cityscapes \
    --split cityscapes_preprocessed \
    --data_path /path/to/fumet/cityscapes_data \
    --height 192 \
    --width 512 \
    --remove_outliers \
    --disable_road_masking \
    --num_layers 50 \
    --num_workers $NUM_WORKERS \
    --num_epochs $NUM_EPOCHS \
    --fx 587.5 \
    --fy 587.5 \
    --cx 267.5 \
    --cy 130.0 \
    --gradual_limit_epoch $GRADUAL_LIMIT_EPOCH \
    --scheduler_step_size $SCHEDULER_STEP_SIZE
```

### Mixed datasets
In the paper, we trained Monodepth2 on mixed datasets including [Argoverse2](https://www.argoverse.org/av2.html), [A2D2](https://www.a2d2.audi/a2d2/en.html), [DDAD](https://github.com/TRI-ML/DDAD), and [Lyft](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles). The preprocessed data will be publicly available soon.

## Trained weights 
You can download the trained models from the links below. `Mix` includes [Argoverse2](https://www.argoverse.org/av2.html), [A2D2](https://www.a2d2.audi/a2d2/en.html), [DDAD](https://github.com/TRI-ML/DDAD), and [Lyft](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles).

| Model | Dataset | Resolution(WxH) | Intrinsics |
|-------|---------|------------|------------|
| [Monodepth2 R50](...) | KITTI | 640x192 | [link](intrinsics/KITTI_640x192.txt) |
| [Lite-Mono R18](...) | KITTI | 640x192 | [link](intrinsics/KITTI_640x192.txt) |
| [VADepth](...) | KITTI | 640x192 | [link](intrinsics/KITTI_640x192.txt) |
| [HR-Depth R18](...) | KITTI | 640x192 | [link](intrinsics/KITTI_640x192.txt) |
| [Monodepth2 R50](...) | Cityscapes | 512x192 | [link](intrinsics/Cityscapes_512x192.txt) |
| [Monodepth2 R50](...) | Mix | 832x512 | [link](intrinsics/Mix_832x512.txt) |
| [Monodepth2 R50](...) | Mix+KITTI | 832x512 | [link](intrinsics/Mix+KITTI_832x512.txt) |
| [Monodepth2 R50](...) | Mix+KITTI+YouTube | 832x512 | [link](intrinsics/Mix+KITTI+YouTube_832x512.txt) |

## Evaluation

### KITTI
Please run the following command to prepare ground-truth depth maps.
```
python export_gt_depths_kitti.py --data_path kitti_data --split eigen
python export_gt_depths_kitti.py --data_path kitti_data --split eigen_benchmark
```

Then, you can evaluate trained models with the following command:
```
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd /path/to/fumet
DEVICE=<CUDA index>

EVAL_SPLIT="eigen" # or "eigen_benchmark"
MODEL_NAME="monodepth2_R50"
CKPT_TIMESTAMP=""

CUDA_VISIBLE_DEVICES=$DEVICE python -OO evaluate_depth.py \
    --model_name $MODEL_NAME \
    --ckpt_timestamp $CKPT_TIMESTAMP \
    --height 192 \
    --width 640 \
    --epoch_for_eval 45 \
    --num_workers 8 \
    --batch_size 1 \
    --post_process \
    --num_layers 50 \
    --eval_split $EVAL_SPLIT
```

### Cityscapes
We use ground-truth depth files uploaded [HERE](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip), which is provided from [ManyDepth](https://github.com/nianticlabs/manydepth). Please download this and unzip into `splits/cityscapes`.

Then, you can evaluate a trained model with the following command:
```
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd /path/to/fumet
DEVICE=<CUDA index>

MODEL_NAME="monodepth2_R50"
CKPT_TIMESTAMP=""

CUDA_VISIBLE_DEVICES=$DEVICE python -OO evaluate_depth.py \
    --model_name $MODEL_NAME \
    --ckpt_timestamp $CKPT_TIMESTAMP \
    --height 192 \
    --width 512 \
    --fx 587.5 \
    --fy 587.5 \
    --cx 267.5 \
    --cy 130.0 \
    --epoch_for_eval 28 \
    --num_workers 8 \
    --batch_size 1 \
    --post_process \
    --num_layers 50 \
    --dataset "cityscapes" \
    --data_path /path/to/downloaded/cityscapes/data \
    --eval_split "cityscapes"
```

Be careful that you have to set `--data_path` as the directory where downloaded Cityscapes images are, not `/path/to/fumet/cityscapes_data`. 

## License
This project is primarily based on [Monodepth2](https://github.com/nianticlabs/monodepth2), which is licensed under the Monodepth2 License. Most of the code derives from Monodepth2, and its usage must comply with the terms of this license. Additionally, the project includes some code borrowed from MIT-licensed repositories ([Lite-Mono](https://github.com/noahzn/Lite-Mono), [VADepth](https://github.com/xjixzz/vadepth-net), and [HR-Depth](https://github.com/shawLyu/HR-Depth)). Copyright and license information are included in the header of each script. Please refer to `license/LICENSE_Monodepth2` for the full terms of the Monodepth2 License and `license/LICENSE_MIT` for details on the MIT-licensed code.

## Acknowledgement
This codes are based on [Monodepth2](https://github.com/nianticlabs/monodepth2). We borrowed the network codes from [Monodepth2](https://github.com/nianticlabs/monodepth2), [Lite-Mono](https://github.com/noahzn/Lite-Mono), [VADepth](https://github.com/xjixzz/vadepth-net), and [HR-Depth](https://github.com/shawLyu/HR-Depth). We appreciate the authors for their awesome open-source contributions.