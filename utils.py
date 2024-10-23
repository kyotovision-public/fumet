# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Modifications made by Genki Kinoshita, 2024.
# These modifications include:
#   - Added compute_scaled_cam_heights() to scale camera heights with object heights
#   - Added compute_projected_obj_heights() to compute object silhouette heights
#   - Added signed_dist_to_horizon() to compute signed distance to horizon lines
#   - Added calc_obj_pix_height_over_dist_to_horizon_with_flat() to compute (object pixel heights) / (distance to horizon)
#   - Added horizon_to_2pts() to compute 2 points on horizon lines
#   - Added masks_to_pix_heights() to compute object pixel heights from masks
#   - Added generate_homo_pix_grid() to generate homogeneous pixel grids
#   - Added cam_pts2normal() to compute normal vectors with pseudo inverse
#   - Added cam_pts2cam_height_with_cross_prod() to compute normal vectors with cross product
#
# This modified version is also licensed under the terms of the Monodepth2
# licence, as outlined in the LICENSE file.

import torch


def compute_scaled_cam_heights(
    segms_flat: torch.Tensor,
    batch_n_inst: torch.Tensor,
    batch_road_normal_neg: torch.Tensor,
    batch_cam_pts: torch.Tensor,
    batch_unscaled_cam_height: torch.Tensor,
    obj_heights_flat: torch.Tensor,
    remove_nan: bool = True,
) -> torch.Tensor:
    """
    segms_flat: [n_inst, h, w], cuda
    batch_road_normal_neg: [bs, 3], cuda
    batch_cam_pts: [bs, h, w, 3], cuda
    batch_unscaled_cam_height: [bs,], cuda
    """
    device = segms_flat.device
    projected_obj_heights_flat = compute_projected_obj_heights(
        segms_flat, batch_n_inst, batch_road_normal_neg, batch_cam_pts, batch_unscaled_cam_height
    )
    split_scales = torch.split(obj_heights_flat / projected_obj_heights_flat, batch_n_inst.tolist())
    frame_scales = torch.tensor([chunk.quantile(0.5) if chunk.numel() > 0 else torch.nan for chunk in split_scales], device=device)
    frame_scaled_cam_heights = frame_scales * batch_unscaled_cam_height
    if remove_nan:
        return frame_scaled_cam_heights[~frame_scaled_cam_heights.isnan()]
    else:
        return frame_scaled_cam_heights


def compute_projected_obj_heights(
    segms_flat: torch.Tensor,
    batch_n_inst: torch.Tensor,
    batch_road_normal_neg: torch.Tensor,
    batch_cam_pts: torch.Tensor,
    batch_unscaled_cam_height: torch.Tensor,
) -> torch.Tensor:
    """
    segms_flat: [n_inst, h, w], cuda
    batch_road_normal_neg: [bs, 3], cuda
    batch_cam_pts: [bs, h, w, 3], cuda
    batch_unscaled_cam_height: [bs,], cuda
    """
    device = segms_flat.device
    bs = batch_cam_pts.shape[0]
    nx, ny, _ = batch_road_normal_neg.T
    batch_origin = torch.stack((torch.zeros(bs, device=device), batch_unscaled_cam_height / ny, torch.zeros(bs, device=device)), dim=1)  # [bs, 3]
    batch_root = torch.sqrt(ny**2 / (nx**2 + ny**2))
    batch_x_basis = torch.stack((batch_root, -ny / ny * batch_root, torch.zeros(bs, device=device)), dim=1)  # [bs, 3]
    batch_z_basis = torch.cross(batch_x_basis, batch_road_normal_neg)  # [bs, 3]
    projected_cam_pts = batch_cam_pts - batch_z_basis[:, None, None, :] * torch.einsum("bijk,bk->bij", batch_cam_pts, batch_z_basis).unsqueeze(-1)
    projected_cam_pts = projected_cam_pts - batch_origin[:, None, None, :]  # [bs, h, w, 3]
    batch_ys = torch.einsum("bijk,bk->bij", projected_cam_pts, batch_road_normal_neg)  # [bs, h, w]

    projected_heights_flat = torch.zeros((segms_flat.shape[0],), device=device)
    prev_n_inst = 0
    for batch_idx, n_inst in enumerate(batch_n_inst):
        for idx in range(prev_n_inst, prev_n_inst + n_inst):
            region = batch_ys[batch_idx, segms_flat[idx]]
            if (segm_max := region.max()) < 0:
                projected_heights_flat[idx] = -region.min()
            else:
                projected_heights_flat[idx] = segm_max - region.min()
        prev_n_inst += n_inst
    return projected_heights_flat


def signed_dist_to_horizon(
    homo_pix_grid: torch.Tensor,
    horizons: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        homo_pix_grid (torch.Tensor): [3, h, w]
        segms_flat (torch.Tensor): [n_inst, h, w]
        horizons (torch.Tensor): [bs, 3], detached

    Returns:
        torch.Tensor:
    """
    bs = horizons.shape[0]
    _, h, w = homo_pix_grid.shape
    device = horizons.device

    # A, Bはhorizon上の2点(A: (x,y)=(0,?), B: (x,y)=(1,?))
    batch_A, batch_B = horizon_to_2pts(horizons)
    batch_AB = batch_B - batch_A  # [bs, 2]
    norm_AB = torch.norm(batch_AB, dim=1)  # [bs,]
    batch_AB = torch.cat((batch_AB, torch.zeros((batch_AB.shape[0], 1), device=device)), dim=1)  # add zeros for torch.cross -> [bs, 3]
    homo_pix_grid = homo_pix_grid.to(device)

    # homo_pix_grid.view(3, -1)[:2]: [2, h*w], batch_A: [bs, 2]
    batch_AP = homo_pix_grid.view(3, -1)[:2, :].unsqueeze(0) - batch_A.unsqueeze(-1)  # [bs, 2, h*w]
    batch_AP = torch.cat((batch_AP, torch.zeros((bs, 1, h * w), device=device)), dim=1)  # [bs, 3, h*w]
    # batch_AB: [bs, 3],
    batch_cross = torch.linalg.cross(batch_AP, batch_AB.unsqueeze(-1), dim=1)[:, -1, :]  # [bs, h*w]

    # |AP x AB| / |AB| == distance from point P to line AB
    # if above the horizon, it should be positive
    return (batch_cross / norm_AB[:, None]).view(bs, h, w)


def calc_obj_pix_height_over_dist_to_horizon_with_flat(
    homo_pix_grid: torch.Tensor,
    segms_flat: torch.Tensor,
    horizons: torch.Tensor,
    batch_n_inst: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        homo_pix_grid (torch.Tensor): [3, h, w]
        segms_flat (torch.Tensor): [n_inst, h, w]
        horizons (torch.Tensor): [bs, 3], detached
        batch_n_insts (torch.Tensor): [bs,]

    Returns:
        torch.Tensor:
    """
    batch_signed_dist = signed_dist_to_horizon(homo_pix_grid, horizons)  # [bs,h,w]
    signed_dist_repeat = batch_signed_dist.repeat_interleave(batch_n_inst, dim=0)  # [n_inst, h, w]
    signed_dist_repeat[~segms_flat] = -100000
    signed_top_to_horizon = signed_dist_repeat.amax((1, 2))
    signed_dist_repeat[~segms_flat] = 100000
    signed_bottom_to_horizon = signed_dist_repeat.amin((1, 2))
    abs_bottom_to_horizon = signed_bottom_to_horizon.abs()
    abs_bottom_to_top = abs_bottom_to_horizon + signed_top_to_horizon + 1  # obj_pix_heights
    ratios = abs_bottom_to_top / abs_bottom_to_horizon
    ratios[signed_bottom_to_horizon > 0] = 10000  # if all pixels are above the horizon, remove the object
    return ratios


def horizon_to_2pts(horizons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bs = horizons.shape[0]
    device = horizons.device
    zeros = torch.zeros((bs,), device=device)
    ones = torch.ones((bs,), device=device)
    return torch.stack((zeros, -horizons[:, 2] / horizons[:, 1]), dim=1), torch.stack(
        (ones, -(horizons[:, 0] + horizons[:, 2]) / horizons[:, 1]), dim=1
    )


def masks_to_pix_heights(masks: torch.Tensor) -> torch.Tensor:
    """
    masks: Size(n_inst, img_h, img_w)
    """
    pix_heights = torch.zeros((masks.shape[0],), device=masks.device, dtype=torch.int16)
    for idx, mask in enumerate(masks):
        y, _ = torch.where(mask != 0)
        pix_heights[idx] = torch.max(y) - torch.min(y)
    return pix_heights


def generate_homo_pix_grid(h: int, w: int) -> torch.Tensor:
    x_pix, y_pix = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    return torch.stack((x_pix, y_pix, torch.ones((h, w))))  # [3, h, w]


def cam_pts2normal(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
    A = cam_pts[road_mask == 1]
    pinvs = torch.pinverse(A.detach())
    ones = torch.ones((A.shape[0], 1), device=A.device).type_as(A)
    normal = pinvs @ ones
    normal = normal / torch.linalg.norm(normal)
    return normal


def cam_pts2cam_height_with_cross_prod(batch_cam_pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        batch_cam_pts (torch.Tensor):

    Returns:
        batch_cam_height, batch_normal
    """
    batch_normal = cam_pts2normal_with_cross_prod(batch_cam_pts)
    batch_cam_height = torch.einsum("ijkl,ijkl->ijk", batch_cam_pts, -batch_normal)  # [bs, h, w]
    return batch_cam_height, batch_normal  # batch_normal is negative


def cam_pts2normal_with_cross_prod(batch_cam_pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # batch_cam_pts: [bs, h, w, 3]
    v0 = batch_cam_pts.roll(-1, dims=2) - batch_cam_pts
    v1 = batch_cam_pts.roll((1, -1), dims=(1, 2)) - batch_cam_pts
    v2 = batch_cam_pts.roll(1, dims=1) - batch_cam_pts
    v3 = batch_cam_pts.roll((1, 1), dims=(1, 2)) - batch_cam_pts
    v4 = batch_cam_pts.roll(1, dims=2) - batch_cam_pts
    v5 = batch_cam_pts.roll((-1, 1), dims=(1, 2)) - batch_cam_pts
    v6 = batch_cam_pts.roll(-1, dims=1) - batch_cam_pts
    v7 = batch_cam_pts.roll((-1, -1), dims=(1, 2)) - batch_cam_pts

    normal_sum = torch.zeros_like(batch_cam_pts, device=batch_cam_pts.device)
    vecs = (v0, v1, v1, v2, v3, v4, v5, v6, v7)
    for i in range(8):
        if i + 2 < 8:
            normal_sum = normal_sum + torch.cross(vecs[i], vecs[i + 2], dim=-1)
        else:
            normal_sum = normal_sum + torch.cross(vecs[i], vecs[i + 2 - 8], dim=-1)
    batch_normal = normal_sum / torch.linalg.norm(normal_sum, dim=-1).unsqueeze(-1)  # [bs, h, w, 3]
    return batch_normal  # batch_normal is negative


def readlines(filename):
    """Read all the lines in a text file and return as a list"""
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]"""
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)
