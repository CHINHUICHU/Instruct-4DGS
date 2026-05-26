#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math
import lpips
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def depth_warp_consistency_loss(depths, cameras):
    """
    Self-supervised depth consistency loss adapted from StableGS (arXiv 2503.18458).

    For each adjacent camera pair in the batch, warps the source depth map into the
    target camera frame using known extrinsics, then computes an L1 loss against the
    target's rendered depth.  This penalises floaters because they produce depth values
    that are geometrically inconsistent across views.

    depths  : list of depth tensors, each (1,H,W) or (H,W), from the rasterizer
    cameras : list of Camera objects matching depths

    Recommended weight: lambda_depth_consistency = 0.05
    """
    n = len(depths)
    if n < 2:
        return torch.tensor(0.0, device=depths[0].device)

    total = torch.tensor(0.0, device=depths[0].device)
    count = 0
    # Use adjacent pairs to keep cost O(batch_size)
    for i in range(n - 1):
        total = total + _depth_pair_loss(depths[i], cameras[i], depths[i + 1], cameras[i + 1])
        total = total + _depth_pair_loss(depths[i + 1], cameras[i + 1], depths[i], cameras[i])
        count += 2
    return total / count


def _depth_pair_loss(depth_src, cam_src, depth_tgt, cam_tgt):
    """
    Warp depth_src into cam_tgt's frame using camera poses, then compute L1 against depth_tgt.

    world_view_transform is stored as W2C^T (transposed world-to-camera) so that
    row-vector multiplication gives:  p_cam = p_world @ world_view_transform
    Unprojection:                     p_world = p_cam @ inv(world_view_transform)
    Re-projection to target:          p_cam_tgt = p_world @ world_view_transform_tgt
    """
    device = depth_src.device

    d = depth_src[0] if depth_src.dim() == 3 else depth_src   # (H, W)
    H, W = d.shape

    tanfovx_s = math.tan(cam_src.FoVx * 0.5)
    tanfovy_s = math.tan(cam_src.FoVy * 0.5)

    us = torch.linspace(-1.0, 1.0, W, device=device)
    vs = torch.linspace(-1.0, 1.0, H, device=device)
    vv, uu = torch.meshgrid(vs, us, indexing='ij')  # (H, W)

    # Camera-src 3-D points (homogeneous row vectors)
    x = uu * tanfovx_s * d
    y = vv * tanfovy_s * d
    ones = torch.ones_like(d)
    pts = torch.stack([x, y, d, ones], dim=-1).reshape(-1, 4)   # (H*W, 4)

    # src camera → world → tgt camera
    vm_src = cam_src.world_view_transform.to(device)             # (4,4)  W2C_src^T
    vm_tgt = cam_tgt.world_view_transform.to(device)             # (4,4)  W2C_tgt^T
    pts_world = pts @ torch.inverse(vm_src)                      # (H*W, 4)
    pts_tgt   = pts_world @ vm_tgt                               # (H*W, 4)

    z_t = pts_tgt[:, 2]
    tanfovx_t = math.tan(cam_tgt.FoVx * 0.5)
    tanfovy_t = math.tan(cam_tgt.FoVy * 0.5)

    u_t = pts_tgt[:, 0] / (tanfovx_t * z_t.clamp(min=1e-4))
    v_t = pts_tgt[:, 1] / (tanfovy_t * z_t.clamp(min=1e-4))

    valid = (z_t > 0.01) & (u_t.abs() < 0.99) & (v_t.abs() < 0.99)
    valid = valid.reshape(H, W)

    if valid.sum() < 100:
        return torch.tensor(0.0, device=device)

    d_tgt = depth_tgt if depth_tgt.dim() == 3 else depth_tgt.unsqueeze(0)  # (1,H,W)
    grid = torch.stack([u_t, v_t], dim=-1).reshape(1, H, W, 2)
    d_tgt_sampled = F.grid_sample(
        d_tgt.unsqueeze(0),       # (1,1,H,W)
        grid,                      # (1,H,W,2)
        mode='bilinear',
        align_corners=True,
        padding_mode='border',
    ).squeeze()                    # (H, W)

    return (z_t.reshape(H, W)[valid] - d_tgt_sampled[valid]).abs().mean()


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

