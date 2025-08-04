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
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

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

def axis_uniformity_loss(scaling):
    """
    限制椭球三个轴的长度不要差距太大
    Args:
        scaling: Tensor, shape (N, 3), 每个椭球的三个轴长度
    Returns:
        loss: Tensor, 一个标量值，表示轴长度不一致性的损失
    """
    max_scaling = torch.max(scaling, dim=1).values  # 每个椭球的最大轴长度
    min_scaling = torch.min(scaling, dim=1).values  # 每个椭球的最小轴长度
    loss = ((max_scaling - min_scaling) ** 2).mean()  # 平方差并求均值
    return loss
def axis_proportion_loss(scaling):
    """
    限制椭球三个轴的比例接近 1:1:1
    Args:
        scaling: Tensor, shape (N, 3), 每个椭球的三个轴长度
    Returns:
        loss: Tensor, 一个标量值，表示轴比例不一致性的损失
    """
    max_scaling = torch.max(scaling, dim=1).values
    min_scaling = torch.min(scaling, dim=1).values
    ratio = max_scaling / (min_scaling + 1e-6)  # 避免除零
    loss = ((ratio - 1) ** 2).mean()
    return loss


def check_for_nan_inf(tensor, tensor_name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {tensor_name}")

def photometric_loss(img1, img2, flow):
    """
    光度一致性损失：衡量光流估计后的亮度一致性。
    Args:
        img1 (torch.Tensor): 当前帧图像, 形状为 [B, C, H, W]
        img2 (torch.Tensor): 下一帧图像, 形状为 [B, C, H, W]
        flow (torch.Tensor): 光流场，形状为 [B, 2, H, W]，表示从 img1 到 img2 的光流
    Returns:
        torch.Tensor: 光度损失
    """
    B, C, H, W = img1.shape

    # 如果 flow 的尺寸不匹配图像尺寸，进行插值
    if flow.shape[-2:] != (H, W):
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)

    # 生成像素坐标网格，确保 grid_x 对应 flow_x，grid_y 对应 flow_y
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")
    grid_x = grid_x.to(img1.device).float()
    grid_y = grid_y.to(img1.device).float()

    # 获取光流的 x 和 y 分量
    flow_x = flow[:, 0, :, :]
    flow_y = flow[:, 1, :, :]

    # 添加光流偏移
    sampling_x = grid_x[None, :, :] + flow_x
    sampling_y = grid_y[None, :, :] + flow_y

    # 归一化采样坐标到 [-1, 1]
    sampling_x = 2.0 * sampling_x / (W - 1) - 1.0
    sampling_y = 2.0 * sampling_y / (H - 1) - 1.0
    sampling_grid = torch.stack((sampling_x, sampling_y), dim=-1)

    # 对 img2 进行采样并计算损失
    warped_img2 = F.grid_sample(img2, sampling_grid, mode='bilinear', padding_mode='border', align_corners=False)
    loss = F.l1_loss(img1, warped_img2)
    return loss

def smoothness_loss(flow, img, weight=1.0, epsilon=1e-8):
    """
    光流平滑损失：鼓励光流场的平滑性。
    Args:
        flow (torch.Tensor): 估计的光流，形状为 [B, 2, H, W]
        img (torch.Tensor): 原始图像，形状为 [B, C, H, W]
        weight (float): 平滑损失的权重
    Returns:
        torch.Tensor: 平滑损失
    """
    # 计算光流的梯度
    flow_dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
    flow_dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])

    # 用图像梯度作为加权
    img_dx = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    img_dy = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

    # 加入 epsilon 防止数值过小导致问题
    img_dx = img_dx + epsilon
    img_dy = img_dy + epsilon

    # 平滑损失计算
    loss = torch.mean(flow_dx * torch.exp(-img_dx / 10)) + torch.mean(flow_dy * torch.exp(-img_dy / 10))
    return weight * loss

def optical_flow_loss(estimated_flow, target_flow, img1, img2, lambda_smooth=0.1):
    """
    计算总的光流损失，包括 L1 损失、平滑损失、光度一致性损失。
    Args:
        estimated_flow (torch.Tensor): 模型预测的光流 [B, 2, H, W]
        target_flow (torch.Tensor): 真实光流 [B, 2, H, W]
        img1 (torch.Tensor): 当前帧图像 [B, C, H, W]
        img2 (torch.Tensor): 下一帧图像 [B, C, H, W]
        lambda_smooth (float): 平滑损失的权重
    Returns:
        torch.Tensor: 总光流损失
    """
    # 确保 img1 和 img2 是 4D
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    # 确保 estimated_flow 和 target_flow 具有相同的形状
    if estimated_flow.dim() == 3:
        estimated_flow = estimated_flow.permute(2, 0, 1).unsqueeze(0)  # [H, W, 2] -> [1, 2, H, W]
    if target_flow.dim() == 3:
        target_flow = target_flow.permute(2, 0, 1).unsqueeze(0)  # [H, W, 2] -> [1, 2, H, W]

    # 在计算 l1_loss 和 smooth_loss 之前检查非法值
    check_for_nan_inf(estimated_flow, "estimated_flow")
    check_for_nan_inf(target_flow, "target_flow")

    # 计算 L1 光流损失
    l1_loss = F.l1_loss(estimated_flow, target_flow)

    # 计算光度一致性损失
    photo_loss = photometric_loss(img1, img2, estimated_flow)
    #print("photo_loss" + str(photo_loss))
    # 计算平滑损失
    smooth_loss = smoothness_loss(estimated_flow, img1, lambda_smooth)
    #print("smooth_loss"+str(smooth_loss))
    # 返回总损失
    total_loss = l1_loss + photo_loss + smooth_loss

    #print(f"l1_loss: {l1_loss}, photo_loss: {photo_loss}, smooth_loss: {smooth_loss}, total_loss: {total_loss}")
    return total_loss*0.1

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

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

