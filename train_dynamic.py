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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, optical_flow_loss, axis_uniformity_loss, axis_proportion_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, Register
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.spring_utils.transform import rot6d_to_rotmat, euler_to_quat, quat_to_rot6d, quat_to_rotmat
from utils.spring_utils.etqdm import etqdm
from utils.spring_utils.misc import bar_perfixes, format_args_cfg
from utils.spring_utils.logger import logger
import imageio
import math
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T
import cv2  # 引入 OpenCV 库
from scene.Register import Register

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def calculate_intrinsics(viewpoint_cam):
    """
    根据视场角和图像尺寸计算相机内参矩阵。

    Args:
        viewpoint_cam: 包含相机属性的对象

    Returns:
        K (torch.Tensor): 相机内参矩阵，形状为 (3, 3)
    """
    # 获取图像宽度和高度
    image_width = viewpoint_cam.image_width
    image_height = viewpoint_cam.image_height

    # 获取水平和垂直视场角 (Field of View)
    FoVx = viewpoint_cam.FoVx  # 水平视场角
    FoVy = viewpoint_cam.FoVy  # 垂直视场角

    # 计算焦距 fx 和 fy
    f_x = image_width / (2.0 * math.tan(FoVx / 2.0))
    f_y = image_height / (2.0 * math.tan(FoVy / 2.0))

    # 计算光心 (cx, cy)
    c_x = image_width / 2.0
    c_y = image_height / 2.0

    # 构建内参矩阵 K
    K = torch.tensor([[f_x, 0, c_x],
                      [0, f_y, c_y],
                      [0, 0, 1]], dtype=torch.float32).cuda()

    return K

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,0.002)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    flow_loss_weight = opt.lambda_flow if hasattr(opt, 'lambda_flow') else 0.1  # 设置光流损失的权重
    flow_frame_interval = 1  # 每n帧计算一次光流损失
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    prev_render_pkg = None  # 初始化 prev_render_pkg 为空

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Pick the next Camera in sequence
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()  # 复制相机列表
        viewpoint_cam = viewpoint_stack.pop(0)  # 每次弹出第一个相机，按顺序处理

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 计算相机内参矩阵 K
        K = calculate_intrinsics(viewpoint_cam)

        # 获取图像的大小 (H, W)
        image_size = (viewpoint_cam.image_height, viewpoint_cam.image_width)
        # print(image_size)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()  
        Ll1 = l1_loss(image, gt_image)
        # new
        Lssim = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # 保存渲染图像，debug
        if iteration > 29400 :
            save_path = os.path.join("./render_image/test02_baishi", f"render_{iteration:06d}.png")
            vutils.save_image(image, save_path)
            # print(f"Rendered image saved at: {save_path}")
        # new
        image_num = 120
        start_num = 100*image_num+1
        
        if 29000 >= iteration >= start_num and ((iteration-start_num) % image_num) % flow_frame_interval == 0:
            # print(iteration)
            # 第一轮先获取渲染包不计算
            if ((iteration-start_num) % image_num) == 0:
                prev_render_pkg = render_pkg
            # 获取前一帧和当前帧的渲染包
            if prev_render_pkg is not None and ((iteration-start_num) % image_num) != 0:
                estimated_flow = estimate_flow_from_render(prev_render_pkg, render_pkg, K, image_size)
                #print(estimated_flow.shape)
                # 加载真实光流
                # 读取 target_flow
                flow_filename = f"flow_{((iteration-start_num) % image_num) % flow_frame_interval:04d}.npy"
                tem_path = os.path.join(os.path.dirname(dataset.model_path), "flow/1")
                # flow_path = os.path.join("./data/Datasets/导弹/flow/1", flow_filename)
                flow_path = os.path.join(tem_path, flow_filename)
                target_flow = torch.from_numpy(np.load(flow_path)).cuda()

                # 打印 estimated_flow 和 target_flow 的形状
                # print(f"estimated_flow shape: {estimated_flow.shape}")
                # print(f"target_flow shape: {target_flow.shape}")

                # 使用裁剪和填充的方法将 target_flow 调整为与 estimated_flow 大小一致
                target_flow_resized = resize_flow_to_match(target_flow, estimated_flow.shape)

                # 打印调整后的 target_flow 形状
                # print(f"Resized target_flow shape: {target_flow_resized.shape}")

                # 当前帧图像
                img1 = viewpoint_cam.original_image.cuda()

                # 获取下一帧图像，若失败则跳过当前迭代
                if hasattr(viewpoint_cam, 'next_image'):
                    img2 = viewpoint_cam.next_image.cuda()
                else:
                    next_viewpoint_cam = get_next_frame(viewpoint_stack)
                    if next_viewpoint_cam is None:
                        continue  # 跳过当前迭代
                    img2 = next_viewpoint_cam.original_image.cuda()
                    viewpoint_stack.append(next_viewpoint_cam)  # 将获取的相机放回栈底

                # 计算光流损失
                flow_loss = optical_flow_loss(estimated_flow, target_flow_resized, img1.unsqueeze(0), img2.unsqueeze(0),lambda_smooth=opt.lambda_flow_smooth)
                loss += flow_loss
                # print(f"Iteration {iteration}: Loss = {loss.item()}")
            # 保存当前帧 下一次计算使用
            prev_render_pkg = render_pkg

        # scaling = gaussians.get_scaling  # 获取所有椭球的轴长度
        # axis_loss1 = axis_uniformity_loss(scaling)
        # axis_loss2 = axis_proportion_loss(scaling)
        # loss += 0.001*axis_loss1 + 0.001* axis_loss2
        loss.backward()

        # # 添加梯度裁剪以防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)

        iter_end.record()

        # scene.save(iteration)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print("SSIM:"+str(Lssim))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 5 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def resize_flow_to_match(flow, target_shape):
    # """
    # 调整光流的大小以匹配目标大小。
    # 如果需要，将裁剪或填充光流以使其与目标形状一致。

    # Args:
    #     flow (torch.Tensor): 需要调整的光流，形状为 [H, W, C]
    #     target_shape (tuple): 目标形状，格式为 (target_H, target_W, C)

    # Returns:
    #     torch.Tensor: 调整后的光流
    # """
    target_H, target_W, _ = target_shape
    H, W, C = flow.shape

    # 如果高度或宽度过大，进行裁剪
    if H > target_H:
        flow = flow[:target_H, :, :]
    if W > target_W:
        flow = flow[:, :target_W, :]

    # 如果高度或宽度过小，进行填充
    if H < target_H:
        padding_H = target_H - H
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 0, 0, padding_H), mode='constant', value=0)
    if W < target_W:
        padding_W = target_W - W
        flow = torch.nn.functional.pad(flow, (0, 0, 0, padding_W), mode='constant', value=0)

    return flow
def get_next_frame(viewpoint_stack):
    if viewpoint_stack:
        return viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    else:
        # print("No more frames in the viewpoint stack, skipping this iteration.")
        return None  # 返回 None 以表示未成功获取
# 可视化光流并保存
def save_image_tensor(img_tensor, save_path):
    vutils.save_image(img_tensor, save_path)
    print(f"Image saved at: {save_path}")

def visualize_flow(flow, save_path):
    flow = flow.detach().cpu().numpy()
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    angle = np.arctan2(flow_y, flow_x)
    normalized_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    hsv_image = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv_image[..., 0] = (angle + np.pi) / (2 * np.pi)
    hsv_image[..., 1] = 1
    hsv_image[..., 2] = normalized_magnitude
    rgb_image = plt.cm.hsv(hsv_image)
    plt.imsave(save_path, rgb_image)
    print(f"Estimated flow saved at: {save_path}")

def estimate_flow_from_render(prev_render_pkg, curr_render_pkg, K, image_size):
    # 获取渲染图像并转换为 NumPy 格式
    prev_image = prev_render_pkg["render"].squeeze().detach().cpu().numpy()
    curr_image = curr_render_pkg["render"].squeeze().detach().cpu().numpy()

    # # 保存前后帧渲染图像
    # prev_image_save_path = os.path.join(".\\render_image\\test03", f"prev_render_{iteration:06d}.png")
    # curr_image_save_path = os.path.join(".\\render_image\\test03", f"curr_render_{iteration:06d}.png")
    # save_image_tensor(torch.from_numpy(prev_image), prev_image_save_path)
    # save_image_tensor(torch.from_numpy(curr_image), curr_image_save_path)

    # 调整图像格式为 [H, W, C]
    if len(prev_image.shape) == 3 and prev_image.shape[0] == 3:  # 如果是 [C, H, W]
        prev_image = prev_image.transpose(1, 2, 0)  # 转换为 [H, W, C]
    if len(curr_image.shape) == 3 and curr_image.shape[0] == 3:  # 如果是 [C, H, W]
        curr_image = curr_image.transpose(1, 2, 0)  # 转换为 [H, W, C]

    # 将图像转换为灰度图像
    prev_gray = cv2.cvtColor((prev_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor((curr_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # 使用 Lucas-Kanade 光流法计算光流
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 选择特征点
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # 计算光流
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

    # 计算光流矢量（仅保留状态为1的点）
    good_new = curr_points[status == 1]
    good_old = prev_points[status == 1]

    # 初始化光流为二维张量 [H, W, 2]，表示 x 和 y 的位移
    flow = np.zeros((prev_image.shape[0], prev_image.shape[1], 2), dtype=np.float32)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        # 限制 a 和 b 坐标在图像范围内，防止越界错误
        a = max(0, min(flow.shape[1] - 1, a))
        b = max(0, min(flow.shape[0] - 1, b))

        flow[int(b), int(a), 0] = a - c  # x 方向位移
        flow[int(b), int(a), 1] = b - d  # y 方向位移

    # 将 flow 转换为 torch.Tensor 并调整大小以匹配目标图像大小
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float().cuda()
    flow_resized = F.interpolate(flow_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

    # 可视化并保存光流
    # flow_save_path = os.path.join("E:\\RAFT\\RAFT\\estimated_flow", f"estimated_flow_{iteration:06d}.png")
    # visualize_flow(flow_resized, flow_save_path)

    return flow_resized

def estimate_flow_from_render_CF(prev_images, curr_render_pkg, K, image_size):
    # 获取渲染图像并转换为 NumPy 格式
    # prev_image = prev_render_pkg["render"].squeeze().detach().cpu().numpy()
    prev_image = prev_images.squeeze().detach().cpu().numpy()
    curr_image = curr_render_pkg["render"].squeeze().detach().cpu().numpy()

    # # 保存前后帧渲染图像
    # prev_image_save_path = os.path.join(".\\render_image\\test03", f"prev_render_{iteration:06d}.png")
    # curr_image_save_path = os.path.join(".\\render_image\\test03", f"curr_render_{iteration:06d}.png")
    # save_image_tensor(torch.from_numpy(prev_image), prev_image_save_path)
    # save_image_tensor(torch.from_numpy(curr_image), curr_image_save_path)

    # 调整图像格式为 [H, W, C]
    if len(prev_image.shape) == 3 and prev_image.shape[0] == 3:  # 如果是 [C, H, W]
        prev_image = prev_image.transpose(1, 2, 0)  # 转换为 [H, W, C]
    if len(curr_image.shape) == 3 and curr_image.shape[0] == 3:  # 如果是 [C, H, W]
        curr_image = curr_image.transpose(1, 2, 0)  # 转换为 [H, W, C]

    # 将图像转换为灰度图像
    prev_gray = cv2.cvtColor((prev_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor((curr_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # 使用 Lucas-Kanade 光流法计算光流
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 选择特征点
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # 计算光流
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

    # 计算光流矢量（仅保留状态为1的点）
    good_new = curr_points[status == 1]
    good_old = prev_points[status == 1]

    # 初始化光流为二维张量 [H, W, 2]，表示 x 和 y 的位移
    flow = np.zeros((prev_image.shape[0], prev_image.shape[1], 2), dtype=np.float32)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        # 限制 a 和 b 坐标在图像范围内，防止越界错误
        a = max(0, min(flow.shape[1] - 1, a))
        b = max(0, min(flow.shape[0] - 1, b))

        flow[int(b), int(a), 0] = a - c  # x 方向位移
        flow[int(b), int(a), 1] = b - d  # y 方向位移

    # 将 flow 转换为 torch.Tensor 并调整大小以匹配目标图像大小
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float().cuda()
    flow_resized = F.interpolate(flow_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

    # 可视化并保存光流
    # flow_save_path = os.path.join("E:\\RAFT\\RAFT\\estimated_flow", f"estimated_flow_{iteration:06d}.png")
    # visualize_flow(flow_resized, flow_save_path)

    return flow_resized

def project_to_image_plane(points_3d, K):
    points_3d = points_3d[:, :3]
    points_2d_homogeneous = points_3d @ K.T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
    N = points_2d.shape[0]
    H, W = find_factors(N)
    points_2d = points_2d.view(H, W, 2)
    return points_2d

def find_factors(n):
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def set_gaussians(gaussians: GaussianModel, load_path=None):
    
    logger.info(f"Load gaussians from {load_path}...")
    gaussians.load_ply(load_path)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

def my_register_gaus(dataset,opt,pipe,debug_from,iterations,cnt,const_scale):
    from simple_knn._C import distCUDA2
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    gaussians = GaussianModel(0,const_scale)
    print(dataset.model_path)
    if cnt == 1:
        point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_30000/point_cloud.ply")
        gaussians.load_ply(point_cloud_path)
        # gaussians.load_ply('/home/c206/zjr/3dgs/data/lvdaodan/output/point_cloud/iteration_30000/point_cloud.ply')
    else :
        point_cloud_path = os.path.join(dataset.model_path, "regist/Regist/regist_gaussians_{}/point_cloud.ply".format(cnt-1))
        gaussians.load_ply(point_cloud_path)
        # file_path_template = '/home/c206/zjr/3dgs/data/lvdaodan/regist/Regist/regist_gaussians_{}/point_cloud.ply'
        # file_path = file_path_template.format(cnt-1)
        # gaussians.load_ply(file_path)

    scene = Scene(dataset, gaussians,regist=True)

    if cnt == 1:
        point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_30000/point_cloud.ply")
        gaussians.load_ply(point_cloud_path)
        # scene.gaussians.load_ply('/home/c206/zjr/3dgs/data/lvdaodan/output/point_cloud/iteration_30000/point_cloud.ply')
    else :
        point_cloud_path = os.path.join(dataset.model_path, "regist/Regist/regist_gaussians_{}/point_cloud.ply".format(cnt-1))
        gaussians.load_ply(point_cloud_path)
        # file_path_template = '/home/c206/zjr/3dgs/data/lvdaodan/regist/Regist/regist_gaussians_{}/point_cloud.ply'
        # file_path = file_path_template.format(cnt-1)
        # scene.gaussians.load_ply(file_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if cnt == 1:
        # #nike
        INIT_R = [0-25, 0-0, 0+0]
        INIT_T = [0.5 * 0.2005, 0.5 * 0.2593, 0.5 * -0.9447]
        INIT_S = [1]
        # 导弹 zhun
        # INIT_R = [0-45, 0+180, 0-45]
        # INIT_T = [0, 0, 0.83]
        # INIT_S = [0.88]
        # boom
        # INIT_R = [0-30, 0+180, 0]
        # INIT_T = [0-0.8, 0+1.5, 0+1.8]
        # INIT_S = [1]
        # #baishi zhun
        # INIT_R = [0, 0-0, 0+40]
        # INIT_T = [0, 0, 0]
        # INIT_S = [1]
        # #nezha zhun
        # INIT_R = [0, 0-0, 0+0]
        # INIT_T = [0+0.5, 0-0.5, 0+0.9]
        # INIT_S = [1]
        # boluo zhun
        # INIT_R = [0+30, 0-10, 0]
        # INIT_T = [0, 0-1.0, 0]
        # INIT_S = [1.5]
        R_LR = 0.0008
        T_LR = 8e-5
        S_LR = 3e-5
    else :
        INIT_R = [0, 0, 0]
        INIT_T = [0, 0, 0]
        INIT_S = [1]
        R_LR = 0.0003
        T_LR = 3e-5
        S_LR = 1e-6
    regist = Register(INIT_R,INIT_T,INIT_S).cuda()

    regist.training_setup(R_LR,T_LR,S_LR)

    train_bar = etqdm(range(1, iterations + 1))


    xyz_i = gaussians.get_xyz.detach().clone()
    # const_scale_orgin = gaussians.const_scale
    scaling_orgin = gaussians._scaling


    for iteration in train_bar:
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack[0]

        gaussians._xyz = regist(xyz_i)
        s = float(regist.s.item())
        gaussians.const_scale = s * const_scale
        gaussians._scaling = s* scaling_orgin

        # dist2 = torch.clamp_min(distCUDA2(xyz_i * regist.s), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2[..., None]))
        # scales = scales.repeat(1,3)
        # gaussians._scaling = scales

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        LAMBDA_DSSIM = 0.0
        loss = (1.0 - LAMBDA_DSSIM) * Ll1 + LAMBDA_DSSIM * (1.0 - ssim(image, gt_image))
        loss.backward()

        regist.optimizer.step()
        regist.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                train_bar.set_description(f"{bar_perfixes['regist']} loss: {ema_loss_for_log:.{7}f}")

            # visualization
            if iteration <10 or iteration % 50 == 0:
                img_list = [
                    image.detach().cpu().permute(1, 2, 0).numpy() * 255,
                    gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255
                ]
                img_list = np.hstack(img_list).astype(np.uint8)
                img_write_dir = os.path.join(dataset.model_path, "regist/Regist/")
                regist_name = f'viz_regist_{cnt}'
                # img_write_dir ='/home/c206/zjr/3dgs/data/regist/Regist/'
                img_save_path = os.path.join(img_write_dir, regist_name)
                os.makedirs(img_save_path, exist_ok=True)
                imageio.imwrite(os.path.join(img_save_path, f"{iteration}.png"), img_list)
    with torch.no_grad():
        print("")
        logger.warning("Saving Registed Gaussians")
        

        regist_name = f'regist_gaussians_{cnt}'
        save_root = os.path.join(dataset.model_path, "regist/Regist/")
        # save_root = '/home/c206/zjr/3dgs/data/lvdaodan/regist/Regist/'
        save_path = os.path.join(save_root, regist_name)
        os.makedirs(save_root, exist_ok=True)

        # scene.save(iteration, save_path=save_path)
        gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
    return gaussians.const_scale

def my_CF_gaus(dataset,opt,pipe,debug_from,iterations,cnt,const_scale,crit_vgg):
    from simple_knn._C import distCUDA2
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    gaussians = GaussianModel(0,const_scale)
    image_path = os.path.join(os.path.dirname(dataset.model_path), "images")
    image_cnt = get_image_num(image_path)
    print(dataset.model_path)
    print(image_cnt)
    prve_gt_image = None
    prev_render_pkg = None

    for i in range(1,image_cnt+1):
        progress_str = "{}/{}".format(i, image_cnt)
        print(progress_str)
        if i == 1:
            file_path = os.path.join(dataset.model_path, "regist/Regist/regist_gaussians_{}/point_cloud.ply".format(cnt))
        else :
            file_path = os.path.join(dataset.model_path, "regist/CF/gaussians/regist_gaussians_{}/point_cloud.ply".format(i-1))
        gaussians.load_ply(file_path)
        scene = Scene(dataset, gaussians,regist=True)
        scene.gaussians.load_ply(file_path)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick the next Camera in sequence
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()  # 复制相机列表
        if i == 100:
            viewpoint_stack.pop(0)
            viewpoint_cam = viewpoint_stack.pop(0)  # 每次弹出第一个相机，按顺序处理
        else:
            viewpoint_cam = viewpoint_stack.pop(0)

        gravity = torch.tensor(viewpoint_cam.R[:, 1], dtype=torch.float32, device="cuda")
        INIT_R = [0, 0, 0]
        INIT_T = [0, 0, 0] 
        INIT_S = [1]
        regist = Register(INIT_R,INIT_T,INIT_S).cuda()

        ### base learning rate
        base_R_LR = 0.007*5
        base_T_LR = 5e-4*3
        base_iterations = 1200
        S_LR = 0

        if i >= 200:
            min_dis = 0.02 #  the smallest dis is about 0.02 
            gravity_project = torch.dot(gaussians.dis[-1].squeeze(), gravity) 
            factor = (gravity_project / min_dis).item()
            R_LR = (base_R_LR * abs(factor))
            T_LR = (base_T_LR * abs(factor)/2)
            iterations = max(int(base_iterations + 100 * abs(factor)), 1200)
            gaussians.update_lrr(R_LR)
            gaussians.update_lrt(T_LR)
            gaussians.update_iter(iterations)
        else:
            R_LR = base_R_LR * 1
            T_LR = base_T_LR * 1
            iterations = 1200

        regist.training_setup(R_LR,T_LR,S_LR)
        train_bar = etqdm(range(1, iterations + 1)
                          )
        xyz_i = gaussians.get_xyz.detach().clone()
        scaling_orgin = gaussians._scaling

        # 配准前的质心
        xyz_0 =  gaussians.get_xyz.detach().clone()
        C0 = torch.mean(xyz_0, dim=0, keepdim = True)
        print(C0)
        gaussians.update_cen(C0)

        for iteration in train_bar:
            # Regist
            regist.update_learning_rate(iteration)

            gaussians._xyz = regist(xyz_i)
            s = float(regist.s.item())
            gaussians.const_scale = s * const_scale
            gaussians._scaling = s* scaling_orgin

            # Render
            pipe.debug = True
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image = render_pkg["render"]

            # Loss
            # L1 loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            LAMBDA_DSSIM = 0.0
            loss = (1.0 - LAMBDA_DSSIM) * Ll1 + LAMBDA_DSSIM * (1.0 - ssim(image, gt_image))

            # flow loss
            # start_iteration = 1
            # end_iteration = 300
            # flow_frame_interval =1

            # if iteration >= start_iteration and iteration <= end_iteration:

            #     if i == 1 :
            #         prve_gt_image = gt_image
            #         prev_render_pkg = render_pkg

            #     else : 
            #         if prev_render_pkg is not None and prve_gt_image is not None:
            #             K = calculate_intrinsics(viewpoint_cam)
            #             # 获取图像的大小 (H, W)
            #             image_size = (viewpoint_cam.image_height, viewpoint_cam.image_width)
            #             estimated_flow = estimate_flow_from_render_CF(prve_gt_image, render_pkg, K, image_size)
            #             # 加载真实光流
            #             flow_filename = f"flow_{(i-1) :04d}.npy"
            #             tem_path = os.path.join(os.path.dirname(dataset.model_path), "flow_orign/1")
            #             flow_path = os.path.join(tem_path, flow_filename)
            #             target_flow = torch.from_numpy(np.load(flow_path)).cuda()
            #             target_flow_resized = resize_flow_to_match(target_flow, estimated_flow.shape)
            #             # 当前帧图像
            #             img1 = render_pkg["render"]
            #             # 获取qian一帧图像
            #             img2 = prve_gt_image
            #             # 计算光流损失
            #             flow_loss = optical_flow_loss(estimated_flow, target_flow_resized, img1.unsqueeze(0), img2.unsqueeze(0),lambda_smooth=opt.lambda_flow_smooth)
            #             loss += flow_loss*0.1 + loss*0.9

            #         prev_render_pkg = render_pkg
            #         prve_gt_image = gt_image

            # VGG/TV loss
            # if end_iteration <= iteration <= 1000 :
            #     target_act = crit_vgg.get_features(gt_image)
            #     loss += 0.001* crit_vgg(image, target_act, target_is_features=True)
            #     loss += 0.001 * crit_tv(image) 

            #Acc loss
            
            if i >= 4 and iteration >= 10 :
                Abefore = gaussians.acc[-1].requires_grad_(True)
                Abefore_gravity = torch.dot(Abefore.squeeze(), gravity).requires_grad_(True) # (A·g) * g 计算加速度在重力方向上的分量，先点积得到常数，再乘到重力向量上。
                Abefore_g = (Abefore_gravity * gravity).requires_grad_(True)

                xyz_t =  gaussians.get_xyz.detach().clone()  # 当前迭代下的质心
                Cnow = torch.mean(xyz_t, dim=0, keepdim = True).requires_grad_(True)
                Cbefore = gaussians.cen[-1]
                St = (Cnow - Cbefore).requires_grad_(True) # 当前迭代下的位移
                
                Anow = ((St - gaussians.dis[-1]) * (gaussians.dis[-1] - gaussians.dis[-2])).requires_grad_(True) # 当前迭代下的加速度
                Anow_gravity = torch.dot(Anow.squeeze(), gravity).requires_grad_(True) #点积
                Anow_g = (Anow_gravity * gravity).requires_grad_(True) #投影
                Anow_others = (Anow - Anow_g).requires_grad_(True)  # 加速度向量减去重力分量，得到垂直重力分量

                gpart = torch.norm(Anow_g - Abefore_g).requires_grad_(True)
                otherpart = torch.norm(Anow_others).requires_grad_(True)
                loss += 0.05 * (gpart + otherpart)

            loss.backward()
            regist.optimizer.step()
            regist.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    train_bar.set_description(f"{bar_perfixes['regist']} loss: {ema_loss_for_log:.{7}f}")

                # visualization
                if iteration < 2 or iteration % 250 == 0 :
                    img_list = [
                        image.detach().cpu().permute(1, 2, 0).numpy() * 255,
                        gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255
                    ]
                    img_list = np.hstack(img_list).astype(np.uint8)
                    regist_name = f'viz_regist_{i}'
                    img_write_dir = os.path.join(dataset.model_path, "regist/CF/images/")
                    img_save_path = os.path.join(img_write_dir, regist_name)
                    os.makedirs(img_save_path, exist_ok=True)
                    imageio.imwrite(os.path.join(img_save_path, f"{iteration}.png"), img_list)
        
        xyz_1 =  gaussians.get_xyz.detach().clone() # 配准后的质心
        C1 = torch.mean(xyz_1, dim=0, keepdim = True)

        S = C1 - gaussians.cen[-1] # 计算质心位移S
        gaussians.update_dis(S)
    
        # 计算加速度（第三帧可以获得第一个加速度值，第四帧可以开始用加速度损失值优化）
        if i >= 3 :
            a = (gaussians.dis[-1] - gaussians.dis[-2]) - (gaussians.dis[-2] - gaussians.dis[-3])
            print(a)
            gaussians.update_acc(a)

        with torch.no_grad():
            print("")
            logger.warning("Saving Registed Gaussians")

            regist_name = f'regist_gaussians_{i}'
            save_root = os.path.join(dataset.model_path, "regist/CF/gaussians/")
            save_path = os.path.join(save_root, regist_name)
            os.makedirs(save_root, exist_ok=True)
            gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
        
    
def get_image_num(path):
    jpg_count = 0
# 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(path):
        for file in files:
            # 检查文件扩展名是否为 .jpg 或 .JPG
            if file.lower().endswith('.jpg'):
                jpg_count += 1
    return jpg_count
    
def test(dataset,opt,pipe,debug_from,save_root,iterations,const_scale):
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    gaussians = GaussianModel(0,const_scale)
    # set_gaussians(gaussians,PLY_ROOT)

    file_path_template = '/home/c206/zjr/3dgs/data/regist/regist_gaussians_{}/point_cloud.ply'
    file_path = file_path_template.format(5)
    gaussians.load_ply(file_path)

    scene = Scene(dataset, gaussians,regist=True)

    file_path_template = '/home/c206/zjr/3dgs/data/regist/regist_gaussians_{}/point_cloud.ply'
    file_path = file_path_template.format(5)
    scene.gaussians.load_ply(file_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        # Pick a random Camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack[0]

    pipe.debug = True
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    image = render_pkg["render"]


    img_list = [
        image.detach().cpu().permute(1, 2, 0).numpy() * 255,
    ]
    img_list = np.hstack(img_list).astype(np.uint8)
    # img_write_dir = os.path.join(scene.exp_path, 'viz_regist')
    regist_name = f'viz_regist_test'
    img_write_dir ='/home/c206/zjr/3dgs/data/regist/'
    img_save_path = os.path.join(img_write_dir, regist_name)
    os.makedirs(img_save_path, exist_ok=True)
    imageio.imwrite(os.path.join(img_save_path, f"{1}.png"), img_list)
def save_constscale(savepath,constscale):
    with open(savepath, 'w') as file:
        # 将 float 变量转换为字符串并写入文件
        file.write(str(constscale))
def load_constscale(path):
    try:
        # 以只读模式打开文件
        with open(path, 'r') as file:
            # 读取文件中的内容
            content = file.read()
            # 将读取的内容转换为 float 类型
            number = float(content)
            return number
    except FileNotFoundError:
        print(f"文件 {path} 未找到。")
    except ValueError:
        print(f"文件 {path} 中的内容无法转换为浮点数。")
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000,29_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000,29_000, 30_000])
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)


    # # Register
    from utils.spring_utils.metrics.vgg_loss import VGGLoss, TVLoss
    crit_vgg = VGGLoss().cuda()
    crit_tv = TVLoss(p=2)
    regist_name = f'regist_gaussians_{0}'
    # SAVE_ROOT = '/home/c206/zjr/3dgs/data/dd107/output'
    const_scale_init = 0.004
    save_root = os.path.join(lp.extract(args).model_path, "regist/constscale.txt")
    for i in range(1,6):
        print("111")
    #     const_scale_init = my_register_gaus(lp.extract(args),op.extract(args),pp.extract(args),
    #                      args.debug_from,
    #                      iterations=5000,cnt = i,
    #                      const_scale=const_scale_init )
    # print(const_scale_init)
    # save_constscale(save_root,const_scale_init)


# 以写入模式打开文件

    # test(lp.extract(args),op.extract(args),pp.extract(args),
    #                      args.debug_from,
    #                      SAVE_ROOT,5000,
    #                      const_scale=const_scale_init )

    # CF
    const_scale_init =load_constscale(save_root)
    print(const_scale_init)
    my_CF_gaus(lp.extract(args),op.extract(args),pp.extract(args),
                    args.debug_from,
                    iterations=1500,cnt = i,
                    const_scale=const_scale_init,crit_vgg = crit_vgg,crit_tv = crit_tv)
    
    # # All done
    # print("\nTraining complete.")
