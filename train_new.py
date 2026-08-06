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
import json
from pathlib import Path
import numpy as np
from torchvision.utils import save_image
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_uncertainty
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from lpipsPyTorch import lpips
import gc
from collections import defaultdict

from utils.filter_utils import create_paths, save_render
from utils.plot_utils import plot_filter, plot_histogram, plot_ause, plot_auce
from utils.ensemble_utils import * 
from filtering.filter import get_filter_variable, get_depth_specific_filter_variable
from filtering.ensemble_filter import get_ens_filter_variable
from uq_metrics.auce import auce
from uq_metrics.ause import ause

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, ens=False):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    if opt.jitter_init:
        gaussians.sample_init_points()
    if opt.randomize_init:
        gaussians.randomize_init_points()
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    evaluated_20k = False
    evaluated_30k = False 

    min_val_loss = 999999
    patience = opt.patience

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
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

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        img_idx = viewpoint_cam.uid

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            val_total_loss = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, opt)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if not torch.isnan(val_total_loss):
                if val_total_loss < min_val_loss:
                    min_val_loss = val_total_loss
                    patience = opt.patience
                else:
                    patience -= 1
                print("Patience is ", patience)

            # VOG Tracking
            if opt.do_filtering == True:
                # Add gradient for tracking variance
                gaussians.add_grads_cam(viewspace_point_tensor, visibility_filter, img_idx, len(scene.getTrainCameras()))
                if iteration > opt.densify_until_iter:
                    gaussians.add_grads_iter(viewspace_point_tensor, visibility_filter, num_iters=500)

                timing_condition_1 = 0 < 20000 - iteration <= 5 * len(scene.getTrainCameras()) and not evaluated_20k
                timing_condition_2 = 0 < 30000 - iteration <= 5 * len(scene.getTrainCameras()) and not evaluated_30k
                grads_cam_condition = torch.count_nonzero(gaussians.get_cam_idxs_grads_stored()) == (len(scene.getTrainCameras()))
                if ((timing_condition_1 or timing_condition_2) and grads_cam_condition) or patience == 0:
                    print("It's Filtering Time!")
                    if iteration < 20000:
                        evaluated_20k = True
                    elif iteration < 30000:
                        evaluated_30k = True
                    evaluate_gaussian_filtering(opt, iteration, scene, (pipe, background), dataset)
                    
                if iteration <= opt.densify_until_iter and iteration % opt.densification_interval == 0:
                        gaussians.reset_grad_cam_tracking()
                elif iteration > opt.densify_until_iter and torch.count_nonzero(gaussians.get_cam_idxs_grads_stored()) == (len(scene.getTrainCameras())): # Once cycled through all images, reset
                        gaussians.reset_grad_cam_tracking()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Early Stopping
            if patience == 0:
                print("Early Stopping at Iteration:", iteration, ", Validation Loss did not improve for", opt.patience, "iterations")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                break


    if ens:
        validation_configs = ({'name': 'eval_test', 'cameras' : scene.getTestCameras()},
                            {'name': 'eval_val', 'cameras' : scene.getValCameras()})
                            # {'name': 'eval_train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 51, 5)]})
        train_renders = []
        val_renders = []
        test_renders = []

        train_img_names = []
        val_img_names = []
        test_img_names = []
        with torch.no_grad():
            for split_idx, config in enumerate(validation_configs):
                if config['cameras'] and len(config['cameras']) > 0:
                    for idx, viewpoint in enumerate(config['cameras']):
                        # Render image
                        render_pkg = render(viewpoint, scene.gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                        image = render_pkg["render"].detach()

                        if viewpoint.alpha_mask is not None:
                            alpha_mask = viewpoint.alpha_mask.cuda()
                            image *= alpha_mask
                        
                        if "train" in config['name'].split("_")[1]:
                            train_renders.append(image)
                            train_img_names.append(viewpoint.image_name)
                        elif "val" in config['name'].split("_")[1]:
                            val_renders.append(image)
                            val_img_names.append(viewpoint.image_name)
                        elif "test" in config['name'].split("_")[1]:
                            test_renders.append(image)
                            test_img_names.append(viewpoint.image_name)
        
        return val_renders, val_img_names, test_renders, test_img_names, scene, bg

                        

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, opt):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    val_total_loss = torch.tensor(float('nan'))

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'val', 'cameras' : scene.getValCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                total_loss = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if FUSED_SSIM_AVAILABLE:
                        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                    else:
                        ssim_value = ssim(image, gt_image)
                    total_loss = (1.0 - opt.lambda_dssim) * l1_test + opt.lambda_dssim * (1.0 - ssim_value)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                total_loss /= len(config['cameras'])            
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} Total Loss {}".format(iteration, config['name'], l1_test, psnr_test, total_loss))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if config['name'] == "val":
                    val_total_loss = total_loss

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return val_total_loss

def get_rendered_uncertainty(viewpoint, scene : Scene, renderArgs, uq_variable, method, iteration, uq_path, save=True):
    render_uncertainty_pkg = render_uncertainty(uq_variable, viewpoint, scene.gaussians, *renderArgs)
    render_uncertainty_image = render_uncertainty_pkg["render"][0].unsqueeze(0)

    if torch.any(torch.logical_or(torch.isnan(render_uncertainty_image), torch.isinf(render_uncertainty_image))):
        valid_uncertainties = render_uncertainty_image[torch.logical_not(torch.logical_or(torch.isnan(render_uncertainty_image), torch.isinf(render_uncertainty_image)))]
        if len(valid_uncertainties) == 0:
            print(f"[eval] everything is nan for UQ")
            not_nan_max = 100.0 # TODO Should look into this value and this general if-statement
        else:
            not_nan_max = torch.max(valid_uncertainties).item()
        render_uncertainty_image = torch.nan_to_num(render_uncertainty_image, not_nan_max, not_nan_max)

    normalized_render_uncertainty_image = torch.clamp(render_uncertainty_image / torch.max(render_uncertainty_image), 0.0, 1.0)

    if save:
        image_name = "norm_uq_" + method + "_no_{}".format(viewpoint.image_name) + "_" + str(iteration) + ".png"
        save_image(normalized_render_uncertainty_image, f"{uq_path}/{image_name}")

    return render_uncertainty_image

def evaluate_gaussian_filtering(opt_params, iteration, scene : Scene, renderArgs, dataset, ens_vars_test=None, ens_vars_val=None):
    torch.cuda.empty_cache()
    assert not torch.is_grad_enabled()
    validation_configs = ({'name': 'eval_val', 'cameras' : scene.getValCameras()},
                            {'name': 'eval_test', 'cameras' : scene.getTestCameras()})
    
    print("Post Processing Filtering Gaussians")

    filter_path, hist_path, image_path, uq_path = create_paths(scene)
    if ens_vars_test is not None and ens_vars_val is not None:
        methods = ["ensemble"]
    else:
        methods = opt_params.filter_criteria.split(",")
    quantiles = torch.tensor([0.8, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1])

    all_l1_losses, all_l_ssims, all_psnrs, all_lpipses = [], [], [], []

    for method in methods:
        # Get filtering variables and thresholds based on method
        print("\nFiltering Based on: ", method)
        if method != "depth_zs" and method != "depth_norm":  # Get filtering variables and thresholds based on method
            filter_variable, filter_variable_const, filter_thresholds = get_filter_variable(method, quantiles, scene, iteration)
        else:
            filter_thresholds = quantiles
        best_quantile = None

        for config in validation_configs:
            l1_losses, l_ssims, psnrs = [], [], []
            lpipses = []
            ause_metric, auce_metric = 0.0, 0.0
            all_auce_coverages = np.zeros(99)
            all_ause_diff, all_ause_err, all_ause_err_by_var = np.zeros(100), np.zeros(100), np.zeros(100)
            
            if config['name'] == "eval_test":
                filter_thresholds = [best_quantile]
                print("Best Quantile Set To: ", best_quantile)
            
            for t_idx, threshold in enumerate(filter_thresholds):
                l1, l_ssim, psnr_metric = 0.0, 0.0, 0.0
                lpips_metric = 0.0
                if ("depth" in method or "ensemble" in method) and config['name'] == "eval_test":
                    thresh_idx = int(threshold)

                if config['cameras'] and len(config['cameras']) > 0:
                    val_or_test = ("test" in config['name'].split("_")[1] or "val" in config['name'].split("_")[1])

                    if "ensemble" in method:
                        if config['name'] != "eval_test":
                            thresh_idx = t_idx
                            filter_variable, threshold = get_ens_filter_variable(scene, config['cameras'], ens_vars_val, quantiles, thresh_idx)
                        else:
                            filter_variable, threshold = get_ens_filter_variable(scene, config['cameras'], ens_vars_test, quantiles, thresh_idx)
                    
                    for idx, viewpoint in enumerate(config['cameras']):
                        if "depth" in method:
                            if config['name'] != "eval_test":
                                thresh_idx = t_idx
                            filter_variable, threshold = get_depth_specific_filter_variable(method, filter_variable_const, quantiles, scene, viewpoint, thresh_idx)

                        # Render image
                        remove_high = method != "depth_zs" and method != "depth_norm" and "inverse" not in method
                        
                        if t_idx == len(quantiles) - 1: # Render without filtering
                            render_pkg = render(viewpoint, scene.gaussians, *renderArgs, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                        else:
                            render_pkg = render(viewpoint, scene.gaussians, *renderArgs, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, filter_criteria=filter_variable, filter_threshold=threshold, filter_high=remove_high)
                        
                        image = render_pkg["render"]
                        if viewpoint.alpha_mask is not None:
                            alpha_mask = viewpoint.alpha_mask.cuda()
                            image *= alpha_mask
                        image = torch.clamp(image, 0.0, 1.0)

                        if ("vog" in method or "random" in method) and config['name'] == "eval_test": # Render uncertainty
                            render_uncertainty_image = get_rendered_uncertainty(viewpoint, scene, renderArgs, filter_variable, method, iteration, uq_path)

                        if config['name'] == "eval_test":
                            save_render(image, image_path, viewpoint, method, iteration, t_idx)

                        # get the groundtruth rgb image
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                        if dataset.train_test_exp:
                            image = image[..., image.shape[-1] // 2:]
                            gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                        l1 += l1_loss(image, gt_image).mean().double()
                        l_ssim += ssim(image, gt_image).mean().double()
                        psnr_metric += psnr(image, gt_image).mean().double()
                        if config['name'] == "eval_test":
                            lpips_metric += lpips(image, gt_image, net_type='vgg')

                        if ("vog" in method or "random" in method) and config['name'] == "eval_test":
                            flat_rgb_uncertainty = render_uncertainty_image.repeat(3,1,1).flatten()
                            normalized_flat_rgb_uncertainty = torch.clamp(flat_rgb_uncertainty / torch.max(flat_rgb_uncertainty), 0.0, 1.0)
                            ratio_removed, ause_err, ause_err_by_var, ause_metric_val = ause(normalized_flat_rgb_uncertainty, ((image - gt_image) ** 2).flatten(), err_type="mse")
                            ause_metric += ause_metric_val
                            all_ause_diff += (ause_err_by_var - ause_err)
                            all_ause_err += ause_err
                            all_ause_err_by_var += ause_err_by_var
                            auce_dict = auce(np.array(image.flatten().cpu()), np.array(normalized_flat_rgb_uncertainty.cpu()), np.array(gt_image.flatten().cpu()))
                            auce_metric += auce_dict["auc_abs_error_values"]
                            all_auce_coverages += auce_dict["coverage_values"]

                    # Plot debugging histogram
                    if "vog" in method and (t_idx == len(quantiles) - 1):
                        tag_header = config['name'] + "_view_{}".format(viewpoint.image_name)
                        plot_histogram(render_uncertainty_image.flatten().tolist(), title=tag_header + "_" + method + "_Uncertainty_Render", folder_path=hist_path, iteration=iteration)

                    l1 /= len(config['cameras'])
                    l_ssim /= len(config['cameras'])
                    psnr_metric /= len(config['cameras'])

                    l1_losses.append(l1.cpu().item())
                    l_ssims.append(l_ssim.cpu().item())
                    psnrs.append(psnr_metric.cpu().item())

                    if config['name'] == "eval_test":
                        lpips_metric /= len(config['cameras'])
                        lpipses.append(lpips_metric.cpu().item())

            if config['name'] == "eval_val":
                highest_psnr = max(psnrs)
                if method != "depth_zs" and method != "depth_norm" and method != "ensemble":
                    best_quantile = filter_thresholds[psnrs.index(highest_psnr)]
                else:
                    best_quantile = psnrs.index(highest_psnr)
                # if method != "depth_zs" and method != "depth_norm":
                print("Best Quantile Set To Idx: ", psnrs.index(highest_psnr), " = quantile: ", quantiles[psnrs.index(highest_psnr)])
                # else:
                #     print("Best Quantile Set To Idx: ", psnrs.index(highest_psnr), " = quantile: ", 1.0 - quantiles[psnrs.index(highest_psnr)])
            
            if config['name'] == "eval_test":
                print("-------------------------------------------")
                print("-------------------------------------------")
                print("Test Metircs for Method ", method, "at Best Threshold of :", best_quantile)
                print("PSNR of test images: ", psnrs)
                print("SSIM of test images: ", l_ssims)
                print("LPIPS of test images: ", lpipses)
                print("-------------------------------------------")
                print("-------------------------------------------")

            tag_header = config['name'] + "_view_{}".format(viewpoint.image_name)
            plot_histogram(filter_variable.flatten().tolist(), title=tag_header + "_" + method + "_Uncertainty_Variable", folder_path=hist_path, iteration=iteration) # Only plot once per image (not every threshold since does not change)

            all_l1_losses.append(l1_losses)
            all_l_ssims.append(l_ssims)
            all_psnrs.append(psnrs)

            print("PSNRS: ", psnrs)
            print("SSIMs: ", l_ssims)
            print("Thresholds: ", filter_thresholds)

            if ("vog" in method or "random" in method) and config['name'] == "eval_test":
                ause_metric /= len(config['cameras'])
                auce_metric /= len(config['cameras'])
                print("Method: ", method)
                print("Ensemble AUSE: ", ause_metric)
                print("Ensemble AUCE: ", auce_metric)

                all_auce_coverages /= len(config['cameras'])
                all_ause_diff /= len(config['cameras'])
                all_ause_err /= len(config['cameras'])
                all_ause_err_by_var /= len(config['cameras'])

                plot_auce(all_auce_coverages, save_dir=scene.model_path, output=method)
                plot_ause(all_ause_diff, all_ause_err_by_var, all_ause_err, save_dir=scene.model_path, output=method)

                metrics = {"AUSE": ause_metric, "AUCE": auce_metric}
                file_name = "uq_metrics_" + str(method) + "_" + config['name'].split("_")[1] + ".json"
                metrics_file = Path(scene.model_path) / file_name
                with open(str(metrics_file), 'w') as f:
                    json.dump(metrics, f)

    # Evaluate unfiltered on test set
    for config in validation_configs:
        l1, l_ssim, psnr_metric = 0.0, 0.0, 0.0
        lpips_metric = 0.0
        if config['name'] == "eval_test" and config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_pkg = render(viewpoint, scene.gaussians, *renderArgs, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                
                image = render_pkg["render"]
                if viewpoint.alpha_mask is not None:
                    alpha_mask = viewpoint.alpha_mask.cuda()
                    image *= alpha_mask
                image = torch.clamp(image, 0.0, 1.0)

                # get the groundtruth rgb image
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                if dataset.train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                l1 += l1_loss(image, gt_image).mean().double()
                l_ssim += ssim(image, gt_image).mean().double()
                psnr_metric += psnr(image, gt_image).mean().double()
                if config['name'] == "eval_test":
                    lpips_metric += lpips(image, gt_image, net_type='vgg')

            l1 /= len(config['cameras'])
            l_ssim /= len(config['cameras'])
            psnr_metric /= len(config['cameras'])
            lpips_metric /= len(config['cameras'])

            print("")
            print("--------Unfiltered Test Metrics--------")
            print("PSNR: ", psnr_metric.cpu().item())
            print("SSIM: ", l_ssim.cpu().item())
            print("LPIPS: ", lpips_metric.cpu().item())
            print("---------------------------------------")
            print("")

    # plot_filter(filter_thresholds, quantiles.cpu().numpy(), all_l1_losses, all_l_ssims, all_lpipses, all_psnrs, filter_path, iteration, methods, validation_configs)

def ensemble(model_params, opt_params, pipe_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, model_path, num_models = 5):
    ### Train Models ###
    all_val_renders, all_test_renders = [], []
    all_val_image_names, all_test_image_names = [], []

    tmp_dir = os.path.join("tmp_preds")
    os.makedirs(tmp_dir, exist_ok=True)

    for m_idx in range(num_models):
        val_renders, val_img_names, test_renders, test_img_names, scene, bg = training(model_params, opt_params, pipe_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, ens=True)

        model_dir_test = os.path.join(tmp_dir, f"model_{m_idx}_test")
        os.makedirs(model_dir_test, exist_ok=True)
        model_dir_val = os.path.join(tmp_dir, f"model_{m_idx}_val")
        os.makedirs(model_dir_val, exist_ok=True)

        for img_name, render in zip(test_img_names, test_renders):
            save_path = os.path.join(model_dir_test, f"{img_name}.pt")
            torch.save(render.detach().cpu(), save_path)

        for img_name, render in zip(val_img_names, val_renders):
            save_path = os.path.join(model_dir_val, f"{img_name}.pt")
            torch.save(render.detach().cpu(), save_path)

        # val_renders = [v.detach().cpu() for v in val_renders]
        # test_renders = [t.detach().cpu() for t in test_renders]

        # all_val_renders.append(val_renders)
        # all_val_image_names.append(val_img_names)

        # all_test_renders.append(test_renders)
        # all_test_image_names.append(test_img_names)

        # ---- DELETE GPU OBJECTS ----
        if m_idx != num_models - 1:
            del scene
            del bg
            # del test_radii
        # del test_renders
        # del test_img_names
        del val_renders
        del test_renders

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  
        
    ### Calculate Mean & Variance ###
    ens_path = create_ens_path(model_params.model_path)
    # _, var_val, ordered_names_val = get_ensemble_variance(all_val_renders, all_val_image_names)
    # mean, var, ordered_names = get_ensemble_variance(all_test_renders, all_test_image_names)

    test_dirs = [os.path.join(tmp_dir, f"model_{i}_test") for i in range(num_models)]

    val_dirs = [os.path.join(tmp_dir, f"model_{i}_val") for i in range(num_models)]

    mean, var, ordered_names = get_ensemble_variance_from_disk(test_dirs)
    mean_val, var_val, ordered_names_val = get_ensemble_variance_from_disk(val_dirs)

    save_ens_uncertainty(var, ordered_names, ens_path)
    save_ens_mean_pred(mean, ordered_names, ens_path)

    save_ens_uncertainty(var_val, ordered_names_val, ens_path)
    save_ens_mean_pred(mean_val, ordered_names_val, ens_path)

    ### Get Ensemble Performance Metrics ###
    # viewpoints = scene.getValCameras()
    viewpoints = scene.getTestCameras()
    calc_ens_metrics(viewpoints, ordered_names, mean, var, model_params.model_path)

    ### Filter High Variance Gaussians ###
    if opt_params.filter_ens:
        with torch.no_grad():
            evaluate_gaussian_filtering(opt_params, 30000, scene, (pp, bg), model_params, ens_vars_test=var, ens_vars_val=var_val)

def get_ensemble_variance_from_disk(tmp_dirs, normalize=False):
    """
    tmp_dirs: list of model directories, e.g.
        ["tmp_preds/model_0_test", "tmp_preds/model_1_test", ...]
    """

    mean = {}
    M2 = {}
    count = defaultdict(int)

    all_names = None

    for m_idx, model_dir in enumerate(tmp_dirs):

        file_names = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

        image_names = [f.replace(".pt", "") for f in file_names]

        if all_names is None:
            all_names = sorted(image_names)

        name_set = set(image_names)

        for img_name in all_names:

            if img_name not in name_set:
                continue  # or raise error if strict matching required

            path = os.path.join(model_dir, f"{img_name}.pt")
            render = torch.load(path, map_location="cpu").float()

            if count[img_name] == 0:
                mean[img_name] = render.clone()
                M2[img_name] = torch.zeros_like(render)
                count[img_name] = 1

            else:
                count[img_name] += 1

                delta = render - mean[img_name]
                mean[img_name] += delta / count[img_name]

                delta2 = render - mean[img_name]
                M2[img_name] += delta * delta2

    ordered_names = sorted(mean.keys())

    pred_mean = [mean[n] for n in ordered_names]
    variance = [M2[n] / count[n] for n in ordered_names]

    if normalize:
        pred_mean = [
            torch.clamp(m / torch.max(m), 0.0, 1.0) for m in pred_mean
        ]
        variance = [
            torch.clamp(v / torch.max(v), 0.0, 1.0) for v in variance
        ]

    return pred_mean, variance, ordered_names
    

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--ens",  action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.ens:
        ensemble(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.model_path)
    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
