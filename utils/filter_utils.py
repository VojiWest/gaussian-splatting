import torch
from torchvision.utils import save_image
import os

from scene import Scene, GaussianModel
from utils.plot_utils import plot_histogram

def create_paths(scene : Scene):
    filter_path = os.path.join(scene.model_path, "Filtering")
    hist_path = os.path.join(scene.model_path, "Histogram")
    image_path = os.path.join(scene.model_path, "Renders")
    uq_path = os.path.join(scene.model_path, "Uncertainty")
    if not os.path.exists(filter_path):
        os.makedirs(filter_path)
    if not os.path.exists(hist_path):
        os.makedirs(hist_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(uq_path):
        os.makedirs(uq_path)

    return filter_path, hist_path, image_path, uq_path

def calculate_filter_variable(filter_criterion, gaussians : GaussianModel, model_path, iteration):
    if "sd_max" in filter_criterion:
        filter_variable = gaussians.get_sd(method='max')
        return filter_variable
    elif "mean" in filter_criterion:
        filter_variable = gaussians.get_sd(method='mean')
        return filter_variable
    
    if "vog" in filter_criterion:
        if "sd" in filter_criterion:
            method = "sd"
        else:
            method = "var"
        if "viewpoint" in filter_criterion:
            grads = gaussians.get_inter_view_gradients()
            variance = get_inter_view_gradient_variance(gradients=grads, method=method, model_path=model_path, iteration=iteration)
        elif "iteration" in filter_criterion:
            grads = gaussians.get_inter_iter_gradients()
            variance = get_inter_view_gradient_variance(gradients=grads, method=method, model_path=model_path, iteration=iteration)
        return variance
    
    if "grad" in filter_criterion:
        grads = gaussians.get_inter_view_gradients()
        norm = get_mean_gradient_norm(gradients=grads)
        return norm

    if "random" in filter_criterion:
        return torch.rand(gaussians.get_xyz.shape[0])
    
    return torch.rand(gaussians.get_xyz.shape[0])

def filter_gaussians(filter_criteria, filter_threshold, means3D, means2D, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp, remove_above_filter=True):
    if remove_above_filter:
        keep = filter_criteria < filter_threshold
    else:
        keep = filter_criteria > filter_threshold

    filtered_means3D = means3D[keep]
    filtered_means2D = means2D[keep]

    if shs is not None:
        filtered_shs = shs[keep]
    else:
        filtered_shs = shs

    if colors_precomp is not None:
        filtered_colors_precomp = colors_precomp[keep]
    else:
        filtered_colors_precomp = colors_precomp

    filtered_opacity = opacity[keep]
    filtered_scales = scales[keep]
    filtered_rotations = rotations[keep]
    filtered_cov3D_precomp = cov3D_precomp

    filtered_pct = round(100 - (100 * (filtered_opacity.numel() / opacity.numel())), 4)
    # print(f"Filtered {filtered_pct} % of Gaussians with Threshold of {round(filter_threshold.item(), 5)}")

    return filtered_means3D, filtered_means2D, filtered_shs, filtered_colors_precomp, filtered_opacity, filtered_scales, filtered_rotations, filtered_cov3D_precomp

def calculate_spread(gaussian_non_zero, method):
    if gaussian_non_zero.numel() > 1:
        if method == 'var':
            var = torch.var(gaussian_non_zero, dim=0)
            spread = var
        elif method == 'sd':
            sd = torch.std(gaussian_non_zero, dim=0)
            spread = sd
    elif gaussian_non_zero.numel() > 0:
        spread = torch.tensor(0.0, device=gaussian_non_zero.device, dtype=gaussian_non_zero.dtype)
    else:
        spread = torch.tensor(-1.0, device=gaussian_non_zero.device, dtype=gaussian_non_zero.dtype)

    return spread

def get_inter_view_gradient_variance(gradients, method='var', model_path="", iteration=0):
    spreads = torch.zeros(gradients.shape[0])
    numels = []
    for idx, gaussian_grads in enumerate(gradients):
        if gaussian_grads.ndim == 2:
            g1 = gaussian_grads[:,0]
            g2 = gaussian_grads[:,1]

            mask_g1 = ~torch.isnan(g1)
            mask_g2 = ~torch.isnan(g2)

            g1_valid = g1[mask_g1]
            g2_valid = g2[mask_g2]

            numels.append(g1_valid.numel())
            numels.append(g2_valid.numel())

            v1 = calculate_spread(g1_valid, method=method)
            v2 = calculate_spread(g2_valid, method=method)

            spreads[idx] = torch.mean(torch.stack([v1, v2]))
        
        elif gaussian_grads.ndim == 1:
            mask = ~torch.isnan(gaussian_grads)
            gaussian_non_zero = gaussian_grads[mask]

            numels.append(gaussian_non_zero.numel())

            spreads[idx] = calculate_spread(gaussian_non_zero, method=method)

    hist_path = os.path.join(model_path, "Histogram")
    plot_histogram(numels, title="Number of Gradient Samples per Gaussian", folder_path=hist_path, iteration=iteration)
            
    return spreads

def get_mean_gradient_norm(gradients):
    norms = torch.zeros(gradients.shape[0])
    for idx, gaussian_grads in enumerate(gradients):
        if gaussian_grads.ndim == 2:
            g1 = gaussian_grads[:,0]
            g2 = gaussian_grads[:,1]

            mask = ~torch.isnan(g1) & ~torch.isnan(g2)

            if mask.any():
                g_valid = gaussian_grads[mask]
                norms[idx] = torch.mean(torch.norm(g_valid, dim=1))
            else:
                norms[idx] = 0.0
        
        elif gaussian_grads.ndim == 1:
            mask = ~torch.isnan(gaussian_grads)
            gaussian_non_zero = gaussian_grads[mask]

            if gaussian_non_zero.numel() > 0:
                norms[idx] = torch.mean(gaussian_non_zero)
            else:
                norms[idx] = 0.0
            
    return norms


def get_depth_weighted_gradient_variance(variances, gaussians, viewpoint_camera, depth_cal = "zs", depth_lambda = 1):
    depths = get_depths(gaussians, viewpoint_camera, depth_cal)
    dw_variances = variances * (depth_lambda / (depths+0.0001))

    # print("Variances: ", variances)
    # print("Depths: ", depths)
    # print("Depth Weighted Variances: ", dw_variances)

    return dw_variances

def homogenize_points(points):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def get_depths(gaussians, viewpoint_camera, depth_cal="zs"):
    # Taken from gaussian_rendeder

    points_world = gaussians.get_xyz # torch.Size([25837, 3]), torch.float32
    T_cam_world = viewpoint_camera.world_view_transform.T # torch.Size([4, 4]), torch.float32

    # homogenize gaussian centers
    points_world_homogenized = homogenize_points(points_world)

    # apply T_cam_world to these points
    #T_cam_world = T_world_cam.inverse()
    points_cam_homogenized = (T_cam_world @ points_world_homogenized.T).T
    points_cam = points_cam_homogenized[:, :3]

    # get the zs and use as color for rendering
    if depth_cal == "zs":
        depths = points_cam[:, 2]
    elif depth_cal == "norm": # Take norm
        depths = torch.norm(points_cam, dim=-1)

    return depths.cpu()

def save_render(image, save_path, viewpoint, method, iteration, t_idx):
    image_name = method + "_no_{}".format(viewpoint.image_name) + "_" + str(iteration) + "_" + str(t_idx) + ".png"
    save_image(image, f"{save_path}/{image_name}")
    