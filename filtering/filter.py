import torch

from utils.filter_utils import *

def get_filter_variable(method, quantiles, scene, iteration):
    filter_variable = calculate_filter_variable(method, scene.gaussians, model_path=scene.model_path, iteration=iteration)
    # if "depth" in method and "weighted" in method:
    filter_variable_const = filter_variable.clone()
    print("Filter Variable Shape: ", filter_variable.shape)
    print("Number of No-Variance: ", torch.sum(filter_variable < 0).item(), "out of ", filter_variable.numel())
    if "inverse" in method:
        filter_thresholds = torch.quantile(filter_variable, 1-quantiles)
    else:
        filter_thresholds = torch.quantile(filter_variable, quantiles)

    return filter_variable, filter_variable_const, filter_thresholds

def get_depth_specific_filter_variable(method, filter_variable_const, quantiles, scene, viewpoint, threshold_idx):
    depth_measure = method.split("_")[1]
    if "weighted" in method:
        filter_variable = get_depth_weighted_gradient_variance(variances=filter_variable_const, gaussians=scene.gaussians, viewpoint_camera=viewpoint, depth_cal=depth_measure)
        threshold = torch.quantile(filter_variable, quantiles[threshold_idx])
    if method == "depth_zs":
        filter_variable = get_depths(gaussians=scene.gaussians, viewpoint_camera=viewpoint, depth_cal=depth_measure) # TODO filtering out lowest zs is just negative values, prolly wanna do lowest positive values
        threshold = torch.quantile(filter_variable, 1.0-quantiles[threshold_idx])
    if method == "depth_norm":
        filter_variable = get_depths(gaussians=scene.gaussians, viewpoint_camera=viewpoint, depth_cal=depth_measure)
        threshold = torch.quantile(filter_variable, 1.0-quantiles[threshold_idx])

    return filter_variable, threshold