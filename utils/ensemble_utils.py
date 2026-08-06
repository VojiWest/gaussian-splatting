import torch
from torchvision.utils import save_image
import os
import numpy as np
import json
from pathlib import Path

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from uq_metrics.auce import auce
from uq_metrics.ause import ause
from utils.plot_utils import plot_ause, plot_auce


def order_renders(renders, all_image_names):
    image_names = all_image_names[0]
    ordered_names = sorted(image_names)

    ordered_renders = []

    for model in range(len(renders)):
        name_to_idx = {name: idx for idx, name in enumerate(all_image_names[model])}
        ordered = torch.stack([renders[model][name_to_idx[name]] for name in ordered_names], dim=0)

        ordered_renders.append(ordered)

    renders = torch.stack(ordered_renders, dim=0)

    return renders, ordered_names

def get_ensemble_variance(renders, all_image_names, normalize=False):
    renders, ordered_names = order_renders(renders, all_image_names)

    pred_mean = torch.mean(renders, dim=0)

    variance = torch.mean(renders ** 2, dim=0) - pred_mean ** 2

    print("Max Variance Value: ", torch.max(variance))
    print("Mean Variance Value: ", torch.mean(variance))

    if normalize:
        pred_mean = torch.clamp(pred_mean / torch.max(pred_mean), 0.0, 1.0)
        variance = torch.clamp(variance / torch.max(variance), 0.0, 1.0) 

    return pred_mean, variance, ordered_names

def create_ens_path(model_path):
    ens_path = os.path.join(model_path, "Ensemble")
    if not os.path.exists(ens_path):
        os.makedirs(ens_path)

    return ens_path

def save_ens_uncertainty(variance, ordered_names, save_path):
    # variance_norm = torch.clamp(variance / torch.max(variance), 0.0, 1.0)
    for var_image, img_name in zip(variance, ordered_names):
        variance_norm = torch.clamp(var_image / torch.max(var_image), 0.0, 1.0)
        var_gray = variance_norm.mean(dim=0, keepdim=True)
        image_name = f"EnsUQ_{img_name}.png"
        save_image(var_gray, f"{save_path}/{image_name}")

def save_ens_mean_pred(pred_mean, ordered_names, save_path):
    for pred, img_name in zip(pred_mean, ordered_names):
        pred_norm = torch.clamp(pred / torch.max(pred), 0.0, 1.0)
        image_name = f"EnsMean_{img_name}.png"
        save_image(pred_norm, f"{save_path}/{image_name}")

def calc_ens_metrics(viewpoints, ordered_names, mean, var, model_path):
    l1, l_ssim, psnr_metric, lpips_metric = 0.0, 0.0, 0.0, 0.0
    ause_metric, auce_metric = 0.0, 0.0
    all_auce_coverages = np.zeros(99)
    all_ause_diff, all_ause_err, all_ause_err_by_var = np.zeros(100), np.zeros(100), np.zeros(100)
    for view in viewpoints:
        view_name = view.image_name
        gt_image = torch.clamp(view.original_image.cpu(), 0.0, 1.0)
        render_idx = ordered_names.index(view_name)

        mean_render = mean[render_idx].cpu()
        var_render = var[render_idx].cpu()

        # Normalize Var
        var_render = torch.clamp(var_render / torch.max(var_render), 0.0, 1.0)

        l1 += l1_loss(mean_render, gt_image).mean().double()
        l_ssim += ssim(mean_render, gt_image).mean().double()
        psnr_metric += psnr(mean_render, gt_image).mean().double()
        lpips_metric += lpips(mean_render, gt_image, net_type='vgg')

        ### Get Ensemble UQ Metrics
        ratio_removed, ause_err, ause_err_by_var, ause_value = ause(var_render.flatten(), ((mean_render - gt_image) ** 2).flatten(), err_type="mse")
        ause_metric += ause_value
        all_ause_diff += (ause_err_by_var - ause_err)
        all_ause_err += ause_err
        all_ause_err_by_var += ause_err_by_var

        auce_dict = auce(np.array(mean_render.flatten()), np.array(var_render.flatten()), np.array(gt_image.flatten()))
        auce_metric += auce_dict["auc_abs_error_values"]
        all_auce_coverages += auce_dict["coverage_values"]

    l1 /= len(viewpoints)
    l_ssim /= len(viewpoints)
    psnr_metric /= len(viewpoints)
    lpips_metric /= len(viewpoints)
    ause_metric /= len(viewpoints)
    auce_metric /= len(viewpoints)

    print("Ensemble L1: ", l1.item())
    print("Ensemble SSIM: ", l_ssim.item())
    print("Ensemble PSNR: ", psnr_metric.item())
    print("Ensemble LPIPS: ", lpips_metric.item())
    print("Ensemble AUSE: ", ause_metric)
    print("Ensemble AUCE: ", auce_metric)

    all_auce_coverages /= len(viewpoints)
    all_ause_diff /= len(viewpoints)
    all_ause_err /= len(viewpoints)
    all_ause_err_by_var /= len(viewpoints)

    plot_auce(all_auce_coverages, save_dir=model_path, output="Ens")
    plot_ause(all_ause_diff, all_ause_err_by_var, all_ause_err, save_dir=model_path, output="Ens")

    metrics = {"SSIM": l_ssim.item(), "PSNR": psnr_metric.item(), "LPIPS": lpips_metric.item(), "AUSE": ause_metric, "AUCE": auce_metric}
    metrics_file = Path(model_path) / "ensemble_metrics.json"
    with open(str(metrics_file), 'w') as f:
        json.dump(metrics, f)
