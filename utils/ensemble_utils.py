import torch
from torchvision.utils import save_image
import os

def get_ensemble_variance(renders):
    # renders = torch.tensor(renders)
    renders = torch.stack([torch.stack(model_renders, dim=0) for model_renders in renders], dim=0)

    pred_mean = torch.mean(renders, dim=0)

    variance = torch.mean(renders ** 2, dim=0) - pred_mean ** 2

    return pred_mean, variance

def create_ens_path(model_path):
    ens_path = os.path.join(model_path, "Ensemble")
    if not os.path.exists(ens_path):
        os.makedirs(ens_path)

    return ens_path

def save_ens_uncertainty(variance, save_path):
    variance_norm = torch.clamp(variance / torch.max(variance), 0.0, 1.0)
    for idx, var_image in enumerate(variance_norm):
        var_gray = var_image.mean(dim=0, keepdim=True)
        image_name = "EnsUQ_" + str(idx) + ".png"
        save_image(var_gray, f"{save_path}/{image_name}")

def save_ens_mean_pred(pred_mean, save_path):
    for idx, pred in enumerate(pred_mean):
        pred_norm = torch.clamp(pred / torch.max(pred), 0.0, 1.0)
        # var_gray = var_image.mean(dim=0, keepdim=True)
        image_name = "EnsMean_" + str(idx) + ".png"
        save_image(pred_norm, f"{save_path}/{image_name}")
