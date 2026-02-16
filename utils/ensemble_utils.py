import torch
from torchvision.utils import save_image
import os

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
