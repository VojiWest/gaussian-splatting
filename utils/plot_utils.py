import matplotlib.pyplot as plt
import numpy as np

def new_plot_filter(filter_thresholds, quantiles, l1_losses, l_ssims, all_lpipses, psnrs, folder_path, iteration, methods, split_names):
    splits = [split_names[0]['name'].split("_")[1], split_names[1]['name'].split("_")[1]]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    nice_method_names = {
        "iteration_vog_var" : "Iteration VOG",
        "iteration_vog_sd" : "Iteration SDOG",
        "depth_norm_weighted_iteration_vog_var" : "Euclidean Depth-Weighted Iteration VOG",
        "depth_zs_weighted_iteration_vog_var" : "Coordinate Depth-Weighted Iteration VOG",
        "depth_norm_weighted_iteration_vog_sd" : "Euclidean Depth-Weighted Iteration SDOG",
        "depth_zs_weighted_iteration_vog_sd" : "Coordinate Depth-Weighted Iteration SDOG",
        "grad_norm" : "Gradient Norm",
        "inverse_viewpoint_vog_var" : "Inverse VOG",
        "inverse_viewpoint_vog_sd" : "Inverse SDOG",
        "viewpoint_vog_var" : "VOG",
        "viewpoint_vog" : "VOG",
        "viewpoint_vog_sd" : "SDOG",
        "depth_norm_weighted_viewpoint_vog_var" : "Euclidean Depth-Weighted VOG",
        "depth_zs_weighted_viewpoint_vog_var" : "Coordinate Depth-Weighted VOG",
        "depth_norm_weighted_viewpoint_vog_sd" : "Euclidean Depth-Weighted SDOG",
        "depth_zs_weighted_viewpoint_vog_sd" : "Coordinate Depth-Weighted SDOG",
        "depth_zs" : "Coordinate Depth",
        "depth_norm" : "Euclidean Depth",
        "sd_max" : "Max Gaussian Scale",
        "sd_mean" : "Mean Gaussian Scale",
        "random" : "Random",
        "ensemble" : "Ensemble",
    }

    x = quantiles

    ### Plot Loss ###
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax_test, ax_train = axes
    for method_idx, method in enumerate(methods):
        color = colors[method_idx]
        # test = even index, train = odd index
        test_loss = l1_losses[2 * method_idx]
        train_loss = l1_losses[2 * method_idx + 1]

        ax_test.plot(x, test_loss, color=color, linestyle='-', label=nice_method_names[method])
        ax_train.plot(x, train_loss, color=color, linestyle='-')
        
    ax_test.set_title("Test")
    ax_train.set_title("Train")
    ax_test.set_ylabel("L1 Loss")
    for ax in axes:
        ax.set_xlabel("Percentile Kept")

    handles, labels = ax_test.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)

    fig.suptitle("L1 Loss Filtering")
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(f"{folder_path}/loss_filter_plot_{iteration}.png")
    plt.close()

    ### Plot PSNR ###
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax_test, ax_train = axes
    for method_idx, method in enumerate(methods):
        color = colors[method_idx]
        test_psnr = psnrs[2 * method_idx]
        train_psnr = psnrs[2 * method_idx + 1]

        ax_test.plot(x, test_psnr, color=color, linestyle='-', label=nice_method_names[method])
        ax_train.plot(x, train_psnr, color=color, linestyle='-')
        
    ax_test.set_title("Test")
    ax_train.set_title("Train")
    ax_test.set_ylabel("PSNR")
    for ax in axes:
        ax.set_xlabel("Percentile Kept")

    handles, labels = ax_test.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)

    fig.suptitle("PSNR Filtering")
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(f"{folder_path}/psnr_filter_plot_{iteration}.png")
    plt.close()

    ### Plot SSIM ###
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax_test, ax_train = axes
    for method_idx, method in enumerate(methods):
        color = colors[method_idx]
        # test = even index, train = odd index
        test_ssim = l_ssims[2 * method_idx]
        train_ssim = l_ssims[2 * method_idx + 1]

        ax_test.plot(x, test_ssim, color=color, linestyle='-', label=nice_method_names[method])
        ax_train.plot(x, train_ssim, color=color, linestyle='-')
        
    ax_test.set_title("Test")
    ax_train.set_title("Train")
    ax_test.set_ylabel("SSIM")
    for ax in axes:
        ax.set_xlabel("Percentile Kept")

    handles, labels = ax_test.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)

    fig.suptitle("SSIM Filtering")
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(f"{folder_path}/ssim_filter_plot_{iteration}.png")
    plt.close()




def plot_filter(filter_thresholds, quantiles, l1_losses, l_ssims, all_lpipses, psnrs, folder_path, iteration, methods, split_names):
    new_plot_filter(filter_thresholds, quantiles, l1_losses, l_ssims, all_lpipses, psnrs, folder_path, iteration, methods, split_names)
    
    # splits = [split_names[0]['name'].split("_")[1], split_names[1]['name'].split("_")[1]]
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    # nice_method_names = {
    #     "iteration_vog_var" : "Iteration VOG",
    #     "depth_norm_weighted_iteration_vog_var" : "Euclidean Depth-Weighted Iteration VOG",
    #     "depth_zs_weighted_iteration_vog_var" : "Coordinate Depth-Weighted Iteration VOG",
    #     "grad_norm" : "Gradient Norm",
    #     "inverse_viewpoint_vog_var" : "Inverse Viewpoint VOG",
    #     "viewpoint_vog_var" : "Viewpoint VOG",
    #     "depth_norm_weighted_viewpoint_vog_var" : "Euclidean Depth-Weighted Viewpoint VOG",
    #     "depth_zs_weighted_viewpoint_vog_var" : "Coordinate Depth-Weighted Viewpoint VOG",
    #     "depth_zs" : "Coordinate Depth",
    #     "depth_norm" : "Euclidean Depth",
    #     "sd_max" : "Max Gaussian Scale",
    #     "sd_mean" : "Mean Gaussian Scale",
    #     "random" : "Random",
    # }

    # # x = filter_thresholds
    # x = quantiles

    # title = "L1 Loss Filtering"
    # for idx, loss in enumerate(l1_losses):
    #     linetype = ':'
    #     if idx % 2 == 1:
    #         linetype = '-'
    #     plt.plot(x, loss, color=colors[idx // 2], label = splits[idx % 2] + " " + nice_method_names[methods[idx // 2]], linestyle=linetype)
    # plt.title(title)
    # plt.ylabel("L1 Loss")
    # x_label = "Percentile Kept"
    # plt.xlabel(x_label)
    # # plt.tight_layout()
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.savefig(f"{folder_path}/loss_filter_plot_{iteration}.png", bbox_inches='tight')
    # plt.close()

    # title = "PSNR Filtering"
    # for idx, psnr in enumerate(psnrs):
    #     linetype = ':'
    #     if idx % 2 == 1:
    #         linetype = '-'
    #     plt.plot(x, psnr, color=colors[idx // 2], label = splits[idx % 2] + " " + nice_method_names[methods[idx // 2]], linestyle=linetype)
    # plt.title(title)
    # plt.ylabel("PSNR")
    # x_label = "Percentile Kept"
    # plt.xlabel(x_label)
    # # plt.tight_layout()
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.savefig(f"{folder_path}/psnr_filter_plot_{iteration}.png", bbox_inches='tight')
    # plt.close()

    # title = "LPIPS Filtering"
    # for idx, lpipses in enumerate(all_lpipses):
    #     linetype = ':'
    #     if idx % 2 == 1:
    #         linetype = '-'
    #     plt.plot(x, lpipses, color=colors[idx // 2], label = splits[idx % 2] + " " + methods[idx // 2], linestyle=linetype)
    # plt.title(title)
    # plt.ylabel("LPIPS")
    # x_label = "Percentile Kept"
    # plt.xlabel(x_label)
    # # plt.tight_layout()
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.savefig(f"{folder_path}/lpips_filter_plot_{iteration}.png")
    # plt.close()

    # title = "SSIM Filtering"
    # for idx, ssim in enumerate(l_ssims):
    #     linetype = ':'
    #     if idx % 2 == 1:
    #         linetype = '-'
    #     plt.plot(x, ssim, color=colors[idx // 2], label = splits[idx % 2] + " " + nice_method_names[methods[idx // 2]], linestyle=linetype)
    # plt.title(title)
    # plt.ylabel("SSIM")
    # x_label = "Percentile Kept"
    # plt.xlabel(x_label)
    # # plt.tight_layout()
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.savefig(f"{folder_path}/ssim_filter_plot_{iteration}.png", bbox_inches='tight')
    # plt.close()

def plot_histogram(data, title, folder_path, iteration):
    plt.hist(data, bins=100)
    plt.yscale('log', nonpositive='clip')
    plt.title(title)
    plt.savefig(f"{folder_path}/hist_{title}_{iteration}.png")
    plt.close()

def plot_ause(ause_diff, ause_err_by_var, ause_err, save_dir="./imgs", output="rgb"):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)

    plt.plot(ratio_removed, ause_diff, label="Difference")
    plt.plot(ratio_removed, ause_err_by_var, color="blue", linestyle=":", label="Error Sorted by Variance")
    plt.plot(ratio_removed, ause_err, color="red", linestyle=":", label="Error Descending")
    plt.ylabel("Normalized Mean Error")
    plt.xlabel("Ratio of Pixels Removed")
    plt.legend()
    plt.savefig("%s/%s_ause_p.png" % (save_dir, output))
    plt.close()

def plot_auce(auce_coverages, save_dir="./imgs", output="rgb"):
    alphas = list(np.arange(start=0.01, stop=1.0, step=0.01))

    plt.plot([0.0, 1.0], [0.0, 1.0], "k:", label="Perfect")
    plt.plot(alphas, np.flip(auce_coverages, 0))
    plt.legend()
    plt.ylabel("Empirical coverage")
    plt.xlabel("p")
    plt.title("Prediction intervals - Empirical coverage")
    plt.savefig("%s/%s_empirical_coverage.png" % (save_dir, output))
    plt.show()
    plt.close()