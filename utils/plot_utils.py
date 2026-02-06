import matplotlib.pyplot as plt

def plot_filter(filter_thresholds, quantiles, l1_losses, l_ssims, all_lpipses, psnrs, folder_path, iteration, methods, split_names):
    splits = [split_names[0]['name'].split("_")[1], split_names[1]['name'].split("_")[1]]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    # x = filter_thresholds
    x = quantiles

    title = "L1 Loss Filtering"
    for idx, loss in enumerate(l1_losses):
        linetype = ':'
        if idx % 2 == 1:
            linetype = '-'
        plt.plot(x, loss, color=colors[idx // 2], label = splits[idx % 2] + " " + methods[idx // 2], linestyle=linetype)
    plt.title(title)
    plt.ylabel("L1 Loss")
    x_label = "Percentile Kept"
    plt.xlabel(x_label)
    # plt.tight_layout()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"{folder_path}/loss_filter_plot_{iteration}.png", bbox_inches='tight')
    plt.close()

    title = "PSNR Filtering"
    for idx, psnr in enumerate(psnrs):
        linetype = ':'
        if idx % 2 == 1:
            linetype = '-'
        plt.plot(x, psnr, color=colors[idx // 2], label = splits[idx % 2] + " " + methods[idx // 2], linestyle=linetype)
    plt.title(title)
    plt.ylabel("PSNR")
    x_label = "Percentile Kept"
    plt.xlabel(x_label)
    # plt.tight_layout()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"{folder_path}/psnr_filter_plot_{iteration}.png", bbox_inches='tight')
    plt.close()

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

    title = "SSIM Filtering"
    for idx, ssim in enumerate(l_ssims):
        linetype = ':'
        if idx % 2 == 1:
            linetype = '-'
        plt.plot(x, ssim, color=colors[idx // 2], label = splits[idx % 2] + " " + methods[idx // 2], linestyle=linetype)
    plt.title(title)
    plt.ylabel("SSIM")
    x_label = "Percentile Kept"
    plt.xlabel(x_label)
    # plt.tight_layout()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"{folder_path}/ssim_filter_plot_{iteration}.png", bbox_inches='tight')
    plt.close()

def plot_histogram(data, title, folder_path, iteration):
    plt.hist(data, bins=100)
    plt.yscale('log', nonpositive='clip')
    plt.title(title)
    plt.savefig(f"{folder_path}/hist_{title}_{iteration}.png")
    plt.close()