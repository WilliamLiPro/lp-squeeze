import numpy as np
import torch
import matplotlib.pyplot as plt


# 普通正文：Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 关键：数学公式不跟随全局字体，使用默认Computer Modern(CM)，才能正常显示\mathcal花体
plt.rcParams['mathtext.fontset'] = 'cm'


def mov_avg(x, wind_sz=7):
    weight = np.ones(wind_sz) / wind_sz
    padded = np.pad(x, wind_sz // 2, mode='reflect')
    return np.convolve(padded, weight, mode='valid')


def show_table(
        alpha_list = [0.0008, 0.0005, 0.00025, 0.0001],
        u_list = [0.15, 0.17, 0.18, 0.2],
):
    method_list = [
        "IHT-AdamW",
        # "Iterative-HTP-AdamW",
        # "CGiht-AdamW",
        # "FISTA",
        "LpISTs2-AdamW",
    ]
    table = []
    for method in method_list:
        loss_list = []
        for i, alpha_p in enumerate(alpha_list):
            u = u_list[i]
            if method is "FISTA":
                data = torch.load(f"TCGA_pancan_{method}_u_{u}.pt")
                loss_list.append(data["loss_list"][-1])
            else:
                data = torch.load(f"TCGA_pancan_{method}_alpha_{alpha_p}.pt")
                loss_list.append(data["loss_list"][-1])
        table.append(loss_list)
    table = np.array(table)
    print(f"{np.array2string(table, formatter={'float_kind' :'{:.2e}'.format})}")


def show_loss_curve(
        alpha_p, u, is_save=False):
    method_list = [
        "IHT-AdamW",
        # "Iterative-HTP-AdamW",
        # "CGiht-AdamW",
        # "FISTA",
        "LpISTs2-AdamW",
    ]
    name_list = [
        "IHT-AdamW",
        # "Iterative-HTP-AdamW",
        # "CGIHT-AdamW",
        # "FISTA",
        "$\ell_p$Squeeze-AdamW",
    ]
    loss_list = []
    for method in method_list:
        if method is "FISTA":
            data = torch.load(f"TCGA_pancan_{method}_u_{u}.pt")
            loss_list.append(data["loss_list"])
        else:
            data = torch.load(f"TCGA_pancan_{method}_alpha_{alpha_p}.pt")
            loss_list.append(data["loss_list"])

    cmap = plt.get_cmap("tab20")
    colors = [cmap(0), cmap(2), cmap(4), cmap(8), cmap(14)]

    plt.figure(figsize=(6, 4), dpi=600)
    for i, loss in enumerate(loss_list):
        plt.plot(mov_avg(loss_list[i], 11), label=name_list[i], color=colors[i])
        plt.plot(loss_list[i], color=colors[i], alpha=0.4)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(fr"Loss ($\alpha={alpha_p}$)")
    plt.tight_layout()
    if is_save:
        plt.savefig(f"tcga_loss_curve_{alpha_p}.pdf")
    plt.show()


if __name__ == "__main__":
    # show_table()
    show_loss_curve(alpha_p=0.00025, u=0.18, is_save=True)
    # show_loss_curve(alpha_p=0.0001, u=0.2, is_save=True)
