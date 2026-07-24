import numpy as np
import torch
from torch.nn import Linear, MSELoss
from torch.optim import SGD, NAdam, AdamW
from tqdm import tqdm
from lp_ist import LpIST, LpISTx, LpISTs, LpISTs2, LpISTsx
from iht import IHT
import matplotlib.pyplot as plt

# 普通正文：Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 关键：数学公式不跟随全局字体，使用默认Computer Modern(CM)，才能正常显示\mathcal花体
plt.rcParams['mathtext.fontset'] = 'cm'


def get_thresholding_result(model, constrain, n, var_list, var_id=0, epoch=0):
    print(constrain.__class__.__name__)
    net = Linear(n, 1, bias=False)
    threshold_re = []
    for var in var_list:
        net.weight.data = model.weight.data.clone()
        constrain.parameters(net, None)
        net.weight.data.view(-1)[var_id] = var
        constrain.update_epoch = 0
        constrain.constrain_param(epoch)
        threshold_re.append(net.weight.data.view(-1)[var_id].item())
    return threshold_re


def show_difference(m, n, use_nonzero_fraction=0.1, sample_e_r=[0, 0.1, 0.2, 0.3, 0.5], epoch_st=0, epochs=100):
    model = Linear(n, 1, bias=False)
    iht = IHT(use_nonzero_fraction, epoch_st, epochs, 1)
    lpist = LpISTs2(use_nonzero_fraction, epoch_st, epochs * 0.6, 2, 0.2, 1)
    name_list = ['IHT', '$\ell_p$Squeeze']

    x = np.linspace(-2, 2, 101) * n ** -0.5
    x_min = x.min() * 1.1
    x_max = x.max() * 1.1

    iht_threshold_re = get_thresholding_result(model, iht, n, x, 0, int(epochs * 0.2))
    lpist_threshold_re_list = []
    p_list = []
    for sample_r in sample_e_r:
        threshold_re = get_thresholding_result(model, lpist, n, x, 0, int(epochs * sample_r))
        lpist_threshold_re_list.append(threshold_re)
        p_list.append(lpist.curr_p)

    # ==================== Hard vs Soft 机制对比示意 ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    # IHT Hard Thresholding
    ax1.plot(x, x, 'k--', linewidth=1, alpha=0.5, label='Original $v_i$')
    ax1.plot(x, iht_threshold_re, 'b-', linewidth=1.5, label='IHT projection')
    threshold = x[np.array(iht_threshold_re) == 0].max()
    ax1.axvline(-threshold, color='red', linestyle=':', alpha=0.7)
    ax1.axvline(threshold, color='red', linestyle=':', alpha=0.7)
    ax1.fill_betweenx([x_min, x_max], -threshold, threshold, alpha=0.1, color='red')
    ax1.text(0, threshold, 'Dead zone', ha='center', fontsize=10, color='darkred')
    ax1.annotate('Abrupt drop', xy=(threshold, 0), xytext=(threshold * 1.4, threshold * 0.5),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(x_min, x_max)
    ax1.set_xlabel('Input $v_i^{(t)}$', fontsize=11)
    ax1.set_ylabel('Output $x_i^{(t+1)}$', fontsize=11)
    ax1.set_title('(a) IHT: Hard Thresholding', fontsize=12, loc='left')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 下半: lpSqueeze Soft Squeezing
    ax2.plot(x, x, 'k--', linewidth=1, alpha=0.5, label='Original $v_i$')
    cmap = plt.get_cmap("summer")
    for i, p in enumerate(p_list):
        ax2.plot(x, lpist_threshold_re_list[i], color=cmap(i / len(p_list)), linewidth=1.5, label=f'$\ell_p$Squeeze, p={p:.2f}')
    ax2.axvline(-threshold, color='orange', linestyle=':', alpha=0.7)
    ax2.axvline(threshold, color='orange', linestyle=':', alpha=0.7)
    ax2.fill_betweenx([x_min, x_max], -threshold, threshold, alpha=0.1, color='orange')
    ax2.text(0, threshold, 'Squeezing zone', ha='center', fontsize=10, color='darkgreen')
    ax2.annotate('Gradual shrinkage', xy=(0.6 * threshold, 0.3 * threshold), xytext=(1.1 * threshold, 0.6 * threshold),
                 arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')
    ax2.annotate('Preserved', xy=(1.5 * threshold, 1.5 * threshold), xytext=(1.8 * threshold, threshold),
                 arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(x_min, x_max)
    ax2.set_xlabel('Input $v_i^{(t)}$', fontsize=11)
    ax2.set_ylabel('Output $x_i^{(t+1)}$', fontsize=11)
    ax2.set_title('(b) $\\ell_p$Squeeze: Progressive Squeezing ', fontsize=12, loc='left')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'figures/mechanism_compare.pdf', bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == "__main__":
    show_difference(64, 256)
