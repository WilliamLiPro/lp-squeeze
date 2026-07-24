"""
3D空间中lp球面演化的两种情况：
目标非0分量个数为2；目标非0分量个数为1
"""
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lp_ist import LpIST, LpISTx, LpISTs, LpISTsx

# 普通正文：Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 关键：数学公式不跟随全局字体，使用默认Computer Modern(CM)，才能正常显示\mathcal花体
plt.rcParams['mathtext.fontset'] = 'cm'


def lp_sphere(r, p, n_phi=81, n_theta=161):
    phi = np.linspace(0, 2 * np.pi, n_phi)  # 方位角
    theta = np.linspace(0, np.pi, n_theta)  # 极角（从北极到南极）
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    norm_p = (np.abs(x) ** p + np.abs(y) ** p + np.abs(z) ** p) ** (1 / p)
    x = x * r / norm_p
    y = y * r / norm_p
    z = z * r / norm_p
    return x, y, z


def get_r_p_and_x_path(x0, p0=2, p_end=0.19, k=2, epochs=191, save_path=None):
    nonzero_fraction = k / 3

    net = torch.nn.Linear(3, 1, bias=False)
    if isinstance(x0, np.ndarray):
        net.weight.data = torch.from_numpy(x0.astype(np.float32))
    else:
        net.weight.data = x0

    constrain = LpIST(nonzero_fraction, 0, epochs, p0, p_end, 1)
    constrain.parameters(net, None)

    r_list = []
    p_list = []
    x_path = []
    for i in range(epochs):
        constrain.constrain_param(i)
        curr_p = constrain.curr_p
        m_p = k * (constrain.lp_st_norm_p[0] / k) ** (curr_p / p0)
        r = m_p ** (1 / curr_p)
        r_list.append(r)
        p_list.append(curr_p)
        x_path.append(net.weight.clone().detach().numpy())

    r_list = np.array(r_list)
    p_list = np.array(p_list)
    x_path = np.array(x_path)

    if save_path is not None:
        np.savez(save_path, r_list=r_list, p_list=p_list, x_path=x_path)
    return r_list, p_list, x_path


def fmt_num(x):
    # 保留2位小数，再去除末尾零及可能多余的小数点
    return f"{x:.2f}".rstrip('0').rstrip('.')


def show_single_lp_sphere_and_x_path(ax, r, p, x_path, elev=25, azim=50):
    show_range = np.abs(x_path).max() * 1.2
    plot_range = np.abs(x_path).max() * 2

    x, y, z = lp_sphere(r, p)
    mask = np.abs(x) > plot_range
    x[mask] = np.sign(x[mask]) * plot_range
    mask = np.abs(y) > plot_range
    y[mask] = np.sign(y[mask]) * plot_range
    mask = np.abs(z) > plot_range
    z[mask] = np.sign(z[mask]) * plot_range

    ax.plot_surface(x, y, z, color='lightblue', alpha=0.4, edgecolor='lightgray', linewidth=0.3, zorder=0)
    ax.plot(x_path[0, 0], x_path[0, 1], x_path[0, 2], marker='o', color='orange', zorder=6)
    if len(x_path) > 1:
        ax.plot(x_path[:, 0], x_path[:, 1], x_path[:, 2], linewidth=3, color='indianred', alpha=0.9, zorder=5)
        ax.plot(x_path[-1, 0], x_path[-1, 1], x_path[-1, 2], marker='*', color='indianred', zorder=6)
        ax.text(x_path[-1, 0] + 0.3, x_path[-1, 1], x_path[-1, 2] - 0.3,
                f'({fmt_num(x_path[-1, 0])}, {fmt_num(x_path[-1, 1])}, {fmt_num(x_path[-1, 2])})',
                color='black', fontsize=11, zorder=10, ha='center')
    else:
        ax.text(x_path[0, 0] + 0.3, x_path[0, 1], x_path[0, 2] + 0.15,
                f'({fmt_num(x_path[0, 0])}, {fmt_num(x_path[0, 1])}, {fmt_num(x_path[0, 2])})',
                color='black', fontsize=11, zorder=10, ha='center')
    ax.set_xlim([-show_range, show_range])
    ax.set_ylim([-show_range, show_range])
    ax.set_zlim([-show_range, show_range])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f'$p={p:.2f}$', fontsize=12)
    ax.view_init(elev=elev, azim=azim)


def show_multi_lp_sphere_and_x_path(show_ps, r_list, p_list, x_path, load_path=None,
                                    output_name="output/lp_sphere_and_x_path.pdf"):
    if load_path is not None:
        data = np.load(load_path)
        r_list = data["r_list"]
        p_list = data["p_list"]
        x_path = data["x_path"]

    assert len(show_ps) > 0, "show_ps must be a non-empty list"
    show_id = 0
    fig, axes = plt.subplots(1, len(show_ps), sharex=True, sharey=True, dpi=300, figsize=(12, 3),
                             subplot_kw={'projection': '3d'})
    for i, p in enumerate(p_list):
        if show_id < len(show_ps) and p <= show_ps[show_id]:
            # plot
            show_single_lp_sphere_and_x_path(axes[show_id], r_list[i], p, x_path[:i + 1])
            show_id += 1
        else:
            continue
    plt.tight_layout(pad=0.3)
    plt.savefig(output_name, format='pdf')
    plt.show()


if __name__ == "__main__":
    r_list, p_list, x_path = get_r_p_and_x_path(np.array([-1, 0.7, 0.5]), k=2)
    show_multi_lp_sphere_and_x_path(
        [2, 1.5, 1, 0.5, 0.2], r_list, p_list, x_path,
        output_name="figures/lp_sphere_and_x_path_k2.pdf")
    r_list, p_list, x_path = get_r_p_and_x_path(np.array([-1, 0.7, 0.5]), k=1)
    show_multi_lp_sphere_and_x_path(
        [2, 1.5, 1, 0.5, 0.2], r_list, p_list, x_path,
        output_name="figures/lp_sphere_and_x_path_k1.pdf")




