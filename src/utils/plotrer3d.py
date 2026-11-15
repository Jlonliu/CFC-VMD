# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(x, y, z, plot_type='scatter', title='3D Plot', 
            xlabel='X', ylabel='Y', zlabel='Z', 
            color='b', cmap='viridis', figsize=(8,6), 
            save_path=None, show=True):
    """
    绘制三维图形。

    参数:
        x, y, z : array_like
            数据点坐标。对于曲面图，x和y应为网格矩阵（如np.meshgrid输出）。
        plot_type : str, 可选 {'scatter', 'line', 'surface'}
            图形类型：
            - 'scatter' : 散点图
            - 'line'    : 三维折线图（按输入顺序连接点）
            - 'surface' : 曲面图（需要x和y为网格，z为对应高度）
        title : str, 图形标题
        xlabel, ylabel, zlabel : str, 坐标轴标签
        color : str 或 list, 散点/线条颜色（曲面图忽略）
        cmap : str, 曲面图的颜色映射（如'viridis', 'plasma'等）
        figsize : tuple, 图形尺寸
        save_path : str or None, 若提供则保存图片到该路径
        show : bool, 是否显示图形

    返回:
        fig, ax : matplotlib Figure 和 Axes3D 对象
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if plot_type == 'scatter':
        ax.scatter(x, y, z, c=color, marker='o', s=20, alpha=0.6)
    elif plot_type == 'line':
        ax.plot(x, y, z, color=color, linewidth=2, marker='o', markersize=3)
    elif plot_type == 'surface':
        # x, y 应为网格矩阵，z 为二维数组
        surf = ax.plot_surface(x, y, z, cmap=cmap, edgecolor='none', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)
    else:
        raise ValueError(f"不支持的 plot_type: {plot_type}，请使用 'scatter', 'line' 或 'surface'")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax

# x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# y = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# X, Y = np.meshgrid(x, y)
# # Z = np.exp(-(X**2 + Y**2)) * np.cos(2*np.pi*X)
# Z=np.linspace(0,1,100)
# Z=Z.reshape(10,10)

# plot_3d(X, Y, Z, plot_type='surface', title='指数衰减余弦曲面', 
#         xlabel='X', ylabel='Y', zlabel='Z', cmap='coolwarm')