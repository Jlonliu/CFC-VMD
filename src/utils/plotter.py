# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)


"""
通用可视化绘图库
支持多种布局：单图、横排、竖排、多线、四宫、六宫、八宫、品字图
全局样式一次设定，所有子图复用
新增：支持置信区间绘制（ci参数）
"""

"""
色盲友好配色
2	High-contrast (来自Paul Tol)	[#4477AA, #EE6677]
3	User-Friendly (来自ColorAny)	[#0072B8, #F6D02F, #D61A24]
4	BlindSafe (来自R包 jumble)	[#AA4499, #88CCEE, #999933, #DDDDDD]
5	Okabe-Ito	[#000000, #E69F00, #56B4E9, #009E73, #F0E442]
"""


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional, Union, Dict, Any
import warnings

# ======================== 全局样式配置 ========================
class PlotStyle:
    """全局样式配置，一次设定全局复用"""

    def __init__(
        self,
        font_family: str = ['SimSun', 'Times New Roman'],# 字体（中文用SimHei,英文新罗马）
        
        # font_family: str = ['Times New Roman'],# 字体
        title_size: int = 6,                  # 标题字号
        label_size: int = 6,                  # 轴标签字号
        tick_size: int = 6,                   # 刻度字号
        legend_size: int = 6,                 # 图例字号
        title_weight: str = "bold",             # 标题粗细
        label_weight: str = "normal",           # 轴标签粗细
        figure_dpi: int = 300,                 # 图像DPI
        bg_color: str = "#FFFFFF",             # 图像背景色
        grid_color: str = "#CCCCCC",            # 网格颜色
        grid_alpha: float = 0.5,               # 网格透明度
        show_grid: bool = False,                 # 是否显示网格
        vline_label_size: int = 6,            # 竖线标注字号 (新增)
    ):
        self.font_family = font_family
        self.title_size = title_size
        self.label_size = label_size
        self.tick_size = tick_size
        self.legend_size = legend_size
        self.title_weight = title_weight
        self.label_weight = label_weight
        self.figure_dpi = figure_dpi
        self.bg_color = bg_color
        self.grid_color = grid_color
        self.grid_alpha = grid_alpha
        self.show_grid = show_grid
        self.vline_label_size = vline_label_size

        # 设置matplotlib全局字体
        plt.rcParams["font.family"] = self.font_family
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        plt.rcParams['mathtext.fontset'] = 'stix'  # 启用 STIX 数学字体

        plt.rcParams['axes.linewidth'] = 0.25   # 设置边框默认线宽
        plt.rcParams['xtick.major.width'] = 0.25
        plt.rcParams['ytick.major.width'] = 0.25
        plt.rcParams['lines.markersize'] = 1

# ======================== 数据结构 ========================

class PlotData:
    """单条曲线/单个子图的数据，支持置信区间"""

    def __init__(
        self,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray, List[List]],
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        linestyles: Optional[List[str]] = None,
        bar_linestyles: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        linewidth: float = 0.25,
        alpha: float = 1,
        vertical_lines: Optional[List[Tuple[float, str]]] = None,
        legend_loc: Optional[str] = None,
        ci: Optional[Union[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,   # 新增：置信区间
    ):
        self.x = np.array(x)
        self.y = np.array(y) if not isinstance(y, list) else [np.array(yy) for yy in y]
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.labels = labels
        self.colors = colors
        self.linestyles = linestyles
        self.bar_linestyles = bar_linestyles
        self.markers = markers
        self.linewidth = linewidth
        self.alpha = alpha
        self.vertical_lines = vertical_lines
        self.legend_loc = legend_loc
        self.ci = ci


        # 处理置信区间格式
        if ci is not None:
            # 如果是列表，表示多条线的置信区间
            if isinstance(ci, list):
                self.ci_list = []
                for single_ci in ci:
                    if isinstance(single_ci, tuple) and len(single_ci) == 2:
                        lower = np.array(single_ci[0])
                        upper = np.array(single_ci[1])
                    elif isinstance(single_ci, np.ndarray) and single_ci.shape[0] == 2:
                        lower = single_ci[0]
                        upper = single_ci[1]
                    else:
                        raise ValueError("Each ci element must be tuple (lower, upper) or 2xn array")
                    if len(lower) != len(self.x) or len(upper) != len(self.x):
                        raise ValueError("ci lower/upper must have same length as x")
                    self.ci_list.append((lower, upper))
            else:
                # 单条线的 ci 处理（原有逻辑）
                if isinstance(ci, tuple) and len(ci) == 2:
                    self.ci_lower = np.array(ci[0])
                    self.ci_upper = np.array(ci[1])
                elif isinstance(ci, np.ndarray) and ci.shape[0] == 2:
                    self.ci_lower = ci[0]
                    self.ci_upper = ci[1]
                else:
                    raise ValueError("ci must be tuple (lower, upper) or 2xn numpy array")
                if len(self.ci_lower) != len(self.x) or len(self.ci_upper) != len(self.x):
                    raise ValueError("ci lower/upper must have same length as x")

# ======================== 核心绘图类 ========================

class Plotter:
    """通用绘图器"""

    # 默认颜色循环
    DEFAULT_COLORS = [
        "#0026FF", "#F0C040", "#E74C3C", "#00FF6A",
        "#C528D0", "#9B59B6", "#E67E22", "#00BFFF",
    ]

    # 默认线型循环
    DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]

    # 默认标记循环
    DEFAULT_MARKERS = ["", "o", "s", "^", "D", "v", "<", ">"]

    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = style if style is not None else PlotStyle()

    # ──────────────────── 内部工具 ────────────────────
    def _apply_style(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str):
        """将全局样式应用到单个axes"""
        ax.set_title(title, fontsize=self.style.title_size, fontweight=self.style.title_weight, y=-0.4, loc='center')
        ax.set_xlabel(xlabel, fontsize=self.style.label_size, fontweight=self.style.label_weight)
        ax.set_ylabel(ylabel, fontsize=self.style.label_size, fontweight=self.style.label_weight)
        ax.tick_params(axis="both", labelsize=self.style.tick_size)
        ax.set_facecolor(self.style.bg_color)

        if self.style.show_grid:
            ax.grid(True, color=self.style.grid_color, alpha=self.style.grid_alpha, linestyle="--")

    def _plot_one_axes(self, ax: plt.Axes, data: PlotData):
        """在一个axes上绘图（支持单条/多条线，支持置信区间），并自动添加红色虚线竖线"""
        y_data = data.y

        # 多条线：y是list of array
        if isinstance(y_data, list):
            n_lines = len(y_data)
            # 判断是否每条线都有对应的 ci
            has_ci_list = hasattr(data, 'ci_list') and data.ci_list is not None and len(data.ci_list) == n_lines
            for i in range(n_lines):
                color = data.colors[i] if data.colors and i < len(data.colors) else self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
                ls = data.linestyles[i] if data.linestyles and i < len(data.linestyles) else self.DEFAULT_LINESTYLES[i % len(self.DEFAULT_LINESTYLES)]
                bar_ls = data.bar_linestyles[i] if data.bar_linestyles and i < len(data.bar_linestyles) else self.DEFAULT_LINESTYLES[i % len(self.DEFAULT_LINESTYLES)]
                marker = data.markers[i] if data.markers and i < len(data.markers) else self.DEFAULT_MARKERS[i % len(self.DEFAULT_MARKERS)]
                label = data.labels[i] if data.labels and i < len(data.labels) else None

                if has_ci_list:
                    # 手动绘制工字型置信区间：每条线独立绘制，zorder 控制叠放顺序
                    # 先画误差棒（zorder=i），再画主线（zorder=i+10），主线总在误差棒上方
                    lower, upper = data.ci_list[i]
                    y_vals = y_data[i]
                    if np.any(lower > y_vals) or np.any(upper < y_vals):
                        warnings.warn(f"曲线 {label or i} 的置信区间与点估计不一致，请检查数据。")

                    zorder_bar = float(i)        # 误差棒 zorder
                    zorder_line = float(i + 10)  # 主线 zorder（总在误差棒上方）
                    bar_linewidth = 0.25

                    # 计算帽子宽度（相对 x 范围的 0.5%，保证在不同尺度下视觉一致）
                    x_range = data.x.max() - data.x.min()
                    cap_w = x_range * 0.008 if x_range > 0 else 0.1

                    for xi, yi, li, ui in zip(data.x, y_vals, lower, upper):
                        # 竖线段：从 lower 到 upper
                        ax.plot([xi, xi], [li, ui],
                                color=color, linewidth=bar_linewidth,linestyle=bar_ls,
                                alpha=data.alpha, zorder=zorder_bar)
                        # 上帽
                        ax.plot([xi - cap_w, xi + cap_w], [ui, ui],
                                color=color, linewidth=bar_linewidth,
                                alpha=data.alpha, zorder=zorder_bar)
                        # 下帽
                        ax.plot([xi - cap_w, xi + cap_w], [li, li],
                                color=color, linewidth=bar_linewidth,
                                alpha=data.alpha, zorder=zorder_bar)

                    # 画主线和标记点
                    ax.plot(
                        data.x, y_vals,
                        color=color,
                        linestyle=ls,
                        marker=marker if marker != "" else None,
                        linewidth=data.linewidth,
                        alpha=data.alpha,
                        label=label,
                        zorder=zorder_line,
                    )
                else:
                    # 原始 plot 逻辑
                    ax.plot(
                        data.x, y_data[i],
                        color=color, linestyle=ls, marker=marker,
                        linewidth=data.linewidth, alpha=data.alpha,
                        label=label,
                    )
            # 处理图例
            if data.labels is not None and any(l is not None for l in data.labels):
                ax.legend(fontsize=self.style.legend_size, loc=data.legend_loc)
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
        else:
            # 单条线
            if data.ci is not None:
                # 手动绘制工字型置信区间，完全独立控制
                color = data.colors[0] if data.colors else self.DEFAULT_COLORS[0]
                ls = data.linestyles[0] if data.linestyles else "-"
                marker = data.markers[0] if data.markers else ""
                label = data.labels[0] if data.labels else None
                if np.any(data.ci_lower > data.y) or np.any(data.ci_upper < data.y):
                    warnings.warn(f"曲线 {label or ''} 的置信区间与点估计不一致，请检查数据。")

                zorder_bar = 0.0
                zorder_line = 10.0
                bar_linewidth = 1.0
                x_range = data.x.max() - data.x.min()
                cap_w = x_range * 0.008 if x_range > 0 else 0.1

                for xi, yi, li, ui in zip(data.x, data.y, data.ci_lower, data.ci_upper):
                    # 竖线段
                    ax.plot([xi, xi], [li, ui],
                            color=color, linewidth=bar_linewidth,
                            alpha=data.alpha, zorder=zorder_bar)
                    # 帽子
                    ax.plot([xi - cap_w, xi + cap_w], [ui, ui],
                            color=color, linewidth=bar_linewidth,
                            alpha=data.alpha, zorder=zorder_bar)
                    ax.plot([xi - cap_w, xi + cap_w], [li, li],
                            color=color, linewidth=bar_linewidth,
                            alpha=data.alpha, zorder=zorder_bar)

                ax.plot(
                    data.x, data.y,
                    color=color,
                    linestyle=ls,
                    marker=marker if marker != "" else None,
                    linewidth=data.linewidth,
                    alpha=data.alpha,
                    label=label,
                    zorder=zorder_line,
                )
                if label:
                    ax.legend(fontsize=self.style.legend_size, loc=data.legend_loc)
            else:
                # 原始 plot 逻辑
                color = data.colors[0] if data.colors else self.DEFAULT_COLORS[0]
                ls = data.linestyles[0] if data.linestyles else "-"
                marker = data.markers[0] if data.markers else ""
                label = data.labels[0] if data.labels else None
                ax.plot(
                    data.x, data.y,
                    color=color, linestyle=ls, marker=marker,
                    linewidth=data.linewidth, alpha=data.alpha,
                    label=label,
                )
                if label:
                    ax.legend(fontsize=self.style.legend_size, loc=data.legend_loc)

        self._apply_style(ax, data.title, data.xlabel, data.ylabel)

        # ========== 绘制红色虚线竖线及标注 ==========
        if hasattr(data, 'vertical_lines') and data.vertical_lines:
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            i = 0
            color_list = [ "#EE6677","#4477AA"]
            for x_val, label in data.vertical_lines:
                i += 1

                if i == 0:
                    hai = 'right'
                    colori = 'green'
                    ylabeli = y_max * 0.90
                    xlabeli = x_max * 0.98
                    alphai = 1
                elif i == 1:
                    hai = 'left'
                    colori = color_list[0]
                    ylabeli = y_max * (1 - 0.2 * i)
                    xlabeli = x_val + 1
                    alphai = 0
                else:
                    hai = 'right'
                    colori = color_list[1]
                    ylabeli = y_max * (1 - 0.2 * i)
                    xlabeli = x_val
                    alphai = 0
                if i != 0:
                    ax.axvline(x=x_val, color=colori, linestyle='--', linewidth=0.25, alpha=0.8)
                ax.text(xlabeli, ylabeli, label,
                        rotation=0, ha=hai, va='top',
                        color=colori, fontsize=self.style.vline_label_size,#bbox=dict(boxstyle='round,pad=0.3',facecolor='white', alpha=alphai)
                        )
                
    # ──────────────────── 1. 单图 ────────────────────
    def single(
        self,
        data: PlotData,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """单图"""
        fig, ax = plt.subplots(figsize=figsize, dpi=self.style.figure_dpi)
        self._plot_one_axes(ax, data)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ──────────────────── 2. 横向并列多图 ────────────────────
    def horizontal(
        self,
        datas: List[PlotData],
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """横向并列（1 × N）"""
        n = len(datas)
        fig, axes = plt.subplots(1, n, figsize=figsize, dpi=self.style.figure_dpi)
        if n == 1:
            axes = [axes]
        for ax, d in zip(axes, datas):
            self._plot_one_axes(ax, d)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ──────────────────── 3. 纵向叠加多图 ────────────────────
    def vertical(
        self,
        datas: List[PlotData],
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """纵向叠加（N × 1）"""
        n = len(datas)
        fig, axes = plt.subplots(n, 1, figsize=figsize, dpi=self.style.figure_dpi)
        if n == 1:
            axes = [axes]
        for ax, d in zip(axes, datas):
            self._plot_one_axes(ax, d)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ──────────────────── 4. 四宫图（2×2） ────────────────────
    def grid_2x2(
        self,
        datas: List[PlotData],
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """四宫图（2×2），datas长度必须为4"""
        assert len(datas) == 4, "四宫图需要恰好4组数据"
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=self.style.figure_dpi)
        for idx, d in enumerate(datas):
            ax = axes[idx // 2][idx % 2]
            self._plot_one_axes(ax, d)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig
    
    def grid_4x1(
        self,
        datas: List[PlotData],
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """四宫图（4×1），datas长度必须为4"""
        assert len(datas) == 4, "四宫图需要恰好4组数据"
        fig, axes = plt.subplots(4, 1, figsize=figsize, dpi=self.style.figure_dpi)
        for idx, d in enumerate(datas):
            ax = axes[idx]
            self._plot_one_axes(ax, d)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ──────────────────── 5. 六宫图（2×3） ────────────────────
    def grid_2x3(
        self,
        datas: List[PlotData],
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """六宫图（2×3），datas长度必须为6"""
        assert len(datas) == 6, "六宫图需要恰好6组数据"
        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=self.style.figure_dpi)
        for idx, d in enumerate(datas):
            ax = axes[idx // 3][idx % 3]
            self._plot_one_axes(ax, d)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def grid_3x2(
        self,
        datas: List[PlotData],
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """六宫图（3×2），datas长度必须为6"""
        assert len(datas) == 6, "六宫图需要恰好6组数据"
        fig, axes = plt.subplots(3, 2, figsize=figsize, dpi=self.style.figure_dpi)
        for idx, d in enumerate(datas):
            ax = axes[idx // 2][idx % 2]
            self._plot_one_axes(ax, d)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ──────────────────── 6. 八宫图（2×4） ────────────────────
    def grid_2x4(
        self,
        datas: List[PlotData],
        figsize: Tuple[int, int] = (18, 8),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """八宫图（2×4），datas长度必须为8"""
        assert len(datas) == 8, "八宫图需要恰好8组数据"
        fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=self.style.figure_dpi)
        for idx, d in enumerate(datas):
            ax = axes[idx // 4][idx % 4]
            self._plot_one_axes(ax, d)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def grid_4x2(
        self,
        datas: List[PlotData],
        figsize: Tuple[int, int] = (12, 12),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """八宫图（4×2），datas长度必须为8"""
        assert len(datas) == 8, "八宮图需要恰好8组数据"
        fig, axes = plt.subplots(4, 2, figsize=figsize, dpi=self.style.figure_dpi)
        for idx, d in enumerate(datas):
            ax = axes[idx // 2][idx % 2]
            self._plot_one_axes(ax, d)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ──────────────────── 7. 品字图 ────────────────────
    def pinzi(
        self,
        top_data: PlotData,
        bottom_left: PlotData,
        bottom_right: PlotData,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        品字图：
           ┌───────────┐
           │    top     │
           ├─────────┬─┤
           │  bottom  │ │
           │   left   │right│
           └─────────┴─┘
        上面1个大图，下面2个并排
        """
        fig = plt.figure(figsize=figsize, dpi=self.style.figure_dpi)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        ax_top = fig.add_subplot(gs[0, :])
        self._plot_one_axes(ax_top, top_data)

        ax_bl = fig.add_subplot(gs[1, 0])
        self._plot_one_axes(ax_bl, bottom_left)

        ax_br = fig.add_subplot(gs[1, 1])
        self._plot_one_axes(ax_br, bottom_right)

        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ──────────────────── 8. 通用网格 ────────────────────
    def grid(
        self,
        datas: List[PlotData],
        nrow: int,
        ncol: int,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """通用 nrow × ncol 网格图"""
        assert len(datas) <= nrow * ncol, f"数据组数({len(datas)})超过网格容量({nrow*ncol})"
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, dpi=self.style.figure_dpi)
        # 处理1维情况
        if nrow == 1 and ncol == 1:
            axes = [[axes]]
        elif nrow == 1:
            axes = [axes]
        elif ncol == 1:
            axes = [[ax] for ax in axes]
        else:
            axes = list(axes)

        for idx, d in enumerate(datas):
            r, c = divmod(idx, ncol)
            if nrow == 1:
                ax = axes[c]
            elif ncol == 1:
                ax = axes[idx][0]
            else:
                ax = axes[r][c]
            self._plot_one_axes(ax, d)

        # 隐藏多余子图
        total_slots = nrow * ncol
        if total_slots > len(datas):
            for idx in range(len(datas), total_slots):
                r, c = divmod(idx, ncol)
                if nrow == 1:
                    ax = axes[c]
                elif ncol == 1:
                    ax = axes[idx][0]
                else:
                    ax = axes[r][c]
                ax.set_visible(False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.style.figure_dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

# ======================== 快速使用入口 ========================
def quick_plot(
    x,
    y,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """一行代码快速画图（单图）"""
    style = PlotStyle()
    plotter = Plotter(style)
    data = PlotData(x=x, y=y, title=title, xlabel=xlabel, ylabel=ylabel)
    return plotter.single(data, figsize=figsize, save_path=save_path, show=show)


# ======================== 使用示例 ========================
if __name__ == "__main__":
    # ── 生成示例数据 ──
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.exp(-0.1 * x)
    y4 = np.cos(x) * np.exp(-0.1 * x)
    y5 = np.sin(2 * x)
    y6 = np.cos(2 * x)
    y7 = np.sin(0.5 * x)
    y8 = np.cos(0.5 * x)

    # ── 设置全局样式（一次设定，后面全部复用）──
    my_style = PlotStyle()
    pt = Plotter(my_style)

    # ── 1. 单图（多条线）──
    data_multi = PlotData(
        x=x, y=[y1, y2, y3],
        title="单图-多线示例",
        xlabel="时间 (s)", ylabel="幅值",
        labels=["sin(x)", "cos(x)", "衰减sin"],
        colors=["#1A1F3C", "#F0C040", "#E74C3C"],
        linestyles=["-", "--", "-."],
        markers=["", "o", ""],
    )
    pt.single(data_multi, figsize=(10, 4), show=True)

    # ── 2. 横向并列 ──
    datas_h = [
        PlotData(x, y1, "子图1", "X", "Y1"),
        PlotData(x, y2, "Wiener-tanh", "X", "Y2"),
        PlotData(x, y3, "子图3", "X", "Y3"),
    ]
    pt.horizontal(datas_h, figsize=(10, 4), show=True)

    # ── 3. 纵向叠加 ──
    datas_v = [
        PlotData(x, y1, "子图1", "", "Y1"),
        PlotData(x, y2, "子图2", "", "Y2"),
        PlotData(x, y3, "子图3", "X", "Y3"),
    ]
    pt.vertical(datas_v, figsize=(10, 8), show=True)

    # ── 4. 四宫图 ──
    datas_4 = [
        PlotData(x, y1, "四宫-图1", "X", "Y"),
        PlotData(x, y2, "四宫-图2", "X", "Y"),
        PlotData(x, y3, "四宫-图3", "X", "Y"),
        PlotData(x, y4, "四宫-图4", "X", "Y"),
    ]
    pt.grid_2x2(datas_4, figsize=(10, 8), show=True)

    # ── 5. 六宫图（2×3）──
    datas_6 = [
        PlotData(x, y1, "图1"), PlotData(x, y2, "图2"), PlotData(x, y3, "图3"),
        PlotData(x, y4, "图4"), PlotData(x, y5, "图5"), PlotData(x, y6, "图6"),
    ]
    pt.grid_2x3(datas_6, figsize=(10, 8), show=True)

    # ── 6. 八宫图（2×4）──
    datas_8 = [
        PlotData(x, y1, "图1"), PlotData(x, y2, "图2"), PlotData(x, y3, "图3"), PlotData(x, y4, "图4"),
        PlotData(x, y5, "图5"), PlotData(x, y6, "图6"), PlotData(x, y7, "图7"), PlotData(x, y8, "图8"),
    ]
    pt.grid_2x4(datas_8, figsize=(10, 8), show=True)

    # ── 7. 品字图 ──
    top = PlotData(x, [y1, y2], "品字-上方大图", "X", "Y", labels=["sin", "cos"])
    bl = PlotData(x, y3, "品字-左下", "X", "Y")
    br = PlotData(x, y4, "品字-右下", "X", "Y")
    pt.pinzi(top, bl, br, figsize=(10, 8), show=True)

    # ── 8. 带置信区间的示例（新功能）──
    grades = np.arange(1, 51)
    success_rates = 0.98 - 0.46 * (grades-1)/49
    ci_lower = success_rates - 0.05
    ci_upper = success_rates + 0.05
    data_ci = PlotData(
        x=grades,
        y=success_rates,
        ci=(ci_lower, ci_upper),
        title="带置信区间的示例",
        xlabel="Grade",
        ylabel="Success Rate",
        labels=["Success rate"]
    )
    pt.single(data_ci, figsize=(12, 6), show=True)