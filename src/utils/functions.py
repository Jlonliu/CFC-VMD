# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)


import numpy as np
import os

def save_list_to_npy(lst, filename):
    """
    将一维列表保存为 NumPy 的 .npy 二进制文件。
    参数:
        lst: 一维列表（元素可为整数、浮点数等数值类型）
        filename: 保存的文件名（建议以 .npy 结尾）
    """
    # 将列表转换为 NumPy 数组（自动推断数据类型）
    arr = np.array(lst)
    # 保存为 .npy 文件
    np.save(filename, arr)
    print(f"列表已保存至 {filename}")

def load_list_from_npy(filename):
    """
    从 .npy 文件中读取数据并转换回 Python 列表。
    参数:
        filename: 要读取的 .npy 文件
    返回:
        一维列表
    """
    arr = np.load(filename)
    return arr

def dedup_keep_one(lst):
    seen = set()
    result = []
    for x in lst:
        if x in seen:
            result.append(0)
        else:
            seen.add(x)
            result.append(x)
    return result


def group_by_interval(x):
    """
    将 x 列表（0~1 的小数）按每 0.01 划分区间，返回每个非空区间的左端点和下标列表。

    参数:
        x: list of float, 值在 [0.0, 1.0] 范围内

    返回:
        x_new: list, 每个非空区间的左端点 (如 [0.0, 0.01, 0.02, ...])
        x_index: list of list, 对应每个非空区间的原 x 的下标列表
    """
    num_intervals = 100  # 0.00~0.99 共 100 个区间
    indices = [[] for _ in range(num_intervals)]

    for idx, val in enumerate(x):
        i = int(val * 100)
        if i < 0:
            i = 0
        elif i >= num_intervals:
            i = num_intervals - 1  # 将 1.0 归入最后一个区间 [0.99, 1.00]
        indices[i].append(idx)

    x_new = []
    x_index = []
    for i, lst in enumerate(indices):
        if lst:
            left_edge = i * 0.01
            x_new.append(round(left_edge, 2))
            x_index.append(lst)
    return x_new, x_index


def save_confidence_data(x, yvmd, yrvmd, civmd_lower, civmd_upper, cirvmd_lower, cirvmd_upper, save_dir):
    """
    将7个列表保存为.npy文件到指定目录。
    文件名固定为: x.npy, y1.npy, y2.npy, ci1_lower.npy, ci1_upper.npy, ci2_lower.npy, ci2_upper.npy
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    np.save(os.path.join(save_dir, 'x.npy'), np.array(x))
    np.save(os.path.join(save_dir, 'yvmd.npy'), np.array(yvmd))
    np.save(os.path.join(save_dir, 'yrvmd.npy'), np.array(yrvmd))
    np.save(os.path.join(save_dir, 'civmd_lower.npy'), np.array(civmd_lower))
    np.save(os.path.join(save_dir, 'civmd_upper.npy'), np.array(civmd_upper))
    np.save(os.path.join(save_dir, 'cirvmd_lower.npy'), np.array(cirvmd_lower))
    np.save(os.path.join(save_dir, 'cirvmd_upper.npy'), np.array(cirvmd_upper))
    print(f"数据已保存到: {save_dir}")

def load_confidence_data(load_dir):
    """
    从指定目录读取7个.npy文件，返回对应的7个列表。
    """
    x = np.load(os.path.join(load_dir, 'x.npy')).tolist()
    yvmd = np.load(os.path.join(load_dir, 'yvmd.npy')).tolist()
    yrvmd = np.load(os.path.join(load_dir, 'yrvmd.npy')).tolist()
    civmd_lower = np.load(os.path.join(load_dir, 'civmd_lower.npy')).tolist()
    civmd_upper = np.load(os.path.join(load_dir, 'civmd_upper.npy')).tolist()
    cirvmd_lower = np.load(os.path.join(load_dir, 'cirvmd_lower.npy')).tolist()
    cirvmd_upper = np.load(os.path.join(load_dir, 'cirvmd_upper.npy')).tolist()
    
    return x, yvmd, yrvmd, civmd_lower, civmd_upper, cirvmd_lower, cirvmd_upper


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=2000, ci=95, random_seed=42):
    """
    计算给定统计量的 Bootstrap 百分位数置信区间。

    参数
    ----------
    data : array_like
        原始样本数据（例如多次实验得到的 NMSE 值列表）。
    statistic : callable, 默认 np.mean
        要估计的统计量函数，例如 np.mean, np.median, np.var 等。
    n_bootstrap : int, 默认 2000
        重采样次数（B 值），越大结果越稳定，建议至少 2000。
    ci : float, 默认 95
        置信水平，介于 0 到 100 之间，如 95 表示 95% 置信区间。
    random_seed : int or None, 默认 None
        随机种子，便于结果复现。
    plot : bool, 默认 False
        是否绘制 Bootstrap 统计量的直方图及置信区间标记。

    返回
    -------
    (lower, upper) : tuple
        置信区间的下界和上界。
    """
    # 设置随机种子（可选）
    if random_seed is not None:
        np.random.seed(random_seed)

    data = np.asarray(data)
    n = len(data)
    if n == 0:
        raise ValueError("输入数据不能为空")

    # 存储每次 Bootstrap 样本计算出的统计量
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # 有放回地重采样，得到与原始数据相同大小的样本
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        # 计算该样本的目标统计量
        boot_stats[i] = statistic(bootstrap_sample)

    # 计算百分位数对应的下标
    alpha = (100 - ci) / 100
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(boot_stats, lower_percentile)
    upper = np.percentile(boot_stats, upper_percentile)



    return lower, upper


def trim_lists(data):
    if not data:
        return []
    min_len = min(len(row) for row in data)
    return [row[:min_len] for row in data]
