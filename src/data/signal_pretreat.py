# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)

"""
此文件为信号处理分析模块，用于将原始信号进行预处理
2026-03-26 刘加龙 南京 陆工大
"""

import sys
from pathlib import Path

# 计算项目根目录的绝对路径
# 向上三级目录：rVMD\src\data\signal_pretreat.py
project_root = Path(__file__).parent.parent.parent  # 注意层级数

# 将项目根目录添加到 sys.path 的开头
sys.path.insert(0, str(project_root))

# 导入必要的库
import os
import yaml
import numpy as np
import scipy as sp

from src.utils.bandpass_filter import bandpass_filter

# 加载配置文件
with open(os.path.join(project_root , "configs" , "signal_configs.yaml"),
          "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
#信号归一化函数
def normalize_signal(signal, method='max',flag=True):
    # 参数说明：
    # signal: 输入的一维信号数据，应该是一个 NumPy 数组或列表。
    # method: 归一化方法，默认为 'max'，还可以选择 'z-score','min-max'。
    # flag: 是否进行归一化，默认为True。

    if not flag:# 不进行归一化
        return signal # 如果 flag 为 False，直接返回原始信号
    
    if method == 'max':
        max_val = np.max(signal)
        if max_val == 0:
            return signal  # 避免除以零
        normalized = signal / max_val
    elif method == 'min-max':
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val - min_val == 0:
            return signal  # 避免除以零
        normalized = (signal - min_val) / (max_val - min_val)
    elif method == 'z-score':
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal  # 避免除以零
        normalized = (signal - mean) / std
    else:
        raise ValueError("Unsupported normalization method. Use 'min-max' or 'z-score'.")
    return normalized

#滤波函数
def filter_signal(signal,fs,low,high, flag=True):
    if not flag:
        return signal

    filtered_signal = bandpass_filter(signal, fs, low, high)

    return filtered_signal

# 重采样
def resample_signal(signal, new_fs,old_fs,flag=True):
    if not flag:
        return signal
    
    # 计算重采样因子
    resample_factor = new_fs / old_fs
    # 使用 scipy 的 resample 函数进行重采样
    resampled_signal = sp.signal.resample(signal, int(len(signal) * resample_factor))
    return resampled_signal

# 去噪函数
def denoise_signal(signal, flag=True):
    if not flag:
        return signal

    # 使用 scipy 的 wiener 函数进行去噪
    denoised_signal = sp.signal.wiener(signal)
    # # 去噪
    # denoised_signal, info = sym4_wavelet_denoise(signal)
    return denoised_signal

# 去除趋势项
def detrend_signal(signal, flag=True):
    if not flag:
        return signal

    # 使用 scipy 的 detrend 函数去除线性趋势
    # detrended_signal = sp.signal.detrend(signal)
    # 去除直流分量和缓慢趋势
    detrended_signal = signal - np.mean(signal)  # 去均值
    return detrended_signal

#信号预处理函数
def SignalPretreat(signal):
    # 归一化信号
    normal_signal = normalize_signal(signal, method='max') 
    
    # 滤波
    filtering_signal = filter_signal(normal_signal, 16000, 1000, 1150, flag=True)

    # 重采样
    resampled_signal = resample_signal(filtering_signal, 4000, 16000, True)
    
    # 去趋势
    detrended_signal = detrend_signal(resampled_signal, True)

    # 去噪
    denoised_signal = denoise_signal(detrended_signal, False) 

    return denoised_signal

