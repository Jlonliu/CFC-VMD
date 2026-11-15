# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)
"""

------------------
频率估计算法完全相同（限幅 -> fftshift 谱 -> 正半轴功率质心），
因此 omega 应与 cfc.CFC 逐位一致；差别仅在于本版把限幅副本也输出来。

    omega = cfc.CFC(u, fs)                  # 正式版，只要读数
    u_lim, u_hat, omega = cfc2.CFC2(u, fs)  # 实验版，连限幅模态一起给

返回值
------
u_lim : (K, T)
    限幅后的模态。real=False 时为单位模复解析信号 exp(j*phi)（推荐，
    负频率能量为零，谱图最干净）；real=True 时取其的实部 cos(phi)。
    低于阈值 0.05*max|z| 的样本相位不可靠，一律置零。
u_hat : (T, K), complex
    fftshift(np.fft.fft(u_lim, axis=1)).T，与 vmdpy 的 u_hat 约定一致。
omega : (1, K), Hz
    由 u_lim 算得的校准载波频率。取一维用 omega.ravel()。

Created: 2026-10-30 刘加龙
"""

import numpy as np
from scipy.signal import hilbert


def _limiter(u, thr_ratio=0.05):
    """解析信号 -> 单位模（=抹去调幅包络）。幅值过小处置零。"""
    z = hilbert(np.asarray(u, dtype=float))
    mag = np.abs(z)
    ok = mag > thr_ratio * mag.max()
    zc = np.zeros_like(z)
    zc[ok] = z[ok] / mag[ok]
    return zc


def CFC(u, fs, real=True):
    """
    u_lim, u_hat, omega = CFC(u, fs, real=False)
    """
    u = np.asarray(u)
    if u.ndim == 1:
        u = u[np.newaxis, :]
    K, T = u.shape

    # 与 vmdpy / cfc.py 相同的归一化频率栅格，已 fftshift 对齐
    freqs = np.arange(1, T + 1) / T - 0.5 - (1.0 / T)

    u_lim = np.zeros((K, T), dtype=float if real else complex)
    u_hat = np.zeros((T, K), dtype=complex)
    omega = np.zeros((1, K))

    for k in range(K):
        zc = _limiter(u[k])
        zc = zc.real if real else zc
        u_lim[k] = zc

        U = np.fft.fftshift(np.fft.fft(zc))
        u_hat[:, k] = U

        Ppos = (np.abs(U) ** 2)[T // 2:]
        fpos = freqs[T // 2:]
        den = Ppos.sum()
        omega[0, k] = float(np.dot(fpos, Ppos) / den) * fs if den else 0.0

    return u_lim, u_hat, omega
