# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)

# ============================================================================
# vmdpy - Python implementation of Variational Mode Decomposition (VMD)
#
# Original work:
#   Copyright (c) 2019 Vinícius Carvalho & Eduardo Mazoni
#   Author: Vinícius Rezende Carvalho (vrcarva@gmail.com)
#   Based on Dominique Zosso's MATLAB VMD implementation:
#     https://www.mathworks.com/matlabcentral/fileexchange/44765
#   Reference: Dragomiretskiy & Zosso (2014), IEEE Trans. Signal Process.,
#     62(3):531-544. doi:10.1109/TSP.2013.2288675
#   Licensed under the MIT License (see LICENSE.txt / vmdpy_LICENSE.txt).
#
# CFC-VMD modifications (carrier-frequency locating bias correction):
#   Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)
#
# CFC-VMD (variant 3) -- VMD with per-iteration limiter calibration FEEDBACK
# ----------------------------------------------------------------------------
# 每次迭代的执行顺序（与要求一致）：
#
#   1) 用上一次迭代得到的中心频率做维纳滤波，重建各模态；
#   2) 由重建模态按 VMD 原始公式求频谱质心 omega_orig
#      （omega_orig = Σ f|u_hat|² / Σ|u_hat|²，即 VMD 代价的精确极小点）；
#   3) 对重建模态做限幅 + 谱质心校准，得到校正中心频率 omega_lim；
#   4) 把 omega_lim 作为 **下一次迭代的滤波中心频率**（feedback=True，默认）；
#   5) 循环直到 uDiff < tol 或达到最大迭代次数。
#
# 收敛性与精度的分工：
#   - omega_orig 每一步仍按原 VMD 公式计算，用于把重建结果“锚”在 VMD 的
#     不动点上（它是 VMD 代价的极小点，保证迭代收缩）。feedback=False 时
#     滤波中心始终用 omega_orig，即退化为“只校准输出、不参与滤波”的老版本；
#   - omega_lim 承载偏差校正，feedback=True 时进入下一次迭代的滤波中心。
#
# 输出（最后一步）：
#   循环结束后，用 **最终校正中心频率** omega_lim 再做一次维纳滤波，
#   得到与之对应的模态 u。因此返回的 u 是在校正中心频率下滤波得到的结果，
#   与返回的 omega[-1, :] 严格对应。
# ============================================================================

"""
CFC-VMD method
2026.11.03 南京 陆工大 刘加龙 创建 (cfcvmd.py)
2026.11.09 南京 陆工大 刘加龙 修改 ->  (per-iteration limiter calibration)
2026.11.15 南京 陆工大 刘加龙 修改 ->  (calibration feedback + final calibrated filtering)
"""

import numpy as np
from scipy.signal import hilbert


def _limiter_center_frequency(u, freqs):
    """
    Carrier (center) frequency of one mode via limiter + spectral centroid.

    Parameters
    ----------
    u : 1D array
        A real single-component mode (one IMF).
    freqs : 1D array
        The normalized frequency grid (cycles per sample), DC-centered
        (i.e. fftshift order: index 0 -> -0.5, index L//2 -> 0, last -> 0.5).
        Its positive half is used for the centroid, matching vmdpy.VMD's
        ``freqs[T//2:T]`` convention.

    Returns
    -------
    omega : float
        Center-frequency estimate in cycles per sample (vmdpy's normalized
        convention, i.e. in [0, 0.5]).
    """
    z = hilbert(u)                        # analytic signal (scipy built-in)
    mag = np.abs(z)
    thr = 0.05 * np.max(mag)
    # limiter: unit-modulus FM signal; near-null samples are zeroed to avoid
    # dividing by a tiny magnitude (they carry no reliable phase anyway).
    zc = np.where(mag > thr, z / np.where(mag > thr, mag, 1.0), 0.0)

    Zc = np.fft.fftshift(np.fft.fft(zc))  # spectrum on the SAME grid as u
    P = np.abs(Zc) ** 2
    L = len(zc)                           # == len(u) == len(freqs)
    # positive-half centroid, matching vmdpy.VMD's freqs[T//2:T] convention
    fpos = freqs[L // 2:]
    Ppos = P[L // 2:]
    denom = np.sum(Ppos)
    if denom == 0:
        return 0.0
    return float(np.dot(fpos, Ppos) / denom)


def _clean_mode(u_hat_full, T, idxs):
    """
    Reconstruct the mirror-removed (clean) real mode from a full-grid spectrum
    ``u_hat_full`` (length T, DC-centered fftshift order), using the SAME
    mirror-removal recipe as VMD's postprocessing. This avoids the mirror
    boundary contaminating the limiter calibration inside the main loop.
    """
    u_hat = np.zeros(T, dtype=complex)
    u_hat[T // 2:T] = u_hat_full[T // 2:T]
    u_hat[idxs] = np.conj(u_hat_full[T // 2:T])
    u_hat[0] = np.conj(u_hat[-1])
    u = np.real(np.fft.ifft(np.fft.ifftshift(u_hat)))
    return u[T // 4:3 * T // 4]


def _cal_grid(L):
    """DC-centered normalized frequency grid of length L (cycles/sample)."""
    t = np.arange(1, L + 1) / L
    return t - 0.5 - (1.0 / L)


def CFCVMD(f, alpha, tau, K, DC, init, tol, feedback=True):
    """
    u, u_hat, omega = CFCVMD(f, alpha, tau, K, DC, init, tol, feedback=True)

    Variational mode decomposition with per-iteration limiter calibration and
    calibration feedback (cfcvmd3).

    每次迭代：维纳滤波 -> 原始质心 omega_orig -> 限幅校准 omega_lim
    -> omega_lim 作为下一次迭代的滤波中心（feedback=True）。
    循环结束后再用最终 omega_lim 做一次滤波，输出与之对应的 u。

    Parameters
    ----------
    feedback : bool
        True（默认）: 校正中心频率进入下一次迭代的滤波（本版新增行为）。
        False        : 滤波中心始终用原始质心，即“只校准输出、不反馈”的老版本
                       （仅用于对照实验）。

    Returns
    -------
    u       : [K, len(f)] 各模态时域波形（在最终校正中心频率下滤波得到）
    u_hat   : [len(f), K] 各模态频谱（由 u 重算）
    omega   : [Niter, K]  每次迭代的中心频率，omega[-1, :] 为最终校正值
    """
    if len(f) % 2:
        f = f[:-1]

    # Period and sampling frequency of input signal
    fs = 1. / len(f)

    ltemp = len(f) // 2
    fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
    fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T + 1) / T

    # Spectral Domain discretization
    freqs = t - 0.5 - (1 / T)

    # Maximum number of iterations
    Niter = 500
    Alpha = alpha * np.ones(K)

    # Construct and center f_hat
    f_hat = np.fft.fftshift(np.fft.fft(fMirr))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[:T // 2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])   # OUTPUT: per-iteration limiter-calibrated estimate
    omega_filt = np.zeros([Niter, K])   # filter center actually used by the Wiener step

    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K)))
    else:
        omega_plus[0, :] = 0
    omega_filt = omega_plus.copy()

    if DC:
        omega_plus[0, 0] = 0
        omega_filt[0, 0] = 0

    lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)

    # mirror-removal index (built once, reused inside the loop for the per
    # iteration limiter calibration on the CLEAN recovered mode)
    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)

    uDiff = tol + np.spacing(1)
    n = 0
    sum_uk = 0
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)

    # *** Main loop for iterative updates ***
    # 每一步：
    #   (a) 维纳滤波 —— 中心频率取 omega_filt[n, :]（feedback=True 时它就是
    #       上一轮校准得到的 omega_lim）；
    #   (b) 原始 VMD 质心 omega_orig —— VMD 代价的极小点，用于锚定收敛；
    #   (c) 限幅 + 谱质心校准 omega_lim —— 偏差校正后的载波频率；
    #   (d) omega_plus 记录 omega_lim；feedback=True 时 omega_filt 也取 omega_lim，
    #       否则取 omega_orig（老行为，仅作对照）。
    while (uDiff > tol and n < Niter - 1):
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]

        # Wiener filter of residuals, centered at the current filter omega
        u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2.) / (1. + Alpha[k] * (freqs - omega_filt[n, k]) ** 2)

        # ORIGINAL VMD update (anchors convergence) + limiter calibration (precision)
        if not (DC):
            omega_orig = np.dot(
                freqs[T // 2:T],
                np.abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2
            ) / np.sum(np.abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)
            u_clean = _clean_mode(u_hat_plus[n + 1, :, k], T, idxs)
            omega_lim = _limiter_center_frequency(u_clean, _cal_grid(len(u_clean)))
            omega_plus[n + 1, k] = omega_lim
            omega_filt[n + 1, k] = omega_lim if feedback else omega_orig

        # update of any other mode
        for k in np.arange(1, K):
            # accumulator
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            # mode spectrum, centered at the current filter omega
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2.) / (1 + Alpha[k] * (freqs - omega_filt[n, k]) ** 2)
            # ORIGINAL VMD update (anchors convergence) + limiter calibration (precision)
            omega_orig = np.dot(
                freqs[T // 2:T],
                np.abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2
            ) / np.sum(np.abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)
            u_clean = _clean_mode(u_hat_plus[n + 1, :, k], T, idxs)
            omega_lim = _limiter_center_frequency(u_clean, _cal_grid(len(u_clean)))
            omega_plus[n + 1, k] = omega_lim
            omega_filt[n + 1, k] = omega_lim if feedback else omega_orig

        # Dual ascent
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)

        # loop counter
        n = n + 1

        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1 / T) * np.dot((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                                             np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])))
        uDiff = np.abs(uDiff)

    # ------------------------------------------------------------------
    # Postprocessing and cleanup
    # ------------------------------------------------------------------
    Niter = np.min([Niter, n])
    omega = omega_plus[:Niter, :]

    # *** 最后一步：用最终校正中心频率 omega[-1, :] 再做一次维纳滤波 ***
    # 取与 omega[-1, :] 对应的那一轮迭代状态（u_hat_plus[Niter-1]）作为参考，
    # 只把滤波中心换成校正值，重建出与之匹配的模态。
    last = Niter - 1
    u_hat_cal = np.zeros([len(freqs), K], dtype=complex)
    sum_uk = np.sum(u_hat_plus[last, :, :], axis=1) - u_hat_plus[last, :, K - 1]
    for k in range(K):
        if k == 0:
            sum_uk = u_hat_plus[last, :, K - 1] + sum_uk - u_hat_plus[last, :, 0]
        else:
            sum_uk = u_hat_cal[:, k - 1] + sum_uk - u_hat_plus[last, :, k]
        u_hat_cal[:, k] = (f_hat_plus - sum_uk - lambda_hat[last, :] / 2.) / (
            1 + Alpha[k] * (freqs - omega_plus[last, k]) ** 2)

    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2:T, :] = u_hat_cal[T // 2:T, :]
    u_hat[idxs, :] = np.conj(u_hat_cal[T // 2:T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # remove mirror part
    u = u[:, T // 4:3 * T // 4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1], K], dtype=complex)
    for k in range(K):
        u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))

    return u, u_hat, omega