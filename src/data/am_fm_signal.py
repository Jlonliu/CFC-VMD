# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)

"""
AM/FM 调幅调频信号生成器（带相位版本）
===============================
提供 gen_signal() 统一函数接口，支持：
  - AM  (调幅)
  - FM  (调频)
  - AM_FM (调幅调频联合调制)

新增：am_phases / fm_phases 参数，完全向后兼容。
返回: (t, signal) 元组
"""

import numpy as np


def _as_mod_index(val, n, name):
    """
    标量 -> 广播为长度 n 的数组；列表 -> 校验长度。

    返回 (arr, is_scalar)。is_scalar 用于决定 AM 是否沿用旧的 /N 归一化。
    """
    arr = np.atleast_1d(np.asarray(val, dtype=float))
    if arr.size == 1:
        return np.full(n, arr.item()), True
    if arr.size != n:
        raise ValueError(
            f"{name} 的长度 {arr.size} 与调制频率个数 {n} 不一致"
        )
    return arr, False


def gen_signal(
    freqs,            # 载波频率列表 [Hz]，例: [50, 100]
    mode='AM',        # 'AM' | 'FM' | 'AM_FM'
    # --- 幅值 ---
    amps=None,        # 各载波幅值，默认全1，与freqs等长
    dc=0.0,           # 直流偏置
    # --- 采样 ---
    fs=1000,          # 采样频率 [Hz]
    duration=1.0,     # 采样时长 [s]
    # --- 调幅 ---
    m=0.5,            # 调幅深度：标量（广播）或列表（逐音 m_h）
    fm_freqs=None,    # 调幅调制频率列表，默认[1.0]
    am_phases=None,   # 调幅各路相位 [rad]，与fm_freqs等长，默认全0
    # --- 调频 ---
    beta=0.2,         # 调频系数：标量（广播）或列表（逐音 beta_i）
    fm_mod_freqs=None,# 调频调制频率列表，默认[5.0]
    fm_phases=None,   # 调频各路相位 [rad]，与fm_mod_freqs等长，默认全0
    # --- 噪声 ---
    noise_std=0.0,    # 高斯噪声标准差，0则无噪声
    noise_seed=None,  # 随机种子（None则不复现）
):
    """
    生成 AM / FM / AM_FM 复合调制信号。

    信号模型
    --------
    单载波多频 AM-FM：
        x(t) = A0 [1 + Σ_h m_h cos(2π f_{a,h} t + φ_h)]
                 · cos[2π f_c t + Σ_i β_i sin(2π f_{f,i} t + θ_i)]

    m / beta 的标量-列表语义
    -----------------------
    - m 为标量：广播到所有 AM 音，并沿用旧版的 /N 归一化（保证向后兼容）
    - m 为列表：每个 AM 音使用自己的 m_h，**不做归一化**
    - beta 为标量：广播到所有 FM 音（与原接口行为一致）
    - beta 为列表：每个 FM 音使用自己的 β_i

    Parameters
    ----------
    freqs : array-like
        载波频率列表 [Hz]，多载波按幅值加权求和。
    mode : str
        'AM' | 'FM' | 'AM_FM'。
    amps : array-like, optional
        各载波幅值，默认全 1。
    dc : float
        直流分量。
    fs : float
        采样频率 [Hz]。
    duration : float
        信号时长 [s]。
    m : float or array-like
        调幅深度。列表长度须等于 len(fm_freqs)。
    fm_freqs : array-like, optional
        调幅（AM）调制频率列表，默认 [1.0]。
    am_phases : array-like, optional
        AM 各路初相 [rad]，与 fm_freqs 等长，默认全 0。
    beta : float or array-like
        调频指数。列表长度须等于 len(fm_mod_freqs)。
    fm_mod_freqs : array-like, optional
        调频（FM）调制频率列表，默认 [5.0]。
    fm_phases : array-like, optional
        FM 各路初相 [rad]，与 fm_mod_freqs 等长，默认全 0。
    noise_std : float
        噪声标准差。
    noise_seed : int, optional
        随机种子。

    Returns
    -------
    t : ndarray
    signal : ndarray
    """
    t = np.arange(0, duration, 1.0 / fs)

    amps = (np.ones(len(freqs)) if amps is None
            else np.array(amps, dtype=float))
    freqs = np.array(freqs, dtype=float)

    # --- AM ---
    if np.all(np.asarray(m, dtype=float) == 0) or mode == 'FM':
        am_component = 1.0
    else:
        fm_arr = np.array([1.0] if fm_freqs is None else fm_freqs,
                          dtype=float)
        if am_phases is None:
            am_phases = np.zeros(len(fm_arr))
        am_phases = np.array(am_phases, dtype=float)
        m_arr, m_is_scalar = _as_mod_index(m, len(fm_arr), 'm')

        mod = sum(mh * np.cos(2 * np.pi * fa * t + phi)
                  for mh, fa, phi in zip(m_arr, fm_arr, am_phases))
        if m_is_scalar:      # 沿用旧版归一化，保证向后兼容
            mod = mod / max(len(fm_arr), 1.0)
        am_component = 1.0 + mod

    # --- FM ---
    # 所有 FM 音累加在同一个载波相位里（多载波时各载波共享同一相位调制）
    if mode in ('FM', 'AM_FM'):
        fm_mod_arr = np.array([5.0] if fm_mod_freqs is None else fm_mod_freqs,
                              dtype=float)
        if fm_phases is None:
            fm_phases = np.zeros(len(fm_mod_arr))
        fm_phases = np.array(fm_phases, dtype=float)
        beta_arr, _ = _as_mod_index(beta, len(fm_mod_arr), 'beta')

        fm_phase = sum(bi * np.sin(2 * np.pi * fmod * t + theta)
                       for bi, fmod, theta
                       in zip(beta_arr, fm_mod_arr, fm_phases))
        carrier = sum(a * np.cos(2 * np.pi * f * t + fm_phase)
                      for a, f in zip(amps, freqs))
    else:
        carrier = sum(a * np.cos(2 * np.pi * f * t)
                      for a, f in zip(amps, freqs))

    signal = carrier * am_component + dc

    if noise_std > 0:
        rng = np.random.default_rng(noise_seed)
        signal = signal + rng.normal(0, noise_std, size=signal.shape)

    return t, signal

# ============================================================
# 快捷函数（新增可选相位参数，默认0保持兼容）
# ============================================================

def gen_am_signal(
    fc=100, fm=5, m=0.5, am_phase=0.0,
    fs=2000, duration=2.0,
    noise_std=0.0, seed=None
):
    """快捷函数：单载波调幅信号。新增 am_phase [rad]，默认 0。"""
    return gen_signal(
        freqs=[fc], mode='AM', m=m,
        fm_freqs=[fm], am_phases=[am_phase],
        fs=fs, duration=duration,
        noise_std=noise_std, noise_seed=seed,
    )


def gen_fm_signal(
    fc=100, fm=5, beta=0.2, fm_phase=0.0,
    fs=2000, duration=2.0,
    noise_std=0.0, seed=None
):
    """快捷函数：单载波调频信号。新增 fm_phase [rad]，默认 0。"""
    return gen_signal(
        freqs=[fc], mode='FM', beta=beta,
        fm_mod_freqs=[fm], fm_phases=[fm_phase],
        fs=fs, duration=duration,
        noise_std=noise_std, noise_seed=seed,
    )


def gen_amfm_signal(
    fc=100, fm_am=5, fm_fm=10, m=0.5, beta=0.2,
    am_phase=0.0, fm_phase=0.0,
    fs=2000, duration=2.0,
    noise_std=0.0, seed=None
):
    """快捷函数：单载波AM-FM联合信号。新增 am_phase/fm_phase [rad]，默认 0。"""
    return gen_signal(
        freqs=[fc], mode='AM_FM', m=m, beta=beta,
        fm_freqs=[fm_am], am_phases=[am_phase],
        fm_mod_freqs=[fm_fm], fm_phases=[fm_phase],
        fs=fs, duration=duration,
        noise_std=noise_std, noise_seed=seed,
    )


# ============================================================
# 演示
# ============================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t, am = gen_signal(
        freqs=[50, 100], mode='AM', m=0.6,
        fm_freqs=[5, 10],
        fs=2000, duration=1.0,
        noise_std=0.05, noise_seed=42,
    )

    plt.figure(figsize=(12, 4))
    plt.plot(t, am)
    plt.title('AM Signal (50Hz + 100Hz carrier, modulated by 5Hz + 10Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
