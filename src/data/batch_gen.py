# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)

"""
batch_gen.py - 批量生成 AM-FM 调制信号（带相位支持）
===================================================
参数可为元组(随机范围)或单值(固定值)。

新增 am_phases / fm_phases 参数，完全向后兼容。

用法示例:
    from batch_gen import batch_gen

    # 全部随机范围（相位默认全0，与原版行为一致）
    signals, components, params = batch_gen(
        K=(2, 5),              # 模态数量范围，随机整数
        base_freqs=(50, 300),  # 基本频率范围 [Hz]
        freq_intervals=(30, 100),  # 频率间隔范围 [Hz]
        mod_freqs=(5, 25),     # 调制频率范围 [Hz]
        am_coeffs=(0.1, 0.9),  # 调幅系数范围
        mod_depths=(0.1, 0.9), # 调制深度范围
        harmonics=[1, 2, 3],   # 倍频列表，fm, 2*fm, 3*fm
        N=20, seed=42,
    )

    # 指定相位
    signals, components, params = batch_gen(
        K=3, mod_freqs=10, harmonics=[1, 2],
        am_phases=(0, 2*pi),   # 各调制频率的调幅相位随机 [0, 2π]
        fm_phases=(0, 2*pi),   # 各调制频率的调频相位随机 [0, 2π]
        N=20, seed=42,
    )
"""
import sys
from pathlib import Path

# 计算项目根目录的绝对路径
# 向上三级目录：rVMD->src->experiment
project_root = Path(__file__).parent.parent.parent  # 注意层级数

# 将项目根目录添加到 sys.path 的开头
sys.path.insert(0, str(project_root))

import numpy as np

from src.data.am_fm_signal import gen_signal


def _sample(rng, spec, is_int=False):
    """
    从参数规格采样一个值。
    - 元组/列表 (lo, hi) → 在 [lo, hi] 内随机采样
    - int/float 单值     → 直接返回，不随机
    """
    if isinstance(spec, (tuple, list)):
        if is_int:
            return int(rng.integers(spec[0], spec[1] + 1))
        else:
            return float(rng.uniform(spec[0], spec[1]))
    return spec


def _sample_array(rng, spec, length):
    """
    从参数规格采样一个长度为 length 的数组。
    - 元组/列表 (lo, hi) → 每个元素在 [lo, hi] 内独立随机采样
    - 标量 float/int     → 所有元素为该固定值
    - 数组/list           → 直接返回（需长度匹配）
    - None               → 返回 None
    """
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2 and isinstance(spec[1], (int, float)):
            # (lo, hi) 范围：每个元素独立随机
            return [float(rng.uniform(spec[0], spec[1])) for _ in range(length)]
        else:
            # 直接给定数组
            return list(spec)
    # 标量：所有元素为该值
    return [float(spec)] * length


def batch_gen(
    K,
    base_freqs,
    freq_intervals,
    mod_freqs,
    am_coeffs,
    mod_depths,
    harmonics=None,
    am_phases=None,       # [新增] 调幅相位范围或固定值，默认 None → 全0
    fm_phases=None,       # [新增] 调频相位范围或固定值，默认 None → 全0
    noise_std=0.0,        # [新增] 噪声标准差，0 则无噪声
    noise_seed=None,      # [新增] 噪声随机种子，None 则每个信号独立
    N=10,
    seed=None,
    fs=1000,
    duration=1.0,
):
    """
    批量生成 AM-FM 信号。每个参数可接受元组(范围，随机)或单值(固定)。

    Parameters
    ----------
    K : tuple[int, int] | int
        模态数量。元组则随机整数，单值则固定。
    base_freqs : tuple[float, float] | float
        基本频率 [Hz]。元组则随机浮点，单值则固定。
    freq_intervals : tuple[float, float] | float
        频率间隔 [Hz]。元组则随机浮点，单值则固定。
    mod_freqs : tuple[float, float] | float
        调制频率 [Hz]。元组则随机浮点，单值则固定。
    am_coeffs : tuple[float, float] | float
        调幅系数。元组则随机浮点，单值则固定。
    mod_depths : tuple[float, float] | float
        调制深度(beta)。元组则随机浮点，单值则固定。
    harmonics : list[int] | None
        倍频列表，默认 [1]。
        例: [1, 2, 3] → 调制频率 = [fm, 2*fm, 3*fm]。
        即除基频外，还叠加各倍频分量的调制。
    am_phases : tuple[float, float] | float | list[float] | None
        [新增] 调幅各路相位 [rad]。
        - None: 默认全 0（与原版行为一致）
        - (lo, hi): 每个调制频率的相位在 [lo, hi] 内独立随机
        - 标量: 所有调制频率相同相位
        - list: 直接指定，需与 actual_mod_freqs 等长
    fm_phases : tuple[float, float] | float | list[float] | None
        [新增] 调频各路相位 [rad]。规则同 am_phases。
    noise_std : tuple[float, float] | float
        [新增] 高斯噪声标准差。元组则随机浮点，单值则固定。默认 0（无噪声）。
    noise_seed : int | None
        [新增] 噪声随机种子。None 则每个信号独立随机（不复现）。
    N : int
        生成信号数量，默认 10。
    seed : int | None
        随机种子。
    fs : float
        采样频率 [Hz]，默认 1000。
    duration : float
        信号时长 [s]，默认 1.0。

    Returns
    -------
    signals : list[ndarray]
        混合信号列表，长度为 N。
    components : list[list[ndarray]]
        各混合信号的分量信号列表。
        例：载波频率 [200, 300] → 对应元素 [signal_200, signal_300]
    params : dict
        生成参数记录。
    """
    rng = np.random.default_rng(seed)
    if harmonics is None:
        harmonics = [1]

    signals = []
    components = []
    params_list = []

    for _ in range(N):
        k = _sample(rng, K, is_int=True)
        fc = _sample(rng, base_freqs, is_int=True)
        interval = _sample(rng, freq_intervals, is_int=True)
        fm = _sample(rng, mod_freqs, is_int=True)
        m = _sample(rng, am_coeffs)
        beta = _sample(rng, mod_depths)
        ns = _sample(rng, noise_std)

        # 载波频率 = [fc, fc + d, fc + 2d, ...]，共 k 个
        carrier_freqs = [fc + i * interval for i in range(k)]

        # 倍频调制频率列表
        actual_mod_freqs = [h * fm for h in harmonics]
        nf = len(actual_mod_freqs)

        # [新增] 采样相位
        _am_phases = _sample_array(rng, am_phases, nf)
        _fm_phases = _sample_array(rng, fm_phases, nf)

        # [新增] 噪声种子：None → 不指定（不复现）
        _noise_seed = None if noise_seed is None else int(noise_seed)

        # 混合信号
        _, mixed = gen_signal(
            freqs=carrier_freqs,
            mode='AM_FM',
            amps=[1.0] * k,
            fs=fs,
            duration=duration,
            m=m,
            fm_freqs=actual_mod_freqs,
            am_phases=_am_phases,
            beta=beta,
            fm_mod_freqs=actual_mod_freqs,
            fm_phases=_fm_phases,
            noise_std=ns,
            noise_seed=_noise_seed,
        )

        # 各分量信号
        singles = []
        for f in carrier_freqs:
            _, sig = gen_signal(
                freqs=[f],
                mode='AM_FM',
                amps=[1.0],
                fs=fs,
                duration=duration,
                m=m,
                fm_freqs=actual_mod_freqs,
                am_phases=_am_phases,
                beta=beta,
                fm_mod_freqs=actual_mod_freqs*2,# 根据调幅调频比调整
                fm_phases=_fm_phases,
                noise_std=ns,
                noise_seed=_noise_seed,
            )
            singles.append(sig)

        signals.append(mixed)
        components.append(singles)
        params_list.append({
            'K': k,
            'carrier_freqs': carrier_freqs,
            'base_freq': fc,
            'interval': interval,
            'mod_freq': fm,
            'actual_mod_freqs': actual_mod_freqs,
            'harmonics': harmonics,
            'am_coeff': m,
            'mod_depth': beta,
            'am_phases': _am_phases,
            'fm_phases': _fm_phases,
            'noise_std': ns,
            'noise_seed': _noise_seed,
        })

    params = {
        'total': N,
        'ranges': {
            'K': K,
            'base_freqs': base_freqs,
            'freq_intervals': freq_intervals,
            'mod_freqs': mod_freqs,
            'am_coeffs': am_coeffs,
            'mod_depths': mod_depths,
            'harmonics': harmonics,
            'am_phases': am_phases,
            'fm_phases': fm_phases,
            'noise_std': noise_std,
            'noise_seed': noise_seed,
        },
        'fs': fs,
        'duration': duration,
        'seed': seed,
        'details': params_list,
    }

    return signals, components, params


from itertools import product

def gen_by_freqs_and_depths(
    freqs,
    mod_depths,
    mod_freq=10,
    am_phases=None,       # [新增] 调幅相位，同 batch_gen 规则
    fm_phases=None,       # [新增] 调频相位，同 batch_gen 规则
    noise_std=0.0,        # [新增] 噪声标准差，0 则无噪声
    noise_seed=None,      # [新增] 噪声随机种子
    fs=1000,
    duration=1.0,
    seed=None,
):
    """
    对 freqs × mod_depths 的笛卡尔积生成 AM-FM 信号。
    新增 am_phases / fm_phases 参数，默认 None → 全 0（向后兼容）。

    Parameters
    ----------
    freqs      : list[float]  载波频率列表 [Hz]
    mod_depths : list[float]  调制深度(beta)列表
    mod_freq   : float        调制频率 [Hz]，默认 10.0
    am_phases  : tuple | float | list | None   [新增] 调幅各路相位
    fm_phases  : tuple | float | list | None   [新增] 调频各路相位
    fs         : float        采样频率 [Hz]，默认 1000
    duration   : float        信号时长 [s]，默认 1.0
    seed       : int | None   随机种子

    Returns
    -------
    signals : list[ndarray]
        信号列表，长度 = len(freqs) * len(mod_depths)
        顺序：先遍历 mod_depths，每个 depth 下遍历所有 freqs
        例：freqs=[100,200,300], mod_depths=[0.2,0.8]
        → [sig(100,0.2), sig(200,0.2), sig(300,0.2),
           sig(100,0.8), sig(200,0.8), sig(300,0.8)]
    params : list[dict]
        每个信号的参数，与 signals 一一对应
    """
    rng = np.random.default_rng(seed)
    signals = []
    params_list = []

    # 默认调制频率列表
    actual_mod_freqs = [10, 20, 30]
    nf = len(actual_mod_freqs)

    for beta, fc in product(mod_depths, freqs):
        # [新增] 每个信号独立采样相位
        _am_phases = _sample_array(rng, am_phases, nf)
        _fm_phases = _sample_array(rng, fm_phases, nf)

        _, sig = gen_signal(
            freqs=[fc],
            mode='AM_FM',
            amps=[1.0],
            fs=fs,
            duration=duration,
            m=0.5,
            fm_freqs=actual_mod_freqs,
            am_phases=_am_phases,
            beta=beta,
            fm_mod_freqs=actual_mod_freqs,
            fm_phases=_fm_phases,
            noise_std=noise_std,
            noise_seed=noise_seed,
        )
        signals.append(sig)
        params_list.append({
            'freq': fc,
            'mod_depth': beta,
            'am_phases': _am_phases,
            'fm_phases': _fm_phases,
            'noise_std': noise_std,
            'noise_seed': noise_seed,
        })

    return signals, params_list


if __name__ == "__main__":
    # 混合用法：部分固定，部分随机（相位默认全0，向后兼容）
    signals, components, params = batch_gen(
        K=3,                   # 固定 3 个模态
        base_freqs=(50, 300),  # 随机
        freq_intervals=60,     # 固定间隔 60Hz
        mod_freqs=10,          # 固定调制基频
        am_coeffs=0.5,         # 固定调幅系数
        mod_depths=(0.1, 0.9), # 随机
        N=10,
        seed=42,
    )

    print(f"信号总数: {params['total']}")
    print(f"信号长度: {len(signals[0])}")
    print()
    for i, d in enumerate(params['details']):
        freqs_str = ", ".join(f"{f:.1f}" for f in d['carrier_freqs'])
        mfs_str = ", ".join(f"{f:.1f}" for f in d['actual_mod_freqs'])
        ap_str = f"am_ph=[{', '.join(f'{p:.2f}' for p in (d.get('am_phases') or []))}]" if d.get('am_phases') else "am_ph=None"
        fp_str = f"fm_ph=[{', '.join(f'{p:.2f}' for p in (d.get('fm_phases') or []))}]" if d.get('fm_phases') else "fm_ph=None"
        print(f"[{i:2d}] K={d['K']}  fc=[{freqs_str}]  "
              f"mod_freqs=[{mfs_str}]  m={d['am_coeff']:.2f}  beta={d['mod_depth']:.2f}  "
              f"noise={d['noise_std']:.3f}  "
              f"{ap_str}  {fp_str}")
