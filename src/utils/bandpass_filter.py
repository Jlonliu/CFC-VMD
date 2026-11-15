# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)

"""
带通滤波器：通过指定频段，阻挡其他频率
适用于旋转机械振动信号 AM-FM 分析

使用 FIR 滤波器 + filtfilt 实现零相位带通滤波
FIR 优势：线性相位，不引入相位畸变，适合需要保留信号包络/瞬时频率的场景
"""

import numpy as np
from scipy import signal


def bandpass_filter(
    x,
    fs,
    lowcut,
    highcut,
    order=1001,
    transition_width=None,
):
    """
    FIR 带通滤波器（零相位，线性相位）

    参数:
        x               : 输入信号 (1D numpy array)
        fs              : 采样率 (Hz)，必须满足 fs > 2 * highcut
        lowcut          : 通带下限频率 (Hz)
        highcut         : 通带上限频率 (Hz)
        order           : FIR 滤波器阶数（必须为奇数，默认 1001）
                          阶数越大，过渡带越陡，但计算量越大、延迟越长
        transition_width: 过渡带宽度 (Hz)，默认自动取 min(lowcut, fs-highcut) * 0.2
                          过渡带 = lowcut - transition_width 到 lowcut（下边）
                                  = highcut 到 highcut + transition_width（上边）

    返回:
        y : 滤波后的信号 (1D numpy array)

    示例:
        >>> y = bandpass_filter(signal, fs=10000, lowcut=1000, highcut=1250)
    """
    # 参数校验
    nyquist = fs / 2
    if highcut >= nyquist:
        raise ValueError(f"highcut ({highcut} Hz) 必须小于奈奎斯特频率 ({nyquist} Hz)，请提高采样率 fs")
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut}) 必须小于 highcut ({highcut})")
    if lowcut <= 0:
        raise ValueError(f"lowcut 必须 > 0")

    # 确保 order 为奇数（对称 FIR）
    if order % 2 == 0:
        order += 1

    # 设计 FIR 带通滤波器
    b = signal.firwin(
        order,
        [lowcut, highcut],
        pass_zero=False,   # 带通（不是低通）
        fs=fs,
    )

    # 零相位滤波（双向滤波，消除相位偏移）
    y = signal.filtfilt(b, [1.0], x)

    return y


def get_filter_response(fs, lowcut, highcut, order=1001):
    """
    获取滤波器的频率响应（用于可视化验证）

    返回:
        freqs : 频率轴 (Hz)
        h     : 频率响应幅值 (dB)
    """
    if order % 2 == 0:
        order += 1
    b = signal.firwin(order, [lowcut, highcut], pass_zero=False, fs=fs)
    w, h = signal.freqz(b, [1.0], worN=4096, fs=fs)
    return w, 20 * np.log10(np.abs(h) + 1e-12)


# ======================== 使用示例 ========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ── 参数设置 ──
    fs = 10000                # 采样率 10kHz
    duration = 1.0            # 信号时长 1 秒
    lowcut = 1000              # 通带下限 1000 Hz
    highcut = 1250             # 通带上限 1250 Hz
    n = int(fs * duration)

    # ── 生成测试信号 ──
    t = np.linspace(0, duration, n, endpoint=False)

    # 通带内：1000 Hz + 1100 Hz（应该保留）
    y_inband = 1.0 * np.sin(2 * np.pi * 1000 * t) + 0.8 * np.sin(2 * np.pi * 1100 * t)

    # 通带外：200 Hz + 3000 Hz（应该被滤掉）
    y_outband = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.6 * np.sin(2 * np.pi * 3000 * t)

    x = y_inband + y_outband

    # ── 滤波 ──
    y_filtered = bandpass_filter(x, fs=fs, lowcut=lowcut, highcut=highcut)

    # ── 绘制频率响应 ──
    freqs, h_db = get_filter_response(fs, lowcut, highcut)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上图：滤波器频率响应
    ax1 = axes[0]
    ax1.plot(freqs, h_db, color="#0026FF", linewidth=1.5)
    ax1.axvspan(lowcut, highcut, alpha=0.15, color="#00FF6A", label=f"通带 [{lowcut}, {highcut}] Hz")
    ax1.axvline(lowcut, color="#E74C3C", linestyle="--", linewidth=1, label=f"下截止 {lowcut} Hz")
    ax1.axvline(highcut, color="#E74C3C", linestyle="--", linewidth=1, label=f"上截止 {highcut} Hz")
    ax1.set_xlim(0, fs / 2)
    ax1.set_ylim(-80, 5)
    ax1.set_xlabel("频率 (Hz)")
    ax1.set_ylabel("幅值 (dB)")
    ax1.set_title(f"FIR 带通滤波器频率响应 (阶数=1001, 通带=[{lowcut}, {highcut}] Hz)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下图：原始信号 vs 滤波后信号（时域局部）
    ax2 = axes[1]
    n_show = min(500, n)  # 只显示前 500 个点
    t_show = t[:n_show] * 1000  # 转换为 ms
    ax2.plot(t_show, x[:n_show], color="#888888", linewidth=0.8, alpha=0.6, label="原始信号（含通带外分量）")
    ax2.plot(t_show, y_filtered[:n_show], color="#E74C3C", linewidth=1.2, label="滤波后信号（仅通带内）")
    ax2.plot(t_show, y_inband[:n_show], color="#0026FF", linewidth=1.0, linestyle="--", alpha=0.7, label="理想通带分量")
    ax2.set_xlabel("时间 (ms)")
    ax2.set_ylabel("幅值")
    ax2.set_title("滤波效果对比")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("=" * 50)
    print(f"采样率: {fs} Hz")
    print(f"通带:   [{lowcut}, {highcut}] Hz")
    print(f"滤波器: FIR 带通，阶数 1001，零相位 (filtfilt)")
    print("=" * 50)
