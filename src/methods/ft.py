import numpy as np
from typing import Optional, Tuple
import scipy as sp
from matplotlib import pyplot as plt

# 傅里叶变换函数
def FFTl(
    signal: np.ndarray,
    sample_rate: float = 1.0,
)->Tuple[np.ndarray, np.ndarray]:
    """
    使用scipy.fft进行傅里叶变换
    
    参数:
    ----------
    signal : np.ndarray
        输入信号序列
    sample_rate : float
        采样率（Hz），默认为1.0
    visualize : bool
        是否可视化原始信号和频域结果，默认为True
    title : str, optional
        图形标题
    figsize : Tuple[int, int]
        图形尺寸，默认为(12, 6)
    返回:
    ----------
    Tuple[np.ndarray, np.ndarray]
        (频率数组, 幅度谱)
    """
    
    # 参数检查
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    if len(signal) == 0:
        raise ValueError("输入信号不能为空")
    
    N = len(signal)  # 信号长度
    

    signal_to_transform = signal.copy()
    
    # 2. 使用scipy.fft进行傅里叶变换
    # scipy.fft.fft 是 numpy.fft.fft 的改进版本
    fft_result = sp.fft.fft(signal_to_transform)
    
    # 计算频率轴（双边）
    freqs = sp.fft.fftfreq(N, 1/sample_rate)
    
    # 计算幅度谱
    magnitude = np.abs(fft_result)
    
    # 右图：单边幅度谱
    # 获取正频率部分
    pos_freq_mask = freqs >= 0
    pos_freqs = freqs[pos_freq_mask]
    pos_magnitude = magnitude[pos_freq_mask]
    
    # 对于实数信号，单边频谱需要乘以2（除DC分量外）
    if np.all(np.isreal(signal)):
        pos_magnitude_single = pos_magnitude.copy()
        if N > 1:
            pos_magnitude_single[1:-1] *= 2  # 除0频率和Nyquist频率外都乘以2
    
    # # 使用对数坐标显示频谱
    # pos_magnitude_db = 20 * np.log10(pos_magnitude_single + 1e-10)  # 避免log(0)
    #不使用对数坐标显示频谱
    pos_magnitude_db = pos_magnitude_single

    # # 3. 如果visualize参数为True，则可视化
    # if visualize:
    #     fig, axes = plt.subplots(1, 2, figsize=figsize)
        
    #     # 左图：原始信号（或加窗后信号）
    #     time = np.arange(N) / sample_rate
        
    #     axes[0].plot(time, signal, 'b-', linewidth=1.5, alpha=0.7, label='原始信号')
    #     axes[0].set_xlabel('时间 (s)')
    #     axes[0].set_ylabel('幅值')
    #     axes[0].set_title(f'时域信号')
    #     axes[0].legend(loc='upper right')
    #     axes[0].grid(True, alpha=0.3)
        
        
    #     axes[1].plot(pos_freqs, pos_magnitude_db, 'r-', linewidth=1.5)
    #     axes[1].set_xlabel('频率 (Hz)')
    #     axes[1].set_ylabel('幅度 (dB)')
    #     axes[1].set_title(f'幅度谱 (单边)')
    #     axes[1].grid(True, alpha=0.3)
    #     axes[1].set_xlim([0, sample_rate/2])
        
    #     # # 自动调整y轴范围
    #     # max_db = np.max(pos_magnitude_db)
    #     # min_db = np.min(pos_magnitude_db)
    #     # axes[1].set_ylim([max( max_db-80, min_db ), max_db+10])
        
    #     # 标记峰值频率
    #     if len(pos_freqs) > 0:
    #         peak_idx = np.argmax(pos_magnitude_single[1:]) + 1  # 跳过DC分量
    #         peak_freq = pos_freqs[peak_idx]
    #         peak_mag = pos_magnitude_db[peak_idx]
            
    #         axes[1].axvline(x=peak_freq, color='g', linestyle='--', alpha=0.5, 
    #                        label=f'峰值: {peak_freq:.2f} Hz')
    #         axes[1].plot(peak_freq, peak_mag, 'go', markersize=8, alpha=0.7)
    #         axes[1].legend(loc='upper right')
        
    #     # 设置总标题
    #     if title:
    #         fig.suptitle(title, fontsize=14, fontweight='bold')
    #     else:
    #         fig.suptitle(f'傅里叶变换分析', fontsize=14, fontweight='bold')
        
    #     plt.tight_layout()
    #     plt.show()
    
    return pos_freqs,pos_magnitude_db

