# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)


"""
CFC-VMD方法试验用程序
2026.10.06 南京 陆工大 刘加龙
"""

import sys
from pathlib import Path

# 计算项目根目录的绝对路径
project_root = Path(__file__).parent.parent.parent  # 注意层级数

# 将项目根目录添加到 sys.path 的开头
sys.path.insert(0, str(project_root))

# 导入本地模块
from src.methods import cfcvmd
from src.methods import cfc
from src.methods import ft
from src.utils import plotter as plt
from src.data.am_fm_signal import gen_signal
from src.data.batch_gen import batch_gen
from src.data import signal_pretreat as spre
from src.utils.array_saver import save_array
from src.utils import statistics as st
from src.data.func_signal import sample_func

#导入必要的库
import yaml
import os
import warnings
import numpy as np
import vmdpy as vmd
from scipy.optimize import linear_sum_assignment


warnings.filterwarnings('ignore')

# 加载配置文件
with open(project_root / "configs" / "signal_configs.yaml",
          "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# 可视化设置全局样式（一次设定，后面全部复用）
my_style = plt.PlotStyle()
pt = plt.Plotter(my_style)

#计算beta边界值
def BetaCrit():
    beta = lambda m: (2+m**2)/(4*m)
    dm,data = sample_func(beta,0,1,1000)
    savepath = os.path.join(project_root , "experiments" , "BetaCrit.csv")
    save_array([dm,data], savepath, header=["m","Beta"])
    
# 计算VMD分解中心频率与真实中心频率的匹配数量
def MatchCount(omega_pred, omega_true, threshold = 1.5):
    match_count = 0
    used = [False] * len(omega_true)#记录omega_true中的元素是否被匹配过
    for vf in omega_pred:
        for i, vt in enumerate(omega_true):
            if not used[i] and abs(vf - vt) <= threshold:
                match_count += 1
                used[i] = True
                break#找到匹配后跳出内层循环
    return match_count

# 计算VMD分解中心频率与真实中心频率的误差
def HungarianMatchWithErrors(list_a, list_b):
    """返回最优匹配的索引对和对应的绝对误差。"""
    cost_matrix = np.abs(np.subtract.outer(list_a, list_b))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = [(i, j) for i, j in zip(row_ind, col_ind)]
    errors = [cost_matrix[i, j] for i, j in matches]
    
    return matches, errors

# 可视化VMD分解结果的函数
def VisualIMFs(signal, fs, duration, u, u_hat, omega):
    # 可视化设置全局样式（一次设定，后面全部复用）
    my_style = plt.PlotStyle()
    pt = plt.Plotter(my_style)
    # 可视化
    data_v = []
    data_v.append(plt.PlotData(
                    np.linspace(0, duration, len(signal)), 
                    signal, 
                    '', "Time (s)", "Amplitude",
                    colors=["#4477AA", "#EE6677"]
                    )
    )

    for i in range(u.shape[0]):
        data_v.append(plt.PlotData(
                        np.linspace(0, fs//2-1, len(signal)//2), 
                        np.abs(u_hat[len(signal)//2:, i]), 
                        "", 
                        "Frequency (Hz)", "Amplitude",
                        colors=["#4477AA", "#EE6677"],
                    #   vertical_lines=[(omega[-1,i]*self.fs, r"$f_{VMD}$"+f'≈{omega[-1,i]*self.fs:.2f}Hz')] 
                        vertical_lines=[(omega[-1,i]*fs, r"$f$"+f'≈{omega[-1,i]*fs:.2f}Hz')] 
                        )
        )
    
    pt.vertical(data_v, figsize=(10, 5), show=True)

# 进行中心频率定位试验
def ExperimentOmegaVisual():
    t,signal = gen_signal(
        freqs=[300],
        mode='AM',
        m=[0.5],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[5],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[4],
        fm_mod_freqs=[5],
        noise_std=0.5,
        noise_seed=42,
    )

    fq,A = ft.FFTl(signal,1200)
    # ── 1. 单图（多条线）──
    data_multi = plt.PlotData(
        x=fq, y=[A/2],
        title="单图-多线示例",
        xlabel="时间 (s)", ylabel="幅值",
        labels=["sin(x)", "cos(x)", "衰减sin"],
        colors=["#1A1F3C", "#F0C040", "#E74C3C"],
        linestyles=["-", "--", "-."],
        markers=["", "o", ""],
    )
    pt.single(data_multi, figsize=(10, 4), show=True)
    
    
    u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega) # 可视化分解结果
    u,u_hat,omega = cfc.CFC(u,1200) # 使用外部库VMD进行分解
    # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    # print(omega)
    VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega/1200) # 可视化分解结果
    
# 显示AM-FM信号的频谱
def ExpShowSpectrum():
    t,signal = gen_signal(
        freqs=[1000],
        mode='FM',
        m=[0.5],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[10],
        amps = [1],
        fs=4000,
        duration=1,
        beta=[4],
        fm_mod_freqs=[10],
        # noise_std=0.5,
        # noise_seed=42,
    )


    # loadnpy = ld.Loadnpy() # 加载信号数据
    # signal = loadnpy[0] # 取第一条信号进行处理
    # signal = spre.normalize_signal(signal, method='max') # 归一化信号
    
    fq,A = ft.FFTl(signal,4000) # FFT频谱分析


    # ── 绘制频谱图 ──
    data_h = [
        plt.PlotData(t, signal, "(a) Carrier wave 300Hz;\nMod wave 20Hz",  "Frequency (Hz)","Amplitude",colors=['#4477AA']),
        plt.PlotData(fq, A, "(b) Carrier wave 300Hz;\nMod wave 20.3Hz", "Frequency (Hz)", "Amplitude",colors=['#EE6677']),
    ]
    pt.horizontal(data_h, figsize=(10, 5), show=True)
    # savepath = os.path.join(project_root , "experiments" , "AMFMSpectrum05.csv")
    # save_array([fq, A,],
    #            savepath, 
    #            header=["Frequency", "Amplitude"]
    #            )

# 进行VMD中心频率定位精度对比试验
def ExperimentVMDOmegaCompareVisual():

    # 生成信号
    signals, _, params = batch_gen(
        K=1,# 随机生成1个模态
        base_freqs=(200,400),# 随机生成基本频率在100到150Hz之间
        freq_intervals=(100,150),# 固定频率在100Hz~150Hz
        mod_freqs=(3,10),# 调制频率在5Hz~10Hz，以满足瞬时频率缓变条件
        harmonics=[1],# 调制频率 = fm
        am_phases=(0, 2*np.pi),# 调幅相位在0到2π之间
        fm_phases=(0, 2*np.pi),# 调频相位在0到2π之间
        am_coeffs=(0, 0.9),# 随机生成调幅系数在0到1之间，保证包络不过零
        mod_depths=(0, 2),# 0到2之间以满足窄带条件
        noise_std=0.562,        # [新增] 噪声标准差，0 则无噪声
        noise_seed=42,      # [新增] 噪声随机种子，None 则每个信号独立
        fs=1200, # 采样率
        duration=1.0, # 信号时长
        N=5000,# 信号数量
        seed=42,# 随机数种子，确保结果可复现
    )

    error_list = []
    fmod_list = []
    fc_list = []
    m_list = []
    beta_list = []
    domega_list = []
    fm_phase_list = []
    am_phase_list = []
    
    for i, d in enumerate(params['details']):
        signal=spre.normalize_signal(signals[i], method='max') # 归一化信号
        signal=spre.detrend_signal(signal) # 去除信号的趋势
        
        # 进行VMD分解
        u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=params['details'][i]['K'], DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        for j in range(0,len(params['details'][i]['carrier_freqs'])):#遍历模态数量
            m_list.append(d['am_coeff'])
            beta_list.append(d['mod_depth'])
            fc_list.append(d['carrier_freqs'][j])
            fmod_list.append(d['actual_mod_freqs'][0])
            fm_phase_list.append(d['fm_phases'][0])
            am_phase_list.append(d['am_phases'][0])
            domega_list.append(omega[-1,:][j]*1200)
            match,errors = HungarianMatchWithErrors(omega[-1,:]*1200, params['details'][i]['carrier_freqs'])
            error_list.append(errors[j])

    savepath = os.path.join(project_root , "experiments" , "VMDOmegaCompareVisualNoise2.csv")
    save_array([fc_list,fmod_list,m_list,beta_list, np.array(beta_list)*np.array(fmod_list),domega_list,error_list], 
            savepath, 
            header=["fc","fmod","m","beta", "delta f","domega","vmderror"])

    error_list = []
    fmod_list = []
    fc_list = []
    m_list = []
    beta_list = []
    domega_list = []
    fm_phase_list = []
    am_phase_list = []
    
    for i, d in enumerate(params['details']):
        signal=spre.normalize_signal(signals[i], method='max') # 归一化信号
        signal=spre.detrend_signal(signal) # 去除信号的趋势
    
        # 进行VMD分解
        u, u_hat, omega = cfcvmd.CFCVMD(signal, alpha=100, tau=0, K=params['details'][i]['K'], DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        for j in range(0,len(params['details'][i]['carrier_freqs'])):#遍历模态数量
            m_list.append(d['am_coeff'])
            beta_list.append(d['mod_depth'])
            fc_list.append(d['carrier_freqs'][j])
            fmod_list.append(d['actual_mod_freqs'][0])
            fm_phase_list.append(d['fm_phases'][0])
            am_phase_list.append(d['am_phases'][0])
            domega_list.append(omega[-1,:][j]*1200)
            match,errors = HungarianMatchWithErrors(omega[-1,:]*1200, params['details'][i]['carrier_freqs'])
            error_list.append(errors[j])

    savepath = os.path.join(project_root , "experiments" , "CFCOmegaCompareVisualNoise3.csv")
    save_array([fc_list,fmod_list,m_list,beta_list, np.array(beta_list)*np.array(fmod_list),domega_list,error_list], 
            savepath, 
            header=["fc","fmod","m","beta", "delta f","domega","vmderror"])
    
# 迭代次数个例实验
def ExperimentIterationCase():

    # 生成信号
    t,signal = gen_signal(
        freqs=[100,200,300,400,500],
        mode='AM_FM',
        m=0.5,
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[20],
        amps = [1,1,1,1,1],
        fs=1200,
        duration=1,
        beta=0.4,
        fm_mod_freqs=[20],
    )

    signal=spre.normalize_signal(signal, method='max') # 归一化信号
    signal=spre.detrend_signal(signal) # 去除信号的趋势

    # 进行VMD分解
    u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=5, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    count = MatchCount(omega[-1,:]*1200, [100,200,300,400,500], threshold=10)
    VisualIMFs(signal, 1200, 1, u, u_hat, omega) # 可视化分解结果
    
    vmd_omega1=[]
    vmd_omega2=[]
    vmd_omega3=[]
    vmd_omega4=[]
    vmd_omega5=[]
    if count == 5:# 如果分解成功
        for j in omega*1200:
            vmd_omega1.append(j[0])
            vmd_omega2.append(j[1])
            vmd_omega3.append(j[2])
            vmd_omega4.append(j[3])
            vmd_omega5.append(j[4])
    else:
        pass

    # 进行VMD分解
    u, u_hat, omega = cfcvmd.CFCVMD(signal, alpha=100, tau=0, K=5, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    count = MatchCount(omega[-1,:]*1200, [100,200,300,400,500], threshold=10)
    VisualIMFs(signal, 1200, 1, u, u_hat, omega) # 可视化分解结果
    
    
    print(omega*1200)
    vmdm_omega1=[]
    vmdm_omega2=[]
    vmdm_omega3=[]
    vmdm_omega4=[]
    vmdm_omega5=[]
    if count == 5:# 如果分解成功
        for j in omega*1200:
            print(j)
            vmdm_omega1.append(j[0])
            vmdm_omega2.append(j[1])
            vmdm_omega3.append(j[2])
            vmdm_omega4.append(j[3])
            vmdm_omega5.append(j[4])
    else:
        pass

    savepath = os.path.join(project_root , "experiments" , "VMDOmegaIterationCase.csv")
    save_array([np.linspace(1,len(vmd_omega1),len(vmd_omega1)),vmd_omega1,vmd_omega2,vmd_omega3,vmd_omega4,vmd_omega5, vmdm_omega1,vmdm_omega2,vmdm_omega3,vmdm_omega4,vmdm_omega5], 
            savepath, 
            header=["Iteration","imf1","imf2","imf3","imf4", "imf5","imf1","imf2","imf3","imf4", "imf5"])

# VMD分解频率个例实验
def ExperimentOmegaCase():
    # 生成信号
    t,signal = gen_signal(
        freqs=[100,200,300,400,500],
        mode='AM_FM',
        m=0.8,
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[10],
        amps = [1,1,1,1,1],
        fs=1200,
        duration=1,
        beta=0.2,
        fm_mod_freqs=[10],
    )

    signal=spre.normalize_signal(signal, method='max') # 归一化信号
    signal=spre.detrend_signal(signal) # 去除信号的趋势

    # 进行VMD分解
    u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=5, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    # count = MatchCount(omega[-1,:]*1200, [100,200,300,400,500], threshold=10)
    VisualIMFs(signal, 1200, 1, u, u_hat, omega) # 可视化分解结果

    savepath = os.path.join(project_root , "experiments" , "OmegaCaseTD1.csv")
    save_array([np.linspace(0, 1, 1200), signal,u[0, :],u[1, :],u[2, :],u[3, :],u[4, :]], 
                savepath, 
                header=["Time (s)", "Signal","IMF1","IMF2","IMF3","IMF4","IMF5"])
    
    savepath = os.path.join(project_root , "experiments" , "OmegaCaseFD1.csv")
    save_array([np.linspace(0, 599, 600), np.abs(u_hat[600:, 0]), np.abs(u_hat[600:, 1]), np.abs(u_hat[600:, 2]), np.abs(u_hat[600:, 3]), np.abs(u_hat[600:, 4])], 
               savepath, 
               header=["Frequency (Hz)", "IMF1","IMF2","IMF3","IMF4","IMF5"])



    # 进行VMD分解
    u, u_hat, omega = cfcvmd.CFCVMD(signal, alpha=100, tau=0, K=5, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    # count = MatchCount(omega[-1,:]*1200, [100,200,300,400,500], threshold=10)
    VisualIMFs(signal, 1200, 1, u, u_hat, omega) # 可视化分解结果
    
    savepath = os.path.join(project_root , "experiments" , "OmegaCaseTD2.csv")
    save_array([np.linspace(0, 1, 1200), signal,u[0, :],u[1, :],u[2, :],u[3, :],u[4, :]], 
                savepath, 
                header=["Time (s)", "Signal","IMF1","IMF2","IMF3","IMF4","IMF5"])
    
    savepath = os.path.join(project_root , "experiments" , "OmegaCaseFD2.csv")
    save_array([np.linspace(0, 599, 600), np.abs(u_hat[600:, 0]), np.abs(u_hat[600:, 1]), np.abs(u_hat[600:, 2]), np.abs(u_hat[600:, 3]), np.abs(u_hat[600:, 4])], 
               savepath, 
               header=["Frequency (Hz)", "IMF1","IMF2","IMF3","IMF4","IMF5"])

# 进行AMFM信号不对称性展示实验：边频功率差
def ExperimentBiasShow():
    t,signal12 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.8],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[10],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    t,signal1 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.8],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[20],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    t,signal2 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.8],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[40],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    t,signal06 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.8],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[6],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    signal_list =[signal12,signal1,signal2,signal06]
    P_list = []
    for signal in signal_list:
        fq,A = ft.FFTl(signal,1200)
        # ── 绘制频谱图 ──
        data_h = [
            plt.PlotData(t, signal, "(a) Carrier wave 300Hz;\nMod wave 20Hz",  "Frequency (Hz)","Amplitude",colors=['#4477AA']),
            plt.PlotData(fq, A, "(b) Carrier wave 300Hz;\nMod wave 20.3Hz", "Frequency (Hz)", "Amplitude",colors=['#EE6677']),
        ]
        pt.horizontal(data_h, figsize=(10, 5), show=True)
        print(np.average(fq, weights=A))
        P=A**2# 计算功率谱
        # P=P/P.sum()# 归一化
        for j in range(14):
            print((j*1/2)," : ",P[300+j*10]-P[300-j*10])
        # u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # omega = cfc.CFC(u,1200) # 使用外部库VMD进行分解
        # # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # print("CFC: ", omega)
        P_list.append(P)
        
        
        
        # u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega) # 可视化分解结果
        
    savepath = os.path.join(project_root , "experiments" , "AM_FMBiasShowT.csv")
    save_array([t,signal_list[0],signal_list[1],signal_list[2],signal_list[3]], 
            savepath, 
            header=["Time", "AMFM1/2","AMFM1","AMFM2","AMFM06"])
    savepath = os.path.join(project_root , "experiments" , "AM_FMBiasShowF.csv")
    save_array([fq,P_list[0],P_list[1],P_list[2],P_list[3]], 
            savepath, 
            header=["Frequency", "AMFM1/2","AMFM1","AMFM2","AMFM06"])
    
    
    # u,u_hat,omega = cfc2.CFC2(u,1200) # 使用外部库VMD进行分解
    # # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    # # print(omega)
    # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega/1200) # 可视化分解结果
    
# 进行AMFM信号不对称性展示实验：频谱质心偏移
def ExperimentBiasShow2():
    t,signal3 = gen_signal(
        freqs=[3000],
        mode='AM_FM',
        m=[0.8],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[60],
        amps = [1],
        fs=12000,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    t,signal32 = gen_signal(
        freqs=[3000],
        mode='AM_FM',
        m=[0.8],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[30],
        amps = [1],
        fs=12000,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    t,signal1 = gen_signal(
        freqs=[3000],
        mode='AM_FM',
        m=[0.8],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[20],
        amps = [1],
        fs=12000,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    t,signal12 = gen_signal(
        freqs=[3000],
        mode='AM_FM',
        m=[0.8],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[10],
        amps = [1],
        fs=12000,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )

    signal_list =[signal3,signal32,signal1,signal12]
    P_list = []
    for signal in signal_list:
        fq,A = ft.FFTl(signal,12000)
        # ── 绘制频谱图 ──
        data_h = [
            plt.PlotData(t, signal, "(a) Carrier wave 300Hz;\nMod wave 20Hz",  "Frequency (Hz)","Amplitude",colors=['#4477AA']),
            plt.PlotData(fq, A, "(b) Carrier wave 300Hz;\nMod wave 20.3Hz", "Frequency (Hz)", "Amplitude",colors=['#EE6677']),
        ]
        pt.horizontal(data_h, figsize=(10, 5), show=True)
        print(np.average(fq, weights=A))
        P=A**2# 计算功率谱
        # P=P/P.sum()# 归一化
        for j in range(12):
            print((j*1/2)," : ",P[3000+j*10]-P[3000-j*10])
        # u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # omega = cfc.CFC(u,1200) # 使用外部库VMD进行分解
        # # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # print("CFC: ", omega)
        P_list.append(P)
        
        
        
        # u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega) # 可视化分解结果
        
    savepath = os.path.join(project_root , "experiments" , "AM_FMBiasShowT2.csv")
    save_array([t,signal_list[0],signal_list[1]], 
            savepath, 
            header=["Time", "AMFM3","AMFM3/2"])
    savepath = os.path.join(project_root , "experiments" , "AM_FMBiasShowF2.csv")
    save_array([fq,P_list[0],P_list[1]], 
            savepath, 
            header=["Frequency", "AMFM3","AMFM3/2"])

    Bias3 = []
    Bias3_2 = []
    Bias1=[]
    Bias1_2 = []
    for k in range(1,20):
        step=20*k+2
        CutP = P_list[0][3000-step:3000+step+1]
        Cutfq = fq[3000-step:3000+step+1]
        Bias3.append(np.average(Cutfq, weights=CutP)-3000)
        print("3:", np.average(Cutfq, weights=CutP)-3000)

    for k in range(1,20):
        step=20*k+2-10
        CutP = P_list[1][3000-step:3000+step+1]
        Cutfq = fq[3000-step:3000+step+1]
        Bias3_2.append(np.average(Cutfq, weights=CutP)-3000)  
        print("3/2:", np.average(Cutfq, weights=CutP)-3000)
        # # ── 1. 单图（多条线）──
        # data_multi = plt.PlotData(
        #     x=Cutfq, y=[CutP],
        #     title="单图-多线示例",
        #     xlabel="时间 (s)", ylabel="幅值",
        #     labels=["sin(x)", "cos(x)", "衰减sin"],
        #     colors=["#1A1F3C", "#F0C040", "#E74C3C"],
        #     linestyles=["-", "--", "-."],
        #     markers=["", "o", ""],
        # )
        # pt.single(data_multi, figsize=(10, 4), show=True)


    for k in range(1,20):
        step=20*k+2
        CutP = P_list[2][3000-step:3000+step+1]
        Cutfq = fq[3000-step:3000+step+1]
        Bias1.append(np.average(Cutfq, weights=CutP)-3000)

    for k in range(1,20):
        step=20*k+2-10
        CutP = P_list[3][3000-step:3000+step+1]
        Cutfq = fq[3000-step:3000+step+1]
        Bias1_2.append(np.average(Cutfq, weights=CutP)-3000)    

    savepath = os.path.join(project_root , "experiments" , "AM_FMBiasShowStatistic.csv")
    save_array([np.linspace(1,20,20)*20*2,Bias3,Bias1,np.linspace(1,20,20)*20*2-20,Bias3_2,Bias1_2], 
            savepath, 
            header=["BW3", "AMFM3","AMFM1","BW3/2","AMFM3/2","AMFM1/2"])
    
    
    # u,u_hat,omega = cfc2.CFC2(u,1200) # 使用外部库VMD进行分解
    # # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    # # print(omega)
    # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega/1200) # 可视化分解结果
    
# 进行AMFM信号不对称性展示实验：多音频谱质心偏移
def ExperimentBiasShow3():
    t,signal1 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.8,0.8],
        am_phases=[0,0],
        fm_phases=[0],
        fm_freqs=[20,33],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    t,signal2 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.8,0.8],
        am_phases=[0,0],
        fm_phases=[0],
        fm_freqs=[27,7],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    t,signal3 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.8,0.8],
        am_phases=[0,0],
        fm_phases=[0],
        fm_freqs=[13,7],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[20],
    )
    
    signal_list =[signal1,signal2,signal3]
    P_list = []
    for signal in signal_list:
        fq,A = ft.FFTl(signal,1200)
        # ── 绘制频谱图 ──
        data_h = [
            plt.PlotData(t, signal, "(a) Carrier wave 300Hz;\nMod wave 20Hz",  "Frequency (Hz)","Amplitude",colors=['#4477AA']),
            plt.PlotData(fq, A, "(b) Carrier wave 300Hz;\nMod wave 20.3Hz", "Frequency (Hz)", "Amplitude",colors=['#EE6677']),
        ]
        pt.horizontal(data_h, figsize=(10, 5), show=True)
        print(np.average(fq, weights=A))
        P=A**2# 计算功率谱
        # P=P/P.sum()# 归一化
        # for j in range(14):
        #     print((j*1/2)," : ",P[300+j*10]-P[300-j*10])
        # u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # omega = cfc.CFC(u,1200) # 使用外部库VMD进行分解
        # # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # print("CFC: ", omega)
        P_list.append(P)
        
        
        
        # u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega) # 可视化分解结果
        
    savepath = os.path.join(project_root , "experiments" , "AM_FMBiasShowT3.csv")
    save_array([t,signal_list[0],signal_list[1],signal_list[2]], 
            savepath, 
            header=["Time", "AMFM1","AMFM2","AMFM3"])
    savepath = os.path.join(project_root , "experiments" , "AM_FMBiasShowF3.csv")
    save_array([fq,P_list[0],P_list[1],P_list[2]], 
            savepath, 
            header=["Frequency", "AMFM1","AMFM2","AMFM3"])
    
    
    # u,u_hat,omega = cfc2.CFC2(u,1200) # 使用外部库VMD进行分解
    # # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    # # print(omega)
    # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega/1200) # 可视化分解结果
    
# 进行CFC校准能力展示
def ExperimentCFCShow():
    t,signal1 = gen_signal(
        freqs=[300],
        mode='AM',
        m=[0.9],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[10],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[10],
    )
    t,signal2 = gen_signal(
        freqs=[300],
        mode='FM',
        m=[0.9],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[10],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[10],
    )
    t,signal3 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.9],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[10],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[10],
    )
    t,signal4 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.9],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[5],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[10],
    )
    t,signal5 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.9],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[20],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[10],
    )
    t,signal6 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.9],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[30],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[10],
    )
    t,signal7 = gen_signal(
        freqs=[300],
        mode='AM_FM',
        m=[0.9],
        am_phases=[0],
        fm_phases=[0],
        fm_freqs=[15],
        amps = [1],
        fs=1200,
        duration=1,
        beta=[2],
        fm_mod_freqs=[10],
    )
    signal_list =[signal1,signal2,signal3,signal4,signal5,signal6,signal7]
    A_list = []
    CFC_Signal_list = []
    CFC_Spectrum_list = []
    for signal in signal_list:
        fq,A = ft.FFTl(signal,1200)
        # ── 绘制频谱图 ──
        data_h = [
            plt.PlotData(t, signal, "(a) Carrier wave 300Hz;\nMod wave 20Hz",  "Frequency (Hz)","Amplitude",colors=['#4477AA']),
            plt.PlotData(fq, A, "(b) Carrier wave 300Hz;\nMod wave 20.3Hz", "Frequency (Hz)", "Amplitude",colors=['#EE6677']),
        ]
        pt.horizontal(data_h, figsize=(10, 5), show=True)
        print(np.average(fq, weights=A))
        A_list.append(A)
        # u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega) # 可视化分解结果
        u,u_hat,omega = cfc.CFC(signal,1200) # 使用外部库VMD进行分解
        VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega/1200) # 可视化分解结果
        # # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        CFC_Signal_list.append(2*u[0,:])
        CFC_Spectrum_list.append(2*np.abs(u_hat[:,0])/len(signal))
        print("CFC: ", omega)
        
        
        
        # u, u_hat, omega = vmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
        # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega) # 可视化分解结果
        
    savepath = os.path.join(project_root , "experiments" , "AM_FMShowSignalT.csv")
    save_array([t,signal_list[0],signal_list[1],signal_list[2], signal_list[3],signal_list[4],signal_list[5],signal_list[6]], 
            savepath, 
            header=["Time","AM","FM","AMFM1", "AMFM1/2","AMFM2","AMFM3","AMFM3/2"])
    savepath = os.path.join(project_root , "experiments" , "AM_FMShowSignalF.csv")
    save_array([fq,A_list[0],A_list[1],A_list[2], A_list[3],A_list[4],A_list[5],A_list[6]], 
            savepath, 
            header=["Time","AM","FM","AMFM1", "AMFM1/2","AMFM2","AMFM3","AMFM3/2"])
    savepath = os.path.join(project_root , "experiments" , "AM_FMShowCFCT.csv")
    save_array([t,CFC_Signal_list[0],CFC_Signal_list[1],CFC_Signal_list[2], CFC_Signal_list[3],CFC_Signal_list[4],CFC_Signal_list[5],CFC_Signal_list[6]], 
            savepath, 
            header=["Time","AM","FM","AMFM1", "AMFM1/2","AMFM2","AMFM3","AMFM3/2"])
    savepath = os.path.join(project_root , "experiments" , "AM_FMShowCFCF.csv")
    save_array([fq,CFC_Spectrum_list[0],CFC_Spectrum_list[1],CFC_Spectrum_list[2], CFC_Spectrum_list[3],CFC_Spectrum_list[4],CFC_Spectrum_list[5],CFC_Spectrum_list[6]], 
            savepath, 
            header=["Time","AM","FM","AMFM1", "AMFM1/2","AMFM2","AMFM3","AMFM3/2"])    
    
    # u,u_hat,omega = cfc2.CFC2(u,1200) # 使用外部库VMD进行分解
    # # u,u_hat,omega = cfcvmd.VMD(signal, alpha=100, tau=0, K=1, DC=0, init=0, tol=1e-7) # 使用外部库VMD进行分解
    # # print(omega)
    # VisualIMFs(u[0,:], 1200, 1, u, u_hat, omega/1200) # 可视化分解结果
    

if __name__ == "__main__":

    """不生成实验数据，只进行可视化"""
    # ExperimentOmegaVisual() # 进行中心频率可视化
    # ExpShowSpectrum() # 进行频谱可视化
    """***********************"""
  
    
    """生成论文用实验数据"""
    # ExperimentBiasShow()# 进行AMFM信号不对称性展示实验
    # ExperimentBiasShow2()# 进行AMFM信号不对称性展示实验
    # ExperimentBiasShow3()# 进行AMFM信号不对称性展示实验
    # ExperimentCFCShow()# 进行CFC校准能力展示
    # ExperimentOmegaCase()# 进行VMD中心频率定位个例实验
    ExperimentVMDOmegaCompareVisual() # 进行VMD中心频率定位精度对比试验
    # ExperimentIterationCase() # 进行VMD与CFCVMD中心频率定位迭代可视化
    """***************"""
    
    pass