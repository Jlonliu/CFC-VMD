# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)



"""
此文件为读取时序数据的模块,
调用LoadData函数可以读取存储[[],[],[],[]...]格式的数据的npy文件,
并返回[[],[],[],[]...]格式的数据列表.
npy文件来源于其他的数据处理文件,此文件不在本项目之内,
若没有此文件,可自行处理数据并保存为npy文件,
要求数据格式为[[],[],[],[]...]格式的列表或者numpy数组.
且,只能用这个文件来操作源数据,其他文件只能调用这个文件的函数来获取数据,
不能直接操作源数据,以保证数据的安全性和一致性
2026-03-16 刘加龙 南京 陆工大
"""
import os
import sys
from pathlib import Path
import yaml
import numpy as np

# 计算项目根目录的绝对路径
# 当前脚本位置: 05/src/data/loaddata.py
# 向上三级目录：data/ -> src/ -> 05/
project_root = Path(__file__).parent.parent.parent  # 注意层级数

# 将项目根目录添加到 sys.path 的开头
sys.path.insert(0, str(project_root))

# 现在可以正常导入 src 下的模块了
from src.utils import functions as func
from src.utils import plotter as plt
from src.methods import ft

# 加载配置文件
with open(project_root / "configs" / "signal_configs.yaml",
          "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def Loadnpy(filename=config['loadsignal']):
    # 定义数据文件的路径
    datapath = os.path.join(project_root , "data" , filename)
    try:
        # 从 .npy 文件中加载数据
        loadnpy = func.load_list_from_npy(datapath)
    # 如果文件不存在，捕获 FileNotFoundError 异常并处理
    except FileNotFoundError :
        # 如果文件不存在，打印错误信息并返回 None 或空列表
        print(f"错误: 数据文件 {datapath} 未找到。请确保文件存在于指定路径。")
        loadnpy = None # 或者 loadnpy = []，根据需要选择返回 None 还是空列表
    return loadnpy
import csv

def Loadcsv(filename=config['loadcsv'],col_name=['Time(s)','CH 1'],delimiter=';'):
    # 定义数据文件的路径
    datapath = os.path.join(project_root , "data" , filename)
    x, y = [], []
    with open(datapath, 'r') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            x.append(float(row[col_name[0]]))
            y.append(float(row[col_name[1]]))
    return x, y

if __name__ == "__main__":
    time,loadcsv = Loadcsv(os.path.join(project_root , "data" , "GA-2400.csv"))
    # time2,loadcsv2 = Loadcsv(os.path.join(project_root , "data" , "A-600.csv"))
    # loadnpy = loadnpy.flatten().tolist() # 将二维数组展平为一维列表
    signal = loadcsv[:32000]
    # signal2=loadcsv2[:32000]
    t=time[:32000]
    fq,A = ft.FFTl(signal,sample_rate=16000)
    # fq2,A2 = ft.FFTl(signal2,sample_rate=16000)
    # ── 设置全局样式（一次设定，后面全部复用）──
    my_style = plt.PlotStyle()
    pt = plt.Plotter(my_style)

    # ── 1. 单图（多条线）──
    data_multi = plt.PlotData(
        x=fq, y=[A],
        title="",
        xlabel="频率 (Hz)", ylabel="幅值",
        labels=["Signal", "Signal 2"],
        colors=["#0026FF", "#FF0000"],
        linestyles=["-","--"],
        markers=["",""],
    )
    pt.single(data_multi, figsize=(6, 4), show=True)
    #── 绘制频谱图 ──
    data_h = [
        plt.PlotData(t, [signal], "(a) AM-FM时域信号", "", ""),
        plt.PlotData(fq, [A], "(b) AM-FM信号频谱", "", ""),
    ]
    pt.horizontal(data_h, figsize=(6, 4), show=True)
    
    # """一行代码快速画图（单图）"""
    # plt.quick_plot(fq,A,figsize=(5.5,4.8))