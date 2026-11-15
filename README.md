# CFC-VMD: Carrier-Frequency-Calibration Variational Mode Decomposition
# CFC-VMD：载波频率校准变分模态分解

This repository provides the code for CFC-VMD, a correction method for the carrier-frequency locating bias of Variational Mode Decomposition (VMD).
本仓库提供 CFC-VMD 的代码，该方法用于修正变分模态分解（VMD）的载波频率定位偏置。

## Overview
## 简介

Variational Mode Decomposition (VMD) locates each mode's center frequency by locking the spectral centroid.
变分模态分解（VMD）通过锁定频谱质心来定位各模态的中心频率。

For combined AM-FM signals, the asymmetric sidebands shift the spectral centroid away from the true carrier frequency.
对于组合的调幅-调频信号，不对称的边频带会把频谱质心推离真实载波频率。

As a result, the center frequency obtained by VMD deviates from the true carrier frequency.
因此，VMD 得到的中心频率会偏离真实的载波频率。

CFC-VMD corrects this bias via clipping.
CFC-VMD 通过限幅的方法修正这一偏置。

## Based on / Derivative work
## 基于的工作 / 衍生作品

This code modifies `vmdpy.py`, the Python implementation of VMD by Vinícius Carvalho and Eduardo Mazoni, released under the MIT License.
本代码修改自 `vmdpy.py`，即 Vinícius Carvalho 与 Eduardo Mazoni 的 VMD Python 实现，以 MIT 许可证发布。

Original VMD paper: Dragomiretskiy and Zosso, "Variational Mode Decomposition," IEEE Transactions on Signal Processing, vol. 62, no. 3, pp. 531–544, 2014.
VMD 原论文：Dragomiretskiy 与 Zosso，"Variational Mode Decomposition"，IEEE Transactions on Signal Processing，第 62 卷第 3 期，531–544 页，2014 年。

The original VMD routine is retained; the carrier-frequency calibration step is the original contribution of this work.
原有的 VMD 例程被保留；载波频率校准步骤是本工作的原创贡献。

## Original contribution
## 原创贡献

The carrier-frequency calibration step (searching the spectral peak near the centroid) is the original contribution of CFC-VMD.
载波频率校准步骤（在质心附近搜索频谱峰值）是 CFC-VMD 的原创贡献。

The method applies when the frequency modulation index satisfies the condition given in the paper (β ≤ β_crit).
该方法在调频指数满足论文所给条件（β ≤ β_crit）时适用。

## License
## 许可证

This project is distributed under the MIT License.
本项目以 MIT 许可证发布。

Copyright (c) 2019 Vinícius Carvalho & Eduardo Mazoni.
版权所有 (c) 2019 Vinícius Carvalho 与 Eduardo Mazoni。

Copyright (c) 2026 Jialong Liu.
版权所有 (c) 2026 Jialong Liu。

See the `LICENSE` file for the full license text.
完整许可文本见 `LICENSE` 文件。

The original vmdpy license is also kept in `vmdpy_LICENSE.txt`.
vmdpy 的原始许可证保留在 `vmdpy_LICENSE.txt` 中。

## AI assistance note
## AI 辅助说明

Some auxiliary scripts (for example, visualization, data loading) were produced with AI assistance and reviewed and verified by the authors.
部分辅助脚本（例如可视化、数据加载）由 AI 辅助生成，并经作者审查与验证。


## Files
## 文件说明

### Codes
### 代码文件说明

`experiments.py` — script that runs the experiments in the paper.
`experiments.py` — 运行论文实验的脚本。

`cfcvmd.py` — the modified VMD implementation containing the CFC-VMD carrier-frequency calibration step.
`cfcvmd.py` — 经修改的 VMD 实现，包含 CFC-VMD 的载波频率校准步骤。

`cfc.py` — the purely clipping-based method is a processing technique used in the data acquisition process.
`cfc.py` — 纯限幅方法，是用于数据采集的过程方法。

`ft.py` — Fourier transform functions.
`ft.py` — 快速傅里叶变换函数。

`functions.py`,`array_saver.py`,`bandpass_filter.py`,`statistics.py` — auxiliary functions used in the program.
`functions.py`,`array_saver.py`,`bandpass_filter.py`,`statistics.py` — 程序中的辅助函数。

`am_fm_signal.py`,`batch_gen.py`,`func_signal.py`,`signal_pretreat.py` — signal generation programs used in the paper.
`am_fm_signal.py`,`batch_gen.py`,`func_signal.py`,`signal_pretreat.py` — 论文用信号生成程序。

`loaddata.py` — data loading program used in the paper.
`loaddata.py` — 论文用数据加载程序。

`ploter.py`,`ploter3d.py` — plotting functions.
`ploter.py`,`ploter3d.py` — 绘图函数。

### Generated data
### 生成数据文件

`AM_FMBiasShowT`, `AM_FMBiasShowF` — Illustration of the power difference between sideband pairs 
`AM_FMBiasShowT`, `AM_FMBiasShowF` — 边频对功率差展示

`AM_FMBiasShowT2`, `AM_FMBiasShowF2` — Illustration of spectral centroid shift for the single-tone case 
`AM_FMBiasShowT2`, `AM_FMBiasShowF2` — 单音时频谱质心偏移展示

`AM_FMBiasShowT3`, `AM_FMBiasShowF3` — Illustration of spectral centroid shift for the multi-tone case 
`AM_FMBiasShowT3`, `AM_FMBiasShowF3` — 多音时频谱质心偏移展示

`AM_FMShowSginalT`, `AM_FMShowSginalF` — VMD results for spectrally asymmetric AM-FM signals 
`AM_FMShowSginalT`, `AM_FMShowSginalF` — 频谱不对称AM-FM信号的VMD结果与限幅方法展示

`AM_FMShowCFCT`, `AM_FMShowCFCF` — Illustration of the clipping method 
`AM_FMShowCFCT`, `AM_FMShowCFCF` — 限幅方法展示

`OmegaCaseTD1.csv`, `OmegaCaseFD1.csv` — Time-domain and frequency-domain data obtained by VMD in a single-case frequency-locating experiment.
`OmegaCaseTD1.csv`、`OmegaCaseFD1.csv` — 频率定位个例实验中由 VMD 得到的时域与频域数据。

`OmegaCaseTD2.csv`, `OmegaCaseFD2.csv` — Time-domain and frequency-domain data obtained by CFC-VMD in a single-case frequency-locating experiment.
`OmegaCaseTD2.csv`、`OmegaCaseFD2.csv` — 频率定位个例实验中由 CFC-VMD 得到的时域与频域数据。

`VMDOmegaCompareVisual.csv`, `CFCOmegaCompareVisual.csv` — Statistical data of the frequency-locating comparison experiment between VMD and CFC-VMD.
`VMDOmegaCompareVisual.csv`、`CFCOmegaCompareVisual.csv` — VMD与CFC-VMD 频率定位对比实验的统计数据。

`VMDOmegaIterationCase.csv` — Result data of the iteration-visualization experiment for center-frequency locating by VMD and CFC-VMD.
`VMDOmegaIterationCase.csv` — VMD 与 CFC-VMD 中心频率定位迭代可视化实验的结果数据。


## Sections to be completed by the author
## 作者补充说明

- 用户仅需运行 `experiments.py` 即可完成实验，实验结果将保存在 `experiments` 所在目录下。默认情况下此程序将所有实验调用语句注释掉了，若要运行某实验，请取消注释该实验的调用语句。
- 用户仅需阅读 `cfcvmd.py`文件即可了解CFC-VMD的实现细节。
- 其他辅助程序可直接调用。
- 作者：刘加龙
- 邮箱：jialong_liu@outlook.com
- 日期：2026年11月10日