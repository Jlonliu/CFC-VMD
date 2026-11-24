# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
VMD(+CFC) 与 CFC-VMD 的分解成功率对比程序
=========================================

包含两个实验
------------
1. 主实验：K=2~10、每个 K 1000 个信号的分解成功率对比。
2. 新增实验：只取 **两种方法都成功** 的模态，比较两者的中心频率绝对误差，
   统计平均误差、分位数以及逐模态胜率，并输出独立 CSV 与对比图。

实验设置
--------
* 模态数量 K = 2, 3, ..., 10；
* 每个 K 独立生成 N=1000 个 AM-FM 混合信号；
* 每个信号的 K 个载波在 0~600 Hz 内按分层均匀方式布置（可通过
  ``--uniform-carriers`` 改为完全独立均匀随机）；
* 每个模态的参数按附件表格采样：
    - 调频频率       f_f   ~ U(3, 10) Hz
    - 调幅频率       f_a   ∈ {f_f, f_f/2}，等概率
    - 调幅指数       m     ~ U(0, 0.9)
    - 调频指数       beta  ~ U(0, 2)
    - 调幅-调频相位差 Δφ   ~ U(0, 2π)
  另取调频初相为 0，调幅初相为 Δφ（整体初相不影响中心频率）。
* VMD 参数：alpha = 100K，tau=0，DC=0，init=1，tol=1e-5；
  （默认按模态数量线性设定 alpha=100K；也可用 --alpha-mode fixed
   指定固定 alpha，或用 --alpha-mode 4k2 改用 alpha=4K²。）
  （tol 取 1e-5 是因为实测继续收紧到 1e-7 时中心频率已基本不再变化，
   收敛容差只增加迭代次数；两种方法使用同一 tol）
* 分解成功判据：K 个中心频率与 K 个真实载波频率经匈牙利算法一一匹配后，
  **所有** |f_est - f_true| < 1 Hz 才记该信号分解成功；
* VMD 的中心频率用 cfc.CFC(u, fs) 校准；CFC-VMD 直接使用
  omega[-1, :] * fs（本身已校准）。

运行
----
    python compare_vmd_cfcvmd_success.py

常用选项
--------
    python compare_vmd_cfcvmd_success.py --n-signals 50 --workers 1  # 快速试跑
    python compare_vmd_cfcvmd_success.py --workers 4                  # 内存充足时加速
    python compare_vmd_cfcvmd_success.py --alpha-mode fixed --alpha 100
    python compare_vmd_cfcvmd_success.py --uniform-carriers            # 严格 i.i.d. 均匀

仅分析已有结果（不重新分解）：
    python compare_vmd_cfcvmd_success.py --analyze-existing ^
        "experiments/VMD_CFCVMD_ModeCount_Success_alpha100K"

输出（默认 experiments/VMD_CFCVMD_ModeCount_Success_alpha100K/）
----------------------------------------------------------------
    success_rate_by_K.csv             主实验绘图数据
    per_signal_results.csv            每个信号的详细结果
    success_rate_by_K.png/pdf         主实验对比图
    run_metadata.json                 本次运行的完整参数

    paired_mode_errors.csv            新增实验：共同成功模态的逐条误差
    paired_mode_quality_by_K.csv      新增实验：按 K 汇总的误差与胜率
    paired_signal_quality_by_K.csv    新增实验补充：两种方法都整体成功的信号
    paired_mode_quality_by_K.png/pdf  新增实验对比图
    paired_mode_quality_metadata.json 新增实验的口径说明

本脚本只新增文件，不修改项目中的任何既有程序。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# 避免 BLAS/OpenMP 在多进程时线程超订。必须在 import numpy 之前设置。
for _env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_name, "1")

import numpy as np
from scipy.optimize import linear_sum_assignment

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 项目路径与导入
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    """同时兼容脚本放在 Code 根目录、Code/experiments 子目录或他处调用。"""
    here = Path(__file__).resolve().parent
    candidates: List[Path] = []
    env_root = os.environ.get("VMD_CFC_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.extend([here, here.parent])
    # 从当前工作目录向上找，方便在项目根目录直接调用其他位置的脚本。
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "src" / "methods" / "cfcvmd.py").is_file():
            return candidate
    raise FileNotFoundError(
        "未找到项目根目录（应包含 src/methods/cfcvmd.py）。"
        f"当前脚本目录：{here}；可设置环境变量 VMD_CFC_PROJECT_ROOT 指定。"
    )


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import vmdpy  # noqa: E402
from src.methods import cfc, cfcvmd  # noqa: E402
from src.data.am_fm_signal import gen_signal  # noqa: E402


# ---------------------------------------------------------------------------
# 默认实验参数
# ---------------------------------------------------------------------------
DEFAULT_K_VALUES = tuple(range(2, 11))
DEFAULT_N_SIGNALS = 1000
DEFAULT_SEED = 20261118
DEFAULT_FS = 1400.0          # > 2 * 600 Hz，并给调制边带留余量
DEFAULT_DURATION = 1.0       # 频率分辨率 1 Hz
DEFAULT_BAND_MAX = 600.0
DEFAULT_THRESHOLD_HZ = 1.0
DEFAULT_TOL = 1e-5
DEFAULT_ALPHA = 2000.0
DEFAULT_CARRIER_MARGIN = 0.15  # 分层采样时每个频段两端各留 15%
DEFAULT_WORKERS = 2

# AM/FM 参数范围（来自附件表格）
FM_FREQ_RANGE = (3.0, 10.0)
AM_INDEX_RANGE = (0.0, 0.9)
FM_INDEX_RANGE = (0.0, 2.0)
PHASE_DIFF_RANGE = (0.0, 2.0 * math.pi)

# 子进程配置（由 initializer 写入）
_WORKER_CFG: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 信号生成
# ---------------------------------------------------------------------------
def _carrier_frequencies(
    K: int,
    rng: np.random.Generator,
    band_max: float,
    stratified: bool,
    margin_ratio: float,
) -> np.ndarray:
    """
    生成 K 个载波频率。

    stratified=True（默认）:
        把 [0, band_max] 等分为 K 个频段，每个频段内均匀随机取一个频率，
        并在频段两端各留 margin_ratio 的空白，避免相邻模态贴得太近。
        这保证任意两个载波至少相隔 2*margin_ratio*band_max/K。

    stratified=False:
        严格在 [0, band_max] 内独立均匀随机取 K 个频率后排序。
    """
    if K <= 0:
        raise ValueError("K 必须为正整数")
    if band_max <= 0:
        raise ValueError("band_max 必须为正数")

    if not stratified:
        return np.sort(rng.uniform(0.0, band_max, size=K))

    width = band_max / K
    margin = float(np.clip(margin_ratio, 0.0, 0.45)) * width
    return np.array(
        [
            k * width + rng.uniform(margin, width - margin)
            for k in range(K)
        ],
        dtype=float,
    )


def _generate_signal(
    K: int,
    rng: np.random.Generator,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成一个由 K 个独立 AM-FM 单载波分量叠加而成的信号。

    x(t) = Σ_k [1 + m_k cos(2π f_a,k t + Δφ_k)]
                 · cos(2π f_c,k t + β_k sin(2π f_f,k t))

    返回 (归一化信号, 真实载波频率数组)。
    """
    fs = float(cfg["fs"])
    duration = float(cfg["duration"])
    n_samples = int(round(fs * duration))
    if n_samples <= 0:
        raise ValueError("fs * duration 必须至少为 1 个采样点")

    carriers = _carrier_frequencies(
        K=K,
        rng=rng,
        band_max=float(cfg["band_max"]),
        stratified=bool(cfg["stratified_carriers"]),
        margin_ratio=float(cfg["carrier_margin"]),
    )

    signal = np.zeros(n_samples, dtype=float)
    for fc in carriers:
        f_fm = rng.uniform(*FM_FREQ_RANGE)
        # 调幅频率以 1/2 概率取 f_f 或 f_f/2
        f_am = f_fm if rng.random() < 0.5 else 0.5 * f_fm
        m = rng.uniform(*AM_INDEX_RANGE)
        beta = rng.uniform(*FM_INDEX_RANGE)
        phase_diff = rng.uniform(*PHASE_DIFF_RANGE)

        _, component = gen_signal(
            freqs=[float(fc)],
            mode="AM_FM",
            amps=[1.0],
            m=[float(m)],
            fm_freqs=[float(f_am)],
            am_phases=[float(phase_diff)],
            beta=[float(beta)],
            fm_mod_freqs=[float(f_fm)],
            fm_phases=[0.0],
            fs=fs,
            duration=duration,
            noise_std=0.0,
        )
        signal += component

    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal /= peak
    return signal, carriers


# ---------------------------------------------------------------------------
# 中心频率匹配与成功判据
# ---------------------------------------------------------------------------
def _score_frequencies(
    predicted_hz: Sequence[float],
    true_hz: Sequence[float],
    threshold_hz: float,
) -> Dict[str, Any]:
    """
    用匈牙利算法把预测中心频率与真实载波频率一一匹配，误差按真实模态顺序返回。

    成功: 所有误差都严格小于 threshold_hz。
    """
    true_arr = np.asarray(true_hz, dtype=float).ravel()
    pred_arr = np.asarray(predicted_hz, dtype=float).ravel()
    K = true_arr.size

    result: Dict[str, Any] = {
        "matched_hz": np.full(K, np.nan),
        "errors_hz": np.full(K, np.nan),
        "max_error_hz": float("nan"),
        "success": False,
        "modes_ok": 0,
    }
    if pred_arr.size != K or not np.all(np.isfinite(pred_arr)):
        return result

    cost = np.abs(pred_arr[:, None] - true_arr[None, :])
    rows, cols = linear_sum_assignment(cost)

    matched = np.full(K, np.nan)
    errors = np.full(K, np.nan)
    matched[cols] = pred_arr[rows]
    errors[cols] = cost[rows, cols]

    max_error = float(np.max(errors))
    result.update(
        matched_hz=matched,
        errors_hz=errors,
        max_error_hz=max_error,
        success=bool(max_error < threshold_hz),
        modes_ok=int(np.sum(errors < threshold_hz)),
    )
    return result


# ---------------------------------------------------------------------------
# 单信号实验（子进程入口）
# ---------------------------------------------------------------------------
def _init_worker(cfg: Dict[str, Any]) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _failure_record(K: int, signal_index: int, seed: int, message: str) -> Dict[str, Any]:
    nan_col = [float("nan")] * K
    return {
        "K": K,
        "signal_index": signal_index,
        "seed": seed,
        "true_freqs_hz": [],
        "vmd_matched_freqs_hz": list(nan_col),
        "vmd_errors_hz": list(nan_col),
        "vmd_max_error_hz": float("nan"),
        "vmd_success": False,
        "vmd_modes_ok": 0,
        "vmd_seconds": float("nan"),
        "vmd_error": message,
        "cfcvmd_matched_freqs_hz": list(nan_col),
        "cfcvmd_errors_hz": list(nan_col),
        "cfcvmd_max_error_hz": float("nan"),
        "cfcvmd_success": False,
        "cfcvmd_modes_ok": 0,
        "cfcvmd_seconds": float("nan"),
        "cfcvmd_error": message,
    }


def _run_one(task: Tuple[int, int, int]) -> Dict[str, Any]:
    """对一个信号分别运行 VMD(+CFC) 与 CFC-VMD。"""
    K, signal_index, seed = task
    cfg = _WORKER_CFG
    threshold = float(cfg["threshold_hz"])
    alpha = float(cfg["alpha"])
    tol = float(cfg["tol"])
    fs = float(cfg["fs"])

    record = _failure_record(K, signal_index, seed, "")
    record["vmd_error"] = ""
    record["cfcvmd_error"] = ""

    # ---- 生成信号 ---------------------------------------------------------
    try:
        rng = np.random.default_rng(seed)
        signal, true_freqs = _generate_signal(K, rng, cfg)
        record["true_freqs_hz"] = true_freqs.tolist()
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        record["vmd_error"] = msg
        record["cfcvmd_error"] = msg
        return record

    # ---- VMD + CFC 校准 ---------------------------------------------------
    try:
        t0 = time.perf_counter()
        u, _u_hat, _omega_vmd = vmdpy.VMD(
            signal,
            alpha=alpha,
            tau=0.0,
            K=K,
            DC=0,
            init=1,
            tol=tol,
        )
        _u_lim, _u_hat_cfc, omega_cfc = cfc.CFC(u, fs)
        record["vmd_seconds"] = time.perf_counter() - t0

        score = _score_frequencies(omega_cfc.ravel(), true_freqs, threshold)
        record.update(
            vmd_matched_freqs_hz=score["matched_hz"].tolist(),
            vmd_errors_hz=score["errors_hz"].tolist(),
            vmd_max_error_hz=score["max_error_hz"],
            vmd_success=score["success"],
            vmd_modes_ok=score["modes_ok"],
        )
    except Exception as exc:  # noqa: BLE001
        record["vmd_error"] = f"{type(exc).__name__}: {exc}"

    # ---- CFC-VMD ----------------------------------------------------------
    try:
        t0 = time.perf_counter()
        _u2, _u_hat2, omega_cfcvmd = cfcvmd.CFCVMD(
            signal,
            alpha=alpha,
            tau=0.0,
            K=K,
            DC=0,
            init=1,
            tol=tol,
            feedback=True,
        )
        record["cfcvmd_seconds"] = time.perf_counter() - t0

        # CFCVMD 的 omega 为归一化频率（cycles/sample），乘采样率转为 Hz。
        cfcvmd_hz = np.asarray(omega_cfcvmd[-1, :], dtype=float) * fs
        score = _score_frequencies(cfcvmd_hz, true_freqs, threshold)
        record.update(
            cfcvmd_matched_freqs_hz=score["matched_hz"].tolist(),
            cfcvmd_errors_hz=score["errors_hz"].tolist(),
            cfcvmd_max_error_hz=score["max_error_hz"],
            cfcvmd_success=score["success"],
            cfcvmd_modes_ok=score["modes_ok"],
        )
    except Exception as exc:  # noqa: BLE001
        record["cfcvmd_error"] = f"{type(exc).__name__}: {exc}"

    return record


# ---------------------------------------------------------------------------
# 汇总、CSV 与绘图
# ---------------------------------------------------------------------------
def _fmt_array(values: Iterable[float]) -> str:
    parts: List[str] = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            parts.append("")
        else:
            parts.append(f"{v:.6f}" if np.isfinite(v) else "")
    return ";".join(parts)


def _wilson_interval(success_count: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = success_count / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _nanmean_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _nanmedian_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _summarize(K: int, alpha: float, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    vmd_success = sum(int(bool(r["vmd_success"])) for r in records)
    cfc_success = sum(int(bool(r["cfcvmd_success"])) for r in records)
    total_modes = n * K
    vmd_modes_ok = sum(int(r["vmd_modes_ok"]) for r in records)
    cfc_modes_ok = sum(int(r["cfcvmd_modes_ok"]) for r in records)

    vmd_all_errors = [e for r in records for e in r["vmd_errors_hz"] if np.isfinite(e)]
    cfc_all_errors = [e for r in records for e in r["cfcvmd_errors_hz"] if np.isfinite(e)]
    vmd_max_errors = [r["vmd_max_error_hz"] for r in records]
    cfc_max_errors = [r["cfcvmd_max_error_hz"] for r in records]

    vmd_ci = _wilson_interval(vmd_success, n)
    cfc_ci = _wilson_interval(cfc_success, n)

    return {
        "K": K,
        "n_signals": n,
        "alpha": alpha,
        "vmd_success_count": vmd_success,
        "vmd_success_rate": vmd_success / n if n else float("nan"),
        "vmd_success_ci95_low": vmd_ci[0],
        "vmd_success_ci95_high": vmd_ci[1],
        "cfcvmd_success_count": cfc_success,
        "cfcvmd_success_rate": cfc_success / n if n else float("nan"),
        "cfcvmd_success_ci95_low": cfc_ci[0],
        "cfcvmd_success_ci95_high": cfc_ci[1],
        "vmd_modes_ok_count": vmd_modes_ok,
        "vmd_mode_success_rate": vmd_modes_ok / total_modes if total_modes else float("nan"),
        "cfcvmd_modes_ok_count": cfc_modes_ok,
        "cfcvmd_mode_success_rate": cfc_modes_ok / total_modes if total_modes else float("nan"),
        "vmd_mean_mode_abs_error_hz": _nanmean_or_nan(vmd_all_errors),
        "cfcvmd_mean_mode_abs_error_hz": _nanmean_or_nan(cfc_all_errors),
        "vmd_mean_max_abs_error_hz": _nanmean_or_nan(vmd_max_errors),
        "cfcvmd_mean_max_abs_error_hz": _nanmean_or_nan(cfc_max_errors),
        "vmd_median_max_abs_error_hz": _nanmedian_or_nan(vmd_max_errors),
        "cfcvmd_median_max_abs_error_hz": _nanmedian_or_nan(cfc_max_errors),
    }


SUMMARY_FIELDS = [
    "K",
    "n_signals",
    "alpha",
    "vmd_success_count",
    "vmd_success_rate",
    "vmd_success_ci95_low",
    "vmd_success_ci95_high",
    "cfcvmd_success_count",
    "cfcvmd_success_rate",
    "cfcvmd_success_ci95_low",
    "cfcvmd_success_ci95_high",
    "vmd_modes_ok_count",
    "vmd_mode_success_rate",
    "cfcvmd_modes_ok_count",
    "cfcvmd_mode_success_rate",
    "vmd_mean_mode_abs_error_hz",
    "cfcvmd_mean_mode_abs_error_hz",
    "vmd_mean_max_abs_error_hz",
    "cfcvmd_mean_max_abs_error_hz",
    "vmd_median_max_abs_error_hz",
    "cfcvmd_median_max_abs_error_hz",
]

PER_SIGNAL_FIELDS = [
    "K",
    "signal_index",
    "seed",
    "true_freqs_hz",
    "vmd_matched_freqs_hz",
    "vmd_errors_hz",
    "vmd_max_error_hz",
    "vmd_success",
    "vmd_modes_ok",
    "vmd_seconds",
    "vmd_error",
    "cfcvmd_matched_freqs_hz",
    "cfcvmd_errors_hz",
    "cfcvmd_max_error_hz",
    "cfcvmd_success",
    "cfcvmd_modes_ok",
    "cfcvmd_seconds",
    "cfcvmd_error",
]


def _record_to_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "K": record["K"],
        "signal_index": record["signal_index"],
        "seed": record["seed"],
        "true_freqs_hz": _fmt_array(record["true_freqs_hz"]),
        "vmd_matched_freqs_hz": _fmt_array(record["vmd_matched_freqs_hz"]),
        "vmd_errors_hz": _fmt_array(record["vmd_errors_hz"]),
        "vmd_max_error_hz": record["vmd_max_error_hz"],
        "vmd_success": int(bool(record["vmd_success"])),
        "vmd_modes_ok": record["vmd_modes_ok"],
        "vmd_seconds": record["vmd_seconds"],
        "vmd_error": record["vmd_error"],
        "cfcvmd_matched_freqs_hz": _fmt_array(record["cfcvmd_matched_freqs_hz"]),
        "cfcvmd_errors_hz": _fmt_array(record["cfcvmd_errors_hz"]),
        "cfcvmd_max_error_hz": record["cfcvmd_max_error_hz"],
        "cfcvmd_success": int(bool(record["cfcvmd_success"])),
        "cfcvmd_modes_ok": record["cfcvmd_modes_ok"],
        "cfcvmd_seconds": record["cfcvmd_seconds"],
        "cfcvmd_error": record["cfcvmd_error"],
    }


def _atomic_write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _plot_success_rate(summary: Sequence[Dict[str, Any]], path_png: Path, path_pdf: Path) -> None:
    """
    绘制 K—分解成功率对比图。

    左图：整体分解成功率（一个信号的 K 个中心频率全部命中才算成功）。
    右图：单模态定位成功率（命中模态数 / 总模态数），用于观察高 K 下
          “部分模态已经命中、但整体因个别模态失配而失败”的情况。
    """
    # Windows 常见中文字体；不存在时 matplotlib 会自动回退。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    K = np.array([row["K"] for row in summary], dtype=int)
    vmd_rate = np.array([row["vmd_success_rate"] for row in summary], dtype=float) * 100.0
    cfc_rate = np.array([row["cfcvmd_success_rate"] for row in summary], dtype=float) * 100.0
    vmd_mode_rate = np.array([row["vmd_mode_success_rate"] for row in summary], dtype=float) * 100.0
    cfc_mode_rate = np.array([row["cfcvmd_mode_success_rate"] for row in summary], dtype=float) * 100.0

    vmd_low = np.array([row["vmd_success_ci95_low"] for row in summary], dtype=float) * 100.0
    vmd_high = np.array([row["vmd_success_ci95_high"] for row in summary], dtype=float) * 100.0
    cfc_low = np.array([row["cfcvmd_success_ci95_low"] for row in summary], dtype=float) * 100.0
    cfc_high = np.array([row["cfcvmd_success_ci95_high"] for row in summary], dtype=float) * 100.0

    # 浮点舍入可能让上/下界比点估计略小/大，误差条必须非负。
    vmd_err = np.clip(np.vstack([vmd_rate - vmd_low, vmd_high - vmd_rate]), 0.0, None)
    cfc_err = np.clip(np.vstack([cfc_rate - cfc_low, cfc_high - cfc_rate]), 0.0, None)

    colors = {"vmd": "#4477AA", "cfc": "#CC3311"}
    markers = {"vmd": "o", "cfc": "s"}
    labels = {"vmd": "VMD + CFC 校准", "cfc": "CFC-VMD"}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    panels = (
        (axes[0], vmd_rate, cfc_rate, vmd_err, cfc_err,
         "(a) 整体分解成功率（K 个模态全部命中）"),
        (axes[1], vmd_mode_rate, cfc_mode_rate, None, None,
         "(b) 单模态定位成功率（命中模态数 / 总模态数）"),
    )

    for ax, vmd_y, cfc_y, vmd_e, cfc_e, title in panels:
        if vmd_e is not None:
            ax.errorbar(
                K, vmd_y, yerr=vmd_e,
                color=colors["vmd"], marker=markers["vmd"], markersize=5.5,
                linewidth=1.8, capsize=3, label=labels["vmd"], zorder=3,
            )
            ax.errorbar(
                K, cfc_y, yerr=cfc_e,
                color=colors["cfc"], marker=markers["cfc"], markersize=5.5,
                linewidth=1.8, capsize=3, label=labels["cfc"], zorder=4,
            )
        else:
            ax.plot(
                K, vmd_y,
                color=colors["vmd"], marker=markers["vmd"], markersize=5.5,
                linewidth=1.8, label=labels["vmd"], zorder=3,
            )
            ax.plot(
                K, cfc_y,
                color=colors["cfc"], marker=markers["cfc"], markersize=5.5,
                linewidth=1.8, label=labels["cfc"], zorder=4,
            )
        ax.set_xlabel("模态数量 $K$", fontsize=11.5)
        ax.set_xticks(K)
        ax.set_xlim(K.min() - 0.4, K.max() + 0.4)
        ax.set_ylim(-2.0, 102.0)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.legend(loc="best", frameon=False, fontsize=9.5)

    axes[0].set_ylabel("分解成功率 (%)", fontsize=11.5)
    fig.suptitle("VMD(+CFC) 与 CFC-VMD 的分解成功率对比", fontsize=13.5, y=1.01)

    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 新增实验：两种方法都成功的模态，比较中心频率误差
# ---------------------------------------------------------------------------
PAIRED_MODE_FIELDS = [
    "K",
    "signal_index",
    "seed",
    "mode_index",
    "true_freq_hz",
    "vmd_abs_error_hz",
    "cfcvmd_abs_error_hz",
    "diff_vmd_minus_cfcvmd_hz",
    "better_method",
]

PAIRED_QUALITY_FIELDS = [
    "K",
    "n_paired_modes",
    "vmd_mean_abs_error_hz",
    "vmd_mean_ci95_low",
    "vmd_mean_ci95_high",
    "cfcvmd_mean_abs_error_hz",
    "cfcvmd_mean_ci95_low",
    "cfcvmd_mean_ci95_high",
    "vmd_median_abs_error_hz",
    "cfcvmd_median_abs_error_hz",
    "vmd_p90_abs_error_hz",
    "cfcvmd_p90_abs_error_hz",
    "vmd_p95_abs_error_hz",
    "cfcvmd_p95_abs_error_hz",
    "mean_vmd_minus_cfcvmd_error_hz",
    "median_vmd_minus_cfcvmd_error_hz",
    "mean_error_reduction_pct",
    "cfc_win_count",
    "cfc_win_rate",
    "vmd_win_count",
    "vmd_win_rate",
    "tie_count",
    "tie_rate",
]

PAIRED_SIGNAL_FIELDS = [
    "K",
    "n_common_success_signals",
    "vmd_mean_max_error_hz",
    "cfcvmd_mean_max_error_hz",
    "vmd_median_max_error_hz",
    "cfcvmd_median_max_error_hz",
    "mean_vmd_minus_cfcvmd_max_error_hz",
    "cfc_win_count",
    "cfc_win_rate",
    "vmd_win_count",
    "vmd_win_rate",
    "tie_count",
    "tie_rate",
]


def _resolve_detail_csv(path: Path) -> Path:
    """接受 per_signal_results.csv 文件或其所在结果目录。"""
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "per_signal_results.csv"
    if not resolved.is_file():
        raise FileNotFoundError(f"未找到逐信号结果文件：{resolved}")
    return resolved


def _split_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(item) for item in value.split(";") if item != ""]


def _parse_paired_results(
    detail_csv: Path,
    threshold_hz: float,
    tie_tol_hz: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    读取 per_signal_results.csv，抽取两类配对样本：

    1) 模态级：某个真实模态在两种方法下的误差都 < threshold_hz；
    2) 信号级（补充）：两种方法都整体成功的信号。

    返回 (paired_modes, common_success_signals)。
    """
    paired_modes: List[Dict[str, Any]] = []
    common_signals: List[Dict[str, Any]] = []

    with detail_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            K = int(row["K"])
            signal_index = int(row["signal_index"])
            seed = row.get("seed", "")
            true_freqs = _split_float_list(row.get("true_freqs_hz", ""))
            vmd_errors = _split_float_list(row.get("vmd_errors_hz", ""))
            cfc_errors = _split_float_list(row.get("cfcvmd_errors_hz", ""))

            if not (
                len(true_freqs) == K
                and len(vmd_errors) == K
                and len(cfc_errors) == K
            ):
                continue

            for mode_index, (true_f, vmd_e, cfc_e) in enumerate(
                zip(true_freqs, vmd_errors, cfc_errors), start=1
            ):
                if not (
                    np.isfinite(vmd_e)
                    and np.isfinite(cfc_e)
                    and vmd_e < threshold_hz
                    and cfc_e < threshold_hz
                ):
                    continue

                diff = vmd_e - cfc_e
                if diff > tie_tol_hz:
                    better = "CFC-VMD"
                elif diff < -tie_tol_hz:
                    better = "VMD+CFC"
                else:
                    better = "Tie"

                paired_modes.append(
                    {
                        "K": K,
                        "signal_index": signal_index,
                        "seed": seed,
                        "mode_index": mode_index,
                        "true_freq_hz": true_f,
                        "vmd_abs_error_hz": vmd_e,
                        "cfcvmd_abs_error_hz": cfc_e,
                        "diff_vmd_minus_cfcvmd_hz": diff,
                        "better_method": better,
                    }
                )

            try:
                vmd_ok = int(row["vmd_success"]) == 1
                cfc_ok = int(row["cfcvmd_success"]) == 1
            except (KeyError, TypeError, ValueError):
                vmd_ok = cfc_ok = False

            if vmd_ok and cfc_ok:
                vmd_max = float(row["vmd_max_error_hz"])
                cfc_max = float(row["cfcvmd_max_error_hz"])
                diff = vmd_max - cfc_max
                if diff > tie_tol_hz:
                    better = "CFC-VMD"
                elif diff < -tie_tol_hz:
                    better = "VMD+CFC"
                else:
                    better = "Tie"
                common_signals.append(
                    {
                        "K": K,
                        "signal_index": signal_index,
                        "vmd_max_error_hz": vmd_max,
                        "cfcvmd_max_error_hz": cfc_max,
                        "diff_vmd_minus_cfcvmd_hz": diff,
                        "better_method": better,
                    }
                )

    return paired_modes, common_signals


def _mean_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    """返回 (均值, 95% CI 下限, 95% CI 上限)。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(arr))
    if arr.size < 2:
        return (mean, mean, mean)
    sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    return (mean, mean - 1.959963984540054 * sem, mean + 1.959963984540054 * sem)


def _win_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    cfc = sum(1 for r in rows if r["better_method"] == "CFC-VMD")
    vmd = sum(1 for r in rows if r["better_method"] == "VMD+CFC")
    tie = n - cfc - vmd
    return {
        "n": n,
        "cfc": cfc,
        "vmd": vmd,
        "tie": tie,
        "cfc_rate": cfc / n if n else float("nan"),
        "vmd_rate": vmd / n if n else float("nan"),
        "tie_rate": tie / n if n else float("nan"),
    }


def _summarize_paired_modes(
    paired_modes: Sequence[Dict[str, Any]],
    common_signals: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按 K 汇总共同成功模态的误差统计和共同成功信号的最大误差统计。"""
    K_values = sorted({int(r["K"]) for r in paired_modes} | {int(r["K"]) for r in common_signals})
    quality_rows: List[Dict[str, Any]] = []
    signal_rows: List[Dict[str, Any]] = []

    for K in K_values:
        pairs = [r for r in paired_modes if int(r["K"]) == K]
        vmd = [float(r["vmd_abs_error_hz"]) for r in pairs]
        cfc = [float(r["cfcvmd_abs_error_hz"]) for r in pairs]
        vmd_mean, vmd_lo, vmd_hi = _mean_ci95(vmd)
        cfc_mean, cfc_lo, cfc_hi = _mean_ci95(cfc)
        mean_diff = (
            float(np.mean(np.asarray(vmd) - np.asarray(cfc)))
            if vmd and cfc else float("nan")
        )
        median_diff = (
            float(np.median(np.asarray(vmd) - np.asarray(cfc)))
            if vmd and cfc else float("nan")
        )
        reduction = (
            100.0 * (vmd_mean - cfc_mean) / vmd_mean
            if vmd_mean and np.isfinite(vmd_mean) and vmd_mean != 0 else float("nan")
        )
        win = _win_stats(pairs)

        quality_rows.append(
            {
                "K": K,
                "n_paired_modes": len(pairs),
                "vmd_mean_abs_error_hz": vmd_mean,
                "vmd_mean_ci95_low": vmd_lo,
                "vmd_mean_ci95_high": vmd_hi,
                "cfcvmd_mean_abs_error_hz": cfc_mean,
                "cfcvmd_mean_ci95_low": cfc_lo,
                "cfcvmd_mean_ci95_high": cfc_hi,
                "vmd_median_abs_error_hz": _nanmedian_or_nan(vmd),
                "cfcvmd_median_abs_error_hz": _nanmedian_or_nan(cfc),
                "vmd_p90_abs_error_hz": float(np.percentile(vmd, 90)) if vmd else float("nan"),
                "cfcvmd_p90_abs_error_hz": float(np.percentile(cfc, 90)) if cfc else float("nan"),
                "vmd_p95_abs_error_hz": float(np.percentile(vmd, 95)) if vmd else float("nan"),
                "cfcvmd_p95_abs_error_hz": float(np.percentile(cfc, 95)) if cfc else float("nan"),
                "mean_vmd_minus_cfcvmd_error_hz": mean_diff,
                "median_vmd_minus_cfcvmd_error_hz": median_diff,
                "mean_error_reduction_pct": reduction,
                "cfc_win_count": win["cfc"],
                "cfc_win_rate": win["cfc_rate"],
                "vmd_win_count": win["vmd"],
                "vmd_win_rate": win["vmd_rate"],
                "tie_count": win["tie"],
                "tie_rate": win["tie_rate"],
            }
        )

        sigs = [r for r in common_signals if int(r["K"]) == K]
        sig_vmd = [float(r["vmd_max_error_hz"]) for r in sigs]
        sig_cfc = [float(r["cfcvmd_max_error_hz"]) for r in sigs]
        sig_win = _win_stats(sigs)
        signal_rows.append(
            {
                "K": K,
                "n_common_success_signals": len(sigs),
                "vmd_mean_max_error_hz": _nanmean_or_nan(sig_vmd),
                "cfcvmd_mean_max_error_hz": _nanmean_or_nan(sig_cfc),
                "vmd_median_max_error_hz": _nanmedian_or_nan(sig_vmd),
                "cfcvmd_median_max_error_hz": _nanmedian_or_nan(sig_cfc),
                "mean_vmd_minus_cfcvmd_max_error_hz": (
                    float(np.mean(np.asarray(sig_vmd) - np.asarray(sig_cfc)))
                    if sigs else float("nan")
                ),
                "cfc_win_count": sig_win["cfc"],
                "cfc_win_rate": sig_win["cfc_rate"],
                "vmd_win_count": sig_win["vmd"],
                "vmd_win_rate": sig_win["vmd_rate"],
                "tie_count": sig_win["tie"],
                "tie_rate": sig_win["tie_rate"],
            }
        )

    return quality_rows, signal_rows


def _plot_paired_mode_quality(
    quality_rows: Sequence[Dict[str, Any]],
    path_png: Path,
    path_pdf: Path,
) -> None:
    """绘制共同成功模态的误差大小与逐模态胜率。"""
    if not quality_rows:
        print("  没有共同成功模态，跳过质量对比图。")
        return

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    K = np.array([row["K"] for row in quality_rows], dtype=int)
    vmd_mean = np.array([row["vmd_mean_abs_error_hz"] for row in quality_rows], dtype=float)
    cfc_mean = np.array([row["cfcvmd_mean_abs_error_hz"] for row in quality_rows], dtype=float)
    vmd_lo = np.array([row["vmd_mean_ci95_low"] for row in quality_rows], dtype=float)
    vmd_hi = np.array([row["vmd_mean_ci95_high"] for row in quality_rows], dtype=float)
    cfc_lo = np.array([row["cfcvmd_mean_ci95_low"] for row in quality_rows], dtype=float)
    cfc_hi = np.array([row["cfcvmd_mean_ci95_high"] for row in quality_rows], dtype=float)
    cfc_win = np.array([row["cfc_win_rate"] for row in quality_rows], dtype=float) * 100.0
    vmd_win = np.array([row["vmd_win_rate"] for row in quality_rows], dtype=float) * 100.0

    vmd_err = np.clip(np.vstack([vmd_mean - vmd_lo, vmd_hi - vmd_mean]), 0.0, None)
    cfc_err = np.clip(np.vstack([cfc_mean - cfc_lo, cfc_hi - cfc_mean]), 0.0, None)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    ax.errorbar(
        K, vmd_mean, yerr=vmd_err,
        color="#4477AA", marker="o", markersize=5.5, linewidth=1.8,
        capsize=3, label="VMD + CFC 校准", zorder=3,
    )
    ax.errorbar(
        K, cfc_mean, yerr=cfc_err,
        color="#CC3311", marker="s", markersize=5.5, linewidth=1.8,
        capsize=3, label="CFC-VMD", zorder=4,
    )
    ax.set_xlabel("模态数量 $K$", fontsize=11.5)
    ax.set_ylabel("平均绝对误差 (Hz)", fontsize=11.5)
    ax.set_xticks(K)
    ax.set_xlim(K.min() - 0.4, K.max() + 0.4)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_title("(a) 共同成功模态的平均绝对误差", fontsize=11.5, pad=8)
    ax.legend(loc="best", frameon=False, fontsize=9.5)

    ax = axes[1]
    ax.plot(
        K, cfc_win,
        color="#CC3311", marker="s", markersize=5.5, linewidth=1.8,
        label="CFC-VMD 误差更小",
    )
    ax.plot(
        K, vmd_win,
        color="#4477AA", marker="o", markersize=5.5, linewidth=1.8,
        label="VMD+CFC 误差更小",
    )
    ax.axhline(50.0, color="0.45", linestyle="--", linewidth=1.0, label="50% 基准")
    win_min = float(np.nanmin(np.concatenate([cfc_win, vmd_win])))
    win_max = float(np.nanmax(np.concatenate([cfc_win, vmd_win])))
    lower = max(0.0, math.floor(win_min - 2.0))
    upper = min(100.0, math.ceil(win_max + 2.0))
    if upper - lower < 10.0:
        mid = 0.5 * (upper + lower)
        lower, upper = mid - 5.0, mid + 5.0
    ax.set_ylim(lower, upper)
    ax.set_xlabel("模态数量 $K$", fontsize=11.5)
    ax.set_ylabel("逐模态比较胜率 (%)", fontsize=11.5)
    ax.set_xticks(K)
    ax.set_xlim(K.min() - 0.4, K.max() + 0.4)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_title("(b) 共同成功模态的逐模态胜率", fontsize=11.5, pad=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=9.0,
    )

    fig.suptitle("共同成功模态的中心频率误差比较", fontsize=13.5, y=1.01)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def _run_paired_quality_analysis(
    detail_csv: Path,
    output_dir: Path,
    threshold_hz: float,
    tie_tol_hz: float,
    make_plot: bool,
) -> List[Dict[str, Any]]:
    """
    新增实验入口：只统计两种方法都成功的模态，比较其绝对误差。

    输出：
      paired_mode_errors.csv          配对模态逐条误差
      paired_mode_quality_by_K.csv    按 K 汇总的误差与胜率
      paired_signal_quality_by_K.csv  补充：两种方法都整体成功的信号
      paired_mode_quality_by_K.png/pdf
      paired_mode_quality_metadata.json
    """
    print("\n" + "-" * 72)
    print("新增实验：共同成功模态的中心频率误差比较")
    print(f"数据来源 : {detail_csv}")
    print(f"筛选条件 : 该真实模态在两种方法下的 |误差| 均 < {threshold_hz:g} Hz")
    print(f"平局容差 : {tie_tol_hz:g} Hz")
    print("-" * 72, flush=True)

    paired_modes, common_signals = _parse_paired_results(
        detail_csv, threshold_hz, tie_tol_hz
    )
    quality_rows, signal_rows = _summarize_paired_modes(
        paired_modes, common_signals
    )

    errors_path = output_dir / "paired_mode_errors.csv"
    quality_path = output_dir / "paired_mode_quality_by_K.csv"
    signal_path = output_dir / "paired_signal_quality_by_K.csv"
    png_path = output_dir / "paired_mode_quality_by_K.png"
    pdf_path = output_dir / "paired_mode_quality_by_K.pdf"
    meta_path = output_dir / "paired_mode_quality_metadata.json"

    _atomic_write_csv(errors_path, paired_modes, PAIRED_MODE_FIELDS)
    _atomic_write_csv(quality_path, quality_rows, PAIRED_QUALITY_FIELDS)
    _atomic_write_csv(signal_path, signal_rows, PAIRED_SIGNAL_FIELDS)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_detail_csv": str(detail_csv),
        "definition": (
            "对每个真实模态，若 VMD+CFC 与 CFC-VMD 的绝对误差均 < threshold_hz，"
            "则纳入配对误差比较；逐模态比较时误差更小者获胜。"
        ),
        "threshold_hz": float(threshold_hz),
        "tie_tolerance_hz": float(tie_tol_hz),
        "n_paired_modes_total": len(paired_modes),
        "n_common_success_signals_total": len(common_signals),
        "note": (
            "diff_vmd_minus_cfcvmd_hz > 0 表示 CFC-VMD 误差更小；"
            "< 0 表示 VMD+CFC 误差更小。"
        ),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if make_plot:
        _plot_paired_mode_quality(quality_rows, png_path, pdf_path)

    print(f"配对模态数 : {len(paired_modes)}")
    print(f"明细数据   : {errors_path}")
    print(f"汇总数据   : {quality_path}")
    print(f"信号级补充 : {signal_path}")
    if make_plot:
        print(f"误差对比图 : {png_path}")
        print(f"矢量图     : {pdf_path}")

    print("\nK   配对模态数   VMD+CFC均值   CFC-VMD均值   CFC胜率")
    for row in quality_rows:
        print(
            f"{row['K']:2d}   {row['n_paired_modes']:8d}   "
            f"{row['vmd_mean_abs_error_hz']:10.4f}   "
            f"{row['cfcvmd_mean_abs_error_hz']:10.4f}   "
            f"{100*row['cfc_win_rate']:7.2f}%"
        )
    print("-" * 72, flush=True)
    return quality_rows


# ---------------------------------------------------------------------------
# 并行执行一个 K
# ---------------------------------------------------------------------------
def _run_k(
    K: int,
    alpha: float,
    n_signals: int,
    base_seed: int,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    tasks = [
        (K, index, base_seed + 1_000_003 * K + index)
        for index in range(n_signals)
    ]
    records: List[Dict[str, Any]] = []
    workers = int(cfg["workers"])
    started = time.perf_counter()

    if workers <= 1:
        _init_worker(cfg)
        for done, task in enumerate(tasks, start=1):
            records.append(_run_one(task))
            if done % max(1, n_signals // 10) == 0 or done == n_signals:
                elapsed = time.perf_counter() - started
                print(
                    f"    K={K:2d}  {done:4d}/{n_signals}  "
                    f"({elapsed:6.1f} s)",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(cfg,),
        ) as executor:
            future_to_task = {executor.submit(_run_one, task): task for task in tasks}
            for done, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    records.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    records.append(
                        _failure_record(
                            K,
                            task[1],
                            task[2],
                            f"worker failed: {type(exc).__name__}: {exc}",
                        )
                    )
                if done % max(1, n_signals // 10) == 0 or done == n_signals:
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed > 0 else float("nan")
                    eta = (n_signals - done) / rate if rate and np.isfinite(rate) else float("nan")
                    print(
                        f"    K={K:2d}  {done:4d}/{n_signals}  "
                        f"({elapsed:6.1f} s, ETA {eta:5.1f} s)",
                        flush=True,
                    )

    records.sort(key=lambda r: r["signal_index"])
    return records


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VMD(+CFC) 与 CFC-VMD 的分解成功率对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-signals", type=int, default=DEFAULT_N_SIGNALS,
                        help="每个模态数量 K 下的信号数")
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_VALUES[0])
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_VALUES[-1])
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="并行进程数；内存较小时建议 1~2")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="基础随机种子")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS,
                        help="采样率 Hz")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="信号时长 s")
    parser.add_argument("--band-max", type=float, default=DEFAULT_BAND_MAX,
                        help="载波频率上限 Hz")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_HZ,
                        help="成功判据：|误差| < threshold Hz")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help="VMD/CFC-VMD 收敛容差")
    parser.add_argument("--alpha-mode", choices=("100k", "4k2", "fixed"), default="100k",
                        help="alpha 规则：100*K、4*K^2 或固定值")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="alpha-mode=fixed 时使用的 alpha")
    parser.add_argument("--carrier-margin", type=float, default=DEFAULT_CARRIER_MARGIN,
                        help="分层采样时每个频段两端的留白比例")
    parser.add_argument("--uniform-carriers", action="store_true",
                        help="载波在 [0, band-max] 内严格 i.i.d. 均匀随机（默认分层均匀）")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录；默认 experiments/VMD_CFCVMD_ModeCount_Success")
    parser.add_argument("--no-plot", action="store_true",
                        help="只生成 CSV，不画图")
    parser.add_argument("--skip-quality", action="store_true",
                        help="主实验后不生成“共同成功模态误差比较”新实验")
    parser.add_argument("--analyze-existing", type=Path, default=None,
                        help="不重新分解，直接分析已有 per_signal_results.csv 或其所在目录")
    parser.add_argument("--quality-tie-tol", type=float, default=1e-6,
                        help="逐模态比较时判定“平局”的误差差容差 Hz")
    return parser.parse_args(argv)


def _alpha_for(K: int, args: argparse.Namespace) -> float:
    if args.alpha_mode == "100k":
        return 100.0 * K
    if args.alpha_mode == "4k2":
        return 4.0 * K * K
    return float(args.alpha)


def _metadata(args: argparse.Namespace, K_values: Sequence[int]) -> Dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "project_root": str(PROJECT_ROOT),
        "K_values": list(K_values),
        "n_signals_per_K": int(args.n_signals),
        "success_definition": "所有 K 个 |f_est - f_true| < threshold Hz（匈牙利匹配）",
        "threshold_hz": float(args.threshold),
        "signal_model": (
            "Σ_k [1 + m_k cos(2π f_a,k t + Δφ_k)] "
            "cos(2π f_c,k t + β_k sin(2π f_f,k t))"
        ),
        "modulation_parameters": {
            "f_f_hz": list(FM_FREQ_RANGE),
            "f_a_hz": "f_f 或 f_f/2，等概率",
            "m": list(AM_INDEX_RANGE),
            "beta": list(FM_INDEX_RANGE),
            "phase_difference_rad": [0.0, 2.0 * math.pi],
            "am_phase": "Δφ",
            "fm_phase": 0.0,
        },
        "carrier_placement": {
            "type": "uniform_iid" if args.uniform_carriers else "stratified_uniform",
            "band_hz": [0.0, float(args.band_max)],
            "margin_ratio": None if args.uniform_carriers else float(args.carrier_margin),
            "note": (
                "严格独立均匀随机后排序"
                if args.uniform_carriers
                else "将 0~band_max 等分为 K 段，每段内均匀随机，段两端留白"
            ),
        },
        "vmd_settings": {
            "alpha_rule": (
                "100*K" if args.alpha_mode == "100k"
                else "4*K^2" if args.alpha_mode == "4k2"
                else "fixed"
            ),
            "alpha_fixed": None if args.alpha_mode != "fixed" else float(args.alpha),
            "tau": 0.0,
            "DC": 0,
            "init": 1,
            "tol": float(args.tol),
            "feedback_cfcvmd": True,
        },
        "sampling": {
            "fs_hz": float(args.fs),
            "duration_s": float(args.duration),
            "n_samples": int(round(args.fs * args.duration)),
            "noise": 0.0,
            "normalization": "每个混合信号除以其最大绝对值",
        },
        "seed": int(args.seed),
        "workers": int(args.workers),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "vmdpy": getattr(vmdpy, "__version__", "unknown"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    mp.freeze_support()  # Windows 打包/多进程兼容
    args = _parse_args(argv)

    # 只分析已有结果，不重新做 VMD/CFC-VMD 分解。
    if args.analyze_existing is not None:
        detail_csv = _resolve_detail_csv(args.analyze_existing)
        if args.output_dir is None:
            quality_dir = detail_csv.parent
        else:
            quality_dir = args.output_dir.expanduser().resolve()
        quality_dir.mkdir(parents=True, exist_ok=True)
        _run_paired_quality_analysis(
            detail_csv=detail_csv,
            output_dir=quality_dir,
            threshold_hz=float(args.threshold),
            tie_tol_hz=float(args.quality_tie_tol),
            make_plot=not bool(args.no_plot),
        )
        return 0

    if args.n_signals <= 0:
        raise SystemExit("--n-signals 必须为正整数")
    if args.fs <= 2.0 * args.band_max:
        print(
            f"警告：fs={args.fs:g} Hz 不高于 2*band_max={2*args.band_max:g} Hz，"
            "可能发生频谱混叠。",
            file=sys.stderr,
        )
    if args.k_min < 1 or args.k_max < args.k_min:
        raise SystemExit("K 范围不合法")

    K_values = list(range(args.k_min, args.k_max + 1))
    if args.output_dir is None:
        suffix = {
            "100k": "_alpha100K",
            "4k2": "_alpha4K2",
            "fixed": f"_alpha{args.alpha:g}",
        }[args.alpha_mode]
        output_dir = (
            PROJECT_ROOT / "experiments" / f"VMD_CFCVMD_ModeCount_Success{suffix}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "success_rate_by_K.csv"
    detail_path = output_dir / "per_signal_results.csv"
    png_path = output_dir / "success_rate_by_K.png"
    pdf_path = output_dir / "success_rate_by_K.pdf"
    meta_path = output_dir / "run_metadata.json"

    metadata = _metadata(args, K_values)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("VMD(+CFC) vs CFC-VMD：分解成功率对比")
    print(f"项目目录 : {PROJECT_ROOT}")
    print(f"输出目录 : {output_dir}")
    print(f"K        : {K_values}")
    print(f"每个 K   : {args.n_signals} 个信号")
    print(f"采样     : fs={args.fs:g} Hz, duration={args.duration:g} s")
    print(
        "alpha    : "
        + ("100*K" if args.alpha_mode == "100k"
           else "4K^2" if args.alpha_mode == "4k2"
           else str(args.alpha))
    )
    print(f"判据     : 全部 |误差| < {args.threshold:g} Hz")
    print(f"并行进程 : {args.workers}")
    print("=" * 72, flush=True)

    all_records: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    run_started = time.perf_counter()

    for K in K_values:
        alpha = _alpha_for(K, args)
        cfg = {
            "fs": float(args.fs),
            "duration": float(args.duration),
            "band_max": float(args.band_max),
            "stratified_carriers": not bool(args.uniform_carriers),
            "carrier_margin": float(args.carrier_margin),
            "threshold_hz": float(args.threshold),
            "alpha": alpha,
            "tol": float(args.tol),
            "workers": int(args.workers),
        }

        print(f"\n[K={K}] alpha={alpha:g}，开始 {args.n_signals} 个信号 ...", flush=True)
        records = _run_k(K, alpha, int(args.n_signals), int(args.seed), cfg)
        all_records.extend(records)

        row = _summarize(K, alpha, records)
        summary.append(row)

        # 每完成一个 K 就落盘，意外中断时仍保留已完成部分。
        _atomic_write_csv(summary_path, summary, SUMMARY_FIELDS)
        _atomic_write_csv(
            detail_path,
            [_record_to_row(r) for r in all_records],
            PER_SIGNAL_FIELDS,
        )

        print(
            f"  K={K:2d} 完成：VMD+CFC {row['vmd_success_count']}/{row['n_signals']} "
            f"({100*row['vmd_success_rate']:.2f}%)，"
            f"CFC-VMD {row['cfcvmd_success_count']}/{row['n_signals']} "
            f"({100*row['cfcvmd_success_rate']:.2f}%)",
            flush=True,
        )

    total_minutes = (time.perf_counter() - run_started) / 60.0
    print("\n" + "=" * 72)
    print(f"全部完成，用时 {total_minutes:.2f} min")
    print(f"汇总数据: {summary_path}")
    print(f"明细数据: {detail_path}")
    print(f"运行参数: {meta_path}")

    if not args.no_plot:
        _plot_success_rate(summary, png_path, pdf_path)
        print(f"对比图  : {png_path}")
        print(f"矢量图  : {pdf_path}")

    if not args.skip_quality:
        _run_paired_quality_analysis(
            detail_csv=detail_path,
            output_dir=output_dir,
            threshold_hz=float(args.threshold),
            tie_tol_hz=float(args.quality_tie_tol),
            make_plot=not bool(args.no_plot),
        )

    print("\nK   VMD+CFC      CFC-VMD")
    for row in summary:
        print(
            f"{row['K']:2d}  "
            f"{100*row['vmd_success_rate']:6.2f}%     "
            f"{100*row['cfcvmd_success_rate']:6.2f}%"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())