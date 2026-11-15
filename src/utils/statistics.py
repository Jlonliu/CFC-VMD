# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)


from statsmodels.stats.proportion import proportion_confint
import numpy as np
from scipy.stats import bootstrap

def calculate_Wilson(signals):
    cil_list=[]
    ciu_list=[]
    success_rate_list=[]
    signals = np.array(signals)
    for i in signals:
        success_count = np.sum(i)
        success_rate_list.append(np.mean(i))
        # 计算 Wilson 95% 置信区间
        ci_lowerr, ci_upperr = proportion_confint(
            count=success_count, 
            nobs=len(i), 
            alpha=0.05,         # 显著性水平，默认0.05，对应95%置信区间
            method='wilson'     # 使用Wilson Score方法
        )
        cil_list.append(ci_lowerr)# 记录置信区间下限
        ciu_list.append(ci_upperr)# 记录置信区间上限

    return success_rate_list,cil_list, ciu_list



def bootstrap_selection_frequency(data_matrix, n_resamples=10000, random_state=42,
                                  tie_breaker='random', return_details=False):
    """
    通过 Bootstrap 重采样计算每个候选 alpha 被选为最优的频率。

    参数
    ----------
    data_matrix : numpy.ndarray, shape (n_samples, n_alphas)
        原始数据矩阵，每一行是一个样本，每一列是一个候选 alpha 值。
        元素应为 1（成功）或 0（失败）。
    n_resamples : int, default=1000
        Bootstrap 重采样次数。
    random_state : int or None, default=None
        随机种子，设定后可保证结果可复现。
    tie_breaker : {'first', 'random'}, default='first'
        当多个 alpha 并列最高成功率时的处理方式：
        - 'first' : 选择第一个（按列索引升序）
        - 'random': 在并列者中随机选择一个
    return_details : bool, default=False
        若为 False，仅返回一维数组，包含每个 alpha 被选中的频率（0~1）。
        若为 True，返回字典，包含 'frequency'（频率数组）和 'counts'（计数数组）。

    返回
    -------
    frequencies : numpy.ndarray, shape (n_alphas,)
        每个 alpha 被选为最优的频率（比例）。
        如果 return_details=True，返回字典 {'frequencies': freq, 'counts': counts}。
    """

    # 参数校验
    if data_matrix.ndim != 2:
        raise ValueError("data_matrix 必须是二维数组 (样本 x Alpha候选)")
    n_samples, n_alphas = data_matrix.shape
    if n_samples == 0 or n_alphas == 0:
        raise ValueError("样本数或Alpha候选数不能为0")

    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)

    # 存储每次重采样选中的最优 alpha 索引
    selected_indices = []

    for _ in range(n_resamples):
        # 有放回地抽取样本索引
        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # 生成重采样数据集
        resampled = data_matrix[sample_indices, :]
        # 计算每个 alpha 的成功率（列均值）
        success_rates = np.mean(resampled, axis=0)

        # 找出最优索引
        max_value = success_rates.max()
        # 找出所有达到最大值的索引
        best_candidates = np.where(success_rates == max_value)[0]

        if tie_breaker == 'first':
            best_idx = best_candidates[0]
        elif tie_breaker == 'random':
            best_idx = np.random.choice(best_candidates)
        else:
            raise ValueError("tie_breaker 必须是 'first' 或 'random'")

        selected_indices.append(best_idx)

    selected_indices = np.array(selected_indices)
    # 统计每个 alpha 被选中的次数
    counts = np.array([np.sum(selected_indices == i) for i in range(n_alphas)])
    frequencies = counts / n_resamples

    if return_details:
        return {'frequencies': frequencies, 'counts': counts}
    else:
        return frequencies


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 生成模拟数据：100个样本，20个alpha候选
    np.random.seed(42)
    n_samples = 100
    n_alphas = 20
    # 模拟：真实最优在索引5附近，生成含噪声的0/1数据
    true_probs = np.zeros(n_alphas)
    true_probs[5] = 0.75  # 设置最优alpha的成功率较高
    for i in range(n_alphas):
        if i != 5:
            true_probs[i] = 0.5 + 0.2 * np.random.rand()  # 其他在0.5~0.7之间随机

    data = np.zeros((n_samples, n_alphas))
    for i in range(n_alphas):
        data[:, i] = np.random.binomial(1, true_probs[i], n_samples)

    # 调用函数（默认返回频率数组）
    freq = bootstrap_selection_frequency(data, n_resamples=1000, random_state=123)
    print("每个alpha被选为最优的频率：")
    print(freq)

    # 查看最优 alpha 的索引
    best_alpha_idx = np.argmax(freq)
    print(f"\n选择最稳定的 alpha 索引：{best_alpha_idx}，频率：{freq[best_alpha_idx]:.2%}")

    # 返回详细信息（频率+计数）
    detailed = bootstrap_selection_frequency(data, n_resamples=1000, random_state=123,
                                             return_details=True, tie_breaker='random')
    print(f"\n计数：{detailed['counts']}")
    print(f"频率：{detailed['frequencies']}")
