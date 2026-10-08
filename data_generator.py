"""
数据生成模块
============

本项目使用「确定性随机种子」生成一份贴近真实业务的模拟贷款数据集，
好处是：
  1. 无需联网下载外部数据集，任何人都能一键复现；
  2. 每次运行结果完全一致（可复现性）；
  3. 每个特征都有明确的业务含义，便于教学与解释。

生成的特征与违约目标之间存在真实的统计关系（信用评分越低、
负债收入比越高、历史违约越多 -> 违约概率越高），因此模型可以
学到有意义的东西，而不是纯粹的噪声。
"""

import numpy as np
import pandas as pd


def generate_loan_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟的银行贷款申请数据集。

    参数
    ----
    n_samples : int
        样本数量（默认 10000）
    seed : int
        随机种子，保证结果可复现

    返回
    ----
    pd.DataFrame
        包含特征列与目标列 default（0=不违约, 1=违约）的数据框
    """
    rng = np.random.default_rng(seed)

    # ---- 基础人口特征 ----
    age = rng.integers(21, 66, size=n_samples).astype(float)          # 年龄 21~65
    annual_income = rng.lognormal(mean=11.0, sigma=0.4, size=n_samples)  # 年收入(美元)，中位数约 6 万
    employment_years = np.clip(age - 18 - rng.integers(0, 6, size=n_samples), 0, 45).astype(float)

    # ---- 类别特征 ----
    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
        size=n_samples,
        p=[0.45, 0.40, 0.10, 0.05],   # 租房 / 有房贷 / 自有 / 其他
    )
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "major_purchase", "medical", "small_business", "car", "education"],
        size=n_samples,
        p=[0.30, 0.15, 0.10, 0.10, 0.08, 0.07, 0.12, 0.08],
    )

    # ---- 信用相关特征 ----
    prior_defaults = rng.poisson(0.3, size=n_samples)  # 历史违约次数
    credit_score = (
        850
        - prior_defaults * 40                                # 历史违约越多，信用分越低
        - rng.normal(0, 50, size=n_samples)
        + (np.log(annual_income) - 11) * 30                  # 收入越高，信用分越高
    )
    credit_score = np.clip(credit_score, 300, 850).astype(int)

    # ---- 贷款相关特征 ----
    loan_amount = np.clip(annual_income * rng.uniform(0.05, 0.4, size=n_samples), 1000, 50000).astype(int)
    loan_term_months = rng.choice([36, 60], size=n_samples, p=[0.6, 0.4]).astype(int)
    debt_to_income = np.clip(rng.beta(2, 5, size=n_samples) * 0.6 + prior_defaults * 0.05, 0, 0.65)
    interest_rate = np.clip(
        5 + (850 - credit_score) / 850 * 15 + prior_defaults * 1.5 + rng.normal(0, 1.5, size=n_samples),
        5, 30,
    )

    # ---- 构造违约概率（逻辑斯蒂形式）----
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)  # 标准化

    logit = (
        -1.5
        - 0.8 * z(credit_score)
        + 1.2 * z(debt_to_income)
        + 0.7 * z(prior_defaults)
        + 0.4 * z(interest_rate)
        - 0.3 * z(annual_income)
        + 0.2 * z(loan_amount)
    )
    # 房产状况对违约的影响
    logit = logit + np.where(home_ownership == "RENT", 0.25, 0.0)
    logit = logit + np.where(home_ownership == "OWN", -0.25, 0.0)

    prob = 1.0 / (1.0 + np.exp(-logit))                 # sigmoid 转成概率
    default = (rng.random(n_samples) < prob).astype(int)  # 按概率采样得到标签

    # ---- 组装 DataFrame ----
    df = pd.DataFrame({
        "age": age.astype(int),
        "annual_income": annual_income.round(2),
        "employment_years": employment_years.astype(int),
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "credit_score": credit_score,
        "prior_defaults": prior_defaults.astype(int),
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "debt_to_income": debt_to_income.round(4),
        "interest_rate": interest_rate.round(2),
        "default": default,
    })
    return df


if __name__ == "__main__":
    # 单独运行本文件时，可快速预览数据
    df = generate_loan_data()
    print(df.head())
    print("\n违约标签分布：")
    print(df["default"].value_counts(normalize=True))
