"""
项目入口
========

一键运行：生成（或读取）数据集 -> 训练模型 -> 评估 -> 输出图表。

用法：
    py main.py
"""

import os
import sys

# 修复 Windows 控制台中文乱码：强制使用 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import pandas as pd

from data_generator import generate_loan_data
from model import run_experiment


def main():
    os.makedirs("data", exist_ok=True)
    data_path = os.path.join("data", "loan_data.csv")

    # 若数据集已存在则直接读取，否则生成并保存
    if os.path.exists(data_path):
        print(f"检测到已有数据集 {data_path}，直接读取。")
        df = pd.read_csv(data_path)
    else:
        print("正在生成模拟数据集 ...")
        df = generate_loan_data(n_samples=10000, seed=42)
        df.to_csv(data_path, index=False)
        print(f"数据集已保存到 {data_path}（{len(df)} 条样本）。")

    print("\n数据概况：")
    print(f"  样本数：{len(df)}")
    print(f"  特征数：{df.shape[1] - 1}")
    print(f"  违约率：{df['default'].mean():.2%}")
    print("\n前 5 行数据：")
    print(df.head().to_string())

    run_experiment(df, output_dir="outputs")
    print("\n全部完成！")


if __name__ == "__main__":
    main()
