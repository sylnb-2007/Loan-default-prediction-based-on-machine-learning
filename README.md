# 贷款违约预测（Loan Default Prediction）

一个简单、可复现的金融机器学习入门项目：根据贷款申请人的个人信息与贷款详情，
预测其是否会违约（二分类）。

- **任务类型**：监督学习 · 二分类
- **模型**：逻辑回归（基线）vs 随机森林
- **数据**：确定性种子生成的模拟贷款数据集（无需联网下载）
- **依赖**：numpy / pandas / scikit-learn / matplotlib

## 快速开始

```bash
pip install -r requirements.txt
python main.py        # Windows 下若 python 不可用，改用 py main.py
```

运行后会：
1. 在 `data/` 生成（或读取）数据集；
2. 训练并对比两个模型，打印准确率 / 精确率 / 召回率 / F1 / ROC-AUC；
3. 在 `outputs/` 保存混淆矩阵图、ROC 曲线图、特征重要性图。

## 目录结构

```
loan-default-prediction/
├── main.py               # 入口
├── data_generator.py     # 数据生成
├── model.py              # 建模与评估
├── requirements.txt      # 依赖
├── README.md             # 本文件
├── 项目说明文档.md        # 详细介绍与使用步骤
├── data/                 # 生成的数据集（自动创建）
└── outputs/              # 评估图表（自动创建）
```

详细说明见 [项目说明文档.md](项目说明文档.md)。
