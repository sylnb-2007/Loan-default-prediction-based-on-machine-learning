"""
建模与评估模块
==============

包含数据预处理、模型训练、评估指标计算，以及结果可视化。
对比两个经典模型：
  - 逻辑回归（LogisticRegression）：线性、可解释的基线模型
  - 随机森林（RandomForestClassifier）：非线性集成模型

评估指标包括准确率、精确率、召回率、F1、ROC-AUC，并输出
混淆矩阵图、ROC 曲线图与特征重要性图。
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境也能保存图片
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

CATEGORICAL_COLS = ["home_ownership", "loan_purpose"]
TARGET = "default"


def prepare_data(df: pd.DataFrame):
    """特征/标签拆分 + 类别变量 One-Hot 编码。"""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    return X, y


def train_models(X_train, y_train):
    """训练逻辑回归与随机森林两个模型。"""
    models = {
        # 逻辑回归对特征尺度敏感，先用 StandardScaler 标准化再训练
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        # 随机森林是树模型，无需标准化
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate(model, X_test, y_test):
    """计算并返回一个模型的各项评估结果。"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "cm": confusion_matrix(y_test, y_pred),
        "y_test": y_test,
        "y_proba": y_proba,
    }


# ---- 绘图函数（均为英文标签，避免中文字体显示问题）----

def plot_roc(y_test, y_proba, auc, name, path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, name, path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    classes = ["No Default", "Default"]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]  # 取前 15 个
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importances[idx][::-1], align="center")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1])
    plt.xlabel("Importance")
    plt.title("Top-15 Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_experiment(df: pd.DataFrame, output_dir: str = "outputs"):
    """完整流程：预处理 -> 划分 -> 训练 -> 评估 -> 可视化。"""
    os.makedirs(output_dir, exist_ok=True)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")
    print(f"特征维度：{X.shape[1]}")

    models = train_models(X_train, y_train)

    for name, model in models.items():
        res = evaluate(model, X_test, y_test)
        print(f"\n{'=' * 40}\n  {name}\n{'=' * 40}")
        print(f"Accuracy  (准确率)  : {res['accuracy']:.4f}")
        print(f"Precision (精确率)  : {res['precision']:.4f}")
        print(f"Recall    (召回率)  : {res['recall']:.4f}")
        print(f"F1-score            : {res['f1']:.4f}")
        print(f"ROC-AUC             : {res['auc']:.4f}")
        print("\n混淆矩阵 (行=真实, 列=预测):")
        print(res["cm"])
        print("\n分类报告：")
        print(classification_report(y_test, model.predict(X_test),
                                    target_names=["No Default", "Default"]))

        plot_roc(res["y_test"], res["y_proba"], res["auc"], name,
                 os.path.join(output_dir, f"roc_{name}.png"))
        plot_confusion_matrix(res["cm"], name,
                              os.path.join(output_dir, f"cm_{name}.png"))

    # 随机森林特征重要性
    rf = models["RandomForest"]
    plot_feature_importance(rf, X.columns,
                            os.path.join(output_dir, "feature_importance.png"))

    print(f"\n图表已保存到 {output_dir}/ 目录。")
