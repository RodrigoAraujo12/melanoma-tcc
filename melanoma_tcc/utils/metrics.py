import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def compute_metrics(labels: list, predictions: list, probabilities: list = None) -> dict:
    report = classification_report(labels, predictions, target_names=["Benign", "Melanoma"])
    print(report)
    results = {"classification_report": report}
    if probabilities is not None:
        auc = roc_auc_score(labels, probabilities)
        print(f"AUC-ROC: {auc:.4f}")
        results["auc_roc"] = auc
    return results


def plot_confusion_matrix(labels: list, predictions: list, save_path: str = None):
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Melanoma"],
        yticklabels=["Benign", "Melanoma"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_roc_curve(labels: list, probabilities: list, save_path: str = None):
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
