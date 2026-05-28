import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def compute_metrics(labels: list, predictions: list, probabilities: list = None,
                    target_names: list = None) -> dict:
    if target_names is None:
        target_names = ["Benign", "Melanoma"]
    report = classification_report(labels, predictions, target_names=target_names)
    print(report)
    results = {"classification_report": report}
    if probabilities is not None and len(target_names) == 2:
        auc = roc_auc_score(labels, probabilities)
        print(f"AUC-ROC: {auc:.4f}")
        results["auc_roc"] = auc
    return results


def plot_confusion_matrix(labels: list, predictions: list, save_path: str = None,
                          target_names: list = None):
    if target_names is None:
        target_names = ["Benign", "Melanoma"]
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
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
