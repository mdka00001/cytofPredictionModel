"""Quality control metrics and plotting for CyTOF label transfer.

This module generates evaluation metrics and plots based on cross-
validated predictions on the training data. These plots help assess
how reliable the classifier is before applying it to the target
(timepoint 5) data, where ground truth labels are unreliable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def evaluate_and_plot_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str] | None,
    output_dir: str | Path,
    n_splits: int = 5,
    random_state: int = 42,
    label_encoder: Any | None = None,
) -> None:
    """Run cross-validated predictions and save QC metrics and plots.

    Parameters
    ----------
    estimator:
        A fitted scikit-learn classifier (used as template for CV).
    X, y:
        Training features and labels.
    class_names:
        Optional list of class names to control plot ordering.
    output_dir:
        Directory where plots and metrics will be written.
    n_splits:
        Number of CV folds.
    random_state:
        Seed used for fold shuffling.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Obtain out-of-fold predictions for a realistic performance estimate.
    # If a label encoder is provided (as in the main training pipeline), we
    # encode labels to integers for fitting and then decode predictions back
    # to the original label names for reporting and plotting.
    if label_encoder is not None:
        y_true = y
        y_encoded = label_encoder.transform(y_true)
        y_pred_encoded = cross_val_predict(
            estimator,
            X,
            y_encoded,
            cv=cv,
            n_jobs=-1,
        )
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_true = y
        y_pred = cross_val_predict(
            estimator,
            X,
            y_true,
            cv=cv,
            n_jobs=-1,
        )

    # Macro F1 summarizes performance across all classes
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # Textual classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    (output_dir / "cv_classification_report.txt").write_text(report)

    # Confusion matrix
    labels = class_names if class_names is not None else np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Cross-validated confusion matrix (macro F1={macro_f1:.3f})")
    fig.tight_layout()
    fig.savefig(output_dir / "cv_confusion_matrix.png", dpi=200)
    plt.close(fig)

    # Per-class F1 bar plot
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), f1, color="tab:blue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 scores (cross-validation)")
    fig.tight_layout()
    fig.savefig(output_dir / "cv_per_class_f1.png", dpi=200)
    plt.close(fig)

    # Save a small summary file
    summary = {
        "macro_f1": float(macro_f1),
        "n_classes": int(len(labels)),
    }
    import json

    with (output_dir / "cv_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
