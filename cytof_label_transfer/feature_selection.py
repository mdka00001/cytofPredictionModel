"""Feature evaluation, selection, and engineering for CyTOF label transfer.

This module provides tools for:
1. Evaluating feature importance before training
2. Visualizing feature importance
3. Manually selecting features or feature groups
4. Feature engineering (combining markers and latent factors)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = "random_forest",
    n_estimators: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Compute feature importance scores using a quick model.

    Parameters
    ----------
    X:
        Feature matrix of shape (n_samples, n_features).
    y:
        Label vector of length n_samples.
    feature_names:
        List of feature names corresponding to columns of X.
    method:
        Method for importance estimation. Currently supports "random_forest".
    n_estimators:
        Number of estimators for RandomForest.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    importances, feature_names
        Array of importance scores and corresponding feature names.
    """

    if method == "random_forest":
        label_encoder = LabelEncoder()
        y_int = label_encoder.fit_transform(y)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y_int)
        importances = rf.feature_importances_
    else:
        raise ValueError(f"Unknown importance method: {method}")

    return importances, feature_names


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 30,
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Plot feature importance scores.

    Parameters
    ----------
    importances:
        Array of feature importance scores.
    feature_names:
        List of feature names.
    top_n:
        Number of top features to display.
    output_path:
        If provided, save the plot to this path.
    figsize:
        Figure size (width, height).
    """

    if len(importances) != len(feature_names):
        raise ValueError("importances and feature_names must have same length")

    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    # Display top N
    display_n = min(top_n, len(sorted_names))
    sorted_importances = sorted_importances[:display_n]
    sorted_names = sorted_names[:display_n]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importances, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance score", fontsize=11)
    ax.set_title(f"Top {display_n} Feature Importances (Random Forest)", fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(fig)


def create_feature_groups(
    feature_names: List[str],
    obsm_key: Optional[str] = None,
) -> Dict[str, List[int]]:
    """Create feature groups based on marker vs. latent features.

    Parameters
    ----------
    feature_names:
        List of all feature names (expression markers + latent features).
    obsm_key:
        The obsm key name used (e.g., "X_scVI_200_epoch").
        If provided, latent features are identified by this prefix.

    Returns
    -------
    feature_groups:
        Dictionary mapping group name to list of column indices.
        Example: {"markers": [0, 1, 2], "latent": [3, 4, 5]}
    """

    groups: Dict[str, List[int]] = {"markers": [], "latent": []}

    for i, name in enumerate(feature_names):
        if obsm_key is not None and name.startswith(f"{obsm_key}_"):
            groups["latent"].append(i)
        else:
            groups["markers"].append(i)

    return groups


def select_features_by_importance(
    importances: np.ndarray,
    feature_names: List[str],
    percentile: float = 90,
) -> Tuple[np.ndarray, List[str]]:
    """Select features above a given importance percentile.

    Parameters
    ----------
    importances:
        Array of feature importance scores.
    feature_names:
        List of feature names.
    percentile:
        Percentile threshold for feature selection (0-100).

    Returns
    -------
    selected_indices, selected_names
        Indices and names of selected features.
    """

    threshold = np.percentile(importances, percentile)
    mask = importances >= threshold
    selected_indices = np.where(mask)[0]
    selected_names = [feature_names[i] for i in selected_indices]

    return selected_indices, selected_names


def select_features_by_groups(
    feature_names: List[str],
    feature_groups: Dict[str, List[int]],
    selected_groups: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Select features based on group membership.

    Parameters
    ----------
    feature_names:
        List of all feature names.
    feature_groups:
        Dictionary mapping group name to list of column indices
        (as returned by create_feature_groups).
    selected_groups:
        List of group names to include (e.g., ["markers", "latent"]).

    Returns
    -------
    selected_indices, selected_names
        Indices and names of selected features.
    """

    selected_indices = []
    for group_name in selected_groups:
        if group_name not in feature_groups:
            raise ValueError(f"Unknown feature group: {group_name}")
        selected_indices.extend(feature_groups[group_name])

    selected_indices = np.array(sorted(selected_indices))
    selected_names = [feature_names[i] for i in selected_indices]

    return selected_indices, selected_names


def select_features_interactive_report(
    importances: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[int]],
    output_dir: str | Path = "feature_reports",
) -> None:
    """Generate an interactive report for feature selection.

    Creates a CSV file with feature importance and group membership,
    useful for manual inspection and selection.

    Parameters
    ----------
    importances:
        Array of feature importance scores.
    feature_names:
        List of feature names.
    feature_groups:
        Dictionary mapping group name to list of column indices.
    output_dir:
        Directory to save the report.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataframe with importance and group info
    data = []
    for i, name in enumerate(feature_names):
        group_names = [g for g, indices in feature_groups.items() if i in indices]
        data.append({
            "feature_index": i,
            "feature_name": name,
            "importance": importances[i],
            "groups": ", ".join(group_names),
        })

    df = pd.DataFrame(data)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    report_path = output_dir / "feature_importance_report.csv"
    df.to_csv(report_path, index=False)

    print(f"Feature importance report saved to: {report_path}")
    print(f"\nTop 10 features:\n{df.head(10)}")
