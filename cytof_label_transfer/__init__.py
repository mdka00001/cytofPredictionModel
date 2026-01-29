"""CyTOF label transfer utilities.

Train a classifier on trusted timepoints (e.g. timepoints 1–4) and
transfer cell-type labels to a target timepoint (e.g. timepoint 5)
in an AnnData object.
"""

from .data_utils import load_anndata, split_timepoints, extract_xy, extract_x_target
from .feature_selection import (
    compute_feature_importance,
    create_feature_groups,
    plot_feature_importance,
    select_features_by_groups,
    select_features_by_importance,
    select_features_interactive_report,
)
from .model import TrainedModelBundle, load_hyperparameters_from_json, predict_timepoint, train_classifier

__all__ = [
    "load_anndata",
    "split_timepoints",
    "extract_xy",
    "extract_x_target",
    "train_classifier",
    "predict_timepoint",
    "TrainedModelBundle",
    "load_hyperparameters_from_json",
    "compute_feature_importance",
    "create_feature_groups",
    "plot_feature_importance",
    "select_features_by_groups",
    "select_features_by_importance",
    "select_features_interactive_report",
]
