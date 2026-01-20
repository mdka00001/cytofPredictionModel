"""CyTOF label transfer utilities.

Train a classifier on trusted timepoints (e.g. timepoints 1–4) and
transfer cell-type labels to a target timepoint (e.g. timepoint 5)
in an AnnData object.
"""

from .data_utils import load_anndata, split_timepoints
from .model import TrainedModelBundle, predict_timepoint, train_classifier

__all__ = [
    "load_anndata",
    "split_timepoints",
    "train_classifier",
    "predict_timepoint",
    "TrainedModelBundle",
]
