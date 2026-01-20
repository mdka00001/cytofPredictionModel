"""Model training and prediction for CyTOF label transfer.

Classifier choice
-----------------
We use :class:`xgboost.XGBClassifier`, a gradient-boosted tree model
that works well for high-dimensional, non-linear data such as CyTOF
marker intensities as well as learned latent spaces (e.g. scVI). It is
efficient, handles class imbalance reasonably well when evaluated with
macro-averaged metrics, and integrates cleanly with scikit-learn's
model selection tools.

The training function performs a randomized hyperparameter search with
cross-validation on timepoints 1-4 (or any user-specified training
set), then refits the best model on all training cells and saves the
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from xgboost.core import XGBoostError


@dataclass
class TrainedModelBundle:
    """Container for a trained classifier and its metadata.

    Attributes
    ----------
    estimator:
        Fitted scikit-learn classifier.
    feature_names:
        Names of the features used during training (ordered).
    label_names:
        Sorted unique label names seen during training.
    best_params:
        Best hyperparameters found by the search.
    cv_best_score:
        Best cross-validated score (macro F1).
    cv_metric:
        Name of the main metric used for model selection.
    """

    estimator: Any
    feature_names: List[str]
    label_names: List[str]
    best_params: Dict[str, Any]
    cv_best_score: float
    cv_metric: str = "f1_macro"
    # Optional label encoder used to map between original labels and
    # integer-encoded classes used by XGBoost.
    label_encoder: Optional[LabelEncoder] = None


def _default_param_distributions() -> Dict[str, Any]:
    """Return a parameter search space for :class:`xgboost.XGBClassifier`."""

    return {
        "n_estimators": [800, 1000],
        "max_depth": [9,12],
        "learning_rate": [0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.3,0.6],
        "min_child_weight": [0.5, 1],
        "gamma": [0.0, 0.01, 0.05],
        "reg_lambda": [0.5,1.0],
        "reg_alpha": [1.0, 2.0],
    }


def train_classifier_fixed_params(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    best_params: Dict[str, Any],
    *,
    random_state: int = 42,
    n_jobs: int = -1,
    output_dir: str | Path = "models",
    use_gpu: bool = False,
) -> TrainedModelBundle:
    """Train a classifier once using pre-selected best hyperparameters.

    This is a fast path that skips hyperparameter search and simply fits
    a single :class:`xgboost.XGBClassifier` defined by ``best_params``.

    Parameters
    ----------
    X, y, feature_names:
        Training data and associated feature names.
    best_params:
        Hyperparameters to use when constructing the classifier. These
        typically come from a previous hyperparameter search.
    random_state, n_jobs, output_dir, use_gpu:
        As in :func:`train_classifier`.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)
    num_classes = int(len(label_encoder.classes_))

    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if num_classes > 2 else "logloss"
    tree_method = "hist"
    device: Optional[str] = "cuda" if use_gpu else "cpu"

    base_kwargs: Dict[str, Any] = {
        "objective": objective,
        "eval_metric": eval_metric,
        "tree_method": tree_method,
        "device": device,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    if num_classes > 2:
        base_kwargs["num_class"] = num_classes

    clf = XGBClassifier(**base_kwargs, **best_params)
    clf.fit(X, y_int)

    # Simple reference metrics on the training set
    y_pred_int = clf.predict(X)
    y_pred = label_encoder.inverse_transform(y_pred_int)
    train_f1 = f1_score(y, y_pred, average="macro")
    train_acc = accuracy_score(y, y_pred)

    metrics = {
        "cv_best_score": None,
        "cv_metric": "f1_macro",
        "train_f1_macro": float(train_f1),
        "train_accuracy": float(train_acc),
        "best_params": best_params,
        "classes_": list(label_encoder.classes_),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }

    import json

    with (output_dir / "training_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    bundle = TrainedModelBundle(
        estimator=clf,
        feature_names=feature_names,
        label_names=list(label_encoder.classes_),
        best_params=best_params,
        cv_best_score=float(metrics["train_f1_macro"]),
        cv_metric="f1_macro",
        label_encoder=label_encoder,
    )

    joblib.dump(bundle, output_dir / "model_bundle.joblib")

    return bundle


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    output_dir: str | Path = "models",
    use_gpu: bool = False,
) -> TrainedModelBundle:
    """Train an optimized classifier with cross-validated hyperparameter search.

    Parameters
    ----------
    X:
        Feature matrix of shape (n_cells, n_features).
    y:
        Label vector of length n_cells (string or categorical labels).
    feature_names:
        List of feature names corresponding to columns of ``X``.
    n_splits:
        Number of cross-validation folds.
    n_iter:
        Number of random hyperparameter settings to sample.
    random_state:
        Random seed for reproducibility.
    n_jobs:
        Number of parallel jobs for hyperparameter search (-1: all cores).
    output_dir:
        Directory where the trained model and CV results are saved.
    use_gpu:
        If True, request GPU acceleration via ``device='cuda'`` in XGBoost
        (requires a GPU-enabled XGBoost build and compatible CUDA drivers).

    Returns
    -------
    TrainedModelBundle
        Object containing the fitted estimator and metadata.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode labels to integers for XGBoost
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)
    num_classes = int(len(label_encoder.classes_))

    # XGBoost classifier suitable for multi-class CyTOF cell-type prediction.
    # We attempt to use GPU acceleration via ``device='cuda'`` when requested,
    # but will fall back to CPU if CUDA is not available.
    tree_method = "hist"
    device: Optional[str] = "cuda" if use_gpu else "cpu"

    param_distributions = _default_param_distributions()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    # Manual randomized search with explicit progress reporting
    sampler = ParameterSampler(
        param_distributions,
        n_iter=n_iter,
        random_state=random_state,
    )

    results: list[dict[str, Any]] = []
    best_score: float = -np.inf
    best_params: Dict[str, Any] | None = None
    warned_gpu_fallback = False

    for i, params in enumerate(sampler, start=1):
        # Objective and evaluation metric follow your example: multi:softprob
        # for multi-class, binary:logistic for binary.
        objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
        eval_metric = "mlogloss" if num_classes > 2 else "logloss"
        base_kwargs: Dict[str, Any] = {
            "objective": objective,
            "eval_metric": eval_metric,
            "tree_method": tree_method,
            "device": device,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        if num_classes > 2:
            base_kwargs["num_class"] = num_classes

        clf = XGBClassifier(**base_kwargs, **params)

        # Use CV to estimate macro F1 for this hyperparameter set
        try:
            scores = cross_val_score(
                clf,
                X,
                y_int,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,  # avoid nested parallelism; XGBoost uses its own threads
            )
        except XGBoostError as e:
            # Fall back to CPU if CUDA/device is not available or not supported
            msg = str(e)
            if "cuda" in msg.lower() or "device" in msg.lower():
                if not warned_gpu_fallback:
                    print(
                        "[Warning] XGBoost CUDA/device not available; "
                        "falling back to CPU (device='cpu')."
                    )
                    warned_gpu_fallback = True
                device = "cpu"
                base_kwargs["device"] = device
                clf = XGBClassifier(**base_kwargs, **params)
                scores = cross_val_score(
                    clf,
                    X,
                    y_int,
                    cv=cv,
                    scoring="f1_macro",
                    n_jobs=1,
                )
            else:
                raise
        mean_score = float(scores.mean())
        std_score = float(scores.std())

        progress = 100.0 * i / float(n_iter)
        print(
            f"[Hyperparam search] Iteration {i}/{n_iter} "
            f"({progress:5.1f}%) - mean CV f1_macro={mean_score:.4f} ± {std_score:.4f}"
        )

        row: Dict[str, Any] = {"iter": i, "mean_test_f1_macro": mean_score, "std_test_f1_macro": std_score}
        row.update(params)
        results.append(row)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is None:
        raise RuntimeError("Hyperparameter search did not evaluate any configurations.")

    # Save CV summary results
    import pandas as pd

    cv_results = pd.DataFrame(results)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)

    # Refit best estimator on the full training set
    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if num_classes > 2 else "logloss"
    base_kwargs = {
        "objective": objective,
        "eval_metric": eval_metric,
        "tree_method": tree_method,
        "device": device,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    if num_classes > 2:
        base_kwargs["num_class"] = num_classes

    best_estimator = XGBClassifier(**base_kwargs, **best_params)

    best_estimator.fit(X, y_int)

    # Compute simple overall metrics on the full training set for reference
    y_pred_int = best_estimator.predict(X)
    y_pred = label_encoder.inverse_transform(y_pred_int)
    train_f1 = f1_score(y, y_pred, average="macro")
    train_acc = accuracy_score(y, y_pred)

    metrics = {
        "cv_best_score": float(best_score),
        "cv_metric": "f1_macro",
        "train_f1_macro": float(train_f1),
        "train_accuracy": float(train_acc),
        "best_params": best_params,
        "classes_": list(label_encoder.classes_),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }

    import json

    with (output_dir / "training_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    bundle = TrainedModelBundle(
        estimator=best_estimator,
        feature_names=feature_names,
        label_names=list(label_encoder.classes_),
        best_params=best_params,
        cv_best_score=float(best_score),
        cv_metric="f1_macro",
        label_encoder=label_encoder,
    )

    # Save model bundle with joblib for reuse
    joblib.dump(bundle, output_dir / "model_bundle.joblib")

    return bundle


def predict_timepoint(
    bundle: TrainedModelBundle,
    X_target: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict labels (and probabilities if available) for a target timepoint.

    Parameters
    ----------
    bundle:
        TrainedModelBundle created by :func:`train_classifier`.
    X_target:
        Feature matrix of the target timepoint, aligned to the same
        features and order used for training.

    Returns
    -------
    y_pred, y_proba
        Predicted labels and class probabilities (or ``None`` if the
        classifier does not support ``predict_proba``).
    """

    clf = bundle.estimator
    y_pred_raw = clf.predict(X_target)

    # Decode labels back to original strings if a label encoder is present
    if getattr(bundle, "label_encoder", None) is not None:
        y_pred = bundle.label_encoder.inverse_transform(y_pred_raw)
    else:
        y_pred = y_pred_raw

    y_proba: Optional[np.ndarray]
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_target)
        except Exception:
            y_proba = None
    else:
        y_proba = None

    return y_pred, y_proba
