"""Data loading and preprocessing helpers for CyTOF label transfer.

This module assumes an AnnData object with:
- `.X` or specified layer containing expression/intensity values
- `.obs[time_col]` containing timepoint labels (e.g. 1, 2, 3, 4, 5)
- `.obs[label_col]` containing cell-type annotations for *trusted* timepoints

You can adapt column names via function arguments.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd


def load_anndata(path: str) -> ad.AnnData:
    """Load an AnnData object from disk.

    Parameters
    ----------
    path:
        Path to a `.h5ad` file.
    """

    return ad.read_h5ad(path)


def split_timepoints(
    adata: ad.AnnData,
    time_col: str,
    train_timepoints: Iterable,
    target_timepoint,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Split AnnData into training and target subsets by timepoint.

    Parameters
    ----------
    adata:
        Full AnnData object containing all timepoints.
    time_col:
        Name of the `.obs` column with timepoint identifiers.
    train_timepoints:
        Iterable of timepoints to use for training (e.g. [1, 2, 3, 4]).
    target_timepoint:
        Timepoint to predict (e.g. 5).
    """

    obs = adata.obs
    train_mask = obs[time_col].isin(train_timepoints)
    target_mask = obs[time_col] == target_timepoint

    adata_train = adata[train_mask].copy()
    adata_target = adata[target_mask].copy()

    return adata_train, adata_target


def _get_expression_matrix(
    adata: ad.AnnData,
    use_layer: str | None = None,
) -> np.ndarray:
    """Return the base expression matrix (from ``.X`` or a layer``)."""

    if use_layer is not None:
        X = adata.layers[use_layer]
    else:
        X = adata.X

    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X)


def _append_latent_features(
    X: np.ndarray,
    adata: ad.AnnData,
    obsm_key: str | None,
) -> Tuple[np.ndarray, List[str]]:
    """Optionally append features from ``adata.obsm[obsm_key]`` to ``X``.

    Returns the extended feature matrix and a list of latent feature
    names (empty list if ``obsm_key`` is None).
    """

    latent_names: List[str] = []
    if obsm_key is None:
        return X, latent_names

    if obsm_key not in adata.obsm_keys():
        raise KeyError(f"obsm_key '{obsm_key}' not found in adata.obsm")

    Z = adata.obsm[obsm_key]
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    Z = np.asarray(Z)

    if Z.shape[0] != X.shape[0]:
        raise ValueError(
            f"Latent space matrix '{obsm_key}' has {Z.shape[0]} rows, "
            f"but expression matrix has {X.shape[0]} rows."
        )

    # Generate synthetic feature names for latent dimensions
    latent_names = [f"{obsm_key}_{i}" for i in range(Z.shape[1])]

    X_ext = np.concatenate([X, Z], axis=1)
    return X_ext, latent_names


def extract_xy(
    adata: ad.AnnData,
    label_col: str,
    use_layer: str | None = None,
    feature_mask: np.ndarray | None = None,
    use_obsm_key: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X and label vector y from AnnData.

    Parameters
    ----------
    adata:
        AnnData subset used for training.
    label_col:
        Name of `.obs` column with cell-type labels.
    use_layer:
        If not None, use `adata.layers[use_layer]` instead of `adata.X`.
    feature_mask:
        Optional boolean mask over variables to select a subset of markers.
    use_obsm_key:
        If not None, append features from `adata.obsm[use_obsm_key]`
        (e.g. a scVI latent space) to the expression features.

    Returns
    -------
    X, y, feature_names
        Feature matrix, label vector, and list of feature names
        (expression + latent, in that order).
    """

    X = _get_expression_matrix(adata, use_layer=use_layer)

    if feature_mask is not None:
        X = X[:, feature_mask]
        expr_names = list(pd.Index(adata.var_names)[feature_mask].astype(str))
    else:
        expr_names = list(adata.var_names.astype(str))

    X, latent_names = _append_latent_features(X, adata, obsm_key=use_obsm_key)
    feature_names = expr_names + latent_names

    y = adata.obs[label_col].astype(str).to_numpy()

    return X, y, feature_names


def extract_x_target(
    adata: ad.AnnData,
    use_layer: str | None = None,
    feature_mask: np.ndarray | None = None,
    use_obsm_key: str | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract feature matrix for the target timepoint.

    Parameters
    ----------
    adata:
        AnnData subset for the target timepoint.
    use_layer:
        Optional layer name to use instead of `.X`.
    feature_mask:
        Optional boolean mask over variables to match training features.
    use_obsm_key:
        If not None, append features from `adata.obsm[use_obsm_key]`.
    """

    X = _get_expression_matrix(adata, use_layer=use_layer)

    if feature_mask is not None:
        X = X[:, feature_mask]
        expr_names = list(pd.Index(adata.var_names)[feature_mask].astype(str))
    else:
        expr_names = list(adata.var_names.astype(str))

    X, latent_names = _append_latent_features(X, adata, obsm_key=use_obsm_key)
    feature_names = expr_names + latent_names

    return X, feature_names
