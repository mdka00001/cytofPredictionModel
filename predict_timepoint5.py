"""Predict cell types for a target timepoint (e.g. timepoint 5).

This script applies a previously trained model bundle to the cells of a
specified timepoint and writes predicted labels back into the AnnData
object.

Usage example
-------------

.. code-block:: bash

   python predict_timepoint5.py \
       --input_h5ad path/to/data.h5ad \
       --time_col timepoint \
       --target_timepoint 5 \
       --model_dir results/model_tp1_4 \
       --label_col celltype \
       --pred_col celltype_predicted \
       --output_h5ad path/to/data_with_predictions.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from cytof_label_transfer.data_utils import extract_x_target, load_anndata, split_timepoints
from cytof_label_transfer.model import TrainedModelBundle, predict_timepoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict cell types for a target timepoint.")

    parser.add_argument("--input_h5ad", required=True, help="Path to input .h5ad file.")
    parser.add_argument("--time_col", required=True, help="Name of .obs column with timepoints.")
    parser.add_argument("--target_timepoint", required=True, help="Timepoint to predict (e.g. 5).")
    parser.add_argument("--model_dir", required=True, help="Directory containing model_bundle.joblib.")
    parser.add_argument("--label_col", required=True, help="Original label column name (for reference).")
    parser.add_argument(
        "--pred_col",
        default="celltype_predicted",
        help="Name of .obs column to store predicted labels (default: celltype_predicted).",
    )
    parser.add_argument(
        "--conf_col",
        default="prediction_confidence",
        help="Name of .obs column to store maximum predicted probability (default: prediction_confidence).",
    )
    parser.add_argument("--use_layer", help="Optional AnnData layer name to use instead of .X.")
    parser.add_argument(
        "--use_obsm_key",
        default=None,
        help=(
            "Optional .obsm key for latent features to append (e.g. 'X_scVI_200_epoch'). "
            "Must match the key used during training."
        ),
    )
    parser.add_argument("--output_h5ad", required=True, help="Path to write updated .h5ad file.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_h5ad)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_h5ad)

    def _cast_timepoint(x: str):
        try:
            return float(x)
        except ValueError:
            return x

    target_timepoint = _cast_timepoint(args.target_timepoint)

    adata = load_anndata(str(input_path))

    # We only need the target subset for prediction
    _, adata_target = split_timepoints(
        adata,
        time_col=args.time_col,
        train_timepoints=[],
        target_timepoint=target_timepoint,
    )

    if adata_target.n_obs == 0:
        raise ValueError("No cells found for the specified target timepoint.")

    # Load trained model bundle
    bundle: TrainedModelBundle = joblib.load(model_dir / "model_bundle.joblib")

    # Extract target features using the same configuration as training
    X_target, feature_names_target = extract_x_target(
        adata_target,
        use_layer=args.use_layer,
        use_obsm_key=args.use_obsm_key,
    )

    # Sanity check: ensure feature ordering matches training
    if list(feature_names_target) != list(bundle.feature_names):
        raise ValueError(
            "Feature names/order in target data do not match those used during training. "
            "Ensure you are using the same layer and obsm key."
        )
    y_pred, y_proba = predict_timepoint(bundle, X_target)

    # Write predictions back into the *full* AnnData object
    pred_col = args.pred_col
    conf_col = args.conf_col

    # Initialize columns with NaN/None and fill only for target cells
    import pandas as pd

    adata.obs[pred_col] = pd.Series(index=adata.obs_names, dtype="object")
    adata.obs.loc[adata_target.obs_names, pred_col] = y_pred

    if y_proba is not None:
        max_conf = y_proba.max(axis=1)
        adata.obs[conf_col] = pd.Series(index=adata.obs_names, dtype="float")
        adata.obs.loc[adata_target.obs_names, conf_col] = max_conf

    adata.write_h5ad(output_path)

    print("Prediction completed.")
    print(f"Updated AnnData with predictions saved to: {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
