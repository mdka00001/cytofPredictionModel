"""Train a CyTOF cell-type classifier on trusted timepoints.

Usage example
-------------

.. code-block:: bash

   python train_model.py \
       --input_h5ad path/to/data.h5ad \
       --time_col timepoint \
       --label_col celltype \
       --train_timepoints 1 2 3 4 \
       --output_dir results/model_tp1_4

This script will:

1. Load the AnnData object.
2. Subset to the specified training timepoints.
3. Extract marker intensities as features and cell types as labels.
4. Train an optimized XGBoost classifier with CV.
5. Save the trained model bundle and QC plots/metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cytof_label_transfer.data_utils import extract_xy, load_anndata, split_timepoints
from cytof_label_transfer.model import TrainedModelBundle, train_classifier
from cytof_label_transfer.qc import evaluate_and_plot_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CyTOF label transfer classifier.")

    parser.add_argument("--input_h5ad", required=True, help="Path to input .h5ad file.")
    parser.add_argument("--time_col", required=True, help="Name of .obs column with timepoints.")
    parser.add_argument("--label_col", required=True, help="Name of .obs column with cell type labels.")
    parser.add_argument(
        "--train_timepoints",
        nargs="+",
        required=True,
        help="Space-separated list of timepoints to use for training (e.g. 1 2 3 4).",
    )
    parser.add_argument(
        "--target_timepoint",
        help="Timepoint intended for prediction (for logging only; not used during training).",
    )
    parser.add_argument("--use_layer", help="Optional AnnData layer name to use instead of .X.")
    parser.add_argument(
        "--use_obsm_key",
        default=None,
        help=(
            "Optional .obsm key for latent features to append (e.g. 'X_scVI_200_epoch'). "
            "If provided, expression features and latent features are concatenated."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the trained model and QC results.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--cv_iter",
        type=int,
        default=30,
        help="Number of random hyperparameter samples (default: 30).",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU-accelerated XGBoost (requires compatible GPU and CUDA).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_h5ad)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert potential numeric timepoints from strings
    def _cast_timepoint(x: str):
        try:
            return float(x)
        except ValueError:
            return x

    train_timepoints = [_cast_timepoint(tp) for tp in args.train_timepoints]

    adata = load_anndata(str(input_path))

    adata_train, _ = split_timepoints(
        adata,
        time_col=args.time_col,
        train_timepoints=train_timepoints,
        target_timepoint=None if args.target_timepoint is None else _cast_timepoint(args.target_timepoint),
    )

    X_train, y_train, feature_names = extract_xy(
        adata_train,
        label_col=args.label_col,
        use_layer=args.use_layer,
        use_obsm_key=args.use_obsm_key,
    )

    bundle: TrainedModelBundle = train_classifier(
        X_train,
        y_train,
        feature_names=feature_names,
        n_splits=args.cv_folds,
        n_iter=args.cv_iter,
        output_dir=output_dir,
        use_gpu=args.use_gpu,
    )

    # QC metrics and plots using CV predictions
    evaluate_and_plot_cv(
        estimator=bundle.estimator,
        X=X_train,
        y=y_train,
        class_names=bundle.label_names,
        output_dir=output_dir / "qc",
        label_encoder=bundle.label_encoder,
    )

    print("Training completed.")
    print(f"Model and metrics saved in: {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
