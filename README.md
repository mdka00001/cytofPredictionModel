# CyTOF Label Transfer (Timepoint 1–4 → 5)

This repository provides a small, reusable codebase to train a
supervised classifier on trusted CyTOF/IMC cell-type annotations from
several timepoints (e.g. 1–4) and transfer those labels to a target
timepoint (e.g. 5) where annotations are unreliable.

The pipeline is built around AnnData objects (`.h5ad` files) and uses a
strong XGBoost classifier with cross-validated hyperparameter
optimization and QC plots. It can combine raw expression markers with a
shared latent space (e.g. scVI) stored in `adata.obsm`.

## Classifier choice

We use `xgboost.XGBClassifier` (gradient-boosted decision trees):

- Handles high-dimensional, non-linear marker data and learned latent
  embeddings well
- Robust to different feature scales (no strict need for normalization)
- Efficient and scalable for large numbers of cells
- Works with imbalanced classes when evaluated with appropriate metrics

Hyperparameters are tuned with `RandomizedSearchCV` using macro F1 as
the main selection metric, which balances performance across rare and
abundant cell types.

## Installation

Create a Python environment (e.g. via `conda` or `venv`) and install
dependencies:

```bash
cd cytofLabelTransfer
pip install -r requirements.txt
```

## Expected AnnData structure

The code assumes your `.h5ad` file has at least:

- `adata.X` (or a chosen `adata.layers["layer_name"]`) with marker
  intensities / features
- `adata.obs[time_col]` with timepoint identifiers (e.g. `1, 2, 3, 4, 5`)
- `adata.obs[label_col]` with trusted cell-type labels for timepoints
  1–4 (timepoint 5 can be noisy / incorrect)
-. Optionally, `adata.obsm["X_scVI_200_epoch"]` (or another key)
  containing a shared latent space across all timepoints, such as
  scVI-derived dimensions.

You can customize the exact column names via command-line arguments.

## 1. Train on timepoints 1–4

Example (adjust paths and column names to your data):

```bash
python train_model.py \
  --input_h5ad path/to/your_data.h5ad \
  --time_col timepoint \
  --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --use_obsm_key X_scVI_200_epoch \
  --use_gpu \
  --target_timepoint 5 \
  --output_dir results/model_tp1_4
```

What this does:

1. Loads the AnnData object.
2. Subsets to cells with `timepoint` in `{1, 2, 3, 4}`.
3. Uses all markers in `adata.var_names` as features and, if
  `--use_obsm_key` is provided, concatenates the corresponding latent
  space (e.g. `X_scVI_200_epoch`).
4. Trains an XGBoost classifier with randomized hyperparameter search
  and 5-fold CV.
5. Saves:
   - `results/model_tp1_4/model_bundle.joblib` (trained model + metadata)
   - `results/model_tp1_4/cv_results.csv` (all CV runs)
   - `results/model_tp1_4/training_metrics.json` (summary metrics)
   - `results/model_tp1_4/qc/` plots and CV metrics:
     - `cv_confusion_matrix.png`
     - `cv_per_class_f1.png`
     - `cv_classification_report.txt`
     - `cv_summary.json`

These QC artifacts allow you to judge how trustworthy the predictions
are before applying them to timepoint 5.

## 2. Predict labels for timepoint 5

Once the model is trained, apply it to the target timepoint (e.g. 5):

```bash
python predict_timepoint5.py \
  --input_h5ad path/to/your_data.h5ad \
  --time_col timepoint \
  --target_timepoint 5 \
  --model_dir results/model_tp1_4 \
  --label_col celltype \
  --use_obsm_key X_scVI_200_epoch \
  --pred_col celltype_predicted \
  --output_h5ad path/to/your_data_with_predictions.h5ad
```

What this does:

1. Loads the same AnnData object.
2. Subsets cells at `timepoint == 5`.
3. Reconstructs the same feature matrix (expression ± latent) that was
  used during training.
4. Predicts cell-type labels and maximum prediction probability.
5. Writes two new columns to `adata.obs` in the **full** object:
   - `celltype_predicted` (configurable via `--pred_col`)
   - `prediction_confidence` (max class probability, configurable via
     `--conf_col`)
6. Saves the updated `.h5ad` to `--output_h5ad`.

## Reproducibility

- The training and CV procedures use fixed `random_state` seeds by
  default for deterministic behavior.
- All key hyperparameters and CV results are stored in
  `training_metrics.json` and `cv_results.csv`.
- The trained model is saved as a single `model_bundle.joblib` file,
  which contains the estimator, feature names, and label names used at
  training time.

## Reusing the codebase

The core functionality is implemented as a small Python package under
`cytof_label_transfer/`:

- `data_utils.py`: helpers to load AnnData, split by timepoint, and
  extract `X, y`.
- `model.py`: training routine with hyperparameter search and
  prediction helpers.
- `qc.py`: cross-validated QC metrics and plots.

You can import these modules into your own analysis notebooks or
pipelines, for example:

```python
from cytof_label_transfer import load_anndata, split_timepoints, train_classifier
from cytof_label_transfer.data_utils import extract_xy

adata = load_anndata("path/to/data.h5ad")
adata_train, adata_target = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)
X_train, y_train, feature_names = extract_xy(adata_train, label_col="celltype")

bundle = train_classifier(X_train, y_train, feature_names, output_dir="results/model_custom")
```

To use a GPU (if available and your XGBoost build supports it), pass
``use_gpu=True`` when calling :func:`train_classifier` directly or add
``--use_gpu`` to the ``train_model.py`` command-line arguments.

You can also swap out the classifier in `model.py` for another
scikit-learn model (e.g. RandomForest, XGBoost) if you want to
experiment with alternative approaches while keeping the same data
handling and QC framework.

For a detailed analysis workflow tutorial. please check the [jupyter notebook]([https://pages.github.com/](https://github.com/mdka00001/cytofPredictionModel/blob/main/cytof_label_transfer_advanced_workflow.ipynb)) file.


