"""
# Advanced CyTOF Label Transfer: Feature Engineering & Hyperparameter Tuning

This document explains the three new features added to the CyTOF label transfer pipeline.

## 1. Custom Hyperparameter Assignment

Instead of using the default hyperparameter search space, you can now provide your own
custom hyperparameters in a JSON file.

### Usage

Create a JSON file with your hyperparameter settings:

```json
{
    "n_estimators": [800, 1000, 1200],
    "max_depth": [8, 10, 12],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.5],
    "min_child_weight": [1],
    "gamma": [0.0]
}
```

Then pass it to the training script:

```bash
python train_model.py \\
  --input_h5ad data.h5ad \\
  --time_col timepoint \\
  --label_col celltype \\
  --train_timepoints 1 2 3 4 \\
  --output_dir results/ \\
  --custom_hyperparams custom_hyperparams.json
```

**Notes:**
- Values can be single numbers or lists
- If a value is a list, it will be sampled during hyperparameter search
- Use `--cv_iter` to control how many random configurations to try

### Recommended Parameters for XGBoost

- **n_estimators**: 500-2000 (number of boosting rounds)
- **max_depth**: 5-15 (tree depth)
- **learning_rate**: 0.01-0.2 (step size)
- **subsample**: 0.5-1.0 (row sampling)
- **colsample_bytree**: 0.3-1.0 (feature sampling)
- **min_child_weight**: 0.5-10 (minimum leaf instance weight)
- **gamma**: 0-1 (minimum loss reduction)
- **reg_lambda**: 0-2 (L2 regularization)
- **reg_alpha**: 0-2 (L1 regularization)

## 2. Feature Evaluation & Visualization

Before training, you can evaluate feature importance using a quick Random Forest model.
This helps identify which markers are most predictive of cell types.

### Usage

```bash
python train_model.py \\
  --input_h5ad data.h5ad \\
  --time_col timepoint \\
  --label_col celltype \\
  --train_timepoints 1 2 3 4 \\
  --output_dir results/ \\
  --eval_features \\
  --feature_importance_output_dir feature_eval/
```

### Output Files

The `feature_eval/` directory will contain:

1. **feature_importance_top30.png**: Bar plot of top 30 most important features
2. **feature_importance_report.csv**: CSV file with:
   - feature_index: 0-based index in the feature matrix
   - feature_name: marker name or latent feature name
   - importance: importance score
   - groups: which feature group(s) the feature belongs to

You can then inspect the CSV file to manually select features based on their importance
and biological relevance.

### Workflow

1. Run with `--eval_features` to see importance rankings
2. Inspect `feature_importance_report.csv`
3. Manually select features you want to use (see next section)
4. Run training again with `--selected_feature_indices` or `--feature_groups`

## 3. Feature Engineering with Grouping

When using both raw markers and latent features (e.g., scVI embeddings), you can
selectively use only certain feature groups.

### Feature Groups

When you provide `--use_obsm_key X_scVI_200_epoch`, features are automatically grouped:

- **markers**: All raw expression markers from `.X`
- **latent**: All dimensions from the specified `.obsm` layer

### Usage

Use only raw markers (no latent features):

```bash
python train_model.py \\
  --input_h5ad data.h5ad \\
  --time_col timepoint \\
  --label_col celltype \\
  --train_timepoints 1 2 3 4 \\
  --use_obsm_key X_scVI_200_epoch \\
  --feature_groups markers \\
  --output_dir results/markers_only/
```

Use only latent features (no raw markers):

```bash
python train_model.py \\
  --input_h5ad data.h5ad \\
  --time_col timepoint \\
  --label_col celltype \\
  --train_timepoints 1 2 3 4 \\
  --use_obsm_key X_scVI_200_epoch \\
  --feature_groups latent \\
  --output_dir results/latent_only/
```

Use both groups (default if not specified):

```bash
python train_model.py \\
  --input_h5ad data.h5ad \\
  --time_col timepoint \\
  --label_col celltype \\
  --train_timepoints 1 2 3 4 \\
  --use_obsm_key X_scVI_200_epoch \\
  --feature_groups markers latent \\
  --output_dir results/combined/
```

### Manual Feature Selection

After running `--eval_features`, you can manually select specific features by their index:

Create a file `selected_features.txt` with indices (0-based, space or newline separated):

```
0 1 2 5 7 10 15 20 25 30
```

Or as JSON:

```json
[0, 1, 2, 5, 7, 10, 15, 20, 25, 30]
```

Then use it in training:

```bash
python train_model.py \\
  --input_h5ad data.h5ad \\
  --time_col timepoint \\
  --label_col celltype \\
  --train_timepoints 1 2 3 4 \\
  --output_dir results/ \\
  --selected_feature_indices selected_features.txt
```

## Complete Example Workflow

Here's a typical workflow combining all three features:

### Step 1: Evaluate Features

```bash
python train_model.py \\
  --input_h5ad data.h5ad \\
  --time_col timepoint \\
  --label_col celltype \\
  --train_timepoints 1 2 3 4 \\
  --use_obsm_key X_scVI_200_epoch \\
  --output_dir results/step1_eval/ \\
  --eval_features \\
  --feature_importance_output_dir feature_reports/
```

### Step 2: Inspect Results

- Check `feature_reports/feature_importance_report.csv`
- Look at `feature_reports/feature_importance_top30.png`
- Decide which features to keep

### Step 3: Train with Selected Features

```bash
python train_model.py \\
  --input_h5ad data.h5ad \\
  --time_col timepoint \\
  --label_col celltype \\
  --train_timepoints 1 2 3 4 \\
  --use_obsm_key X_scVI_200_epoch \\
  --output_dir results/final_model/ \\
  --feature_groups markers latent \\
  --selected_feature_indices selected_features.txt \\
  --custom_hyperparams custom_hyperparams.json \\
  --cv_iter 50 \\
  --use_gpu
```

## Python API

You can also use these features programmatically in notebooks:

```python
from cytof_label_transfer import (
    load_anndata,
    split_timepoints,
    extract_xy,
    compute_feature_importance,
    create_feature_groups,
    plot_feature_importance,
    select_features_by_groups,
    train_classifier,
    load_hyperparameters_from_json,
)

# Load data
adata = load_anndata("data.h5ad")
adata_train, _ = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)

# Extract features
X_train, y_train, feature_names = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch",
)

# Evaluate features
importances, _ = compute_feature_importance(X_train, y_train, feature_names)
plot_feature_importance(importances, feature_names, output_path="importance.png")

# Select features by groups
feature_groups = create_feature_groups(feature_names, obsm_key="X_scVI_200_epoch")
selected_indices, _ = select_features_by_groups(
    feature_names, feature_groups, ["markers"]
)

# Re-extract with selection
X_train, y_train, feature_names = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch",
    selected_feature_indices=selected_indices,
)

# Load custom hyperparameters
params = load_hyperparameters_from_json("custom_hyperparams.json")

# Train with custom settings
bundle = train_classifier(
    X_train, y_train, feature_names,
    param_distributions=params,
    n_iter=50,
    output_dir="results/",
)
```

## Troubleshooting

**Q: Feature importance takes too long to compute**
- A: It uses a quick Random Forest. For very large datasets, reduce training set size or use GPU

**Q: Selected features don't match my expectations**
- A: Random Forest might not capture biological relationships. Consider domain knowledge + importance scores

**Q: Hyperparameter search is slow**
- A: Reduce `--cv_iter` (default 30) or use faster CPU/GPU. Check that `n_jobs=-1` is used.

**Q: Feature indices don't align after re-running**
- A: Always use the same `--use_obsm_key` and feature selection strategy to ensure consistency
"""
