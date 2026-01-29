# Summary of New Features Added

## Overview
Three major enhancements have been added to the CyTOF label transfer pipeline to provide more flexibility and control over model training:

---

## 1. Custom Hyperparameter Assignment ✅

### What's New
- **Function**: `load_hyperparameters_from_json()` in [model.py](model.py)
- **CLI Argument**: `--custom_hyperparams <path_to_json>`
- Users can now provide custom XGBoost hyperparameters via JSON file instead of relying on default settings

### How It Works
1. Create a JSON file with hyperparameter distributions
2. Pass it to `train_model.py` with `--custom_hyperparams`
3. The training pipeline will use your custom search space

### Example
```bash
python train_model.py \
  --input_h5ad data.h5ad \
  --time_col timepoint \
  --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --output_dir results/ \
  --custom_hyperparams custom_hyperparams.json
```

### Files
- Template: [example_custom_hyperparams.json](example_custom_hyperparams.json)
- Modified: [model.py](cytof_label_transfer/model.py) - `load_hyperparameters_from_json()`, `train_classifier()` updated

---

## 2. Feature Evaluation & Visualization ✅

### What's New
- **New Module**: [feature_selection.py](cytof_label_transfer/feature_selection.py)
- **CLI Argument**: `--eval_features` (boolean flag)
- **Output Directory**: `--feature_importance_output_dir <path>`
- Users can evaluate feature importance BEFORE training to identify the most predictive markers

### What It Does
1. Computes feature importance using a quick Random Forest model
2. Creates visualization of top 30 features
3. Generates CSV report with importance scores and feature group membership
4. Helps users make informed decisions about which features to use

### Example
```bash
python train_model.py \
  --input_h5ad data.h5ad \
  --time_col timepoint \
  --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --eval_features \
  --feature_importance_output_dir feature_reports/
```

### Output Files
- `feature_importance_top30.png` - Bar plot of top features
- `feature_importance_report.csv` - Complete ranking with indices and groups

### Functions
- `compute_feature_importance()` - Quick importance estimation
- `plot_feature_importance()` - Create visualizations
- `select_features_by_importance()` - Filter by percentile
- `select_features_interactive_report()` - Generate CSV report

---

## 3. Feature Engineering with Grouping ✅

### What's New
- **CLI Arguments**: `--feature_groups` (space-separated), `--selected_feature_indices <file>`
- Users can selectively use feature groups (markers, latent, custom combinations)
- Users can manually select specific features by index

### Feature Groups
When using `--use_obsm_key X_scVI_200_epoch`, features are automatically categorized as:
- **markers**: Raw expression markers from `.X`
- **latent**: Latent dimensions from `.obsm[key]`

### How It Works

#### By Groups
```bash
# Use only raw markers
python train_model.py \
  --input_h5ad data.h5ad \
  --use_obsm_key X_scVI_200_epoch \
  --feature_groups markers \
  --train_timepoints 1 2 3 4 \
  --output_dir results/markers_only/
```

#### By Manual Selection
Create `selected_features.txt`:
```
0 1 2 5 7 10 15 20 25 30
```

Then use it:
```bash
python train_model.py \
  --input_h5ad data.h5ad \
  --train_timepoints 1 2 3 4 \
  --selected_feature_indices selected_features.txt \
  --output_dir results/
```

### Functions
- `create_feature_groups()` - Automatically categorize markers vs latent
- `select_features_by_groups()` - Select by group membership
- Modified `extract_xy()` - Now accepts `selected_feature_indices` parameter

### Files
- Template: [example_selected_features.txt](example_selected_features.txt)
- Modified: [data_utils.py](cytof_label_transfer/data_utils.py) - `extract_xy()`, `extract_x_target()`

---

## Integration Points

### Modified Files
1. **[cytof_label_transfer/model.py](cytof_label_transfer/model.py)**
   - Added `load_hyperparameters_from_json()` function
   - Updated `train_classifier()` to accept `param_distributions` parameter

2. **[cytof_label_transfer/data_utils.py](cytof_label_transfer/data_utils.py)**
   - Updated `extract_xy()` with `selected_feature_indices` parameter
   - Updated `extract_x_target()` with `selected_feature_indices` parameter

3. **[train_model.py](train_model.py)**
   - Added new CLI arguments for all three features
   - Updated `parse_args()` with 5 new arguments
   - Updated `main()` to handle feature evaluation, selection, and custom hyperparams

4. **[cytof_label_transfer/__init__.py](cytof_label_transfer/__init__.py)**
   - Exported new feature selection functions

### New Files
1. **[cytof_label_transfer/feature_selection.py](cytof_label_transfer/feature_selection.py)** (280 lines)
   - Complete feature evaluation and selection toolkit

2. **[ADVANCED_USAGE.md](ADVANCED_USAGE.md)** (200+ lines)
   - Comprehensive documentation with examples and workflow

3. **[example_custom_hyperparams.json](example_custom_hyperparams.json)**
   - Template for custom hyperparameters

4. **[example_selected_features.txt](example_selected_features.txt)**
   - Template for feature indices

---

## Typical Workflow

### 1. Evaluate Features
```bash
python train_model.py \
  --input_h5ad data.h5ad \
  --time_col timepoint \
  --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --use_obsm_key X_scVI_200_epoch \
  --eval_features \
  --feature_importance_output_dir feature_reports/
```

### 2. Inspect `feature_reports/feature_importance_report.csv`

### 3. Train with Optimized Settings
```bash
python train_model.py \
  --input_h5ad data.h5ad \
  --time_col timepoint \
  --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --use_obsm_key X_scVI_200_epoch \
  --feature_groups markers latent \
  --selected_feature_indices selected_features.txt \
  --custom_hyperparams custom_hyperparams.json \
  --output_dir results/final_model/
```

---

## Backward Compatibility

All changes are **fully backward compatible**:
- Default behavior is unchanged when new arguments are not used
- Existing code that calls functions directly will still work
- Optional parameters have sensible defaults

---

## Testing Recommendations

1. **Feature Evaluation**: Verify plots and CSV reports are generated correctly
2. **Feature Selection**: Confirm that selected features reduce dimensionality without hurting performance
3. **Custom Hyperparams**: Compare models trained with default vs custom hyperparameters
4. **Feature Groups**: Test training with different group combinations (markers-only, latent-only, combined)

---

## Documentation

See [ADVANCED_USAGE.md](ADVANCED_USAGE.md) for:
- Detailed usage examples
- Python API examples
- Recommended hyperparameter ranges
- Troubleshooting guide
- Complete workflow walkthrough
