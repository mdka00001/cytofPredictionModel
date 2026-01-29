# Quick Reference Guide

## New Features at a Glance

### 1️⃣ Custom Hyperparameters
```bash
python train_model.py ... --custom_hyperparams custom_params.json
```
- Define XGBoost hyperparameter search space in JSON
- File: [example_custom_hyperparams.json](example_custom_hyperparams.json)

### 2️⃣ Feature Evaluation
```bash
python train_model.py ... --eval_features --feature_importance_output_dir reports/
```
- Compute feature importance before training
- Outputs: plots + CSV report with rankings
- Guides feature selection decisions

### 3️⃣ Feature Engineering
```bash
python train_model.py ... --feature_groups markers --selected_feature_indices top_features.txt
```
- Select features by group (markers, latent)
- Or by manual index list
- Reduces dimensionality and training time

---

## File Mapping

### Core Implementation
| File | Purpose | New/Modified |
|------|---------|---|
| [feature_selection.py](cytof_label_transfer/feature_selection.py) | Feature evaluation & selection toolkit | **NEW** |
| [model.py](cytof_label_transfer/model.py) | Custom hyperparams support | Modified |
| [data_utils.py](cytof_label_transfer/data_utils.py) | Feature selection support | Modified |
| [train_model.py](train_model.py) | CLI exposure | Modified |
| [__init__.py](cytof_label_transfer/__init__.py) | API exports | Modified |

### Documentation
| File | Purpose |
|------|---------|
| [ADVANCED_USAGE.md](ADVANCED_USAGE.md) | **START HERE**: Complete guide with workflows |
| [PRACTICAL_EXAMPLES.md](PRACTICAL_EXAMPLES.md) | 7 ready-to-use code examples |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design & technical details |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | What changed and where |
| [CHECKLIST.md](CHECKLIST.md) | Implementation status |

### Examples & Templates
| File | Purpose |
|------|---------|
| [example_custom_hyperparams.json](example_custom_hyperparams.json) | Template for custom hyperparameters |
| [example_selected_features.txt](example_selected_features.txt) | Template for feature indices |

---

## Quick Decision Tree

### Do you want to...

**...evaluate features before training?**
→ Use `--eval_features`
→ Read: [ADVANCED_USAGE.md#2-feature-evaluation](ADVANCED_USAGE.md#2-feature-evaluation--visualization)
→ Example: [PRACTICAL_EXAMPLES.md#Example-1](PRACTICAL_EXAMPLES.md#example-1-basic-feature-evaluation)

**...use only certain markers (no latent)?**
→ Use `--feature_groups markers`
→ Read: [ADVANCED_USAGE.md#3-feature-engineering](ADVANCED_USAGE.md#3-feature-engineering-with-grouping)
→ Example: Usage example in same section

**...manually select specific features?**
→ Use `--selected_feature_indices <file>`
→ Read: [ADVANCED_USAGE.md#manual-feature-selection](ADVANCED_USAGE.md#manual-feature-selection)
→ Example: [PRACTICAL_EXAMPLES.md#Example-6](PRACTICAL_EXAMPLES.md#example-6-save-feature-indices-for-cli-use)

**...use custom hyperparameters?**
→ Use `--custom_hyperparams <file>`
→ Read: [ADVANCED_USAGE.md#1-custom-hyperparameter-assignment](ADVANCED_USAGE.md#1-custom-hyperparameter-assignment)
→ Example: [PRACTICAL_EXAMPLES.md#Example-2](PRACTICAL_EXAMPLES.md#example-2-training-with-custom-hyperparameters)

**...do everything optimized?**
→ Read: [ADVANCED_USAGE.md#complete-example-workflow](ADVANCED_USAGE.md#complete-example-workflow)
→ See: [PRACTICAL_EXAMPLES.md#Example-5](PRACTICAL_EXAMPLES.md#example-5-complete-workflow-with-all-features)

---

## Common Workflows

### Workflow A: Quick Evaluation
```bash
# 1. Evaluate features
python train_model.py \
  --input_h5ad data.h5ad \
  --time_col timepoint --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --output_dir results/eval \
  --eval_features --feature_importance_output_dir reports/

# 2. Review reports/feature_importance_report.csv
# 3. Create selected_features.txt with chosen indices
```

### Workflow B: Feature Selection
```bash
# Train with markers only
python train_model.py \
  --input_h5ad data.h5ad \
  --time_col timepoint --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --use_obsm_key X_scVI_200_epoch \
  --feature_groups markers \
  --output_dir results/markers_only
```

### Workflow C: Full Optimization
```bash
# 1. Evaluate
python train_model.py \
  --input_h5ad data.h5ad \
  --time_col timepoint --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --use_obsm_key X_scVI_200_epoch \
  --eval_features --feature_importance_output_dir reports/

# 2. Select features from report

# 3. Create custom_hyperparams.json

# 4. Train with everything optimized
python train_model.py \
  --input_h5ad data.h5ad \
  --time_col timepoint --label_col celltype \
  --train_timepoints 1 2 3 4 \
  --use_obsm_key X_scVI_200_epoch \
  --selected_feature_indices selected.txt \
  --custom_hyperparams custom_params.json \
  --output_dir results/optimized
```

---

## Python API Quick Reference

### Import What You Need
```python
from cytof_label_transfer import (
    load_anndata, split_timepoints, extract_xy,
    compute_feature_importance, plot_feature_importance,
    create_feature_groups, select_features_by_groups,
    train_classifier, load_hyperparameters_from_json,
)
```

### Evaluate Features
```python
importances, names = compute_feature_importance(X, y, feature_names)
plot_feature_importance(importances, feature_names, output_path="importance.png")
```

### Create Feature Groups
```python
groups = create_feature_groups(feature_names, obsm_key="X_scVI_200_epoch")
# Returns: {"markers": [0,1,2,...], "latent": [...]}
```

### Select Features
```python
# By group
selected_idx, selected_names = select_features_by_groups(
    feature_names, groups, ["markers"]
)

# Re-extract with selection
X_selected, y_selected, names_selected = extract_xy(
    adata, label_col="celltype",
    selected_feature_indices=selected_idx
)
```

### Train with Custom Hyperparameters
```python
params = load_hyperparameters_from_json("custom_params.json")
bundle = train_classifier(
    X_train, y_train, feature_names,
    param_distributions=params,
    output_dir="results/"
)
```

---

## Key Functions Reference

### Feature Selection Module

```python
compute_feature_importance(X, y, feature_names, method="random_forest")
    → importances (array), feature_names (list)

plot_feature_importance(importances, feature_names, top_n=30, output_path=None)
    → None (saves plot)

create_feature_groups(feature_names, obsm_key=None)
    → Dict[str, List[int]]  {group_name: [indices]}

select_features_by_importance(importances, feature_names, percentile=90)
    → selected_indices, selected_names

select_features_by_groups(feature_names, feature_groups, selected_groups)
    → selected_indices, selected_names

select_features_interactive_report(importances, feature_names, feature_groups, output_dir)
    → None (saves CSV report)
```

### Enhanced Model Module

```python
load_hyperparameters_from_json(json_path)
    → Dict[str, Any]

train_classifier(..., param_distributions=None)
    → TrainedModelBundle
    # If param_distributions=None, uses default
```

### Enhanced Data Utils

```python
extract_xy(..., selected_feature_indices=None)
    → X, y, feature_names

extract_x_target(..., selected_feature_indices=None)
    → X, feature_names
```

---

## Common Questions

**Q: Where do I start?**
A: Read [ADVANCED_USAGE.md](ADVANCED_USAGE.md) first. It's the main guide.

**Q: Can I use the old way without new features?**
A: Yes! All features are optional. Existing scripts work unchanged.

**Q: How do I create the custom hyperparameters file?**
A: Copy [example_custom_hyperparams.json](example_custom_hyperparams.json) and modify values.

**Q: What's the difference between --feature_groups and --selected_feature_indices?**
A: Groups are semantic (markers vs latent), indices are specific features you choose.

**Q: How long does feature importance take?**
A: Usually minutes for typical datasets. Uses parallel processing (all cores).

**Q: Can I use Python API in notebooks?**
A: Yes! See [PRACTICAL_EXAMPLES.md](PRACTICAL_EXAMPLES.md) for notebook examples.

---

## Links to Detailed Docs

- **Complete Guide**: [ADVANCED_USAGE.md](ADVANCED_USAGE.md)
- **Code Examples**: [PRACTICAL_EXAMPLES.md](PRACTICAL_EXAMPLES.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **What Changed**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Status**: [CHECKLIST.md](CHECKLIST.md)

---

## Summary

✅ **3 new features implemented**
✅ **Fully backward compatible**
✅ **Extensively documented**
✅ **Ready to use**

Start with [ADVANCED_USAGE.md](ADVANCED_USAGE.md) and pick the feature you need! 🚀
