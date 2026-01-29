"""
# Practical Code Examples

This notebook demonstrates how to use the new features programmatically.

## Example 1: Basic Feature Evaluation

```python
from cytof_label_transfer import (
    load_anndata,
    split_timepoints,
    extract_xy,
    compute_feature_importance,
    plot_feature_importance,
)

# Load and prepare data
adata = load_anndata("data.h5ad")
adata_train, _ = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)

# Extract features
X_train, y_train, feature_names = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch"
)

# Evaluate importance
importances, _ = compute_feature_importance(
    X_train, y_train, feature_names, method="random_forest"
)

# Visualize
plot_feature_importance(
    importances,
    feature_names,
    top_n=30,
    output_path="feature_importance.png"
)

print(f"Features evaluated. Top feature: {feature_names[importances.argmax()]}")
```

## Example 2: Training with Custom Hyperparameters

```python
from cytof_label_transfer import (
    load_anndata,
    split_timepoints,
    extract_xy,
    train_classifier,
    load_hyperparameters_from_json,
)

# Load data
adata = load_anndata("data.h5ad")
adata_train, _ = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)

# Extract features
X_train, y_train, feature_names = extract_xy(
    adata_train,
    label_col="celltype"
)

# Load custom hyperparameters
custom_params = load_hyperparameters_from_json("custom_hyperparams.json")

# Train with custom hyperparameters
bundle = train_classifier(
    X_train,
    y_train,
    feature_names=feature_names,
    n_splits=5,
    n_iter=50,  # Try 50 random configurations
    param_distributions=custom_params,
    output_dir="results/custom_model/"
)

print(f"Best F1 score: {bundle.cv_best_score:.4f}")
print(f"Best params: {bundle.best_params}")
```

## Example 3: Feature Selection by Group

```python
from cytof_label_transfer import (
    load_anndata,
    split_timepoints,
    extract_xy,
    create_feature_groups,
    select_features_by_groups,
    train_classifier,
)

# Load data
adata = load_anndata("data.h5ad")
adata_train, _ = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)

# Extract all features
X_train, y_train, feature_names = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch"
)

# Create and examine feature groups
feature_groups = create_feature_groups(feature_names, obsm_key="X_scVI_200_epoch")
print(f"Feature groups: {list(feature_groups.keys())}")
print(f"  - Markers: {len(feature_groups['markers'])} features")
print(f"  - Latent: {len(feature_groups['latent'])} features")

# Select only markers
selected_indices, selected_names = select_features_by_groups(
    feature_names,
    feature_groups,
    ["markers"]
)

print(f"Selected {len(selected_indices)} marker features")

# Re-extract with selection
X_train_selected, y_train_selected, names_selected = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch",
    selected_feature_indices=selected_indices
)

# Train with selected features
bundle = train_classifier(
    X_train_selected,
    y_train_selected,
    feature_names=names_selected,
    output_dir="results/markers_only/"
)
```

## Example 4: Feature Selection by Importance Percentile

```python
from cytof_label_transfer import (
    load_anndata,
    split_timepoints,
    extract_xy,
    compute_feature_importance,
    select_features_by_importance,
    train_classifier,
)

# Load and prepare data
adata = load_anndata("data.h5ad")
adata_train, _ = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)

# Extract features
X_train, y_train, feature_names = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch"
)

# Compute importance
importances, _ = compute_feature_importance(
    X_train, y_train, feature_names
)

# Select top 90th percentile features
selected_indices, selected_names = select_features_by_importance(
    importances,
    feature_names,
    percentile=90
)

print(f"Selected {len(selected_indices)} features above 90th percentile")

# Re-extract with selection
X_train_selected, y_train_selected, names_selected = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch",
    selected_feature_indices=selected_indices
)

# Train
bundle = train_classifier(
    X_train_selected,
    y_train_selected,
    feature_names=names_selected,
    output_dir="results/top90_features/"
)
```

## Example 5: Complete Workflow with All Features

```python
from pathlib import Path
from cytof_label_transfer import (
    load_anndata,
    split_timepoints,
    extract_xy,
    compute_feature_importance,
    plot_feature_importance,
    create_feature_groups,
    select_features_interactive_report,
    select_features_by_groups,
    load_hyperparameters_from_json,
    train_classifier,
)

# Step 1: Load and evaluate
print("Step 1: Loading data and evaluating features...")
adata = load_anndata("data.h5ad")
adata_train, _ = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)

X_train, y_train, feature_names = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch"
)

# Compute importance
importances, _ = compute_feature_importance(X_train, y_train, feature_names)

# Create outputs
output_dir = Path("analysis_results")
output_dir.mkdir(exist_ok=True)

plot_feature_importance(
    importances,
    feature_names,
    output_path=output_dir / "feature_importance.png"
)

feature_groups = create_feature_groups(feature_names, obsm_key="X_scVI_200_epoch")
select_features_interactive_report(
    importances,
    feature_names,
    feature_groups,
    output_dir=output_dir / "reports"
)

# Step 2: Select features by group
print("Step 2: Selecting features by group...")
selected_indices, selected_names = select_features_by_groups(
    feature_names,
    feature_groups,
    ["markers", "latent"]  # Use both
)

# Step 3: Train with custom hyperparameters
print("Step 3: Training with custom hyperparameters...")

custom_params = {
    "n_estimators": [1000],
    "max_depth": [10, 12],
    "learning_rate": [0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.5, 0.7],
}

X_selected, y_selected, names_selected = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch",
    selected_feature_indices=selected_indices
)

bundle = train_classifier(
    X_selected,
    y_selected,
    feature_names=names_selected,
    n_splits=5,
    n_iter=20,
    param_distributions=custom_params,
    output_dir=output_dir / "trained_model",
    use_gpu=True,
)

print(f"Training complete!")
print(f"Best CV F1 score: {bundle.cv_best_score:.4f}")
print(f"Number of features used: {len(bundle.feature_names)}")
print(f"Results saved to: {output_dir}")
```

## Example 6: Save Feature Indices for CLI Use

```python
import json
from pathlib import Path
from cytof_label_transfer import (
    load_anndata,
    split_timepoints,
    extract_xy,
    compute_feature_importance,
    select_features_by_importance,
)

# Compute importance
adata = load_anndata("data.h5ad")
adata_train, _ = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)
X_train, y_train, feature_names = extract_xy(adata_train, label_col="celltype")

importances, _ = compute_feature_importance(X_train, y_train, feature_names)

# Select top 90% features
selected_indices, _ = select_features_by_importance(importances, feature_names, percentile=90)

# Save for CLI use
with open("my_selected_features.json", "w") as f:
    json.dump(selected_indices.tolist(), f)

print(f"Saved {len(selected_indices)} feature indices to my_selected_features.json")

# Now can use in CLI:
# python train_model.py --selected_feature_indices my_selected_features.json ...
```

## Example 7: Comparing Different Feature Sets

```python
import pandas as pd
from cytof_label_transfer import (
    load_anndata,
    split_timepoints,
    extract_xy,
    create_feature_groups,
    select_features_by_groups,
    train_classifier,
)

# Load data
adata = load_anndata("data.h5ad")
adata_train, _ = split_timepoints(adata, "timepoint", [1, 2, 3, 4], 5)

# Create feature groups
X_all, y_all, all_names = extract_xy(
    adata_train,
    label_col="celltype",
    use_obsm_key="X_scVI_200_epoch"
)

feature_groups = create_feature_groups(all_names, obsm_key="X_scVI_200_epoch")

# Compare different feature combinations
results = []

for group_set in [["markers"], ["latent"], ["markers", "latent"]]:
    print(f"Training with: {group_set}...")
    
    selected_indices, selected_names = select_features_by_groups(
        all_names,
        feature_groups,
        group_set
    )
    
    X_train, y_train, names = extract_xy(
        adata_train,
        label_col="celltype",
        use_obsm_key="X_scVI_200_epoch",
        selected_feature_indices=selected_indices
    )
    
    bundle = train_classifier(
        X_train,
        y_train,
        feature_names=names,
        output_dir=f"results/compare_{'+'.join(group_set)}"
    )
    
    results.append({
        "feature_groups": ",".join(group_set),
        "n_features": len(selected_indices),
        "cv_f1_score": bundle.cv_best_score,
    })

# Display comparison
df = pd.DataFrame(results)
print(df)
```
"""
