# New Jupyter Notebook: Advanced Workflow

## Overview

A new comprehensive Jupyter notebook has been created to demonstrate the complete CyTOF label transfer workflow with all new features:

**File**: [cytof_label_transfer_advanced_workflow.ipynb](cytof_label_transfer_advanced_workflow.ipynb)

---

## Notebook Structure

### 1. **Import Required Libraries** (1 cell)
- All necessary imports for data handling, visualization, and CyTOF analysis
- Configuration of plotting style and display settings
- Imports all new feature selection and evaluation functions

### 2. **Load and Prepare Data** (4 cells)
- Configuration of paths and parameters
- Load AnnData object
- Data exploration and statistics
- Split into training and target subsets
- Extract features with optional latent space

### 3. **Feature Importance Evaluation** (3 cells)
- Compute feature importance using Random Forest
- Display top features with importance scores
- Create visualization of top 30 features
- Generate CSV report with feature rankings and group membership

### 4. **Feature Selection (Optional)** (2 cells)
- Options for feature selection mode:
  - Use all features
  - Markers only (exclude latent)
  - Latent features only
  - Top percentile by importance
  - Manually selected features
- Re-extract features with selection if needed

### 5. **Custom Hyperparameters (Optional)** (1 cell)
- Toggle for using custom vs default hyperparameters
- Load custom hyperparameters from JSON file
- Display hyperparameter search space

### 6. **Train the Model** (2 cells)
- Configure training parameters (CV folds, iterations, GPU)
- Train classifier with cross-validation
- Display training metrics and best parameters

### 7. **Evaluate the Model** (2 cells)
- Generate QC plots (confusion matrix, per-class F1)
- Display evaluation metrics
- Review classification report

### 8. **Make Predictions** (2 cells)
- Extract features for target timepoint
- Generate predictions with confidence scores
- Add predictions to AnnData object
- Save updated AnnData with predictions

### 9. **Summary** (1 cell)
- Final statistics and output locations
- Reference to additional documentation

---

## Key Features

✅ **Complete Workflow**: From data loading to prediction in one notebook

✅ **Modular Design**: Each step is independent and can be toggled on/off

✅ **Feature Evaluation**: Built-in visualization and reporting of feature importance

✅ **Feature Selection**: Multiple selection strategies (groups, percentile, manual)

✅ **Custom Hyperparameters**: Optional custom hyperparameter configuration

✅ **Output Generation**: Saves models, plots, metrics, and predictions

✅ **Well-Documented**: Extensive comments and markdown explanations

---

## Usage

### Basic Workflow (5 minutes)
```python
# 1. Update INPUT_H5AD path in cell 2
# 2. Run all cells in order
# 3. View results in results/ directory
```

### Custom Configuration
```python
# In cell 4: Change FEATURE_SELECTION_MODE to:
#   - 'markers_only' (use only markers, no latent)
#   - 'top_percentile' (use top 90% features)
#   - 'manual' (manually select top N features)

# In cell 5: Set USE_CUSTOM_HYPERPARAMS = True
#   and provide path to custom_hyperparams.json
```

---

## Output Files

The notebook generates:

```
results/
├── feature_evaluation/
│   ├── feature_importance_top30.png
│   └── feature_importance_report.csv
│
├── trained_model/
│   ├── model_bundle.joblib
│   ├── cv_results.csv
│   ├── training_metrics.json
│   └── qc/
│       ├── cv_confusion_matrix.png
│       ├── cv_per_class_f1.png
│       ├── cv_classification_report.txt
│       └── cv_summary.json
│
└── data_with_predictions.h5ad
```

---

## Configuration Parameters

### Data Configuration (Cell 2)
```python
INPUT_H5AD = 'path/to/your_data.h5ad'
TIMEPOINT_COL = 'timepoint'
LABEL_COL = 'celltype'
OBSM_KEY = 'X_scVI_200_epoch'
TRAIN_TIMEPOINTS = [1, 2, 3, 4]
TARGET_TIMEPOINT = 5
```

### Feature Selection (Cell 4)
```python
FEATURE_SELECTION_MODE = 'all'  # Options: 'all', 'markers_only', 'top_percentile'
```

### Custom Hyperparameters (Cell 5)
```python
USE_CUSTOM_HYPERPARAMS = False
CUSTOM_HYPERPARAMS_FILE = 'custom_hyperparams.json'
```

### Training Configuration (Cell 6)
```python
CV_FOLDS = 5
CV_ITERATIONS = 30
USE_GPU = False
```

---

## Comparison with Original Notebooks

| Feature | Original | Advanced |
|---------|----------|----------|
| Feature Evaluation | ❌ | ✅ |
| Feature Selection | ❌ | ✅ |
| Custom Hyperparams | ❌ | ✅ |
| Model Training | ✅ | ✅ |
| Evaluation Plots | ✅ | ✅ |
| Prediction | ✅ | ✅ |
| Visualization | ✅ | ✅ |
| Documentation | Limited | Extensive |

---

## Code Examples

### Enable Feature Evaluation
```python
# Features are evaluated by default
# Results saved to results/feature_evaluation/
```

### Select Specific Features
```python
# In cell 4, change:
FEATURE_SELECTION_MODE = 'markers_only'

# Or select by importance:
FEATURE_SELECTION_MODE = 'top_percentile'
```

### Use Custom Hyperparameters
```python
# In cell 5, change:
USE_CUSTOM_HYPERPARAMS = True
CUSTOM_HYPERPARAMS_FILE = 'custom_hyperparams.json'
```

---

## Quick Start

1. **Open the notebook** in Jupyter or VS Code
2. **Update path** in cell 2: `INPUT_H5AD = 'your_data.h5ad'`
3. **Update column names** (timepoint, celltype) if different
4. **Run all cells** or run section by section
5. **Review outputs** in the `results/` directory

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'cytof_label_transfer'`
- **Solution**: Ensure conda environment is activated and installed

**Issue**: `FileNotFoundError: path/to/your_data.h5ad`
- **Solution**: Update INPUT_H5AD in cell 2 with correct path

**Issue**: Column name errors
- **Solution**: Update TIMEPOINT_COL, LABEL_COL to match your data

**Issue**: OBSM_KEY not found
- **Solution**: Set OBSM_KEY = None if your data doesn't have latent space

---

## Related Documentation

- [ADVANCED_USAGE.md](ADVANCED_USAGE.md) - Feature documentation
- [PRACTICAL_EXAMPLES.md](PRACTICAL_EXAMPLES.md) - Code examples
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick start guide

---

## Statistics

- **Total cells**: 28
- **Code cells**: 20
- **Markdown cells**: 8
- **File size**: 15 KB
- **Execution time**: ~5-30 minutes (depends on data size and GPU)

---

## Features Demonstrated

✅ **Feature Importance Evaluation**: Compute and visualize feature importance before training

✅ **Feature Selection**: Select features by group (markers/latent) or by importance threshold

✅ **Custom Hyperparameters**: Use custom XGBoost hyperparameter search space

✅ **Model Training**: Train with cross-validated hyperparameter optimization

✅ **QC & Evaluation**: Generate confusion matrices, F1 plots, and metrics

✅ **Prediction**: Predict labels for target timepoint with confidence scores

✅ **Visualization**: Multiple plots for understanding model behavior and predictions

---

## File Information

- **Format**: Jupyter Notebook (.ipynb)
- **Python Version**: 3.9+
- **Required Packages**: All standard (anndata, scikit-learn, xgboost, matplotlib, seaborn, pandas, numpy)
- **Dependencies**: cytof_label_transfer package (with new features)

---

**Created**: January 29, 2026  
**Status**: ✅ Ready to Use  
**Location**: [cytof_label_transfer_advanced_workflow.ipynb](cytof_label_transfer_advanced_workflow.ipynb)
