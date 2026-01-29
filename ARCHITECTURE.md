# Updated Architecture & Design

## Overview

The CyTOF label transfer pipeline has been enhanced with three major capabilities:
1. **Custom hyperparameter tuning**
2. **Feature importance evaluation and visualization**
3. **Feature engineering with selective grouping**

## New Module Structure

```
cytof_label_transfer/
├── __init__.py                    # Updated: Export new functions
├── data_utils.py                  # Modified: extract_xy() now supports selected_feature_indices
├── model.py                       # Modified: Added load_hyperparameters_from_json(), param_distributions param
├── qc.py                          # Unchanged
├── feature_selection.py           # NEW: Feature evaluation, selection, and grouping
└── __pycache__/

train_model.py                      # Modified: New CLI args for all 3 features
predict_timepoint5.py              # Unchanged

Configuration/Example Files:
├── example_custom_hyperparams.json # NEW: Template for custom hyperparameters
├── example_selected_features.txt   # NEW: Template for feature indices
├── ADVANCED_USAGE.md              # NEW: Comprehensive guide
├── PRACTICAL_EXAMPLES.md          # NEW: Code examples
└── IMPLEMENTATION_SUMMARY.md      # NEW: This implementation guide
```

## Data Flow Diagram

### Original Pipeline
```
Input Data (.h5ad)
    ↓
split_timepoints() → Train subset
    ↓
extract_xy() → Features + Labels
    ↓
train_classifier() → Trained Model
    ↓
evaluate_and_plot_cv() → QC Plots
    ↓
Output: model_bundle.joblib + metrics
```

### Enhanced Pipeline
```
Input Data (.h5ad)
    ↓
split_timepoints() → Train subset
    ↓
extract_xy() → Full Features + Labels
    ↓
[OPTIONAL] Feature Evaluation:
    compute_feature_importance() → Importance scores
    plot_feature_importance() → Visualization
    select_features_interactive_report() → CSV report
    ↓
[OPTIONAL] Feature Selection:
    create_feature_groups() → Group assignment
    select_features_by_groups() → By group membership
    OR
    select_features_by_importance() → By percentile
    ↓
[OPTIONAL] Re-extract with selection:
    extract_xy(..., selected_feature_indices=...) → Reduced Features
    ↓
[OPTIONAL] Load custom hyperparameters:
    load_hyperparameters_from_json() → Custom param distributions
    ↓
train_classifier(..., param_distributions=...) → Trained Model
    ↓
evaluate_and_plot_cv() → QC Plots
    ↓
Output: model_bundle.joblib + metrics
```

## API Changes

### New Functions

#### feature_selection.py (New Module)
```python
def compute_feature_importance(X, y, feature_names, method="random_forest", 
                               n_estimators=100, random_state=42) → (importances, feature_names)
def plot_feature_importance(importances, feature_names, top_n=30, output_path=None, figsize=(12,8)) → None
def create_feature_groups(feature_names, obsm_key=None) → Dict[str, List[int]]
def select_features_by_importance(importances, feature_names, percentile=90) → (indices, names)
def select_features_by_groups(feature_names, feature_groups, selected_groups) → (indices, names)
def select_features_interactive_report(importances, feature_names, feature_groups, output_dir) → None
```

#### model.py (Modified)
```python
def load_hyperparameters_from_json(json_path) → Dict[str, Any]  # NEW

def train_classifier(X, y, feature_names, *, ..., 
                     param_distributions=None) → TrainedModelBundle  # param_distributions NEW
```

#### data_utils.py (Modified)
```python
def extract_xy(adata, label_col, use_layer=None, feature_mask=None, 
               use_obsm_key=None, selected_feature_indices=None) → (X, y, feature_names)  # NEW param

def extract_x_target(adata, use_layer=None, feature_mask=None, 
                     use_obsm_key=None, selected_feature_indices=None) → (X, feature_names)  # NEW param
```

### Updated CLI Arguments (train_model.py)

**New Arguments:**
- `--custom_hyperparams <path>` - Path to JSON with custom hyperparameters
- `--eval_features` - Enable feature importance evaluation
- `--feature_importance_output_dir <path>` - Where to save feature reports
- `--selected_feature_indices <path>` - Path to file with selected feature indices
- `--feature_groups [markers|latent|...]` - Space-separated feature groups to use

## Backward Compatibility

✅ **Fully backward compatible**
- All new parameters are optional
- Default behavior unchanged when new args not used
- Existing scripts work without modification
- New features are additive, not breaking

## Usage Patterns

### Pattern 1: Simple Evaluation Only
```bash
python train_model.py ... --eval_features --feature_importance_output_dir reports/
# Generates reports, but doesn't change training
```

### Pattern 2: Feature Selection Only
```bash
python train_model.py ... --feature_groups markers --selected_feature_indices top_features.txt
# Trains with selected features using default hyperparameters
```

### Pattern 3: Custom Hyperparameters Only
```bash
python train_model.py ... --custom_hyperparams custom_params.json
# Uses custom search space with all available features
```

### Pattern 4: Complete Optimization
```bash
python train_model.py ... \
  --eval_features \
  --feature_groups markers latent \
  --selected_feature_indices selected.txt \
  --custom_hyperparams custom_params.json
# Full workflow with everything optimized
```

## File Formats

### Custom Hyperparameters JSON
```json
{
    "param_name": [value1, value2, ...] or single_value,
    ...
}
```

### Selected Features (Text)
```
0 1 2 5 7 10 15 20
```

### Selected Features (JSON)
```json
[0, 1, 2, 5, 7, 10, 15, 20]
```

## Performance Considerations

### Feature Importance Computation
- Uses Random Forest (quick, parallelizable)
- Time: O(n_samples × n_features × n_trees)
- Memory: Reasonable for typical CyTOF datasets

### Feature Selection Options
| Method | Speed | Selectivity | Use Case |
|--------|-------|-------------|----------|
| By percentile | Very fast | Flexible | Quick filtering |
| By groups | Very fast | Fixed | Biological grouping |
| By importance + manual | Fast | High | Expert curation |

### Training with Fewer Features
- Reduces model complexity
- Faster training
- Better generalization (if features chosen well)
- Can improve inference speed

## Testing Strategy

### Unit Tests Needed
- Feature importance calculation
- Feature group creation
- Feature selection operations
- Hyperparameter loading from JSON
- Train/predict roundtrip with selected features

### Integration Tests
```bash
# Test 1: Full evaluation workflow
python train_model.py ... --eval_features

# Test 2: Feature group selection
python train_model.py ... --feature_groups markers

# Test 3: Custom hyperparameters
python train_model.py ... --custom_hyperparams test_params.json

# Test 4: All together
python train_model.py ... --eval_features --feature_groups markers --custom_hyperparams test_params.json
```

## Future Extensions

### Possible Enhancements
1. Support additional importance methods (XGBoost, permutation, etc.)
2. Interactive feature selection GUI
3. Automated feature group discovery
4. Cross-validation across feature selections
5. Feature importance stability analysis
6. Support for feature interactions

### Integration Points
- Can wrap in Jupyter notebooks for interactive workflows
- Can integrate with automated hyperparameter optimization tools
- Can extend with domain-specific feature engineering

## Summary of Changes

| File | Changes | Lines |
|------|---------|-------|
| feature_selection.py | NEW | 280 |
| model.py | Modified | +40 |
| data_utils.py | Modified | +6 |
| train_model.py | Modified | +70 |
| __init__.py | Modified | +12 |
| ADVANCED_USAGE.md | NEW | 200+ |
| PRACTICAL_EXAMPLES.md | NEW | 300+ |
| IMPLEMENTATION_SUMMARY.md | NEW | 150+ |
| example_custom_hyperparams.json | NEW | 12 |
| example_selected_features.txt | NEW | 5 |

**Total New Code**: ~800 lines  
**Total Documentation**: ~650 lines  
**Backward Compatibility**: ✅ 100%
