# GA Feature Selection Implementation Guide

## Overview
This implementation uses a Genetic Algorithm (GA) with Decision Tree classifier for feature selection on the diabetes diagnosis dataset (multiclass classification).

## Key Components

### 1. **Individual Representation**
- Binary mask over features (1 = selected, 0 = excluded)
- Length = number of features (51 in diabetes dataset, after removing target column)
- Each individual represents a feature subset

### 2. **Classifier**
- **Decision Tree** from scikit-learn
- Configuration: `max_depth=10`, `min_samples_split=5`
- Chosen for interpretability and computational efficiency

### 3. **Fitness Function**
- Multi-objective: `fitness = accuracy_weight × accuracy + feature_reduction_weight × (1 - n_features/total_features)`
- Default weights: `accuracy_weight=0.7`, `feature_reduction_weight=0.3`
- Optimizes for: **high accuracy + low feature count**

### 4. **GA Configuration** (as in notebook)
- Population size: 20
- Generations: 10 (test run; increase for production)
- Selection: Tournament (size=3)
- Crossover: Uniform (prob=0.8)
- Mutation: Bit-flip (prob=0.01)
- Elitism rate: 0.1 (10% best preserved)
- Cross-validation: 3-fold

### 5. **Minimum Features Constraint**
- **Enforced: At least 20 selected features** in final results
- If GA selects fewer, additional features are automatically added
- Ensures meaningful subset for analysis

## How to Run

### Step 1: Navigate to notebooks
```bash
cd "e:/code projects/ga-feature-select/notebooks"
```

### Step 2: Open and run the notebook
Open `02_baseline_experiments.ipynb` in Jupyter/VS Code and run all cells sequentially:

1. **Cell 1**: Import libraries
2. **Cell 2**: Load diabetes dataset (multiclass, 10 diagnosis classes)
3. **Cell 3**: Train/test split (80/20, stratified)
4. **Cell 4**: Baseline Decision Tree with all 51 features
5. **Cell 5**: Configure GA
6. **Cell 6**: Run GA evolution
7. **Cell 7**: Enforce minimum 20 features
8. **Cell 8**: Train final model with selected features
9. **Cell 9**: Summary comparison
10. **Cell 10**: Save results to CSV

### Step 3: Expected output location
```
data/results/ga_selected_features_diabetes_multiclass.csv
```

## Output CSV Structure

The saved CSV contains:
- **feature_index**: 0-based index
- **feature_name**: Column name from dataset
- **selected**: 1 if selected by GA (adjusted), 0 otherwise
- **baseline_accuracy**: Accuracy with all features
- **ga_accuracy**: Accuracy with selected features
- **accuracy_improvement**: ga_accuracy - baseline_accuracy
- **baseline_f1**: F1-score (weighted) with all features
- **ga_f1**: F1-score (weighted) with selected features
- **f1_improvement**: ga_f1 - baseline_f1
- **total_features**: Total number of features (51)
- **selected_features**: Number selected (≥20)
- **feature_reduction_pct**: Percentage of features eliminated

## Expected Results

### Typical outcomes (small test run):
- **Feature reduction**: 30-70% (depending on GA convergence)
- **Selected features**: 20-35 (enforced minimum 20)
- **Accuracy**: Comparable or improved vs baseline
- **F1-score**: Stable or improved (weighted for multiclass)

### For production runs:
Increase in notebook cell 5:
```python
population_size=50  # or 100
n_generations=50    # or 100
```

## Project Structure Reminder

```
ga-feature-select/
├── data/
│   ├── processed/
│   │   └── cleaned_diabetes_data.csv  # Input dataset
│   └── results/
│       └── ga_selected_features_diabetes_multiclass.csv  # Output
├── notebooks/
│   └── 02_baseline_experiments.ipynb  # Main notebook (updated)
├── src/
│   ├── ga/
│   │   ├── genetic_algorithm.py  # GA core logic
│   │   ├── operators.py          # Selection, crossover, mutation
│   │   └── fitness.py            # Fitness evaluation
│   └── models/
│       └── classifier.py         # Decision Tree factory
└── requirements.txt
```

## Key Design Decisions

### Why Multiclass (not binary)?
- Dataset has 10 diagnosis classes
- Represents real-world medical diagnosis complexity
- Weighted F1-score handles class imbalance

### Why Decision Tree?
- Interpretable (can visualize splits)
- Fast training (important for GA fitness evaluation)
- Works well with tabular medical data
- No feature scaling required

### Why at least 20 features?
- User requirement for meaningful analysis
- Prevents over-aggressive feature reduction
- Ensures diverse feature representation

### Why Binary Mask Encoding?
- Simple and efficient
- Direct mapping to feature inclusion/exclusion
- Easy to interpret and validate
- Standard approach in GA feature selection

## Troubleshooting

### If GA selects too few features:
- Increase `feature_reduction_weight` in fitness_cfg (e.g., 0.4 or 0.5)
- Decrease `accuracy_weight` correspondingly

### If convergence is slow:
- Increase `population_size`
- Increase `n_generations`
- Try different selection methods ('rank' or 'roulette')

### If accuracy drops significantly:
- Reduce `feature_reduction_weight`
- Increase `accuracy_weight`
- Ensure CV folds ≥ 3

## Next Steps for Analysis

1. **Feature Importance**: Identify which features are consistently selected
2. **Operator Comparison**: Run multiple experiments with different GA operators
3. **Stability Analysis**: Run multiple seeds and check selection consistency
4. **Visualization**: Plot convergence curves, confusion matrices
5. **Report**: Document findings with the CSV results

## References

- GA implementation: `src/ga/genetic_algorithm.py`
- Fitness function: `src/ga/fitness.py`
- Project README: `README.md`
- Requirements: `projectRequirements.md`
