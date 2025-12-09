"""
Fitness evaluation functions for feature selection.

Fitness is computed as:
    fitness = accuracy - lambda * (n_selected_features / n_total_features)

Where accuracy is obtained via cross-validation on a decision tree classifier.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from . import config


def evaluate_fitness(
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = config.CV_FOLDS,
    lambda_penalty: float = config.LAMBDA_PENALTY
) -> float:
    """
    Evaluate fitness for a single binary feature mask.
    
    Fitness is computed as cross-validation accuracy minus a penalty
    for the number of selected features.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape (n_features,) where 1 indicates selected feature.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    cv_folds : int, optional
        Number of cross-validation folds (default: from config).
    lambda_penalty : float, optional
        Penalty weight for feature count (default: from config).
        
    Returns
    -------
    fitness : float
        Fitness score = mean_accuracy - lambda * (n_selected / n_total).
        Higher is better.
        
    Notes
    -----
    - If no features are selected (all zeros in mask), returns a very low fitness.
    - Decision tree hyperparameters are taken from config.py.
    """
    n_selected = np.sum(mask)
    n_total = len(mask)
    
    # If no features selected, return very low fitness
    if n_selected == 0:
        return -1.0
    
    # Select features using the mask
    X_selected = X[:, mask == 1]
    
    # Create decision tree classifier
    dt = DecisionTreeClassifier(
        max_depth=config.DT_MAX_DEPTH,
        min_samples_split=config.DT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.DT_MIN_SAMPLES_LEAF,
        random_state=config.DT_RANDOM_STATE
    )
    
    # Perform cross-validation
    try:
        cv_scores = cross_val_score(dt, X_selected, y, cv=cv_folds, scoring='accuracy')
        mean_accuracy = np.mean(cv_scores)
    except Exception as e:
        # If cross-validation fails, return low fitness
        print(f"Warning: Cross-validation failed: {e}")
        return -1.0
    
    # Compute fitness with feature penalty
    feature_penalty = lambda_penalty * (n_selected / n_total)
    fitness = mean_accuracy - feature_penalty
    
    return fitness
