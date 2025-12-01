"""
Fitness evaluation for GA feature selection.
Combines classification accuracy with feature reduction using cross-validation.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


def evaluate_individual(mask, X, y, classifier_factory, fitness_cfg, eval_cfg):
    """
    Evaluate a single individual (feature subset) using cross-validation.
    
    Combines classification accuracy with feature reduction in a weighted fitness.
    Higher fitness is better.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape (n_features,) where 1 means feature is selected.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    classifier_factory : callable
        Function that returns a new classifier instance (e.g., lambda: DecisionTreeClassifier()).
    fitness_cfg : dict
        Fitness configuration with keys:
            - accuracy_weight: float
            - feature_reduction_weight: float
            - penalty_threshold: float (fraction above which to penalize)
    eval_cfg : dict
        Evaluation configuration with keys:
            - cv_folds: int
            - random_state: int
            - metrics: list of str (e.g., ['accuracy', 'f1_score'])
            
    Returns
    -------
    fitness : float
        Scalar fitness value (higher is better).
    """
    n_features = X.shape[1]
    selected_features = np.sum(mask)
    
    # Ensure at least one feature is selected
    if selected_features == 0:
        return -1e9  # Very poor fitness
    
    # Apply mask to select features
    X_selected = X[:, mask == 1]
    
    # Extract config values
    cv_folds = eval_cfg.get("cv_folds", 5)
    random_state = eval_cfg.get("random_state", 42)
    accuracy_weight = fitness_cfg.get("accuracy_weight", 0.7)
    feature_reduction_weight = fitness_cfg.get("feature_reduction_weight", 0.3)
    penalty_threshold = fitness_cfg.get("penalty_threshold", 0.95)
    
    # Perform k-fold cross-validation
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    accuracies = []
    f1_scores = []
    
    for train_idx, val_idx in kfold.split(X_selected, y):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create fresh classifier for this fold
        clf = classifier_factory()
        
        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        # Use weighted average for multiclass, binary for binary classification
        n_classes = len(np.unique(y))
        if n_classes == 2:
            f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
        else:
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        accuracies.append(acc)
        f1_scores.append(f1)
    
    # Aggregate metrics across folds
    mean_accuracy = np.mean(accuracies)
    
    # Compute feature fraction
    feature_fraction = selected_features / n_features
    
    # Multi-objective fitness: combine accuracy and feature reduction
    # Higher accuracy is better, lower feature_fraction is better
    fitness = (
        accuracy_weight * mean_accuracy +
        feature_reduction_weight * (1 - feature_fraction)
    )
    
    # Apply penalty if using too many features
    if feature_fraction > penalty_threshold:
        penalty = 0.1 * (feature_fraction - penalty_threshold)
        fitness -= penalty
    
    return fitness


def evaluate_population(population, X, y, classifier_factory, fitness_cfg, eval_cfg):
    """
    Evaluate all individuals in a population.
    
    Parameters
    ----------
    population : np.ndarray
        Population matrix of shape (pop_size, n_features).
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    classifier_factory : callable
        Function that returns a new classifier instance.
    fitness_cfg : dict
        Fitness configuration.
    eval_cfg : dict
        Evaluation configuration.
        
    Returns
    -------
    fitnesses : np.ndarray
        Array of fitness values of shape (pop_size,).
    """
    fitnesses = np.array([
        evaluate_individual(individual, X, y, classifier_factory, fitness_cfg, eval_cfg)
        for individual in population
    ])
    return fitnesses
