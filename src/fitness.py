"""
Fitness Evaluation Module for Feature Selection using Genetic Algorithms

This module evaluates the quality of feature subsets by:
1. Training a Decision Tree classifier on selected features
2. Computing cross-validation accuracy
3. Applying a penalty for using too many features

Fitness Formula:
    fitness = accuracy - lambda * (n_selected_features / n_total_features)

Supports two modes:
- Continuous mode: Individual is [0.3, 0.8, 0.1, ...] (threshold-based selection)
- Binary mode: Individual is [0, 1, 0, 1, ...] (direct selection)

Features:
- Uses DecisionTreeWrapper for consistent model configuration
- Cross-validation for robust accuracy estimation
- Fitness caching to avoid redundant evaluations
- Optional parallel processing for large populations
- Configurable threshold for continuous-valued individuals
"""

import numpy as np
from joblib import Parallel, delayed

from .decision_tree import DecisionTreeWrapper
from .config import (
    CV_FOLDS, 
    LAMBDA_PENALTY, 
    USE_PARALLEL,
    N_JOBS,
    WEIGHT_THRESHOLD
)


class FitnessEvaluator:
    
    def __init__(self, X, y, lambda_penalty=None, cv_folds=None, threshold=None):
        # Convert to numpy arrays if needed
        self.X = np.array(X) if not isinstance(X, np.ndarray) else X
        self.y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        self.n_features = self.X.shape[1]
        self.lambda_penalty = lambda_penalty if lambda_penalty is not None else LAMBDA_PENALTY
        self.cv_folds = cv_folds if cv_folds is not None else CV_FOLDS
        self.threshold = threshold if threshold is not None else WEIGHT_THRESHOLD
        
        # Cache for fitness scores (avoids re-evaluating identical individuals)
        self._fitness_cache = {}
        
    def _convert_to_binary(self, individual):
        individual_array = np.array(individual)
        binary_mask = (individual_array >= self.threshold).astype(int)
        return binary_mask
    
    def evaluate_individual(self, individual):
        individual_array = np.array(individual)
        mask = self._convert_to_binary(individual_array)
        mask_tuple = tuple(mask)
        if mask_tuple in self._fitness_cache:
            return self._fitness_cache[mask_tuple]
        
        n_selected = np.sum(mask)
        if n_selected == 0:
            fitness = -1.0
            self._fitness_cache[mask_tuple] = fitness
            return fitness
        X_selected = self.X[:, mask == 1]
        
        try:
            accuracy = self._cross_validate(X_selected, self.y)
        except Exception as e:
            print(f"Warning: Cross-validation failed for individual: {e}")
            fitness = -1.0
            self._fitness_cache[mask_tuple] = fitness
            return fitness
        
        feature_penalty = self.lambda_penalty * (n_selected / self.n_features)
        fitness = accuracy - feature_penalty
        
        self._fitness_cache[mask_tuple] = fitness
        
        return fitness
    
    def evaluate_population(self, population):
        if USE_PARALLEL and len(population) > 10:
            fitness_scores = Parallel(n_jobs=N_JOBS)(
                delayed(self.evaluate_individual)(ind) 
                for ind in population
            )
        else:
            fitness_scores = [
                self.evaluate_individual(ind) 
                for ind in population
            ]
        
        return fitness_scores
    
    def _cross_validate(self, X_selected, y):
        dt_wrapper = DecisionTreeWrapper()
        mean_accuracy = dt_wrapper.cross_validate(X_selected, y, cv=self.cv_folds)
        
        return mean_accuracy
    
    def get_selected_features(self, individual):
        mask = self._convert_to_binary(individual)
        selected_indices = np.where(mask == 1)[0].tolist()
        return selected_indices
    
    def get_feature_count(self, individual):
        mask = self._convert_to_binary(individual)
        return int(np.sum(mask))
    
    def get_binary_mask(self, individual):
        return self._convert_to_binary(individual)
    
    def clear_cache(self):
        self._fitness_cache.clear()
        
    def get_cache_size(self):
        return len(self._fitness_cache)
    
    def get_statistics(self):
        return {
            'n_features': self.n_features,
            'lambda_penalty': self.lambda_penalty,
            'cv_folds': self.cv_folds,
            'threshold': self.threshold,
            'cache_size': self.get_cache_size()
        }