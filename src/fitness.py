"""
Fitness Evaluation Module for Feature Selection using Genetic Algorithms

This module evaluates the quality of feature subsets by:
1. Training a Decision Tree classifier on selected features
2. Computing cross-validation accuracy
3. Applying a penalty for using too many features

Fitness Formula:
    fitness = accuracy - lambda * (n_selected_features / n_total_features)

Supports two modes:
- Binary mode: Individual is [0, 1, 0, 1, ...] (direct selection)
- Continuous mode: Individual is [0.3, 0.8, 0.1, ...] (threshold-based selection)

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
    """
    Evaluates fitness of feature subsets for Genetic Algorithm optimization.
    
    This class handles:
    - Individual fitness evaluation using Decision Trees
    - Population-level evaluation (with optional parallelization)
    - Fitness caching for improved performance
    - Both binary and continuous-valued individuals
    
    Attributes
    ----------
    X : np.ndarray
        Complete feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target labels of shape (n_samples,)
    n_features : int
        Total number of features in the dataset
    lambda_penalty : float
        Weight for feature reduction penalty
    cv_folds : int
        Number of cross-validation folds
    threshold : float
        Threshold for converting continuous weights to binary (default: 0.5)
    """
    
    def __init__(self, X, y, lambda_penalty=None, cv_folds=None, threshold=None):
        """
        Initialize the FitnessEvaluator.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Complete feature matrix (n_samples, n_features)
        y : np.ndarray or pd.Series
            Target labels (n_samples,)
        lambda_penalty : float, optional
            Penalty weight for feature count (default: from config.LAMBDA_PENALTY)
        cv_folds : int, optional
            Number of CV folds (default: from config.CV_FOLDS)
        threshold : float, optional
            Threshold for continuous->binary conversion (default: 0.5)
        """
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
        """
        Convert continuous-valued individual to binary mask.
        
        Parameters
        ----------
        individual : list or np.ndarray
            Individual with values in [0, 1] range
            
        Returns
        -------
        binary_mask : np.ndarray
            Binary array where 1 = feature selected (weight >= threshold)
            
        Example
        -------
        >>> individual = [0.2, 0.8, 0.5, 0.3]
        >>> threshold = 0.5
        >>> result = [0, 1, 1, 0]  # 0.8 and 0.5 >= 0.5
        """
        individual_array = np.array(individual)
        binary_mask = (individual_array >= self.threshold).astype(int)
        return binary_mask
    
    def evaluate_individual(self, individual):
        """
        Evaluate fitness of a single individual.
        
        Parameters
        ----------
        individual : list or np.ndarray
            Can be either:
            - Binary: [0, 1, 0, 1, ...] where 1 = feature selected
            - Continuous: [0.3, 0.8, 0.1, ...] converted to binary via threshold
            
        Returns
        -------
        fitness : float
            Fitness score (higher is better)
            = accuracy - lambda * (n_selected / n_total)
        
        Notes
        -----
        - Results are cached to avoid redundant evaluations
        - Returns -1.0 if no features are selected
        - Continuous values are converted to binary using threshold
        """
        # Convert to numpy array
        individual_array = np.array(individual)
        
        # Convert to binary mask (handles both binary and continuous inputs)
        mask = self._convert_to_binary(individual_array)
        
        # Convert to tuple for caching (lists aren't hashable)
        mask_tuple = tuple(mask)
        
        # Check cache first
        if mask_tuple in self._fitness_cache:
            return self._fitness_cache[mask_tuple]
        
        # Count selected features
        n_selected = np.sum(mask)
        
        # Edge case: no features selected
        if n_selected == 0:
            fitness = -1.0
            self._fitness_cache[mask_tuple] = fitness
            return fitness
        
        # Select features using the binary mask
        X_selected = self.X[:, mask == 1]
        
        # Compute cross-validation accuracy using DecisionTreeWrapper
        try:
            accuracy = self._cross_validate(X_selected, self.y)
        except Exception as e:
            print(f"Warning: Cross-validation failed for individual: {e}")
            fitness = -1.0
            self._fitness_cache[mask_tuple] = fitness
            return fitness
        
        # Calculate feature penalty
        feature_penalty = self.lambda_penalty * (n_selected / self.n_features)
        
        # Compute final fitness
        fitness = accuracy - feature_penalty
        
        # Cache the result
        self._fitness_cache[mask_tuple] = fitness
        
        return fitness
    
    def evaluate_population(self, population):
        """
        Evaluate fitness for an entire population.
        
        Uses parallel processing if enabled in config and population is large enough.
        
        Parameters
        ----------
        population : list of lists/arrays
            List of individuals (each can be binary or continuous)
            
        Returns
        -------
        fitness_scores : list of float
            Fitness values for each individual in the population
        """
        # Use parallel processing for large populations (>10 individuals)
        if USE_PARALLEL and len(population) > 10:
            fitness_scores = Parallel(n_jobs=N_JOBS)(
                delayed(self.evaluate_individual)(ind) 
                for ind in population
            )
        else:
            # Sequential evaluation for small populations
            fitness_scores = [
                self.evaluate_individual(ind) 
                for ind in population
            ]
        
        return fitness_scores
    
    def _cross_validate(self, X_selected, y):
        """
        Perform cross-validation using DecisionTreeWrapper.
        
        Parameters
        ----------
        X_selected : np.ndarray
            Feature matrix with only selected features (n_samples, n_selected)
        y : np.ndarray
            Target labels (n_samples,)
            
        Returns
        -------
        mean_accuracy : float
            Average accuracy across all CV folds
        """
        # Create Decision Tree using the wrapper
        dt_wrapper = DecisionTreeWrapper()
        
        # Perform cross-validation
        mean_accuracy = dt_wrapper.cross_validate(X_selected, y, cv=self.cv_folds)
        
        return mean_accuracy
    
    def get_selected_features(self, individual):
        """
        Get indices of features selected by an individual.
        
        Parameters
        ----------
        individual : list or np.ndarray
            Individual (binary or continuous)
            
        Returns
        -------
        selected_indices : list of int
            Indices where mask value is 1 (after threshold conversion)
            
        Example
        -------
        >>> individual = [0.2, 0.8, 0.5, 0.3]  # continuous
        >>> evaluator.get_selected_features(individual)
        [1, 2]  # Features with weights >= 0.5
        """
        mask = self._convert_to_binary(individual)
        selected_indices = np.where(mask == 1)[0].tolist()
        return selected_indices
    
    def get_feature_count(self, individual):
        """
        Get number of features selected by an individual.
        
        Parameters
        ----------
        individual : list or np.ndarray
            Individual (binary or continuous)
            
        Returns
        -------
        count : int
            Number of selected features (sum of 1s in binary mask)
        """
        mask = self._convert_to_binary(individual)
        return int(np.sum(mask))
    
    def get_binary_mask(self, individual):
        """
        Get the binary mask representation of an individual.
        
        Useful for debugging or visualization.
        
        Parameters
        ----------
        individual : list or np.ndarray
            Individual (binary or continuous)
            
        Returns
        -------
        binary_mask : np.ndarray
            Binary array [0, 1, 0, 1, ...]
            
        Example
        -------
        >>> individual = [0.2, 0.8, 0.5, 0.3]
        >>> evaluator.get_binary_mask(individual)
        array([0, 1, 1, 0])
        """
        return self._convert_to_binary(individual)
    
    def clear_cache(self):
        """
        Clear the fitness cache.
        
        Useful when starting a new experiment or run to free memory.
        """
        self._fitness_cache.clear()
        
    def get_cache_size(self):
        """
        Get current size of fitness cache.
        
        Returns
        -------
        cache_size : int
            Number of cached fitness evaluations
        """
        return len(self._fitness_cache)
    
    def get_statistics(self):
        """
        Get statistics about the evaluator state.
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - n_features: Total number of features
            - lambda_penalty: Current penalty weight
            - cv_folds: Number of CV folds
            - threshold: Feature selection threshold
            - cache_size: Number of cached evaluations
        """
        return {
            'n_features': self.n_features,
            'lambda_penalty': self.lambda_penalty,
            'cv_folds': self.cv_folds,
            'threshold': self.threshold,
            'cache_size': self.get_cache_size()
        }