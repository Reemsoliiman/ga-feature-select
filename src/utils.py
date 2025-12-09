"""
Utility functions for data loading, preprocessing, and visualization.
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, List


def load_data(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load dataset from CSV file and split into features (X) and target (y).
    
    The last column is assumed to be the target variable.
    All other columns are treated as features.
    
    Parameters
    ----------
    dataset_path : str
        Absolute or relative path to the CSV file.
        
    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    feature_names : list of str
        List of feature column names.
        
    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the CSV has fewer than 2 columns.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    # Load CSV
    df = pd.read_csv(dataset_path)
    
    if df.shape[1] < 2:
        raise ValueError(f"Dataset must have at least 2 columns (features + target). Found {df.shape[1]} columns.")
    
    # Split features and target
    # Last column is target, all others are features
    feature_columns = df.columns[:-1].tolist()
    target_column = df.columns[-1]
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    return X, y, feature_columns


def weights_to_mask(weights: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert continuous feature weights to binary mask.
    
    Parameters
    ----------
    weights : np.ndarray
        Array of feature weights in [0, 1].
    threshold : float, optional
        Threshold for selecting features (default: 0.5).
        Features with weight >= threshold are selected.
        
    Returns
    -------
    mask : np.ndarray
        Binary mask where 1 indicates selected feature.
    """
    return (weights >= threshold).astype(int)


def get_selected_features(weights: np.ndarray, 
                         feature_names: List[str], 
                         threshold: float = 0.5) -> List[str]:
    """
    Get names of selected features based on weights and threshold.
    
    Parameters
    ----------
    weights : np.ndarray
        Array of feature weights in [0, 1].
    feature_names : list of str
        List of feature names.
    threshold : float, optional
        Threshold for selecting features (default: 0.5).
        
    Returns
    -------
    selected : list of str
        Names of features with weight >= threshold.
    """
    mask = weights_to_mask(weights, threshold)
    return [feature_names[i] for i in range(len(feature_names)) if mask[i] == 1]
