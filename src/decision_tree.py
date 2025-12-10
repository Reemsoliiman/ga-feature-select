"""
Decision Tree Classifier Wrapper
Provides a clean interface for training and evaluating decision trees.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from .config import DT_MAX_DEPTH, DT_MIN_SAMPLES_SPLIT, DT_MIN_SAMPLES_LEAF, RANDOM_STATE

class DecisionTreeWrapper:
    """
    Wrapper for sklearn DecisionTreeClassifier with project-specific defaults.
    """
    
    def __init__(self, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        """
        Initialize decision tree with configurable parameters.
        
        Args:
            max_depth: Maximum tree depth (defaults to config)
            min_samples_split: Minimum samples to split (defaults to config)
            min_samples_leaf: Minimum samples in leaf (defaults to config)
        """
        self.max_depth = max_depth or DT_MAX_DEPTH
        self.min_samples_split = min_samples_split or DT_MIN_SAMPLES_SPLIT
        self.min_samples_leaf = min_samples_leaf or DT_MIN_SAMPLES_LEAF
        
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=RANDOM_STATE
        )
    
    def train(self, X_train, y_train):
        """
        Train the decision tree on training data.
        
        Args:
            X_train: Training features (numpy array or pandas DataFrame)
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model accuracy on test data.
        
        Args:
            X_test: Test features
            y_test: True test labels
            
        Returns:
            Accuracy score (float between 0 and 1)
        """
        return self.model.score(X_test, y_test)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform k-fold cross-validation.
        
        Args:
            X: All features
            y: All labels
            cv: Number of folds (default from config)
            
        Returns:
            Mean accuracy across all folds
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return np.mean(scores)

def get_decision_tree():
    """
    Factory function to create a configured decision tree.
    
    Returns:
        DecisionTreeWrapper instance with default config
    """
    return DecisionTreeWrapper() 