"""
Classifier module for GA feature selection.
Provides decision tree wrapper and factory functions.
"""

from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier


def get_decision_tree(max_depth=None, min_samples_split=2, random_state=42):
    """
    Factory function to create a Decision Tree classifier.
    
    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    classifier : DecisionTreeClassifier
        Configured scikit-learn Decision Tree classifier.
    """
    return SklearnDecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )


# Re-export for README compatibility: from src.models import DecisionTreeClassifier
DecisionTreeClassifier = SklearnDecisionTreeClassifier
