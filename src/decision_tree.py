"""
Decision Tree Classifier Wrapper
Provides a clean interface for training and evaluating decision trees.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from .config import DT_MAX_DEPTH, DT_MIN_SAMPLES_SPLIT, DT_MIN_SAMPLES_LEAF, RANDOM_STATE

class DecisionTreeWrapper:
    
    def __init__(self, max_depth=None, min_samples_split=None, min_samples_leaf=None):
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
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)
    
    def cross_validate(self, X, y, cv=5):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return np.mean(scores)

def get_decision_tree():
    return DecisionTreeWrapper() 