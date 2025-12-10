"""
Utility Functions
Data loading, preprocessing, and visualization helpers.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import HANDLE_MISSING, NORMALIZE, TEST_SIZE, RANDOM_STATE

def load_dataset(filepath):
    """
    Load dataset from CSV file with preprocessing.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        X: Feature matrix (numpy array)
        y: Target labels (numpy array)
        feature_names: List of feature column names
    """
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = X.columns.tolist()
    
    # Handle missing values
    if HANDLE_MISSING == 'mean':
        X = X.fillna(X.mean())
    elif HANDLE_MISSING == 'median':
        X = X.fillna(X.median())
    elif HANDLE_MISSING == 'drop':
        df = df.dropna()
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # Normalize features
    if NORMALIZE:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y, feature_names

def split_data(X, y, test_size=None, random_state=None):
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    test_size = test_size or TEST_SIZE
    random_state = random_state or RANDOM_STATE
    
    return train_test_split(X, y, test_size=test_size, 
                           random_state=random_state, stratify=y)

def plot_convergence(history, title="GA Convergence", save_path=None):
    """
    Plot fitness evolution over generations.
    """
    plt.figure(figsize=(10, 6))
    generations = range(len(history['best_fitness']))
    
    plt.plot(generations, history['best_fitness'], 
             label='Best Fitness', linewidth=2.5, color='tab:blue')
    plt.plot(generations, history['mean_fitness'], 
             label='Average Fitness', linewidth=2, alpha=0.8, color='tab:orange')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_feature_reduction(history, title="Feature Reduction Over Time", save_path=None):
    plt.figure(figsize=(10, 6))
    generations = range(len(history['n_selected_features']))
    
    plt.plot(generations, history['n_selected_features'], 
             linewidth=2.5, color='green')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Number of Selected Features', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_operator_comparison(results_df, metric='accuracy', save_path=None):
    """
    Create box plot comparing different operator configurations.
    
    Args:
        results_df: DataFrame with columns: [selection, crossover, mutation, accuracy, ...]
        metric: Which metric to plot ('accuracy', 'n_features', etc.)
        save_path: If provided, save plot to this path
    """
    plt.figure(figsize=(14, 6))
    
    # Create combined label
    results_df['config'] = (results_df['selection'] + '_' + 
                           results_df['crossover'] + '_' + 
                           results_df['mutation'])
    
    # Box plot
    sns.boxplot(data=results_df, x='config', y=metric)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Configuration (Selection_Crossover_Mutation)', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'{metric.capitalize()} Comparison Across Configurations', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_heatmap(results_df, selection_method, save_path=None):
    """
    Create heatmap showing crossover vs mutation performance for a selection method.
    
    Args:
        results_df: DataFrame with experiment results
        selection_method: Which selection method to filter for
        save_path: If provided, save plot to this path
    """
    # Filter for specific selection method
    filtered = results_df[results_df['selection'] == selection_method]
    
    # Pivot to create matrix
    pivot = filtered.pivot_table(
        values='accuracy', 
        index='mutation', 
        columns='crossover', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy'})
    plt.title(f'Performance Heatmap - {selection_method.capitalize()} Selection', fontsize=14)
    plt.xlabel('Crossover Method', fontsize=12)
    plt.ylabel('Mutation Method', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_feature_frequency(all_best_individuals, feature_names, top_n=20, save_path=None):
    """
    Plot frequency of features being selected across all runs.
    
    Args:
        all_best_individuals: List of binary masks from different runs
        feature_names: List of feature names
        top_n: Show top N most frequently selected features
        save_path: If provided, save plot to this path
    """
    # Count selections
    feature_counts = np.sum(all_best_individuals, axis=0)
    
    # Get top N
    top_indices = np.argsort(feature_counts)[-top_n:][::-1]
    top_counts = feature_counts[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_names)), top_counts, color='steelblue')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Selection Frequency', fontsize=12)
    plt.title(f'Top {top_n} Most Frequently Selected Features', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def save_results_to_csv(results, filepath):
    """
    Save experiment results to CSV.
    
    Args:
        results: List of dictionaries or DataFrame
        filepath: Output CSV path
    """
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results
    
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def load_results_from_csv(filepath):
    """
    Load experiment results from CSV.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with results
    """
    return pd.read_csv(filepath)