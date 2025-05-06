"""
Feature importance analysis utilities
"""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


def calculate_tree_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate tree-based feature importance scores.
    
    Args:
        X: Feature matrix
        y: Target labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with feature importance scores
    """
    # Train a random forest model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def calculate_mutual_information(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate mutual information between features and target.
    
    Args:
        X: Feature matrix
        y: Target labels
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with mutual information scores
    """
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    
    # Create DataFrame with feature names and MI scores
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mutual_info': mi_scores
    })
    
    # Sort by MI score
    mi_df = mi_df.sort_values('mutual_info', ascending=False)
    
    return mi_df


def calculate_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation importance scores.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target labels
        n_repeats: Number of times to permute each feature
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with permutation importance scores
    """
    from sklearn.inspection import permutation_importance

    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    
    # Create DataFrame with feature names and importances
    perm_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    })
    
    # Sort by importance
    perm_df = perm_df.sort_values('importance_mean', ascending=False)
    
    return perm_df


def analyze_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Optional[Any] = None,
    output_dir: Optional[str] = None,
    top_n: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    Analyze feature importance using multiple methods and generate plots.
    
    Args:
        X: Feature matrix
        y: Target labels
        model: Trained model (optional, for permutation importance)
        output_dir: Directory to save plots (optional)
        top_n: Number of top features to display in plots
        
    Returns:
        Dictionary with DataFrames of importance scores
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate tree-based importance
    tree_imp = calculate_tree_importance(X, y)
    
    # Calculate mutual information
    mi_imp = calculate_mutual_information(X, y)
    
    # Calculate permutation importance if model provided
    perm_imp = None
    if model:
        perm_imp = calculate_permutation_importance(model, X, y)
    
    # Create importance plot for tree-based importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=tree_imp.head(top_n))
    plt.title(f'Top {top_n} Features (Tree-based Importance)')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'tree_importance.png'), dpi=300)
        plt.close()
    
    # Create MI plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='mutual_info', y='feature', data=mi_imp.head(top_n))
    plt.title(f'Top {top_n} Features (Mutual Information)')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'mutual_info.png'), dpi=300)
        plt.close()
    
    # Create permutation importance plot if available
    if perm_imp is not None:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance_mean', y='feature', data=perm_imp.head(top_n))
        plt.title(f'Top {top_n} Features (Permutation Importance)')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'permutation_importance.png'), dpi=300)
            plt.close()
    
    # Return all importance scores
    result = {
        'tree_importance': tree_imp,
        'mutual_info': mi_imp
    }
    if perm_imp is not None:
        result['permutation_importance'] = perm_imp
    
    return result


def group_feature_importance(
    importance_df: pd.DataFrame,
    group_prefixes: List[str],
    importance_col: str = 'importance'
) -> pd.DataFrame:
    """
    Group feature importance by feature type/prefix.
    
    Args:
        importance_df: DataFrame with feature importances
        group_prefixes: List of prefixes to group by
        importance_col: Name of column containing importance values
        
    Returns:
        DataFrame with grouped importance scores
    """
    # Create dictionary to store grouped importance
    grouped_imp = {prefix: 0.0 for prefix in group_prefixes}
    grouped_imp['other'] = 0.0
    
    # Sum importance for each group
    for _, row in importance_df.iterrows():
        feature = row['feature']
        importance = row[importance_col]
        
        # Check if feature belongs to any group
        matched = False
        for prefix in group_prefixes:
            if feature.startswith(prefix):
                grouped_imp[prefix] += importance
                matched = True
                break
        
        # If no match, add to "other"
        if not matched:
            grouped_imp['other'] += importance
    
    # Create DataFrame from grouped importance
    grouped_df = pd.DataFrame({
        'feature_group': list(grouped_imp.keys()),
        'importance': list(grouped_imp.values())
    })
    
    # Sort by importance
    grouped_df = grouped_df.sort_values('importance', ascending=False)
    
    return grouped_df


def plot_grouped_importance(
    grouped_df: pd.DataFrame,
    title: str = 'Grouped Feature Importance',
    output_path: Optional[str] = None
) -> None:
    """
    Plot grouped feature importance.
    
    Args:
        grouped_df: DataFrame with grouped importance scores
        title: Plot title
        output_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Create pie chart
    plt.pie(
        grouped_df['importance'],
        labels=grouped_df['feature_group'],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True
    )
    plt.axis('equal')
    plt.title(title)
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show() 