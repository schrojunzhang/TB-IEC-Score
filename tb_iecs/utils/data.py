"""
Data handling utilities
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Dataset loader for TB-IEC-Score"""
    
    def __init__(self, descriptors_path: str, label_csv: Optional[str] = None):
        """
        Initialize dataset loader.
        
        Args:
            descriptors_path: Path to descriptors CSV file
            label_csv: Path to CSV file with labels (optional)
        """
        self.descriptors_path = descriptors_path
        self.label_csv = label_csv
        self.data = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load descriptor data and optionally merge with labels.
        
        Returns:
            DataFrame with descriptors and labels
        """
        # Load descriptors
        df = pd.read_csv(self.descriptors_path)
        
        # Merge with labels if provided
        if self.label_csv:
            labels = pd.read_csv(self.label_csv)
            df = pd.merge(df, labels, on='name', how='inner')
        
        return df
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded data.
        
        Returns:
            DataFrame with descriptors and labels
        """
        return self.data
    
    def get_features_and_labels(
        self, 
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Get features and labels from data.
        
        Args:
            exclude_cols: List of columns to exclude from features
            
        Returns:
            Tuple of (features, labels) if labels are available, otherwise (features, None)
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Always exclude 'name' column from features
        exclude_cols.append('name')
        
        # Create features DataFrame
        X = self.data.loc[:, [col for col in self.data.columns if col not in exclude_cols]]
        
        # Get labels if available
        y = self.data.loc[:, 'label'] if 'label' in self.data.columns else None
        
        return X, y
    
    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split by label
            exclude_cols: List of columns to exclude from features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if 'label' not in self.data.columns:
            raise ValueError("Cannot split data without labels")
        
        # Get features and labels
        X, y = self.get_features_and_labels(exclude_cols)
        
        # Stratify by label if requested
        stratify_param = y if stratify else None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test


def combine_descriptors(
    descriptor_files: List[str],
    output_file: Optional[str] = None,
    merge_on: str = 'name'
) -> pd.DataFrame:
    """
    Combine multiple descriptor files into a single dataset.
    
    Args:
        descriptor_files: List of descriptor CSV files
        output_file: Path to save combined data (optional)
        merge_on: Column to merge on
        
    Returns:
        Combined DataFrame
    """
    if not descriptor_files:
        raise ValueError("No descriptor files provided")
    
    # Load first file
    combined_df = pd.read_csv(descriptor_files[0])
    
    # Merge with remaining files
    for file in descriptor_files[1:]:
        df = pd.read_csv(file)
        combined_df = pd.merge(combined_df, df, on=merge_on, how='inner')
    
    # Save combined file if requested
    if output_file:
        combined_df.to_csv(output_file, index=False)
    
    return combined_df


def normalize_data(data: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize data to mean=0, std=1.
    
    Args:
        data: Input DataFrame
        exclude_cols: Columns to exclude from normalization
        
    Returns:
        Normalized DataFrame
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Create copy of data
    normalized = data.copy()
    
    # Normalize each column except excluded ones
    for col in normalized.columns:
        if col not in exclude_cols:
            mean = normalized[col].mean()
            std = normalized[col].std()
            if std > 0:  # Avoid division by zero
                normalized[col] = (normalized[col] - mean) / std
    
    return normalized


def handle_missing_values(
    data: pd.DataFrame,
    strategy: str = 'mean',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in data.
    
    Args:
        data: Input DataFrame
        strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'value')
        fill_value: Value to use if strategy is 'value'
        
    Returns:
        DataFrame with missing values handled
    """
    # Create copy of data
    cleaned = data.copy()
    
    # Handle missing values according to strategy
    for col in cleaned.columns:
        if cleaned[col].dtype.kind in 'fiu':  # Check if column is numeric
            if strategy == 'mean':
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
            elif strategy == 'median':
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            elif strategy == 'mode':
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
            elif strategy == 'value' and fill_value is not None:
                cleaned[col] = cleaned[col].fillna(fill_value)
            else:
                raise ValueError(f"Invalid strategy '{strategy}' or fill_value not provided")
    
    return cleaned 