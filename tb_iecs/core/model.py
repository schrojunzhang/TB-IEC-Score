"""
Machine learning models for TB-IEC-Score
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import Normalizer, StandardScaler
from xgboost import XGBClassifier


def calculate_enrichment_factor(test_y: np.ndarray, pred_prob: np.ndarray, top: float = 0.01) -> float:
    """
    Calculate enrichment factor at a specific percentage of the dataset.
    
    Args:
        test_y: True labels
        pred_prob: Predicted probabilities
        top: Top percentage (0-1) of compounds to consider
        
    Returns:
        Enrichment factor value
    """
    # Create dataframe
    df = pd.DataFrame({"label": test_y, "pred": pred_prob})
    
    # Sort by prediction scores
    df.sort_values(by="pred", inplace=True, ascending=False)
    
    # Calculate statistics
    N_total = len(df)
    N_active = len(df[df["label"] == 1])
    topb_total = int(N_total * top + 0.5)
    
    # Get top n% data
    topb_data = df.iloc[:topb_total, :]
    topb_active = len(topb_data[topb_data["label"] == 1])
    
    # Calculate enrichment factor
    ef = (topb_active / topb_total) / (N_active / N_total)
    return ef


class ModelBase:
    """Base class for all ML models in TB-IEC-Score"""
    
    def __init__(self, over_sampling: bool = False, hyper_opt: bool = True):
        """
        Initialize the model.
        
        Args:
            over_sampling: Whether to use SMOTE oversampling for imbalanced data
            hyper_opt: Whether to optimize hyperparameters
        """
        self.over_sampling = over_sampling
        self.hyper_opt = hyper_opt
        self.scaler = None
        self.threshold = None
        self.normalizer = None
        self.clf = None
        
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the input data.
        
        Args:
            X: Input features
            
        Returns:
            Preprocessed features
        """
        if self.scaler is None:
            self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        
        if self.threshold is None:
            self.threshold = VarianceThreshold().fit(X_scaled)
        X_thresholded = self.threshold.transform(X_scaled)
        
        if self.normalizer is None:
            self.normalizer = Normalizer(norm='l2').fit(X_thresholded)
        X_normalized = self.normalizer.transform(X_thresholded)
        
        return X_normalized
    
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, train_size: float = 0.8) -> Tuple:
        """
        Prepare data for training or testing.
        
        Args:
            X: Input features
            y: Target labels (if None, only X is processed)
            train_size: Fraction of data to use for training
            
        Returns:
            Prepared data as tuple
        """
        if y is not None:
            # Split dataset
            train_x, test_x, train_y, test_y = train_test_split(
                X, y, train_size=train_size, random_state=42, stratify=y, shuffle=True
            )
            
            # Over sampling for imbalanced data
            if self.over_sampling:
                sampler = SMOTE()
                train_x, train_y = sampler.fit_resample(train_x, train_y)
            
            # Preprocess data
            train_x = self.preprocess_data(train_x)
            test_x = self.preprocess_data(test_x)
            
            return train_x, test_x, train_y, test_y
        else:
            # Only preprocess X for prediction
            return self.preprocess_data(X)
    
    def evaluate(self, test_y: np.ndarray, pred: np.ndarray, pred_prob: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_y: True labels
            pred: Predicted labels
            pred_prob: Predicted probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate enrichment factors
        ef_1 = calculate_enrichment_factor(test_y, pred_prob, top=0.01)
        ef_2 = calculate_enrichment_factor(test_y, pred_prob, top=0.02)
        ef_5 = calculate_enrichment_factor(test_y, pred_prob, top=0.05)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()
        roc_auc = roc_auc_score(y_true=test_y, y_score=pred_prob)
        acc = accuracy_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        mcc = matthews_corrcoef(test_y, pred)
        kappa = cohen_kappa_score(test_y, pred)
        
        return {
            "ef_1": ef_1,
            "ef_2": ef_2,
            "ef_5": ef_5,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": acc,
            "f1": f1,
            "mcc": mcc,
            "kappa": kappa,
            "roc_auc": roc_auc
        }


class SVMModel(ModelBase):
    """Support Vector Machine model for TB-IEC-Score"""
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Train SVM model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Evaluation metrics and predicted probabilities
        """
        # Prepare data
        train_x, test_x, train_y, test_y = self.prepare_data(X, y)
        
        # Optimize hyperparameters
        if self.hyper_opt:
            def optimize_model(hyper_parameter):
                clf = svm.SVC(**hyper_parameter, class_weight='balanced', random_state=42)
                score = cross_val_score(
                    clf, train_x, train_y, 
                    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                    scoring='f1', n_jobs=-1
                ).mean()
                return -score

            hyper_parameter = {
                'C': hp.uniform('C', 0.1, 10),
                'gamma': hp.uniform('gamma', 0.001, 1)
            }
            
            best = fmin(
                optimize_model, hyper_parameter, 
                algo=tpe.suggest, max_evals=100,
                rstate=np.random.RandomState(42)
            )
            
            self.clf = svm.SVC(
                C=best['C'], gamma=best['gamma'], 
                class_weight='balanced', random_state=42,
                probability=True
            )
        else:
            self.clf = svm.SVC(class_weight='balanced', random_state=42, probability=True)
        
        # Train model
        self.clf.fit(train_x, train_y)
        
        # Make predictions
        pred = self.clf.predict(test_x)
        pred_proba = self.clf.predict_proba(test_x)[:, 1]
        
        # Evaluate model
        metrics = self.evaluate(test_y, pred, pred_proba)
        
        return metrics, pred_proba
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels and probabilities
        """
        if self.clf is None:
            raise ValueError("Model not trained yet")
        
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        pred = self.clf.predict(X_processed)
        pred_proba = self.clf.predict_proba(X_processed)[:, 1]
        
        return pred, pred_proba


class RandomForestModel(ModelBase):
    """Random Forest model for TB-IEC-Score"""
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Train Random Forest model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Evaluation metrics and predicted probabilities
        """
        # Prepare data
        train_x, test_x, train_y, test_y = self.prepare_data(X, y)
        
        # Optimize hyperparameters
        if self.hyper_opt:
            def optimize_model(hyper_parameter):
                clf = RandomForestClassifier(
                    **hyper_parameter, n_jobs=-1, 
                    random_state=42, class_weight='balanced'
                )
                score = cross_val_score(
                    clf, train_x, train_y, 
                    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                    scoring='f1', n_jobs=-1
                ).mean()
                return -score

            # Define hyperparameter space
            estimators = list(range(100, 301, 10))
            depths = list(range(6, 100))
            features = ['sqrt', 'log2']
            leaves = list(range(3, 10))
            
            hyper_parameter = {
                'n_estimators': hp.choice('n_estimators', estimators),
                'max_depth': hp.choice('max_depth', depths),
                'max_features': hp.choice('max_features', features),
                'min_samples_leaf': hp.choice('min_samples_leaf', leaves)
            }
            
            best = fmin(
                optimize_model, hyper_parameter, 
                algo=tpe.suggest, max_evals=100,
                rstate=np.random.RandomState(42)
            )
            
            self.clf = RandomForestClassifier(
                n_estimators=estimators[best['n_estimators']],
                max_depth=depths[best['max_depth']],
                max_features=features[best['max_features']],
                min_samples_leaf=leaves[best['min_samples_leaf']],
                n_jobs=-1, random_state=42, class_weight='balanced'
            )
        else:
            self.clf = RandomForestClassifier(
                n_jobs=-1, random_state=42, class_weight='balanced'
            )
        
        # Train model
        self.clf.fit(train_x, train_y)
        
        # Make predictions
        pred = self.clf.predict(test_x)
        pred_proba = self.clf.predict_proba(test_x)[:, 1]
        
        # Evaluate model
        metrics = self.evaluate(test_y, pred, pred_proba)
        
        return metrics, pred_proba
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels and probabilities
        """
        if self.clf is None:
            raise ValueError("Model not trained yet")
        
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        pred = self.clf.predict(X_processed)
        pred_proba = self.clf.predict_proba(X_processed)[:, 1]
        
        return pred, pred_proba


class XGBoostModel(ModelBase):
    """XGBoost model for TB-IEC-Score"""
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Train XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Evaluation metrics and predicted probabilities
        """
        # Prepare data
        train_x, test_x, train_y, test_y = self.prepare_data(X, y)
        
        # Optimize hyperparameters
        if self.hyper_opt:
            def optimize_model(hyper_parameter):
                clf = XGBClassifier(**hyper_parameter, n_jobs=-1, random_state=42)
                score = cross_val_score(
                    clf, train_x, train_y, 
                    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                    scoring='f1', n_jobs=-1
                ).mean()
                return -score

            # Define hyperparameter space
            estimators = list(range(100, 301, 10))
            depths = list(range(3, 11))
            
            hyper_parameter = {
                'n_estimators': hp.choice('n_estimators', estimators),
                'max_depth': hp.choice('max_depth', depths),
                'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
                'reg_lambda': hp.uniform('reg_lambda', 0.5, 3)
            }
            
            best = fmin(
                optimize_model, hyper_parameter, 
                algo=tpe.suggest, max_evals=100,
                rstate=np.random.RandomState(42)
            )
            
            self.clf = XGBClassifier(
                n_estimators=estimators[best['n_estimators']],
                max_depth=depths[best['max_depth']],
                learning_rate=best['learning_rate'],
                reg_lambda=best['reg_lambda'],
                n_jobs=-1, random_state=42
            )
        else:
            self.clf = XGBClassifier(n_jobs=-1, random_state=42)
        
        # Train model
        self.clf.fit(train_x, train_y)
        
        # Make predictions
        pred = self.clf.predict(test_x)
        pred_proba = self.clf.predict_proba(test_x)[:, 1]
        
        # Evaluate model
        metrics = self.evaluate(test_y, pred, pred_proba)
        
        return metrics, pred_proba
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels and probabilities
        """
        if self.clf is None:
            raise ValueError("Model not trained yet")
        
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        pred = self.clf.predict(X_processed)
        pred_proba = self.clf.predict_proba(X_processed)[:, 1]
        
        return pred, pred_proba


def get_model(model_type: str = "xgboost", **kwargs) -> ModelBase:
    """
    Factory function to get appropriate model.
    
    Args:
        model_type: Type of model ('xgboost', 'svm', or 'rf')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model instance
    """
    if model_type.lower() == "xgboost":
        return XGBoostModel(**kwargs)
    elif model_type.lower() == "svm":
        return SVMModel(**kwargs)
    elif model_type.lower() == "rf":
        return RandomForestModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 