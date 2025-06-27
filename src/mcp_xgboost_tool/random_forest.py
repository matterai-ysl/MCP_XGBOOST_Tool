"""
Random Forest Core Implementation

This module provides the RandomForestWrapper class that encapsulates 
scikit-learn's random forest algorithms with enhanced functionality.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import warnings
from .cross_validation import CrossValidationStrategy
import asyncio
import json
import os
import uuid
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)



class RandomForestWrapper:
    """
    Wrapper class for Random Forest algorithms with enhanced functionality.
    
    Supports both regression and classification tasks with automatic
    feature importance calculation and model interpretability enhancements.
    """
    
    def __init__(self, 
                 task_type: str = "auto",
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = "sqrt",
                 bootstrap: bool = True,
                 n_jobs: int = -1,
                 random_state: Optional[int] = 42,
                 **kwargs):
        """
        Initialize RandomForestWrapper.
        
        Args:
            task_type: "auto", "regression", or "classification"
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at each leaf node
            max_features: Number of features to consider when looking for the best split
            bootstrap: Whether bootstrap samples are used when building trees
            n_jobs: Number of jobs to run in parallel (-1 means using all processors)
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the random forest model
        """
        self.task_type = task_type.lower()
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'n_jobs': n_jobs,
            'random_state': random_state,
            **kwargs
        }
        
        self.model = None
        self.feature_names = None
        self.feature_importances = None
        self.permutation_importances = None
        self.is_fitted = False
        self.actual_task_type = None
        
        logger.info(f"Initialized RandomForestWrapper with task_type={task_type}, n_estimators={n_estimators}")
        
    def _detect_task_type(self, y: np.ndarray) -> str:
        """
        Automatically detect task type based on target variable.
        
        Args:
            y: Target variable
            
        Returns:
            Detected task type: "regression" or "classification"
        """
        if self.task_type != "auto":
            return self.task_type
            
        # Check if target is continuous or discrete
        unique_values = np.unique(y)
        
        # Check if target is string/categorical (classification)
        if y.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object (typically string)
            detected_type = "classification"
        # Check if target has very few unique values and they are integers (classification)
        elif len(unique_values) <= 10:
            try:
                # Try to convert to int to check if they are integer values
                if np.all(unique_values == unique_values.astype(int)):
                    detected_type = "classification"
                else:
                    detected_type = "regression"
            except (ValueError, TypeError):
                # If conversion fails, treat as classification (categorical)
                detected_type = "classification"
        else:
            detected_type = "regression"
            
        logger.info(f"Auto-detected task type: {detected_type} (unique values: {len(unique_values)}, dtype: {y.dtype})")
        return detected_type
        
    def _initialize_model(self, task_type: str):
        """
        Initialize the appropriate sklearn model based on task type.
        
        Args:
            task_type: "regression" or "classification"
        """
        if task_type == "regression":
            self.model = RandomForestRegressor(**self.model_params)
        elif task_type == "classification":
            # For classification, adjust max_features default if needed
            if self.model_params.get('max_features') == 'sqrt':
                # sqrt is good for classification
                pass
            self.model = RandomForestClassifier(**self.model_params)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
        logger.info(f"Initialized {task_type} model with {self.model_params['n_estimators']} estimators")
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            compute_permutation_importance: bool = True) -> 'RandomForestWrapper':
        """
        Fit the random forest model.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional feature names
            compute_permutation_importance: Whether to compute permutation importance
            
        Returns:
            Self for method chaining
        """
        try:
            # Convert to numpy array if pandas DataFrame
            if isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = X.columns.tolist()
                X = X.values
                
            # Store feature names
            if feature_names is not None:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            # Detect task type
            self.actual_task_type = self._detect_task_type(y)
            
            # Initialize appropriate model
            self._initialize_model(self.actual_task_type)
            
            # Fit the model
            logger.info(f"Training {self.actual_task_type} model with {X.shape[0]} samples and {X.shape[1]} features")
            self.model.fit(X, y)
            
            # Calculate feature importances
            self._calculate_feature_importances(X, y, compute_permutation_importance)
            
            self.is_fitted = True
            logger.info("Model training completed successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray, 
                                     compute_permutation_importance: bool = True):
        """
        Calculate both tree-based and permutation feature importances.
        
        Args:
            X: Feature matrix
            y: Target vector
            compute_permutation_importance: Whether to compute permutation importance
        """
        # Tree-based feature importance (fast)
        self.feature_importances = {
            name: importance 
            for name, importance in zip(self.feature_names, self.model.feature_importances_)
        }
        
        # Permutation importance (more reliable but slower)
        if compute_permutation_importance:
            try:
                logger.info("Computing permutation importance...")
                perm_importance = permutation_importance(
                    self.model, X, y, 
                    n_repeats=5, 
                    random_state=self.model_params.get('random_state', 42),
                    n_jobs=1  # Use single job to avoid nested parallelism
                )
                
                self.permutation_importances = {
                    name: {
                        'importance_mean': float(mean),
                        'importance_std': float(std)
                    }
                    for name, mean, std in zip(
                        self.feature_names, 
                        perm_importance.importances_mean,
                        perm_importance.importances_std
                    )
                }
                logger.info("Permutation importance calculation completed")
                
            except Exception as e:
                logger.warning(f"Failed to compute permutation importance: {str(e)}")
                self.permutation_importances = None
                
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Prediction array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (only for classification).
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Class probability array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if self.actual_task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
            
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict_proba(X)
        
    def get_feature_importance(self, importance_type: str = "tree") -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: "tree" for tree-based importance, 
                           "permutation" for permutation importance
                           
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        if importance_type == "tree":
            return self.feature_importances.copy()
        elif importance_type == "permutation":
            if self.permutation_importances is None:
                raise ValueError("Permutation importance not computed")
            return {
                name: data['importance_mean'] 
                for name, data in self.permutation_importances.items()
            }
        else:
            raise ValueError("importance_type must be 'tree' or 'permutation'")
            
    def get_feature_importance_detailed(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed feature importance with both tree-based and permutation scores.
        
        Returns:
            Dictionary with detailed importance information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        detailed_importance = {}
        
        for name in self.feature_names:
            detailed_importance[name] = {
                'tree_importance': self.feature_importances[name]
            }
            
            if self.permutation_importances is not None:
                perm_data = self.permutation_importances[name]
                detailed_importance[name].update({
                    'permutation_importance_mean': perm_data['importance_mean'],
                    'permutation_importance_std': perm_data['importance_std']
                })
                
        return detailed_importance
        
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on given data.
        
        Args:
            X: Feature matrix
            y_true: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        y_pred = self.predict(X)
        
        if self.actual_task_type == "regression":
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
            }
        else:  # classification
            # Handle different averaging strategies for multiclass
            n_classes = len(np.unique(y_true))
            average = 'binary' if n_classes == 2 else 'weighted'
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = {
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
                    'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0))
                }
                
        return metrics
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'task_type': self.task_type,
                'model_params': self.model_params
            }
            
        info = {
            'is_fitted': True,
            'task_type': self.actual_task_type,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'max_features': self.model.max_features
        }
        
        if hasattr(self.model, 'classes_'):
            info['classes'] = self.model.classes_.tolist()
            info['n_classes'] = len(self.model.classes_)
            
        return info
        
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
            'permutation_importances': self.permutation_importances,
            'actual_task_type': self.actual_task_type,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importances = model_data['feature_importances']
        self.permutation_importances = model_data['permutation_importances']
        self.actual_task_type = model_data['actual_task_type']
        self.model_params = model_data['model_params']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        
    def get_tree_count(self) -> int:
        """Get the number of trees in the forest."""
        if not self.is_fitted:
            return 0
        return len(self.model.estimators_)
        
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if bootstrap=True.
        
        Returns:
            OOB score or None if bootstrap=False
        """
        if not self.is_fitted:
            return None
            
        if hasattr(self.model, 'oob_score_'):
            return float(self.model.oob_score_)
        else:
            return None
            
    def cross_validate(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: np.ndarray,
                      cv_folds: int = 5,
                      metrics: Optional[List[str]] = None,
                      stratify: bool = True,
                      return_train_score: bool = True,
                      random_state: Optional[int] = None,
                      save_data: bool = False,
                      output_dir: str = "cv_results",
                      data_name: str = "cv_data",
                      preprocessor: Optional[Any] = None,
                      feature_names: Optional[List[str]] = None,
                      original_X: Optional[pd.DataFrame] = None,
                      original_y: Optional[np.ndarray] = None,
                      original_feature_names: Optional[List[str]] = None,
                      original_target_name: Optional[str] = None,
                      task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Cross-validate the model, allowing explicit task_type传递。
        """
        # 优先用外部传入的task_type
        if task_type is not None:
            use_task_type = task_type
        else:
            use_task_type = self.actual_task_type if self.is_fitted else self._detect_task_type(y)
        # Create a fresh model instance with current parameters for CV
        fresh_model = self._create_fresh_model(use_task_type)
        # Create cross-validation strategy
        cv_strategy = CrossValidationStrategy(
            cv_folds=cv_folds,
            random_state=random_state,
            stratify=stratify
        )
        # Perform cross-validation
        results = cv_strategy.cross_validate_model(
            estimator=fresh_model,
            X=X,
            y=y,
            task_type=use_task_type,
            metrics=metrics,
            return_train_score=return_train_score,
            save_data=save_data,
            output_dir=output_dir,
            data_name=data_name,
            preprocessor=preprocessor,
            feature_names=feature_names,
            original_X=original_X,
            original_y=original_y,
            original_feature_names=original_feature_names,
            original_target_name=original_target_name
        )
        return results
    
    def simple_cross_validate(self,
                            X: Union[np.ndarray, pd.DataFrame],
                            y: np.ndarray,
                            cv_folds: int = 5,
                            metric: str = 'default',
                            random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Simple cross-validation with a single metric.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            metric: Metric to evaluate ('default' uses MAE for regression, accuracy for classification)
            random_state: Random state for CV splitter
            
        Returns:
            Dictionary with cross-validation scores
        """
        # Use model's random state if not specified
        if random_state is None:
            random_state = self.model_params.get('random_state', 42)
            
        # Detect task type if model is not fitted
        task_type = self.actual_task_type if self.is_fitted else self._detect_task_type(y)
        
        # Create a fresh model instance
        fresh_model = self._create_fresh_model(task_type)
        
        # Create cross-validation strategy
        cv_strategy = CrossValidationStrategy(
            cv_folds=cv_folds,
            random_state=random_state
        )
        
        # Perform simple cross-validation
        results = cv_strategy.simple_cross_val_score(
            estimator=fresh_model,
            X=X,
            y=y,
            metric=metric,
            task_type=task_type
        )
        
        return results
    
    def _create_fresh_model(self, task_type: str):
        """
        Create a fresh model instance with current parameters.
        
        Args:
            task_type: "regression" or "classification"
            
        Returns:
            Fresh model instance
        """
        # Get the most up-to-date parameters
        current_params = self.get_params()
        
        # Remove task_type if present (it's not a sklearn parameter)
        current_params = {k: v for k, v in current_params.items() if k != 'task_type'}
        
        if task_type == "regression":
            return RandomForestRegressor(**current_params)
        elif task_type == "classification":
            return RandomForestClassifier(**current_params)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def get_supported_cv_metrics(self, task_type: Optional[str] = None) -> List[str]:
        """
        Get list of supported cross-validation metrics.
        
        Args:
            task_type: Optional task type. Uses fitted model's task type if None.
            
        Returns:
            List of supported metric names
        """
        if task_type is None:
            if not self.is_fitted:
                raise ValueError("Must specify task_type or fit model first")
            task_type = self.actual_task_type
            
        cv_strategy = CrossValidationStrategy()
        return cv_strategy.get_supported_metrics(task_type)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of current model parameters
        """
        if self.is_fitted and self.model is not None:
            # Return the actual fitted model's parameters
            return self.model.get_params()
        else:
            # Return the stored parameters
            return self.model_params.copy()
    
    def set_params(self, **params) -> 'RandomForestWrapper':
        """
        Set model parameters and update internal model_params.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        # Update internal model_params
        self.model_params.update(params)
        
        # If model is already fitted, update its parameters too
        if self.is_fitted and self.model is not None:
            self.model.set_params(**params)
            logger.info(f"Updated model parameters: {params}")
        
        return self

class EnhancedRandomForestTool:
    def __init__(self, base_dir: str = "trained_models"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def _detect_feature_types(self, df: pd.DataFrame, target_column: str) -> Dict[str, List[str]]:
        """智能检测特征类型"""
        feature_types = {
            'numerical': [],
            'categorical': [],
            'binary': [],
            'datetime': [],
            'text': []
        }
        
        features = [col for col in df.columns if col != target_column]
        
        for col in features:
            col_data = df[col]
            
            if col_data.dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(col_data):
                feature_types['datetime'].append(col)
                continue
            
            if pd.api.types.is_numeric_dtype(col_data):
                unique_vals = col_data.dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                    feature_types['binary'].append(col)
                else:
                    feature_types['numerical'].append(col)
            else:
                unique_count = col_data.nunique()
                total_count = len(col_data.dropna())
                
                if unique_count / total_count < 0.05 or unique_count < 20:
                    if unique_count == 2:
                        feature_types['binary'].append(col)
                    else:
                        feature_types['categorical'].append(col)
                else:
                    feature_types['text'].append(col)
        
        return feature_types
    
    def _create_preprocessor(self, df: pd.DataFrame, target_column: str, feature_types: Dict[str, List[str]]) -> Tuple[ColumnTransformer, Dict[str, Any]]:
        """创建数据预处理器"""
        preprocessing_steps = []
        preprocessing_config = {
            'feature_types': feature_types,
            'preprocessing_steps': []
        }
        
        if feature_types['numerical']:
            preprocessing_steps.append(
                ('numerical', StandardScaler(), feature_types['numerical'])
            )
            preprocessing_config['preprocessing_steps'].append({
                'type': 'StandardScaler',
                'features': feature_types['numerical']
            })
        
        if feature_types['categorical']:
            preprocessing_steps.append(
                ('categorical', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 feature_types['categorical'])
            )
            preprocessing_config['preprocessing_steps'].append({
                'type': 'OneHotEncoder',
                'features': feature_types['categorical']
            })
        
        if feature_types['binary']:
            preprocessing_steps.append(
                ('binary', 'passthrough', feature_types['binary'])
            )
            preprocessing_config['preprocessing_steps'].append({
                'type': 'passthrough',
                'features': feature_types['binary']
            })
        
        if preprocessing_steps:
            preprocessor = ColumnTransformer(
                transformers=preprocessing_steps,
                remainder='passthrough',
                sparse_threshold=0
            )
        else:
            preprocessor = None
        
        return preprocessor, preprocessing_config
    
    def _preprocess_target(self, y: pd.Series, task_type: str) -> Tuple[np.ndarray, Optional[LabelEncoder], Dict[str, Any]]:
        """预处理目标变量"""
        target_info = {
            'original_type': str(y.dtype),
            'preprocessing_applied': None,
            'unique_values': None
        }
        
        if task_type == 'classification':
            if not pd.api.types.is_numeric_dtype(y) or y.dtype == 'object':
                label_encoder = LabelEncoder()
                y_processed = label_encoder.fit_transform(y)
                target_info['preprocessing_applied'] = 'LabelEncoder'
                target_info['unique_values'] = label_encoder.classes_.tolist()
            else:
                label_encoder = None
                y_processed = y.values
                target_info['unique_values'] = sorted(y.unique().tolist())
        else:
            label_encoder = None
            y_processed = y.values
            target_info['statistics'] = {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        
        return y_processed, label_encoder, target_info
    
    def _perform_cross_validation_with_predictions(self, model, X_processed: np.ndarray, y_processed: np.ndarray, 
                                                 task_type: str, cv_folds: int = 5) -> Dict[str, Any]:
        """执行交叉验证并保存每个fold的预测结果"""
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'r2'
        
        from sklearn.model_selection import cross_validate
        cv_results = cross_validate(model, X_processed, y_processed, cv=cv, 
                                  scoring=scoring, return_train_score=True)
        
        # 获取每个fold的详细预测
        cv_predictions = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_processed, y_processed)):
            X_train, X_test = X_processed[train_idx], X_processed[test_idx]
            y_train, y_test = y_processed[train_idx], y_processed[test_idx]
            
            # 训练模型
            fold_model = type(model)(**model.get_params())
            fold_model.fit(X_train, y_train)
            
            # 预测
            y_pred = fold_model.predict(X_test)
            
            # 保存fold结果
            fold_data = {
                'fold': fold_idx + 1,
                'test_indices': test_idx.tolist(),
                'features': X_test.tolist(),
                'true_values': y_test.tolist(),
                'predicted_values': y_pred.tolist()
            }
            
            if task_type == 'regression':
                fold_data['metrics'] = {
                    'r2_score': float(r2_score(y_test, y_pred)),
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'mae': float(mean_absolute_error(y_test, y_pred))
                }
            else:
                from sklearn.metrics import accuracy_score
                fold_data['metrics'] = {
                    'accuracy': float(accuracy_score(y_test, y_pred))
                }
            
            cv_predictions.append(fold_data)
        
        return {
            'cv_scores': cv_results['test_score'].tolist(),
            'cv_mean': float(cv_results['test_score'].mean()),
            'cv_std': float(cv_results['test_score'].std()),
            'train_scores': cv_results['train_score'].tolist(),
            'fold_predictions': cv_predictions
        }
    
    def _inverse_transform_features(self, X_processed: np.ndarray, preprocessor: ColumnTransformer, 
                                  original_feature_names: List[str]) -> pd.DataFrame:
        """将预处理后的特征还原为原始格式"""
        try:
            # 尝试逆变换数值特征
            restored_features = {}
            feature_idx = 0
            
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'numerical' and hasattr(transformer, 'inverse_transform'):
                    # 获取数值特征的列数
                    n_features = len(columns)
                    features_subset = X_processed[:, feature_idx:feature_idx + n_features]
                    
                    # 逆变换
                    restored_subset = transformer.inverse_transform(features_subset)
                    
                    # 添加到结果
                    for i, col_name in enumerate(columns):
                        restored_features[col_name] = restored_subset[:, i]
                    
                    feature_idx += n_features
                    
                elif name == 'categorical' and hasattr(transformer, 'inverse_transform'):
                    # 对于OneHot编码，获取特征数量
                    if hasattr(transformer, 'get_feature_names_out'):
                        encoded_feature_names = transformer.get_feature_names_out(columns)
                        n_features = len(encoded_feature_names)
                    else:
                        n_features = len(columns) * len(transformer.categories_[0]) - len(columns)  # drop='first'
                    
                    features_subset = X_processed[:, feature_idx:feature_idx + n_features]
                    
                    # 尝试逆变换OneHot编码
                    try:
                        restored_subset = transformer.inverse_transform(features_subset)
                        for i, col_name in enumerate(columns):
                            restored_features[col_name] = restored_subset[:, i]
                    except:
                        # 如果无法逆变换，使用占位符
                        for col_name in columns:
                            restored_features[col_name] = ['categorical_feature'] * X_processed.shape[0]
                    
                    feature_idx += n_features
                    
                elif name == 'binary' or isinstance(columns, list):
                    # 二进制特征或passthrough特征
                    n_features = len(columns) if isinstance(columns, list) else 1
                    features_subset = X_processed[:, feature_idx:feature_idx + n_features]
                    
                    if isinstance(columns, list):
                        for i, col_name in enumerate(columns):
                            restored_features[col_name] = features_subset[:, i]
                    else:
                        restored_features[columns] = features_subset[:, 0]
                    
                    feature_idx += n_features
            
            # 如果还有剩余特征（remainder='passthrough'）
            if feature_idx < X_processed.shape[1]:
                remaining_features = X_processed[:, feature_idx:]
                remaining_names = [f'remainder_{i}' for i in range(remaining_features.shape[1])]
                
                for i, name in enumerate(remaining_names):
                    restored_features[name] = remaining_features[:, i]
            
            return pd.DataFrame(restored_features)
            
        except Exception as e:
            print(f"警告: 特征还原失败: {e}")
            # 返回处理后的特征作为备选
            feature_names = [f'processed_feature_{i}' for i in range(X_processed.shape[1])]
            return pd.DataFrame(X_processed, columns=feature_names)
    
    def _save_datasets(self, model_dir: str, original_df: pd.DataFrame, processed_features: np.ndarray, 
                      processed_target: np.ndarray, feature_names: List[str], target_column: str,
                      preprocessor: Optional[ColumnTransformer], cv_results: Dict[str, Any]):
        """保存多版本数据集"""
        datasets_dir = os.path.join(model_dir, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)
        
        # 保存原始数据集
        original_df.to_csv(os.path.join(datasets_dir, 'original_dataset.csv'), index=False)
        
        # 保存预处理后的数据集
        processed_df = pd.DataFrame(processed_features, columns=feature_names)
        processed_df[target_column] = processed_target
        processed_df.to_csv(os.path.join(datasets_dir, 'processed_dataset.csv'), index=False)
        
        # 保存特征映射
        feature_mapping_df = pd.DataFrame({
            'processed_feature_name': feature_names,
            'processed_feature_index': range(len(feature_names))
        })
        feature_mapping_df.to_csv(os.path.join(datasets_dir, 'feature_mapping.csv'), index=False)
        
        # 保存交叉验证详细结果
        cv_dir = os.path.join(datasets_dir, 'cross_validation')
        os.makedirs(cv_dir, exist_ok=True)
        
        for fold_data in cv_results['fold_predictions']:
            fold_num = fold_data['fold']
            
            # 创建fold数据DataFrame
            fold_df = pd.DataFrame({
                'sample_index': fold_data['test_indices'],
                'true_value': fold_data['true_values'],
                'predicted_value': fold_data['predicted_values']
            })
            
            # 添加特征列
            features_df = pd.DataFrame(fold_data['features'], columns=feature_names)
            fold_df = pd.concat([fold_df, features_df], axis=1)
            
            # 保存fold数据
            fold_df.to_csv(os.path.join(cv_dir, f'fold_{fold_num}_predictions.csv'), index=False)
        
        # 保存CV汇总
        cv_summary = {
            'cv_scores': cv_results['cv_scores'],
            'cv_mean': cv_results['cv_mean'],
            'cv_std': cv_results['cv_std'],
            'train_scores': cv_results['train_scores']
        }
        with open(os.path.join(cv_dir, 'cv_summary.json'), 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        # 尝试保存特征还原后的数据集
        if preprocessor:
            try:
                original_feature_names = original_df.drop(columns=[target_column]).columns.tolist()
                restored_df = self._inverse_transform_features(processed_features, preprocessor, original_feature_names)
                restored_df[target_column] = processed_target
                restored_df.to_csv(os.path.join(datasets_dir, 'restored_dataset.csv'), index=False)
                print(f"✅ 特征还原数据集已保存: restored_dataset.csv")
            except Exception as e:
                print(f"⚠️ 特征还原失败: {e}")
    
    def _create_plots(self, model_dir: str, original_df: pd.DataFrame, processed_features: np.ndarray, 
                     feature_names: List[str], target_column: str, preprocessor: Optional[ColumnTransformer] = None):
        """创建可视化图表"""
        plots_dir = os.path.join(model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 原始特征分布图
        original_features = original_df.drop(columns=[target_column])
        
        if len(original_features.columns) > 0:
            plt.figure(figsize=(15, 10))
            original_features.hist(bins=30, figsize=(15, 10))
            plt.suptitle('Original Features Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'original_features_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 处理后特征分布图
        if processed_features.shape[1] > 0:
            plt.figure(figsize=(15, 10))
            processed_df = pd.DataFrame(processed_features, columns=feature_names)
            processed_df.hist(bins=30, figsize=(15, 10))
            plt.suptitle('Processed Features Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'processed_features_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 特征相关性热图
        if len(original_features.columns) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = original_features.select_dtypes(include=[np.number]).corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Original Features Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'original_features_correlation.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 还原后特征分布图
        if preprocessor:
            try:
                original_feature_names = original_df.drop(columns=[target_column]).columns.tolist()
                restored_df = self._inverse_transform_features(processed_features, preprocessor, original_feature_names)
                
                plt.figure(figsize=(15, 10))
                restored_df.hist(bins=30, figsize=(15, 10))
                plt.suptitle('Restored Features Distribution')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'restored_features_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 还原特征与原始特征的对比
                if len(restored_df.columns) > 1 and len(original_features.columns) > 1:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                    
                    # 原始特征箱线图
                    original_features.select_dtypes(include=[np.number]).boxplot(ax=ax1)
                    ax1.set_title('Original Features Box Plot')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # 还原特征箱线图
                    restored_df.select_dtypes(include=[np.number]).boxplot(ax=ax2)
                    ax2.set_title('Restored Features Box Plot')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'original_vs_restored_comparison.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                print(f"✅ 还原后可视化图表已生成")
                
            except Exception as e:
                print(f"⚠️ 还原后可视化生成失败: {e}")

    def predict_enhanced_model(self, model_uuid: str, new_data: pd.DataFrame, 
                              output_path: Optional[str] = None) -> Dict[str, Any]:
        """使用训练好的模型进行预测，包含完整的预处理和结果保存"""
        
        model_dir = os.path.join(self.base_dir, model_uuid)
        if not os.path.exists(model_dir):
            raise ValueError(f"模型目录不存在: {model_dir}")
        
        # 加载元数据
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # 加载模型和预处理器
        model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        
        preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
        preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
        
        target_encoder_path = os.path.join(model_dir, 'target_encoder.joblib')
        target_encoder = joblib.load(target_encoder_path) if os.path.exists(target_encoder_path) else None
        
        # 预处理新数据
        if preprocessor:
            X_processed = preprocessor.transform(new_data)
        else:
            X_processed = new_data.values
        
        # 进行预测
        predictions = model.predict(X_processed)
        
        # 如果是分类问题且有标签编码器，进行逆变换
        if target_encoder and metadata['task_type'] == 'classification':
            predictions_decoded = target_encoder.inverse_transform(predictions.astype(int))
        else:
            predictions_decoded = predictions
        
        # 创建预测结果DataFrame
        prediction_results = pd.DataFrame({
            'prediction': predictions_decoded,
            'prediction_raw': predictions
        })
        
        # 添加原始特征
        for i, col in enumerate(new_data.columns):
            prediction_results[f'input_{col}'] = new_data[col].values
        
        # 添加预处理后的特征
        processed_feature_names = metadata['feature_engineering']['feature_names_processed']
        for i, col in enumerate(processed_feature_names):
            prediction_results[f'processed_{col}'] = X_processed[:, i]
        
        # 尝试特征还原
        if preprocessor:
            try:
                original_feature_names = metadata['feature_engineering']['feature_names_original']
                restored_df = self._inverse_transform_features(X_processed, preprocessor, original_feature_names)
                
                for col in restored_df.columns:
                    prediction_results[f'restored_{col}'] = restored_df[col].values
                    
            except Exception as e:
                print(f"警告: 特征还原失败: {e}")
        
        # 保存预测结果
        if output_path:
            prediction_results.to_csv(output_path, index=False)
            print(f"✅ 预测结果已保存到: {output_path}")
        
        # 创建预测目录和文件
        predictions_dir = os.path.join(model_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_file = os.path.join(predictions_dir, f'predictions_{timestamp}.csv')
        prediction_results.to_csv(prediction_file, index=False)
        
        # 创建预测摘要
        summary = {
            'prediction_timestamp': datetime.now().isoformat(),
            'model_uuid': model_uuid,
            'samples_predicted': len(new_data),
            'input_features': new_data.columns.tolist(),
            'task_type': metadata['task_type'],
            'predictions_file': prediction_file
        }
        
        # 添加预测统计
        if metadata['task_type'] == 'regression':
            summary['prediction_statistics'] = {
                'mean': float(predictions.mean()),
                'std': float(predictions.std()),
                'min': float(predictions.min()),
                'max': float(predictions.max())
            }
        else:
            unique_preds, counts = np.unique(predictions_decoded, return_counts=True)
            summary['prediction_distribution'] = {
                str(pred): int(count) for pred, count in zip(unique_preds, counts)
            }
        
        with open(os.path.join(predictions_dir, f'prediction_summary_{timestamp}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return {
            'predictions': predictions_decoded.tolist(),
            'predictions_raw': predictions.tolist(),
            'prediction_results': prediction_results,
            'summary': summary,
            'predictions_file': prediction_file
        }


