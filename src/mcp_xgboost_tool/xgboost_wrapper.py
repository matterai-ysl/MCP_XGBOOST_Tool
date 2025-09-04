"""
XGBoost Core Implementation

This module provides the XGBoostWrapper class that encapsulates 
XGBoost algorithms with enhanced functionality.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import warnings
from .cross_validation import CrossValidationStrategy
from .xgboost_error_handler import XGBoostErrorHandler, xgboost_error_handler, validate_xgboost_data
from .xgboost_data_optimizer import XGBoostDataOptimizer
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


class XGBoostWrapper:
    """
    Wrapper class for XGBoost algorithms with enhanced functionality.
    
    Supports both regression and classification tasks with automatic
    feature importance calculation and model interpretability enhancements.
    """
    
    def __init__(self, 
                 task_type: str = "auto",
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.3,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 colsample_bylevel: float = 1.0,
                 colsample_bynode: float = 1.0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0,
                 min_child_weight: int = 1,
                 tree_method: Optional[str] = None,
                 device: str = "auto",
                 enable_gpu: bool = True,
                 n_jobs: int = -1,
                 random_state: Optional[int] = 42,
                 **kwargs):
        """
        Initialize XGBoostWrapper with GPU support.
        
        Args:
            task_type: "auto", "regression", or "classification"
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            colsample_bylevel: Subsample ratio of columns for each level
            colsample_bynode: Subsample ratio of columns for each node
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            gamma: Minimum loss reduction required to make a further partition
            min_child_weight: Minimum sum of instance weight needed in a child
            tree_method: Tree construction method ("auto", "exact", "approx", "hist", "gpu_hist")
            device: Device to use ("auto", "cpu", "cuda", "gpu")
            enable_gpu: Whether to enable GPU training if available
            n_jobs: Number of parallel threads (-1 means using all processors)
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the XGBoost model
        """
        self.task_type = task_type.lower()
        
        # GPU support configuration
        self.enable_gpu = enable_gpu
        self.device = device.lower()
        self.gpu_available = self._detect_gpu_support()
        self.effective_device = self._determine_effective_device()
        self.tree_method = self._determine_tree_method(tree_method)
        
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'colsample_bynode': colsample_bynode,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'tree_method': self.tree_method,
            'device': self.effective_device,
            'n_jobs': n_jobs if self.effective_device == 'cpu' else 1,  # GPU training uses single thread
            'random_state': random_state,
            **kwargs
        }
        
        # XGBoost specific parameters
        self.model = None
        self.feature_names = None
        self.feature_importances = None
        self.permutation_importances = None
        self.is_fitted = False
        self.actual_task_type = None
        self.evals_result = None  # Store evaluation results
        
        # Initialize data optimizer
        self.data_optimizer = XGBoostDataOptimizer(enable_memory_optimization=True)
        
        logger.info(f"Initialized XGBoostWrapper with task_type={task_type}, n_estimators={n_estimators}")
        logger.info(f"GPU Support: Available={self.gpu_available}, Enabled={self.enable_gpu}, Effective Device={self.effective_device}")
        logger.info(f"Tree Method: {self.tree_method}")
        
    def _detect_gpu_support(self) -> bool:
        """
        Detect if GPU support is available in the current environment.
        
        Returns:
            True if GPU support is available, False otherwise
        """
        try:
            # Method 1: Check if CUDA is available via XGBoost
            import subprocess
            import os
            
            # Check if nvidia-smi is available (indicates NVIDIA driver)
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("NVIDIA GPU detected via nvidia-smi")
                    
                    # Method 2: Try to create a simple XGBoost model with GPU
                    try:
                        # Create a dummy XGBRegressor with GPU settings
                        dummy_model = xgb.XGBRegressor(
                            n_estimators=1, 
                            tree_method='hist',
                            device='cuda',
                            verbosity=0
                        )
                        
                        # Test with minimal data
                        X_test = np.array([[1, 2], [3, 4]])
                        y_test = np.array([1.0, 2.0])
                        dummy_model.fit(X_test, y_test)
                        
                        logger.info("XGBoost GPU training test successful")
                        return True
                        
                    except Exception as e:
                        logger.warning(f"XGBoost GPU test failed: {e}")
                        return False
                        
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("nvidia-smi not found or timed out")
                return False
                
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return False
            
    def _determine_effective_device(self) -> str:
        """
        Determine the effective device to use based on settings and availability.
        
        Returns:
            Effective device string ("cpu", "cuda", or "gpu")
        """
        if not self.enable_gpu:
            return "cpu"
            
        if self.device == "auto":
            return "cuda" if self.gpu_available else "cpu"
        elif self.device in ["cuda", "gpu"]:
            if self.gpu_available:
                return "cuda"
            else:
                logger.warning("GPU requested but not available, falling back to CPU")
                return "cpu"
        else:
            return "cpu"
            
    def _determine_tree_method(self, tree_method: Optional[str]) -> str:
        """
        Determine the optimal tree method based on device and user preference.
        
        Args:
            tree_method: User-specified tree method
            
        Returns:
            Optimal tree method string
        """
        # For XGBoost 2.0+, 'gpu_hist' is deprecated. Use 'hist' and control device with `device` parameter.
        if tree_method is not None and tree_method.lower() == 'gpu_hist':
            logger.info("tree_method 'gpu_hist' is deprecated, automatically using 'hist' instead. GPU usage is controlled by the 'device' parameter.")
            return 'hist'
        
        if tree_method is not None:
            return tree_method.lower()
        
        # For XGBoost >= 2.0, 'hist' is the default and works for both CPU and GPU.
        return "hist"
            
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU information and status.
        
        Returns:
            Dictionary with GPU information
        """
        info = {
            'gpu_available': self.gpu_available,
            'gpu_enabled': self.enable_gpu,
            'effective_device': self.effective_device,
            'tree_method': self.tree_method,
            'using_gpu': self.effective_device in ["cuda", "gpu"]
        }
        
        # Try to get additional GPU details
        if self.gpu_available:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split('\n')[0].split(', ')
                    if len(gpu_info) >= 3:
                        info['gpu_name'] = gpu_info[0]
                        info['gpu_memory_total_mb'] = int(gpu_info[1])
                        info['gpu_memory_used_mb'] = int(gpu_info[2])
                        info['gpu_memory_available_mb'] = int(gpu_info[1]) - int(gpu_info[2])
            except Exception as e:
                logger.debug(f"Could not get detailed GPU info: {e}")
                
        return info
        
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
        Initialize the appropriate XGBoost model based on task type.
        
        Args:
            task_type: "regression" or "classification"
         """

            
        # Set model-specific parameters
        model_params = self.model_params.copy()
        
        if task_type == "regression":
            model_params['objective'] = 'reg:squarederror'
            self.model = xgb.XGBRegressor(**model_params)
        elif task_type == "classification":
            model_params['objective'] = 'binary:logistic'  # Will be auto-adjusted for multiclass
            self.model = xgb.XGBClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
        logger.info(f"Initialized XGBoost {task_type} model")
        
    @xgboost_error_handler("model training")
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_val: Optional[np.ndarray] = None,
            compute_permutation_importance: bool = True,
            verbose: bool = True) -> 'XGBoostWrapper':
        """
        Fit the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional feature names
            X_val: Optional validation feature matrix
            y_val: Optional validation target vector
            compute_permutation_importance: Whether to compute permutation importance
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        try:
            # Validate input data
            validation_result = validate_xgboost_data(X, y, "training")
            if not validation_result['is_valid']:
                logger.warning(f"Data validation issues found: {validation_result['issues']}")
                
            # Prepare optimized data
            optimized_data = self.data_optimizer.prepare_xgboost_data(
                X, y, X_val, y_val, feature_names, optimize_memory=True
            )
            
            # Use optimized data
            X = optimized_data['X']
            y = optimized_data['y']
            X_val = optimized_data['X_val']
            y_val = optimized_data['y_val']
            feature_names = optimized_data['feature_names']
            
            logger.info(f"Data optimization: Memory saved {optimized_data['memory_stats']['saved_mb']:.1f}MB")
                
            # Convert to numpy array if pandas DataFrame
            if isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = X.columns.tolist()
                X = X.values
                
            if X_val is not None and isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
                
            # Store feature names
            if feature_names is not None:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            # Detect task type
            self.actual_task_type = self._detect_task_type(y)
            
            # Initialize appropriate model
            self._initialize_model(self.actual_task_type)
            
            # Prepare evaluation set (simplified - no early stopping)
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X, y), (X_val, y_val)]
            
            # Fit the model
            logger.info(f"Training XGBoost {self.actual_task_type} model with {X.shape[0]} samples and {X.shape[1]} features")
            
            fit_params = {}
            if eval_set is not None:
                fit_params['eval_set'] = eval_set
                if not verbose:
                    fit_params['verbose'] = False
            
            self.model.fit(X, y, **fit_params)
            
            # Store evaluation results
            if hasattr(self.model, 'evals_result_'):
                self.evals_result = self.model.evals_result_
            
            # Calculate feature importances
            self._calculate_feature_importances(X, y, compute_permutation_importance)
            
            self.is_fitted = True
            
            # Log training completion info
            if hasattr(self.model, 'best_iteration'):
                logger.info(f"Model training completed successfully. Best iteration: {self.model.best_iteration}")
            else:
                logger.info("Model training completed successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray, 
                                     compute_permutation_importance: bool = True):
        """
        Calculate multiple types of feature importances available in XGBoost.
        
        Args:
            X: Feature matrix
            y: Target vector
            compute_permutation_importance: Whether to compute permutation importance
        """
        # Initialize feature importances dictionary
        self.feature_importances = {}
        
        # XGBoost native feature importance types
        importance_types = ['weight', 'gain', 'cover']
        
        for imp_type in importance_types:
            try:
                importance_scores = self.model.get_booster().get_score(importance_type=imp_type)
                
                # Convert to consistent format with all features
                importance_dict = {}
                for i, name in enumerate(self.feature_names):
                    # XGBoost uses f0, f1, f2... as default feature names
                    xgb_feature_name = f"f{i}"
                    importance_dict[name] = importance_scores.get(xgb_feature_name, 0.0)
                
                self.feature_importances[imp_type] = importance_dict
                
            except Exception as e:
                logger.warning(f"Failed to compute {imp_type} feature importance: {str(e)}")
                self.feature_importances[imp_type] = {name: 0.0 for name in self.feature_names}
        
        # Also get sklearn-compatible feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances['sklearn_gain'] = {
                name: float(importance) 
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
                
    @xgboost_error_handler("prediction")
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
            
        # Validate input data
        validation_result = validate_xgboost_data(X, operation="prediction")
        if not validation_result['is_valid']:
            logger.warning(f"Prediction data validation issues: {validation_result['issues']}")
            
        # Optimize data for inference
        optimized_inference = self.data_optimizer.optimize_for_inference(self.model, X)
        X = optimized_inference['data']
        use_optimized = optimized_inference['optimized']
        
        if use_optimized:
            logger.info("Using optimized data format for prediction")
            
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
        
    @xgboost_error_handler("probability prediction")
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
        
    def get_feature_importance(self, importance_type: str = "gain", 
                             return_sorted: bool = True) -> Dict[str, float]:
        """
        Get feature importance scores (unified method).
        
        Args:
            importance_type: Type of importance:
                           - "weight": Number of times a feature is used to split
                           - "gain": Average gain across all splits using feature  
                           - "cover": Average coverage across all splits using feature
                           - "sklearn_gain": Sklearn-compatible gain importance
                           - "permutation": Permutation importance (mean)
            return_sorted: Whether to return sorted by importance
                           
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        # Get importance scores
        if importance_type == "permutation":
            if self.permutation_importances is None:
                raise ValueError("Permutation importance not computed")
            importance_dict = {
                name: data['importance_mean'] 
                for name, data in self.permutation_importances.items()
            }
        elif importance_type in self.feature_importances:
            importance_dict = self.feature_importances[importance_type].copy()
        else:
            available_types = list(self.feature_importances.keys())
            if self.permutation_importances is not None:
                available_types.append('permutation')
            raise ValueError(f"importance_type must be one of {available_types}")
        
        # Sort if requested
        if return_sorted:
            importance_dict = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], reverse=True))
            
        return importance_dict
        
    def get_all_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """
        Get all available feature importance types in a comprehensive format.
        
        Returns:
            Dictionary with all importance types for each feature
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        result = {}
        
        # Organize by feature name
        for name in self.feature_names:
            result[name] = {}
            
            # Add all XGBoost importance types
            for imp_type, imp_dict in self.feature_importances.items():
                result[name][imp_type] = imp_dict[name]
            
            # Add permutation importance if available
            if self.permutation_importances is not None:
                perm_data = self.permutation_importances[name]
                result[name]['permutation_mean'] = perm_data['importance_mean']
                result[name]['permutation_std'] = perm_data['importance_std']
                
        return result
        
    def get_top_features(self, n_features: int = 10, 
                        importance_type: str = "gain") -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n_features: Number of top features to return
            importance_type: Type of importance to use for ranking
            
        Returns:
            List of tuples (feature_name, importance_score)
        """
        importance_dict = self.get_feature_importance(importance_type, return_sorted=True)
        return list(importance_dict.items())[:n_features]
        
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
        Get comprehensive model information including GPU status.
        
        Returns:
            Dictionary with model information and GPU details
        """
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'task_type': self.task_type,
                'model_params': self.model_params,
                'gpu_info': self.get_gpu_info()
            }
            
        info = {
            'is_fitted': True,
            'task_type': self.actual_task_type,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'subsample': self.model.subsample,
            'colsample_bytree': self.model.colsample_bytree,
            'gpu_info': self.get_gpu_info()
        }
        
        # Add XGBoost specific info
        if hasattr(self.model, 'best_iteration'):
            info['best_iteration'] = self.model.best_iteration
            info['best_score'] = self.model.best_score
            
        if hasattr(self.model, 'classes_'):
            info['classes'] = self.model.classes_.tolist()
            info['n_classes'] = len(self.model.classes_)
            
        if self.evals_result is not None:
            info['training_history'] = self.evals_result
            
        return info
        
    def get_booster_info(self) -> Dict[str, Any]:
        """
        Get XGBoost booster specific information.
        
        Returns:
            Dictionary with booster information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting booster info")
            
        booster = self.model.get_booster()
        
        info = {
            'num_boosted_rounds': booster.num_boosted_rounds(),
            'num_features': booster.num_features(),
        }
        
        # Get booster attributes
        try:
            info['booster_config'] = booster.save_config()
        except:
            logger.warning("Could not retrieve booster config")
            
        return info
        
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
        Cross-validate the model, allowing explicit task_type passing.
        """
        # Use external task_type if provided
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
            current_params['objective'] = 'reg:squarederror'
            return xgb.XGBRegressor(**current_params)
        elif task_type == "classification":
            current_params['objective'] = 'binary:logistic'
            return xgb.XGBClassifier(**current_params)
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
            params = self.model.get_params()
            # Add XGBoost-specific parameters
            return params
        else:
            # Return the stored parameters
            params = self.model_params.copy()
            return params
    
    def set_params(self, **params) -> 'XGBoostWrapper':
        """
        Set model parameters and update internal model_params.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        # Handle XGBoost-specific parameters
            
        # Update internal model_params
        self.model_params.update(params)
        
        # If model is already fitted, update its parameters too
        if self.is_fitted and self.model is not None:
            self.model.set_params(**params)
            logger.info(f"Updated model parameters: {params}")
        
        return self
        
    def save_model(self, filepath: str, save_format: str = "json") -> None:
        """
        Save the trained XGBoost model.
        
        Args:
            filepath: Path to save the model
            save_format: Format to save ("json", "ubj", or "deprecated")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        # Save using XGBoost's native save method
        if save_format in ["json", "ubj", "deprecated"]:
            self.model.save_model(filepath)
        else:
            # Save using joblib (compatible with sklearn)
            joblib.dump(self.model, filepath)
            
        logger.info(f"Model saved to {filepath} in {save_format} format")
        
    def load_model(self, filepath: str) -> 'XGBoostWrapper':
        """
        Load a trained XGBoost model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        try:
            # Try to load as XGBoost native format first
            if filepath.endswith(('.json', '.ubj', '.model')):
                # Create a dummy model to load into
                dummy_model = xgb.XGBRegressor()  # Will be overwritten
                dummy_model.load_model(filepath)
                self.model = dummy_model
            else:
                # Load using joblib
                self.model = joblib.load(filepath)
                
            self.is_fitted = True
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
        return self 