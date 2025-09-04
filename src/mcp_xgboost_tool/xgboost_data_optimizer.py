"""
XGBoost Data Flow Optimizer

This module provides optimized data handling specifically for XGBoost operations,
including DMatrix conversion, memory optimization, and efficient data preprocessing.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import gc
import psutil
import os

logger = logging.getLogger(__name__)

class XGBoostDataOptimizer:
    """
    Optimize data flow for XGBoost operations.
    Provides memory-efficient data handling and format conversion.
    """
    
    def __init__(self, enable_memory_optimization: bool = True):
        self.enable_memory_optimization = enable_memory_optimization
        self.memory_threshold_mb = 1000  # Switch to optimized handling above 1GB
        self._original_dtypes = {}
        
    def optimize_dataframe_dtypes(self, df: pd.DataFrame, 
                                 preserve_categorical: bool = True) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency.
        
        Args:
            df: Input DataFrame
            preserve_categorical: Whether to preserve categorical columns
            
        Returns:
            DataFrame with optimized dtypes
        """
        if not self.enable_memory_optimization:
            return df.copy()
            
        logger.info("Optimizing DataFrame data types for XGBoost...")
        
        df_optimized = df.copy()
        memory_before = df_optimized.memory_usage(deep=True).sum() / 1024**2
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            # Store original dtype
            self._original_dtypes[col] = col_type
            
            # Skip if already optimal or categorical
            if preserve_categorical and col_type.name == 'category':
                continue
                
            # Optimize numeric columns
            if pd.api.types.is_numeric_dtype(col_type):
                # Integer columns
                if pd.api.types.is_integer_dtype(col_type):
                    c_min = df_optimized[col].min()
                    c_max = df_optimized[col].max()
                    
                    if c_min >= 0:  # Unsigned integers
                        if c_max <= np.iinfo(np.uint8).max:
                            df_optimized[col] = df_optimized[col].astype(np.uint8)
                        elif c_max <= np.iinfo(np.uint16).max:
                            df_optimized[col] = df_optimized[col].astype(np.uint16)
                        elif c_max <= np.iinfo(np.uint32).max:
                            df_optimized[col] = df_optimized[col].astype(np.uint32)
                    else:  # Signed integers
                        if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                            df_optimized[col] = df_optimized[col].astype(np.int8)
                        elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                            df_optimized[col] = df_optimized[col].astype(np.int16)
                        elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                            df_optimized[col] = df_optimized[col].astype(np.int32)
                
                # Float columns
                elif pd.api.types.is_float_dtype(col_type):
                    # Use float32 if precision allows
                    if col_type == np.float64:
                        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            
            # Convert object columns to category if beneficial
            elif col_type == 'object':
                num_unique_values = len(df_optimized[col].unique())
                num_total_values = len(df_optimized[col])
                
                # Convert to category if cardinality is low
                if num_unique_values / num_total_values < 0.5:
                    df_optimized[col] = df_optimized[col].astype('category')
        
        memory_after = df_optimized.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = ((memory_before - memory_after) / memory_before) * 100
        
        logger.info(f"Memory optimization complete: {memory_before:.1f}MB -> {memory_after:.1f}MB "
                   f"({memory_reduction:.1f}% reduction)")
        
        return df_optimized
    
    def create_dmatrix(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Optional[np.ndarray] = None,
                      feature_names: Optional[List[str]] = None,
                      enable_categorical: bool = True,
                      missing_value: Optional[float] = None) -> xgb.DMatrix:
        """
        Create optimized XGBoost DMatrix from input data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            feature_names: Feature names
            enable_categorical: Enable categorical feature support
            missing_value: Value to treat as missing
            
        Returns:
            XGBoost DMatrix object
        """
        logger.info("Creating optimized XGBoost DMatrix...")
        
        # Handle pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # Extract categorical features for XGBoost
            categorical_features = []
            if enable_categorical:
                categorical_features = [
                    i for i, dtype in enumerate(X.dtypes) 
                    if dtype.name == 'category' or dtype == 'object'
                ]
            
            # Get feature names
            if feature_names is None:
                feature_names = X.columns.tolist()
            
            # Convert to numpy for DMatrix
            X_values = X.values
        else:
            X_values = X
            categorical_features = []
        
        # Create DMatrix with optimizations
        dmatrix = xgb.DMatrix(
            data=X_values,
            label=y,
            feature_names=feature_names,
            missing=missing_value,
            enable_categorical=enable_categorical
        )
        
        # Set categorical features if any
        if categorical_features:
            dmatrix.set_categorical_features(categorical_features)
            logger.info(f"Set {len(categorical_features)} categorical features")
        
        logger.info(f"DMatrix created with shape: {dmatrix.num_row()} x {dmatrix.num_col()}")
        
        return dmatrix
    
    def prepare_xgboost_data(self, X: Union[np.ndarray, pd.DataFrame],
                           y: np.ndarray,
                           X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                           y_val: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None,
                           optimize_memory: bool = True) -> Dict[str, Any]:
        """
        Prepare data optimally for XGBoost training.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
            optimize_memory: Whether to apply memory optimizations
            
        Returns:
            Dictionary with optimized data components
        """
        logger.info("Preparing optimized data for XGBoost training...")
        
        # Memory usage before optimization
        memory_before = self._get_memory_usage()
        
        # Optimize DataFrame if applicable
        if isinstance(X, pd.DataFrame) and optimize_memory:
            X = self.optimize_dataframe_dtypes(X)
            if X_val is not None and isinstance(X_val, pd.DataFrame):
                X_val = self.optimize_dataframe_dtypes(X_val)
        
        # Check if we should use DMatrix (for large datasets)
        use_dmatrix = self._should_use_dmatrix(X, y)
        
        result = {
            'use_dmatrix': use_dmatrix,
            'memory_optimized': optimize_memory,
            'X': X,
            'y': y,
            'X_val': X_val,
            'y_val': y_val,
            'feature_names': feature_names
        }
        
        if use_dmatrix:
            # Create training DMatrix
            dtrain = self.create_dmatrix(X, y, feature_names)
            result['dtrain'] = dtrain
            
            # Create validation DMatrix if provided
            if X_val is not None and y_val is not None:
                dval = self.create_dmatrix(X_val, y_val, feature_names)
                result['dval'] = dval
                result['eval_set'] = [dtrain, dval]
            else:
                result['eval_set'] = [dtrain]
        
        # Memory usage after optimization
        memory_after = self._get_memory_usage()
        memory_saved = memory_before - memory_after
        
        if memory_saved > 0:
            logger.info(f"Memory optimization saved {memory_saved:.1f}MB")
        
        result['memory_stats'] = {
            'before_mb': memory_before,
            'after_mb': memory_after,
            'saved_mb': memory_saved
        }
        
        return result
    
    def _should_use_dmatrix(self, X: Union[np.ndarray, pd.DataFrame], 
                          y: np.ndarray) -> bool:
        """
        Determine if DMatrix should be used based on data size and characteristics.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Boolean indicating whether to use DMatrix
        """
        # Calculate approximate memory usage
        if isinstance(X, pd.DataFrame):
            data_size_mb = X.memory_usage(deep=True).sum() / 1024**2
        else:
            data_size_mb = X.nbytes / 1024**2
        
        # Use DMatrix for large datasets or when memory is limited
        system_memory_gb = psutil.virtual_memory().total / 1024**3
        memory_usage_ratio = data_size_mb / 1024 / system_memory_gb
        
        use_dmatrix = (
            data_size_mb > self.memory_threshold_mb or  # Large dataset
            memory_usage_ratio > 0.1 or                # High memory usage ratio
            self.enable_memory_optimization             # Memory optimization enabled
        )
        
        logger.info(f"Data size: {data_size_mb:.1f}MB, Memory ratio: {memory_usage_ratio:.3f}, "
                   f"Use DMatrix: {use_dmatrix}")
        
        return use_dmatrix
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2
    
    def optimize_for_inference(self, model: Any, 
                             X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Optimize data for inference/prediction.
        
        Args:
            model: Trained XGBoost model
            X: Input features
            
        Returns:
            Optimized data for prediction
        """
        logger.info("Optimizing data for XGBoost inference...")
        
        # For prediction, we usually don't need DMatrix unless very large
        if isinstance(X, pd.DataFrame):
            # Check if we need to optimize memory
            data_size_mb = X.memory_usage(deep=True).sum() / 1024**2
            
            if data_size_mb > self.memory_threshold_mb:
                X = self.optimize_dataframe_dtypes(X)
                
                # Create DMatrix for very large predictions
                if data_size_mb > self.memory_threshold_mb * 2:
                    feature_names = X.columns.tolist()
                    dmatrix = self.create_dmatrix(X, feature_names=feature_names)
                    return {
                        'data': dmatrix,
                        'use_dmatrix': True,
                        'optimized': True
                    }
        
        return {
            'data': X,
            'use_dmatrix': False,
            'optimized': True
        }
    
    def batch_predict(self, model: Any, X: Union[np.ndarray, pd.DataFrame],
                     batch_size: int = 10000) -> np.ndarray:
        """
        Perform batch prediction for large datasets.
        
        Args:
            model: Trained XGBoost model
            X: Input features
            batch_size: Size of each batch
            
        Returns:
            Prediction array
        """
        logger.info(f"Performing batch prediction with batch size: {batch_size}")
        
        n_samples = len(X)
        predictions = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            if isinstance(X, pd.DataFrame):
                X_batch = X.iloc[start_idx:end_idx]
            else:
                X_batch = X[start_idx:end_idx]
            
            # Optimize batch data
            optimized_batch = self.optimize_for_inference(model, X_batch)
            
            if optimized_batch['use_dmatrix']:
                batch_pred = model.predict(optimized_batch['data'])
            else:
                batch_pred = model.predict(optimized_batch['data'])
            
            predictions.append(batch_pred)
            
            # Log progress for large datasets
            if n_samples > 50000:
                progress = (end_idx / n_samples) * 100
                logger.info(f"Batch prediction progress: {progress:.1f}%")
        
        # Combine predictions
        result = np.concatenate(predictions)
        logger.info(f"Batch prediction completed for {n_samples} samples")
        
        return result
    
    def cleanup_memory(self):
        """Force garbage collection and memory cleanup."""
        if self.enable_memory_optimization:
            logger.info("Performing memory cleanup...")
            gc.collect()
            
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        memory_info = psutil.virtual_memory()
        
        return {
            'memory_optimization_enabled': self.enable_memory_optimization,
            'memory_threshold_mb': self.memory_threshold_mb,
            'current_memory_usage_mb': self._get_memory_usage(),
            'system_memory_gb': memory_info.total / 1024**3,
            'memory_usage_percent': memory_info.percent,
            'optimized_columns': len(self._original_dtypes)
        } 