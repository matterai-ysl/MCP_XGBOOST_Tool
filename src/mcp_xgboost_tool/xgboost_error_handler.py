"""
XGBoost Error Handler

This module provides specialized error handling for XGBoost operations,
including common error patterns, diagnostics, and recovery strategies.
"""

import logging
from typing import Dict, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
import traceback
from functools import wraps

logger = logging.getLogger(__name__)

class XGBoostError(Exception):
    """Base exception for XGBoost-related errors."""
    pass

class XGBoostDataError(XGBoostError):
    """Raised when there are data-related issues in XGBoost operations."""
    pass

class XGBoostTrainingError(XGBoostError):
    """Raised when training fails."""
    pass

class XGBoostPredictionError(XGBoostError):
    """Raised when prediction fails."""
    pass

class XGBoostConfigurationError(XGBoostError):
    """Raised when configuration/parameters are invalid."""
    pass

class XGBoostErrorHandler:
    """
    Centralized error handling for XGBoost operations.
    Provides error diagnosis, logging, and recovery suggestions.
    """
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common XGBoost error patterns and their solutions."""
        return {
            # Data-related errors
            "feature_mismatch": {
                "keywords": ["feature", "dimension", "shape", "mismatch"],
                "error_type": XGBoostDataError,
                "message": "Feature dimension mismatch detected",
                "suggestions": [
                    "Check that training and prediction data have the same number of features",
                    "Verify feature preprocessing is consistent",
                    "Ensure no missing columns in prediction data"
                ]
            },
            "missing_values": {
                "keywords": ["nan", "inf", "null", "missing"],
                "error_type": XGBoostDataError,
                "message": "Missing or invalid values detected",
                "suggestions": [
                    "Handle NaN values before training/prediction",
                    "Use XGBoost's built-in missing value handling",
                    "Check for infinite values in the dataset"
                ]
            },
            "data_type": {
                "keywords": ["dtype", "type", "float64", "int64"],
                "error_type": XGBoostDataError,
                "message": "Data type incompatibility",
                "suggestions": [
                    "Ensure numeric data types (float32/float64)",
                    "Convert categorical variables to numeric",
                    "Check for object columns in DataFrame"
                ]
            },
            
            # Training-related errors
            "convergence": {
                "keywords": ["convergence", "optimization", "gradient"],
                "error_type": XGBoostTrainingError,
                "message": "Training convergence issues",
                "suggestions": [
                    "Reduce learning rate",
                    "Increase number of estimators",
                    "Adjust regularization parameters",
                    "Check data quality and preprocessing"
                ]
            },
            "memory": {
                "keywords": ["memory", "allocation", "oom", "out of memory"],
                "error_type": XGBoostTrainingError,
                "message": "Memory allocation error",
                "suggestions": [
                    "Reduce dataset size or use sampling",
                    "Decrease tree depth or number of estimators",
                    "Use sparse data formats if applicable",
                    "Increase system memory or use smaller data types"
                ]
            },
            "early_stopping": {
                "keywords": ["early_stopping", "validation", "eval_set"],
                "error_type": XGBoostTrainingError,
                "message": "Early stopping configuration error",
                "suggestions": [
                    "Provide validation set when using early stopping",
                    "Check eval_metric compatibility with task type",
                    "Ensure validation set has same features as training set"
                ]
            },
            
            # Configuration errors
            "parameter": {
                "keywords": ["parameter", "param", "config", "invalid"],
                "error_type": XGBoostConfigurationError,
                "message": "Invalid parameter configuration",
                "suggestions": [
                    "Check parameter names and value ranges",
                    "Verify parameter compatibility with objective",
                    "Use XGBoost documentation for valid parameter combinations"
                ]
            },
            "objective": {
                "keywords": ["objective", "eval_metric", "num_class"],
                "error_type": XGBoostConfigurationError,
                "message": "Objective function configuration error",
                "suggestions": [
                    "Ensure objective matches task type (regression/classification)",
                    "Set num_class for multi-class classification",
                    "Check eval_metric compatibility with objective"
                ]
            },
            
            # Model/prediction errors
            "model_not_trained": {
                "keywords": ["not trained", "fitted", "model"],
                "error_type": XGBoostPredictionError,
                "message": "Model not properly trained",
                "suggestions": [
                    "Train the model before making predictions",
                    "Check if model training completed successfully",
                    "Verify model loading if using saved model"
                ]
            },
            "prediction_shape": {
                "keywords": ["prediction", "shape", "dimension"],
                "error_type": XGBoostPredictionError,
                "message": "Prediction output shape error",
                "suggestions": [
                    "Check input data dimensions",
                    "Verify model was trained on compatible data",
                    "Ensure consistent feature preprocessing"
                ]
            }
        }
    
    def diagnose_error(self, error: Exception, operation: str = "unknown") -> Dict[str, Any]:
        """
        Diagnose an XGBoost-related error and provide suggestions.
        
        Args:
            error: The exception that occurred
            operation: The operation that was being performed
            
        Returns:
            Dictionary with diagnosis information
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Look for pattern matches
        matched_patterns = []
        for pattern_name, pattern_info in self.error_patterns.items():
            if any(keyword in error_str for keyword in pattern_info["keywords"]):
                matched_patterns.append((pattern_name, pattern_info))
        
        # Create diagnosis
        diagnosis = {
            "operation": operation,
            "original_error": str(error),
            "error_type": error_type,
            "matched_patterns": matched_patterns,
            "suggestions": [],
            "severity": "high" if matched_patterns else "medium"
        }
        
        # Collect suggestions from matched patterns
        for pattern_name, pattern_info in matched_patterns:
            diagnosis["suggestions"].extend(pattern_info["suggestions"])
        
        # Add general suggestions if no patterns matched
        if not matched_patterns:
            diagnosis["suggestions"] = [
                "Check XGBoost documentation for the specific error",
                "Verify data format and preprocessing steps",
                "Try with default parameters first",
                "Check system resources (memory, disk space)"
            ]
        
        return diagnosis
    
    def handle_training_error(self, error: Exception, 
                            model_params: Optional[Dict] = None,
                            data_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle training-specific errors with detailed context."""
        diagnosis = self.diagnose_error(error, "training")
        
        # Add training-specific context
        if model_params:
            diagnosis["model_params"] = model_params
            
        if data_info:
            diagnosis["data_info"] = data_info
            
        # Training-specific suggestions
        training_suggestions = [
            "Start with default parameters and gradually tune",
            "Validate data preprocessing pipeline",
            "Check for class imbalance in classification tasks",
            "Monitor memory usage during training"
        ]
        
        diagnosis["training_suggestions"] = training_suggestions
        
        return diagnosis
    
    def handle_prediction_error(self, error: Exception,
                              model_info: Optional[Dict] = None,
                              input_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle prediction-specific errors with context."""
        diagnosis = self.diagnose_error(error, "prediction")
        
        # Add prediction-specific context
        if model_info:
            diagnosis["model_info"] = model_info
            
        if input_info:
            diagnosis["input_info"] = input_info
            
        # Prediction-specific suggestions
        prediction_suggestions = [
            "Verify input data format matches training data",
            "Check feature names and order",
            "Ensure model is properly loaded and trained",
            "Validate data preprocessing consistency"
        ]
        
        diagnosis["prediction_suggestions"] = prediction_suggestions
        
        return diagnosis
    
    def log_error_diagnosis(self, diagnosis: Dict[str, Any], 
                          logger_instance: Optional[logging.Logger] = None):
        """Log error diagnosis in a structured format."""
        log = logger_instance or logger
        
        log.error(f"XGBoost Error in {diagnosis['operation']}:")
        log.error(f"  Original Error: {diagnosis['original_error']}")
        log.error(f"  Error Type: {diagnosis['error_type']}")
        log.error(f"  Severity: {diagnosis['severity']}")
        
        if diagnosis['matched_patterns']:
            log.error("  Matched Error Patterns:")
            for pattern_name, _ in diagnosis['matched_patterns']:
                log.error(f"    - {pattern_name}")
        
        log.error("  Suggestions:")
        for suggestion in diagnosis['suggestions']:
            log.error(f"    - {suggestion}")
            
        # Log additional context if available
        if 'training_suggestions' in diagnosis:
            log.error("  Training-specific suggestions:")
            for suggestion in diagnosis['training_suggestions']:
                log.error(f"    - {suggestion}")
                
        if 'prediction_suggestions' in diagnosis:
            log.error("  Prediction-specific suggestions:")
            for suggestion in diagnosis['prediction_suggestions']:
                log.error(f"    - {suggestion}")

def xgboost_error_handler(operation: str = "unknown"):
    """
    Decorator for XGBoost operations that provides automatic error handling.
    
    Args:
        operation: Description of the operation being performed
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = XGBoostErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Diagnose the error
                diagnosis = error_handler.diagnose_error(e, operation)
                
                # Log the diagnosis
                error_handler.log_error_diagnosis(diagnosis)
                
                # Re-raise with enhanced message
                enhanced_message = f"XGBoost {operation} failed: {str(e)}"
                if diagnosis['suggestions']:
                    enhanced_message += f"\nSuggestions: {'; '.join(diagnosis['suggestions'][:3])}"
                
                raise type(e)(enhanced_message) from e
                
        return wrapper
    return decorator

def validate_xgboost_data(X: Union[np.ndarray, pd.DataFrame], 
                         y: Optional[np.ndarray] = None,
                         operation: str = "operation") -> Dict[str, Any]:
    """
    Validate data for XGBoost operations and return diagnostic information.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        operation: Operation description
        
    Returns:
        Validation results and any issues found
    """
    issues = []
    data_info = {}
    
    # Check X
    if X is None:
        raise XGBoostDataError("Feature matrix X cannot be None")
    
    # Convert to numpy if pandas DataFrame
    if isinstance(X, pd.DataFrame):
        data_info['original_type'] = 'DataFrame'
        data_info['feature_names'] = X.columns.tolist()
        
        # Check for object columns
        object_columns = X.select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            issues.append(f"Object columns detected: {object_columns}")
            
        X_array = X.values
    else:
        data_info['original_type'] = 'numpy'
        X_array = X
    
    # Check data shape
    if len(X_array.shape) != 2:
        raise XGBoostDataError(f"X must be 2D array, got shape {X_array.shape}")
    
    data_info['n_samples'], data_info['n_features'] = X_array.shape
    
    # Check for NaN/inf values
    if np.any(np.isnan(X_array)):
        issues.append("NaN values detected in features")
    if np.any(np.isinf(X_array)):
        issues.append("Infinite values detected in features")
    
    # Check data types
    if not np.issubdtype(X_array.dtype, np.number):
        issues.append(f"Non-numeric data type: {X_array.dtype}")
    
    # Check y if provided
    if y is not None:
        if len(y) != X_array.shape[0]:
            raise XGBoostDataError(f"X and y length mismatch: {X_array.shape[0]} vs {len(y)}")
        
        data_info['target_type'] = type(y).__name__
        if hasattr(y, 'dtype'):
            data_info['target_dtype'] = str(y.dtype)
        
        # Check for NaN in target
        try:
            if np.any(np.isnan(y)):
                issues.append("NaN values detected in target")
        except (TypeError, ValueError):
            # Non-numeric target, check with pandas
            if pd.Series(y).isna().any():
                issues.append("Missing values detected in target")
    
    return {
        'data_info': data_info,
        'issues': issues,
        'is_valid': len(issues) == 0,
        'operation': operation
    }

def format_xgboost_error_report(diagnosis: Dict[str, Any]) -> str:
    """
    Format error diagnosis into a readable report.
    
    Args:
        diagnosis: Error diagnosis dictionary
        
    Returns:
        Formatted error report string
    """
    report = []
    report.append("=" * 60)
    report.append("XGBOOST ERROR REPORT")
    report.append("=" * 60)
    report.append(f"Operation: {diagnosis['operation']}")
    report.append(f"Error Type: {diagnosis['error_type']}")
    report.append(f"Severity: {diagnosis['severity']}")
    report.append("")
    report.append("Original Error:")
    report.append(f"  {diagnosis['original_error']}")
    report.append("")
    
    if diagnosis['matched_patterns']:
        report.append("Identified Error Patterns:")
        for pattern_name, pattern_info in diagnosis['matched_patterns']:
            report.append(f"  - {pattern_name}: {pattern_info['message']}")
        report.append("")
    
    report.append("Suggestions:")
    for i, suggestion in enumerate(diagnosis['suggestions'], 1):
        report.append(f"  {i}. {suggestion}")
    
    if 'training_suggestions' in diagnosis:
        report.append("")
        report.append("Training-specific Suggestions:")
        for i, suggestion in enumerate(diagnosis['training_suggestions'], 1):
            report.append(f"  {i}. {suggestion}")
    
    if 'prediction_suggestions' in diagnosis:
        report.append("")
        report.append("Prediction-specific Suggestions:")
        for i, suggestion in enumerate(diagnosis['prediction_suggestions'], 1):
            report.append(f"  {i}. {suggestion}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report) 