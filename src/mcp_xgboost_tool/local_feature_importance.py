#!/usr/bin/env python3
"""
Local Feature Importance Analysis Module

This module provides functionality for analyzing local feature importance
using SHAP (SHapley Additive exPlanations) for individual samples or batches.
Generates waterfall plots, force plots, and decision plots.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Check for SHAP availability
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP library is available")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available. Install with: pip install shap")

# Check for sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")


class LocalFeatureImportanceAnalyzer:
    """
    Analyzer for local (sample-specific) feature importance using SHAP.
    
    Provides methods to analyze individual samples or small batches and generate
    detailed visualizations including waterfall plots, force plots, and decision plots.
    """
    
    def __init__(self, output_dir: str = "local_feature_analysis"):
        """
        Initialize the local feature importance analyzer.
        
        Args:
            output_dir: Directory to save analysis results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results storage
        self.shap_explainer = None
        self.shap_values = None
        self.expected_value = None
        self.feature_names = None
        self.X_data = None
        
        # Analysis metadata
        self.model_info = {}
        self.analysis_timestamp = None
        self.target_names = None
        self.target_dimension = 1
        
        # Preprocessor for target inverse transformation
        self.preprocessor = None
        
        logger.info(f"LocalFeatureImportanceAnalyzer initialized with output directory: {self.output_dir}")
    
    def _get_target_name(self, target_index: int) -> str:
        """
        Get the target name for a given target index.
        
        Args:
            target_index: Index of the target
            
        Returns:
            Target name (either from stored target_names or fallback format)
        """
        if (hasattr(self, 'target_names') and self.target_names and 
            isinstance(self.target_names, list) and target_index < len(self.target_names)):
            return self.target_names[target_index]
        else:
            return f"Target_{target_index}"
    
    def _get_intelligent_class_label(self, class_index: int, target_column_name: str = None) -> str:
        """
        智能生成分类任务的类别标签，优先使用label_mapping中的真实类别名称
        
        Args:
            class_index: 类别索引
            target_column_name: 目标列名称
            
        Returns:
            智能生成的类别标签
        """
        # 获取基础目标列名称
        if target_column_name:
            base_name = target_column_name
        elif (hasattr(self, 'target_names') and self.target_names and 
              isinstance(self.target_names, list) and len(self.target_names) > 0):
            base_name = self.target_names[0]
        else:
            base_name = "Target"
        
        # 检查是否有模型信息和类别映射
        if hasattr(self, 'model_metadata') and self.model_metadata:
            # 优先使用label_mapping获取真实的类别名称
            label_mapping = self.model_metadata.get('label_mapping', {})
            if label_mapping:
                class_to_label = label_mapping.get('class_to_label', {})
                classes_list = label_mapping.get('classes', [])
                
                # 方法1: 通过class_to_label映射查找
                if class_to_label:
                    # 尝试使用class_index作为键查找
                    class_label = (class_to_label.get(str(class_index)) or 
                                 class_to_label.get(class_index))
                    
                    if class_label:
                        # 如果找到了真实的类别名称，使用更简洁的格式
                        if base_name and base_name.lower() not in ['target', 'label', 'class']:
                            return f"{base_name}_{class_label}"
                        else:
                            return str(class_label)
                
                # 方法2: 通过classes列表查找
                if classes_list and class_index < len(classes_list):
                    class_label = classes_list[class_index]
                    if base_name and base_name.lower() not in ['target', 'label', 'class']:
                        return f"{base_name}_{class_label}"
                    else:
                        return str(class_label)
            
            # 回退到原来的model_info方式
            model_info = self.model_metadata.get('model_info', {})
            classes = model_info.get('classes', [])
            n_classes = model_info.get('n_classes', 0)
            task_type = self.model_metadata.get('task_type', 'unknown')
            
            if task_type == 'classification' and classes and class_index < len(classes):
                class_value = classes[class_index]
                
                # 对于二分类，使用更有意义的标签
                if n_classes == 2:
                    if isinstance(class_value, (int, float)):
                        # 如果原始类别是数字，保持原始的数字值
                        if base_name and base_name.lower() not in ['target', 'label', 'class']:
                            # 使用格式：目标列名_原始类别值 (如 "Survived_0", "Survived_1")
                            return f"{base_name}_{int(class_value)}"
                        else:
                            return f"Class_{int(class_value)}"
                    else:
                        # 如果原始类别就是字符串，直接使用
                        return f"{base_name}_{class_value}"
                
                # 对于多分类
                else:
                    if isinstance(class_value, (int, float)):
                        # 对于数字类别，保持原始数字值
                        return f"{base_name}_{int(class_value)}"
                    else:
                        # 对于字符串类别，直接使用
                        return f"{base_name}_{class_value}"
        
        # 最终回退到原始逻辑
        return f"{base_name}_Class{class_index}"
    
    def analyze_sample_importance(
        self,
        model: Union[RandomForestClassifier, RandomForestRegressor],
        sample_data: Union[np.ndarray, pd.DataFrame, Dict[str, float], List[float]],
        background_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        sample_index: Optional[int] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
        original_sample_data: Optional[Union[np.ndarray, pd.DataFrame, Dict[str, float], List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze local feature importance for a single sample.
        
        Args:
            model: Trained RandomForest model
            sample_data: Single sample to analyze (various input formats supported)
            background_data: Background data for SHAP explainer
            feature_names: Names of features
            sample_index: Optional index for the sample (for naming)
            model_metadata: Model metadata including target information for multi-target handling
            original_sample_data: Original sample data before preprocessing (for feature_contributions)
            
        Returns:
            Dictionary with local importance analysis results
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP library not available for local feature importance analysis")
            return {'error': 'SHAP library not available'}
        
        try:
            logger.info("Starting local feature importance analysis for single sample...")
            
            # Prepare and validate input data
            X_sample, feature_names = self._prepare_sample_data(sample_data, feature_names)
            X_background, _ = self._prepare_background_data(background_data, feature_names)
            
            # Store data for visualization
            self.X_data = X_background
            self.feature_names = feature_names
            
            # Determine if this is a multi-target task
            is_multi_target = False
            target_names = None
            target_dimension = 1
            
            if model_metadata:
                target_dimension = model_metadata.get('target_dimension', 1)
                is_multi_target = target_dimension > 1
                target_names = model_metadata.get('target_name', [])
                if isinstance(target_names, str):
                    target_names = [target_names]
                # Store target names and model metadata in the analyzer for use in plotting
                self.target_names = target_names if target_names else None
                self.target_dimension = target_dimension
                self.model_metadata = model_metadata  # Store full metadata for intelligent labeling
                logger.info(f"Multi-target analysis: {is_multi_target}, Target dimension: {target_dimension}")
            
            # Create SHAP explainer
            if self.shap_explainer is None:
                logger.info("Creating SHAP TreeExplainer...")
                self.shap_explainer = shap.TreeExplainer(model)
                self.expected_value = self.shap_explainer.expected_value
            
            # Calculate SHAP values for the sample
            logger.info("Calculating SHAP values for sample...")
            logger.info(f"X_sample shape: {X_sample.shape}, is_multi_target: {is_multi_target}, target_dimension: {target_dimension}")
            
            try:
                sample_shap_values = self.shap_explainer.shap_values(X_sample)
                logger.info(f"SHAP values computed successfully - shape: {getattr(sample_shap_values, 'shape', 'N/A')}, type: {type(sample_shap_values)}")
                if isinstance(sample_shap_values, list):
                    logger.info(f"SHAP values is a list with {len(sample_shap_values)} elements")
                    for i, sv in enumerate(sample_shap_values):
                        logger.info(f"  List element {i} shape: {sv.shape}")
                else:
                    logger.info(f"SHAP values single array shape: {sample_shap_values.shape}")
            except Exception as shap_error:
                logger.error(f"Error computing SHAP values: {str(shap_error)}")
                logger.error(f"X_sample shape: {X_sample.shape}, X_sample type: {type(X_sample)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise
            
            # Get model prediction
            model_prediction = self._get_model_prediction(model, X_sample)
            
            # Handle multi-output case
            if isinstance(sample_shap_values, list):
                # Classification case - store all class SHAP values
                self.shap_values = sample_shap_values
                if len(sample_shap_values) == 2:
                    # Binary classification - analyze positive class
                    shap_values_to_analyze = sample_shap_values[1]
                    expected_value = self.expected_value[1] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                else:
                    # Multi-class - analyze first class (can be extended)
                    shap_values_to_analyze = sample_shap_values[0]
                    expected_value = self.expected_value[0] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                
                # Extract SHAP values for single sample
                if shap_values_to_analyze.ndim > 1:
                    shap_values_single = shap_values_to_analyze[0]
                else:
                    shap_values_single = shap_values_to_analyze
                
                prediction_scalar = float(model_prediction[0]) if hasattr(model_prediction, '__len__') else float(model_prediction)
                
                # Generate intelligent target name for classification
                target_name = None
                if (hasattr(self, 'model_metadata') and self.model_metadata and 
                    self.model_metadata.get('task_type') == 'classification'):
                    target_column_name = self.model_metadata.get('target_column', 
                                         self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                    if len(sample_shap_values) == 2:
                        # Binary classification - use positive class
                        target_name = self._get_intelligent_class_label(1, target_column_name)
                    else:
                        # Multi-class - use first class
                        target_name = self._get_intelligent_class_label(0, target_column_name)
                
                print(f"DEBUG: In analyze_sample_importance, generated target_name: {target_name}")
                
                # Create classification results
                results = self._create_single_target_results(
                    sample_index, expected_value, prediction_scalar, 
                    shap_values_single, X_sample, feature_names, model, X_background,
                    target_name=target_name,
                    original_sample_data=original_sample_data
                )
                
            elif is_multi_target and target_dimension > 1:
                # Multi-target regression case
                logger.info(f"Processing multi-target regression with {target_dimension} targets")
                logger.info(f"SHAP values shape: {sample_shap_values.shape}")
                logger.info(f"Expected value type: {type(self.expected_value)}, shape: {getattr(self.expected_value, 'shape', 'N/A')}")
                logger.info(f"Model prediction shape: {getattr(model_prediction, 'shape', 'N/A')}")
                
                self.shap_values = sample_shap_values
                
                # Create results for each target
                target_results = {}
                overall_feature_importance = {}
                
                for target_idx in range(target_dimension):
                    target_name = target_names[target_idx] if target_names and target_idx < len(target_names) else f"Target_{target_idx + 1}"
                    
                    # Extract SHAP values for this target
                    # For multi-target regression, SHAP values shape can be (n_samples, n_features, n_targets) or (n_samples, n_targets, n_features)
                    if sample_shap_values.ndim == 3:
                        # Determine format based on dimensions
                        if sample_shap_values.shape[1] == len(feature_names) and sample_shap_values.shape[2] == target_dimension:
                            # Shape: (n_samples, n_features, n_targets)
                            target_shap = sample_shap_values[0, :, target_idx]
                        elif sample_shap_values.shape[1] == target_dimension and sample_shap_values.shape[2] == len(feature_names):
                            # Shape: (n_samples, n_targets, n_features)
                            target_shap = sample_shap_values[0, target_idx, :]
                        else:
                            logger.warning(f"Unexpected SHAP values shape: {sample_shap_values.shape}")
                            target_shap = sample_shap_values[0, target_idx, :] if sample_shap_values.shape[1] >= target_idx else sample_shap_values[0, :, target_idx]
                    elif sample_shap_values.ndim == 2 and target_dimension == 1:
                        # Single target case
                        target_shap = sample_shap_values[0, :]
                    else:
                        # Handle unexpected shapes
                        logger.warning(f"Unexpected SHAP values shape: {sample_shap_values.shape}")
                        if sample_shap_values.ndim == 2:
                            target_shap = sample_shap_values[0, :]
                        else:
                            target_shap = sample_shap_values.flatten()[:len(feature_names)]
                    
                    # Extract expected value for this target
                    if isinstance(self.expected_value, np.ndarray) and len(self.expected_value) > target_idx:
                        target_expected = float(self.expected_value[target_idx])
                    else:
                        target_expected = float(self.expected_value)
                    
                    # Extract prediction for this target
                    if isinstance(model_prediction, np.ndarray) and model_prediction.ndim > 1:
                        target_prediction = float(model_prediction[0, target_idx])
                    elif isinstance(model_prediction, np.ndarray) and len(model_prediction) > target_idx:
                        target_prediction = float(model_prediction[target_idx])
                    else:
                        target_prediction = float(model_prediction)
                    
                    logger.info(f"Target {target_name}: expected={target_expected:.4f}, prediction={target_prediction:.4f}, shap_shape={target_shap.shape}")
                    
                    # Create results for this target
                    target_result = self._create_single_target_results(
                        sample_index, target_expected, target_prediction,
                        target_shap, X_sample, feature_names, model, X_background,
                        target_name=target_name,
                        original_sample_data=original_sample_data
                    )
                    
                    target_results[target_name] = target_result
                    
                    # Accumulate feature importance across targets
                    for contrib in target_result['feature_contributions']:
                        feature = contrib['feature']
                        if feature not in overall_feature_importance:
                            overall_feature_importance[feature] = {
                                'total_abs_shap': 0,
                                'target_contributions': {},
                                'feature_value': contrib['value']
                            }
                        overall_feature_importance[feature]['total_abs_shap'] += abs(contrib['shap_value'])
                        overall_feature_importance[feature]['target_contributions'][target_name] = contrib['shap_value']
                
                # Create overall summary
                overall_contributions = []
                for feature, data in overall_feature_importance.items():
                    overall_contributions.append({
                        'feature': feature,
                        'value': data['feature_value'],
                        'total_abs_shap': data['total_abs_shap'],
                        'target_contributions': data['target_contributions']
                    })
                
                # Sort by total absolute SHAP importance
                overall_contributions.sort(key=lambda x: x['total_abs_shap'], reverse=True)
                for i, contrib in enumerate(overall_contributions):
                    contrib['overall_rank'] = i + 1
                
                # Create comprehensive multi-target results
                results = {
                    'sample_index': sample_index or 0,
                    'is_multi_target': True,
                    'target_dimension': target_dimension,
                    'target_names': target_names if target_names else [f"Target_{i+1}" for i in range(target_dimension)],
                    'target_results': target_results,
                    'overall_feature_importance': overall_contributions,
                    'analysis_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'n_features': len(feature_names),
                        'n_targets': target_dimension,
                        'model_type': type(model).__name__,
                        'background_samples': X_background.shape[0],
                        'analysis_based_on_preprocessed_data': True,
                        'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.'
                    }
                }
                
                logger.info(f"Multi-target local importance analysis completed")
                logger.info(f"Top 3 overall contributing features: {', '.join([c['feature'] for c in overall_contributions[:3]])}")
                
            else:
                # Single-target regression case
                self.shap_values = sample_shap_values
                shap_values_to_analyze = sample_shap_values
                expected_value = self.expected_value
                
                # Extract SHAP values for single sample
                if shap_values_to_analyze.ndim > 1:
                    shap_values_single = shap_values_to_analyze[0]
                else:
                    shap_values_single = shap_values_to_analyze
                
                prediction_scalar = float(model_prediction[0]) if hasattr(model_prediction, '__len__') else float(model_prediction)
                
                # Create single-target results
                results = self._create_single_target_results(
                    sample_index, expected_value, prediction_scalar,
                    shap_values_single, X_sample, feature_names, model, X_background,
                    target_name=target_names[0],
                    original_sample_data=original_sample_data
                )
            
            # Store analysis timestamp
            self.analysis_timestamp = results['analysis_metadata']['timestamp']
            
            return results
            
        except Exception as e:
            logger.error(f"Error in local sample importance analysis: {str(e)}")
            raise
    
    def analyze_batch_importance(
        self,
        model: Union[RandomForestClassifier, RandomForestRegressor],
        batch_data: Union[np.ndarray, pd.DataFrame, List[Dict[str, float]]],
        background_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        max_samples: int = 10,
        model_metadata: Optional[Dict[str, Any]] = None,
        original_batch_data: Optional[Union[np.ndarray, pd.DataFrame, List[Dict[str, float]]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze local feature importance for a batch of samples.
        
        Args:
            model: Trained RandomForest model
            batch_data: Batch of samples to analyze
            background_data: Background data for SHAP explainer
            feature_names: Names of features
            max_samples: Maximum number of samples to analyze
            model_metadata: Model metadata including target information for multi-target handling
            original_batch_data: Original batch data before preprocessing (for feature_contributions)
            
        Returns:
            Dictionary with batch importance analysis results
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP library not available for local feature importance analysis")
            return {'error': 'SHAP library not available'}
        
        try:
            logger.info(f"Starting local feature importance analysis for batch...")
            
            # Prepare and validate input data
            X_batch, feature_names = self._prepare_batch_data(batch_data, feature_names, max_samples)
            X_background, _ = self._prepare_background_data(background_data, feature_names)
            
            # Store data for visualization
            self.X_data = X_background
            self.feature_names = feature_names
            
            # Determine if this is a multi-target task
            is_multi_target = False
            target_names = None
            target_dimension = 1
            
            if model_metadata:
                target_dimension = model_metadata.get('target_dimension', 1)
                is_multi_target = target_dimension > 1
                target_names = model_metadata.get('target_name', [])
                if isinstance(target_names, str):
                    target_names = [target_names]
                logger.info(f"Batch multi-target analysis: {is_multi_target}, Target dimension: {target_dimension}")
            
            # Create SHAP explainer
            if self.shap_explainer is None:
                logger.info("Creating SHAP TreeExplainer...")
                self.shap_explainer = shap.TreeExplainer(model)
                self.expected_value = self.shap_explainer.expected_value
            
            # Calculate SHAP values for the batch
            logger.info(f"Calculating SHAP values for {X_batch.shape[0]} samples...")
            batch_shap_values = self.shap_explainer.shap_values(X_batch)
            
            # Get model predictions for the batch
            predictions = self._get_model_prediction(model, X_batch)
            
            # Handle multi-output case
            if isinstance(batch_shap_values, list):
                # Classification case
                self.shap_values = batch_shap_values
                if len(batch_shap_values) == 2:
                    # Binary classification - analyze positive class
                    shap_values_to_analyze = batch_shap_values[1]
                    expected_value = self.expected_value[1] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                else:
                    # Multi-class - analyze first class (can be extended)
                    shap_values_to_analyze = batch_shap_values[0]
                    expected_value = self.expected_value[0] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                
                # Prepare original batch data for each sample
                original_samples = []
                if original_batch_data is not None:
                    if isinstance(original_batch_data, pd.DataFrame):
                        original_samples = [original_batch_data.iloc[i:i+1] for i in range(min(len(original_batch_data), X_batch.shape[0]))]
                    elif isinstance(original_batch_data, list):
                        original_samples = original_batch_data[:X_batch.shape[0]]
                    elif isinstance(original_batch_data, np.ndarray):
                        original_samples = [original_batch_data[i:i+1] for i in range(X_batch.shape[0])]
                    else:
                        original_samples = [None] * X_batch.shape[0]
                else:
                    original_samples = [None] * X_batch.shape[0]
                
                # Create detailed results for each sample
                sample_results = []
                for i in range(X_batch.shape[0]):
                    sample_result = self._create_single_target_results(
                        i, expected_value, 
                        float(predictions[i]) if hasattr(predictions, '__len__') else float(predictions),
                        shap_values_to_analyze[i], 
                        X_batch[i:i+1], feature_names, model, X_background,
                        original_sample_data=original_samples[i]
                    )
                    sample_results.append(sample_result)
                
                # Calculate original scale batch statistics if preprocessor is available (classification)
                original_scale_predictions = predictions
                original_scale_expected_value = expected_value
                
                if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                    try:
                        # Transform batch predictions to original scale
                        pred_array = np.array(predictions)
                        if pred_array.ndim == 1:
                            pred_array = pred_array.reshape(-1, 1)
                        original_pred_transformed = self.preprocessor.inverse_transform_target(pred_array)
                        original_scale_predictions = original_pred_transformed.ravel() if original_pred_transformed.shape[1] == 1 else original_pred_transformed
                        
                        # Transform expected value to original scale
                        if isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
                            if expected_value.ndim == 1:
                                expected_array = expected_value.reshape(-1, 1)
                            else:
                                expected_array = expected_value
                        else:
                            expected_array = np.array([[float(expected_value)]])
                        original_expected_transformed = self.preprocessor.inverse_transform_target(expected_array)
                        original_scale_expected_value = float(original_expected_transformed[0])
                        
                        logger.info(f"Transformed batch mean prediction from {np.mean(predictions)} to original scale: {np.mean(original_scale_predictions)}")
                    except Exception as e:
                        logger.warning(f"Failed to transform batch predictions to original scale: {e}")
                        original_scale_predictions = predictions
                        original_scale_expected_value = expected_value

                # Create batch summary
                results = {
                    'batch_size': X_batch.shape[0],
                    'is_multi_target': False,
                    'expected_value': float(expected_value),
                    'expected_value_original_scale': float(original_scale_expected_value),
                    'sample_results': sample_results,
                    'batch_summary': {
                        'mean_prediction': float(np.mean(predictions)),
                        'mean_prediction_original_scale': float(np.mean(original_scale_predictions)),
                        'prediction_std': float(np.std(predictions)),
                        'prediction_std_original_scale': float(np.std(original_scale_predictions)),
                        'top_features_across_batch': self._get_top_features_across_batch(sample_results)
                    },
                    'analysis_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'n_features': len(feature_names),
                        'model_type': type(model).__name__,
                        'background_samples': X_background.shape[0],
                        'has_original_scale_values': hasattr(self, 'preprocessor') and self.preprocessor is not None,
                        'analysis_based_on_preprocessed_data': True,
                        'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.'
                    }
                }
                
            elif is_multi_target and target_dimension > 1:
                # Multi-target regression case
                logger.info(f"Processing multi-target regression batch with {target_dimension} targets")
                logger.info(f"Batch SHAP values shape: {batch_shap_values.shape}")
                logger.info(f"Expected shape interpretation: (n_samples, n_features, n_targets) or (n_samples, n_targets, n_features)")
                
                # For multi-target regression, SHAP values shape should be (n_samples, n_features, n_targets)
                # But some versions might return (n_samples, n_targets, n_features)
                # Let's determine the correct interpretation
                if batch_shap_values.shape[1] == len(feature_names) and batch_shap_values.shape[2] == target_dimension:
                    # Shape: (n_samples, n_features, n_targets)
                    shap_format = "samples_features_targets"
                    logger.info("SHAP format detected: (n_samples, n_features, n_targets)")
                elif batch_shap_values.shape[1] == target_dimension and batch_shap_values.shape[2] == len(feature_names):
                    # Shape: (n_samples, n_targets, n_features) 
                    shap_format = "samples_targets_features"
                    logger.info("SHAP format detected: (n_samples, n_targets, n_features)")
                else:
                    logger.error(f"Unexpected SHAP shape: {batch_shap_values.shape} for {len(feature_names)} features and {target_dimension} targets")
                    raise ValueError(f"Unexpected SHAP values shape: {batch_shap_values.shape}")
                logger.info(f"Batch SHAP values shape: {batch_shap_values.shape}")
                logger.info(f"Predictions shape: {getattr(predictions, 'shape', 'N/A')}")
                
                self.shap_values = batch_shap_values
                
                # Create results for each sample and target
                sample_results = []
                target_batch_summaries = {}
                overall_batch_importance = {}
                
                # Initialize target summaries
                for target_idx in range(target_dimension):
                    target_name = target_names[target_idx] if target_names and target_idx < len(target_names) else f"Target_{target_idx + 1}"
                    target_batch_summaries[target_name] = {
                        'predictions': [],
                        'sample_results': []
                    }
                
                # Prepare original batch data for each sample (multi-target case)
                original_samples = []
                if original_batch_data is not None:
                    if isinstance(original_batch_data, pd.DataFrame):
                        original_samples = [original_batch_data.iloc[i:i+1] for i in range(min(len(original_batch_data), X_batch.shape[0]))]
                    elif isinstance(original_batch_data, list):
                        original_samples = original_batch_data[:X_batch.shape[0]]
                    elif isinstance(original_batch_data, np.ndarray):
                        original_samples = [original_batch_data[i:i+1] for i in range(X_batch.shape[0])]
                    else:
                        original_samples = [None] * X_batch.shape[0]
                else:
                    original_samples = [None] * X_batch.shape[0]
                
                # Process each sample
                for i in range(X_batch.shape[0]):
                    # Create multi-target result for this sample
                    sample_target_results = {}
                    sample_overall_importance = {}
                    
                    for target_idx in range(target_dimension):
                        target_name = target_names[target_idx] if target_names and target_idx < len(target_names) else f"Target_{target_idx + 1}"
                        
                        # Extract SHAP values for this target and sample
                        if batch_shap_values.ndim == 3:
                            if shap_format == "samples_features_targets":
                                # Shape: (n_samples, n_features, n_targets)
                                target_shap = batch_shap_values[i, :, target_idx]
                            else:
                                # Shape: (n_samples, n_targets, n_features)
                                target_shap = batch_shap_values[i, target_idx, :]
                        elif batch_shap_values.ndim == 2:
                            target_shap = batch_shap_values[i, :]
                        else:
                            logger.warning(f"Unexpected batch SHAP values shape: {batch_shap_values.shape}")
                            target_shap = batch_shap_values.flatten()[:len(feature_names)]
                        
                        # Extract expected value for this target
                        if isinstance(self.expected_value, np.ndarray) and len(self.expected_value) > target_idx:
                            target_expected = float(self.expected_value[target_idx])
                        else:
                            target_expected = float(self.expected_value)
                        
                        # Extract prediction for this target and sample
                        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
                            target_prediction = float(predictions[i, target_idx])
                        elif isinstance(predictions, np.ndarray) and len(predictions) > i:
                            if hasattr(predictions[i], '__len__') and len(predictions[i]) > target_idx:
                                target_prediction = float(predictions[i][target_idx])
                            else:
                                target_prediction = float(predictions[i])
                        else:
                            target_prediction = float(predictions)
                        
                        # Create result for this target
                        target_result = self._create_single_target_results(
                            i, target_expected, target_prediction,
                            target_shap, X_batch[i:i+1], feature_names, model, X_background,
                            target_name=target_name,
                            original_sample_data=original_samples[i]
                        )
                        
                        sample_target_results[target_name] = target_result
                        target_batch_summaries[target_name]['predictions'].append(target_prediction)
                        target_batch_summaries[target_name]['sample_results'].append(target_result)
                        
                        # Accumulate overall importance for this sample
                        for contrib in target_result['feature_contributions']:
                            feature = contrib['feature']
                            if feature not in sample_overall_importance:
                                sample_overall_importance[feature] = {
                                    'total_abs_shap': 0,
                                    'target_contributions': {},
                                    'feature_value': contrib['value']
                                }
                            sample_overall_importance[feature]['total_abs_shap'] += abs(contrib['shap_value'])
                            sample_overall_importance[feature]['target_contributions'][target_name] = contrib['shap_value']
                    
                    # Create overall contributions for this sample
                    sample_overall_contributions = []
                    for feature, data in sample_overall_importance.items():
                        sample_overall_contributions.append({
                            'feature': feature,
                            'value': data['feature_value'],
                            'total_abs_shap': data['total_abs_shap'],
                            'target_contributions': data['target_contributions']
                        })
                    
                    sample_overall_contributions.sort(key=lambda x: x['total_abs_shap'], reverse=True)
                    for j, contrib in enumerate(sample_overall_contributions):
                        contrib['overall_rank'] = j + 1
                    
                    # Add to batch-wide importance tracking
                    for contrib in sample_overall_contributions:
                        feature = contrib['feature']
                        if feature not in overall_batch_importance:
                            overall_batch_importance[feature] = {
                                'total_abs_shap': 0,
                                'sample_contributions': []
                            }
                        overall_batch_importance[feature]['total_abs_shap'] += contrib['total_abs_shap']
                        overall_batch_importance[feature]['sample_contributions'].append(contrib['total_abs_shap'])
                    
                    # Store sample result
                    sample_result = {
                        'sample_index': i,
                        'is_multi_target': True,
                        'target_dimension': target_dimension,
                        'target_names': target_names if target_names else [f"Target_{i+1}" for i in range(target_dimension)],
                        'target_results': sample_target_results,
                        'overall_feature_importance': sample_overall_contributions
                    }
                    sample_results.append(sample_result)
                
                # Calculate batch-wide statistics
                batch_overall_importance = []
                for feature, data in overall_batch_importance.items():
                    batch_overall_importance.append({
                        'feature': feature,
                        'mean_total_abs_shap': data['total_abs_shap'] / len(sample_results),
                        'std_total_abs_shap': float(np.std(data['sample_contributions'])),
                        'total_abs_shap_sum': data['total_abs_shap']
                    })
                
                batch_overall_importance.sort(key=lambda x: x['mean_total_abs_shap'], reverse=True)
                for i, contrib in enumerate(batch_overall_importance):
                    contrib['batch_rank'] = i + 1
                
                # Calculate target-specific batch summaries
                for target_name, data in target_batch_summaries.items():
                    target_batch_summaries[target_name]['mean_prediction'] = float(np.mean(data['predictions']))
                    target_batch_summaries[target_name]['prediction_std'] = float(np.std(data['predictions']))
                    target_batch_summaries[target_name]['top_features'] = self._get_top_features_across_batch(data['sample_results'])
                    # Remove large sample_results to reduce output size
                    del target_batch_summaries[target_name]['sample_results']
                
                # Create comprehensive batch results
                results = {
                    'batch_size': X_batch.shape[0],
                    'is_multi_target': True,
                    'target_dimension': target_dimension,
                    'target_names': target_names if target_names else [f"Target_{i+1}" for i in range(target_dimension)],
                    'sample_results': sample_results,
                    'target_batch_summaries': target_batch_summaries,
                    'batch_overall_importance': batch_overall_importance,
                    'analysis_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'n_features': len(feature_names),
                        'n_targets': target_dimension,
                        'model_type': type(model).__name__,
                        'background_samples': X_background.shape[0],
                        'analysis_based_on_preprocessed_data': True,
                        'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.'
                    }
                }
                
                logger.info(f"Multi-target batch analysis completed for {X_batch.shape[0]} samples")
                logger.info(f"Top 3 batch features: {', '.join([c['feature'] for c in batch_overall_importance[:3]])}")
                
            else:
                # Single-target regression case
                self.shap_values = batch_shap_values
                shap_values_to_analyze = batch_shap_values
                expected_value = self.expected_value
                
                # Prepare original batch data for each sample (single-target case)
                original_samples = []
                if original_batch_data is not None:
                    if isinstance(original_batch_data, pd.DataFrame):
                        original_samples = [original_batch_data.iloc[i:i+1] for i in range(min(len(original_batch_data), X_batch.shape[0]))]
                    elif isinstance(original_batch_data, list):
                        original_samples = original_batch_data[:X_batch.shape[0]]
                    elif isinstance(original_batch_data, np.ndarray):
                        original_samples = [original_batch_data[i:i+1] for i in range(X_batch.shape[0])]
                    else:
                        original_samples = [None] * X_batch.shape[0]
                else:
                    original_samples = [None] * X_batch.shape[0]
                
                # Create detailed results for each sample
                sample_results = []
                for i in range(X_batch.shape[0]):
                    sample_result = self._create_single_target_results(
                        i, expected_value, 
                        float(predictions[i]) if hasattr(predictions, '__len__') else float(predictions),
                        shap_values_to_analyze[i], 
                        X_batch[i:i+1], feature_names, model, X_background,
                        target_name=target_names[0] if target_names else None,
                        original_sample_data=original_samples[i]
                    )
                    sample_results.append(sample_result)
                
                # Calculate original scale batch statistics if preprocessor is available (single-target regression)
                original_scale_predictions = predictions
                original_scale_expected_value = expected_value
                
                if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                    try:
                        # Transform batch predictions to original scale
                        pred_array = np.array(predictions)
                        if pred_array.ndim == 1:
                            pred_array = pred_array.reshape(-1, 1)
                        original_pred_transformed = self.preprocessor.inverse_transform_target(pred_array)
                        original_scale_predictions = original_pred_transformed.ravel() if original_pred_transformed.shape[1] == 1 else original_pred_transformed
                        
                        # Transform expected value to original scale
                        if isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
                            if expected_value.ndim == 1:
                                expected_array = expected_value.reshape(-1, 1)
                            else:
                                expected_array = expected_value
                        else:
                            expected_array = np.array([[float(expected_value)]])
                        original_expected_transformed = self.preprocessor.inverse_transform_target(expected_array)
                        original_scale_expected_value = float(original_expected_transformed[0])
                        
                        logger.info(f"Transformed batch mean prediction from {np.mean(predictions)} to original scale: {np.mean(original_scale_predictions)}")
                    except Exception as e:
                        logger.warning(f"Failed to transform batch predictions to original scale: {e}")
                        original_scale_predictions = predictions
                        original_scale_expected_value = expected_value

                # Create batch summary
                results = {
                    'batch_size': X_batch.shape[0],
                    'is_multi_target': False,
                    'expected_value': float(expected_value),
                    'expected_value_original_scale': float(original_scale_expected_value),
                    'sample_results': sample_results,
                    'batch_summary': {
                        'mean_prediction': float(np.mean(predictions)),
                        'mean_prediction_original_scale': float(np.mean(original_scale_predictions)),
                        'prediction_std': float(np.std(predictions)),
                        'prediction_std_original_scale': float(np.std(original_scale_predictions)),
                        'top_features_across_batch': self._get_top_features_across_batch(sample_results)
                    },
                    'analysis_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'n_features': len(feature_names),
                        'model_type': type(model).__name__,
                        'background_samples': X_background.shape[0],
                        'has_original_scale_values': hasattr(self, 'preprocessor') and self.preprocessor is not None,
                        'analysis_based_on_preprocessed_data': True,
                        'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.'
                    }
                }
            
            # Store analysis timestamp
            self.analysis_timestamp = results['analysis_metadata']['timestamp']
            
            logger.info(f"Batch local importance analysis completed for {X_batch.shape[0]} samples")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch importance analysis: {str(e)}")
            raise
    
    def _create_single_target_results(
        self,
        sample_index: Optional[int],
        expected_value: float,
        prediction: float,
        shap_values: np.ndarray,
        X_sample: np.ndarray,
        feature_names: List[str],
        model,
        X_background: np.ndarray,
        target_name: Optional[str] = None,
        original_sample_data: Optional[Union[np.ndarray, pd.DataFrame, Dict[str, float], List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Create results dictionary for a single target.
        
        Args:
            sample_index: Index of the sample
            expected_value: Expected value from SHAP explainer
            prediction: Model prediction for this target
            shap_values: SHAP values for this target
            X_sample: Sample feature values (preprocessed)
            feature_names: List of feature names
            model: The trained model
            X_background: Background data
            target_name: Name of the target (for multi-target)
            original_sample_data: Original sample data before preprocessing
            
        Returns:
            Dictionary with single target results
        """
        # Prepare original feature values
        original_values = None
        if original_sample_data is not None:
            if isinstance(original_sample_data, dict):
                # Single sample as dictionary
                original_values = [original_sample_data.get(feature_names[i], X_sample[0][i]) for i in range(len(feature_names))]
            elif isinstance(original_sample_data, pd.DataFrame):
                # DataFrame - get first row
                if len(original_sample_data) > 0:
                    original_values = [original_sample_data.iloc[0][feature_names[i]] if feature_names[i] in original_sample_data.columns else X_sample[0][i] for i in range(len(feature_names))]
                else:
                    original_values = [X_sample[0][i] for i in range(len(feature_names))]
            elif isinstance(original_sample_data, (list, np.ndarray)):
                # List or array
                if len(original_sample_data) >= len(feature_names):
                    original_values = [float(original_sample_data[i]) for i in range(len(feature_names))]
                else:
                    original_values = [X_sample[0][i] for i in range(len(feature_names))]
            else:
                original_values = [X_sample[0][i] for i in range(len(feature_names))]
        else:
            # Fallback to preprocessed values
            original_values = [X_sample[0][i] for i in range(len(feature_names))]
        
        # Ensure shap_values is 1-dimensional
        if shap_values.ndim > 1:
            if shap_values.shape[0] == 1:
                # Shape is (1, n_features) - extract the features
                shap_values_1d = shap_values[0]
            else:
                # Unexpected shape - flatten and take first n_features elements
                logger.warning(f"Unexpected SHAP values shape in _create_single_target_results: {shap_values.shape}")
                shap_values_1d = shap_values.flatten()[:len(feature_names)]
        else:
            shap_values_1d = shap_values
        
        # Ensure we have the right number of elements
        if len(shap_values_1d) != len(feature_names):
            logger.error(f"SHAP values length ({len(shap_values_1d)}) doesn't match feature names length ({len(feature_names)})")
            # Pad with zeros or truncate as needed
            if len(shap_values_1d) < len(feature_names):
                shap_values_1d = np.pad(shap_values_1d, (0, len(feature_names) - len(shap_values_1d)))
            else:
                shap_values_1d = shap_values_1d[:len(feature_names)]
        
        logger.debug(f"Final SHAP values shape: {shap_values_1d.shape}, feature names count: {len(feature_names)}")
        
        # Create detailed results
        feature_contributions = []
        for i in range(len(feature_names)):
            # Handle original value conversion (may be string for categorical features)
            original_val = original_values[i]
            if isinstance(original_val, str):
                # For categorical features, keep as string but also provide a note
                original_val_display = original_val
            else:
                try:
                    original_val_display = float(original_val)
                except (ValueError, TypeError):
                    # If conversion fails, keep as string
                    original_val_display = str(original_val)
            
            feature_contributions.append({
                'feature': feature_names[i],
                'value': original_val_display,  # Original value (user input, may be string)
                'value_preprocessed': float(X_sample[0][i]),  # Preprocessed value (used in analysis)
                'shap_value': float(shap_values_1d[i]),
                'contribution_rank': 0  # Will be filled after sorting
            })
        
        # Sort by absolute SHAP values and add ranks
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        for i, contrib in enumerate(feature_contributions):
            contrib['contribution_rank'] = i + 1
        
        # Calculate original scale prediction if preprocessor is available
        original_prediction = prediction
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            try:
                # Convert prediction to numpy array format for inverse transform
                if isinstance(prediction, (int, float)):
                    pred_array = np.array([[prediction]])
                else:
                    pred_array = np.array(prediction).reshape(-1, 1)
                
                # Inverse transform prediction to original scale
                original_pred_transformed = self.preprocessor.inverse_transform_target(pred_array)
                original_prediction = float(original_pred_transformed[0]) if original_pred_transformed.size == 1 else original_pred_transformed.tolist()
                logger.info(f"Transformed prediction from {prediction} to original scale: {original_prediction}")
            except Exception as e:
                logger.warning(f"Failed to transform prediction to original scale: {e}")
                original_prediction = prediction

        # Calculate original scale base value if preprocessor is available
        original_base_value = expected_value
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            try:
                # Convert base value to numpy array format for inverse transform
                if isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
                    # expected_value is already an array
                    if expected_value.ndim == 1:
                        base_array = expected_value.reshape(-1, 1)
                    else:
                        base_array = expected_value
                else:
                    # expected_value is a scalar
                    base_array = np.array([[float(expected_value)]])
                
                # Inverse transform base value to original scale
                original_base_transformed = self.preprocessor.inverse_transform_target(base_array)
                original_base_value = float(original_base_transformed[0]) if original_base_transformed.size == 1 else original_base_transformed.tolist()
                logger.info(f"Transformed base value from {expected_value} to original scale: {original_base_value}")
            except Exception as e:
                logger.warning(f"Failed to transform base value to original scale: {e}")
                original_base_value = expected_value

        # Handle expected_value conversion (may be array for multi-class)
        if isinstance(expected_value, np.ndarray):
            if expected_value.size == 1:
                base_value_display = float(expected_value.item())
            else:
                # For multi-class, use the first value or average
                base_value_display = float(expected_value[0]) if len(expected_value) > 0 else 0.0
                logger.debug(f"Multi-class base value, using first element: {base_value_display} from {expected_value}")
        else:
            base_value_display = float(expected_value)
        
        # Handle original_base_value conversion
        if isinstance(original_base_value, np.ndarray):
            if original_base_value.size == 1:
                original_base_value_display = float(original_base_value.item())
            else:
                original_base_value_display = float(original_base_value[0]) if len(original_base_value) > 0 else 0.0
        else:
            original_base_value_display = float(original_base_value)
        
        results = {
            'sample_index': sample_index or 0,
            'target_name': target_name,
            'base_value': base_value_display,
            'base_value_original_scale': original_base_value_display,
            'prediction': prediction,
            'prediction_original_scale': original_prediction,
            'feature_contributions': feature_contributions,
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_features': len(feature_names),
                'model_type': type(model).__name__,
                'background_samples': X_background.shape[0],
                'has_original_scale_values': hasattr(self, 'preprocessor') and self.preprocessor is not None,
                'analysis_based_on_preprocessed_data': True,
                'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.'
            }
        }
        
        return results
    
    def create_waterfall_plot(
        self,
        sample_index: int = 0,
        figsize: Tuple[int, int] = (12, 8),
        save_plot: bool = True,
        max_display: int = 10,
        target_index: Optional[int] = None
    ) -> str:
        """
        Create SHAP waterfall plot for a specific sample.
        For multi-target models, creates plots for each target.
        
        Args:
            sample_index: Index of sample to visualize
            figsize: Figure size for the plot
            save_plot: Whether to save the plot to disk
            max_display: Maximum number of features to display
            target_index: For multi-target models, which target to plot (None = all targets)
            
        Returns:
            Path to saved plot file or "displayed"
        """
        # print("*"*100)
        # print("sample_index",sample_index)
        # print("target_index",target_index)
        # print("*"*100)
        if not SHAP_AVAILABLE:
            return "SHAP not available"
        
        if self.shap_values is None:
            logger.error("No SHAP values available. Run analysis first.")
            return "No SHAP values"
        
        try:
            logger.info(f"Creating waterfall plot for sample {sample_index}...")
            
            # DEBUG: Print SHAP values structure
            print(f"DEBUG: SHAP values type: {type(self.shap_values)}")
            if isinstance(self.shap_values, list):
                print(f"DEBUG: SHAP values list length: {len(self.shap_values)}")
                print(f"DEBUG: SHAP values[0] shape: {self.shap_values[0].shape if hasattr(self.shap_values[0], 'shape') else 'no shape'}")
            elif hasattr(self.shap_values, 'shape'):
                print(f"DEBUG: SHAP values shape: {self.shap_values.shape}")
            
            # Handle multi-output case
            if isinstance(self.shap_values, list):
                # print("*"*100)
                # print("list")
                # print("*"*100)
                # Classification case
                if len(self.shap_values) == 2:
                    # Binary classification - use positive class
                    print("*"*100)
                    print("self.shap_values",self.shap_values)
                    print("Binary classification")
                    print("*"*100)
                    shap_values_to_plot = self.shap_values[1]
                    expected_value = self.expected_value[1] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    print(f"DEBUG: Binary classification detected")
                    print(f"DEBUG: Has model_metadata: {hasattr(self, 'model_metadata')}")
                    if hasattr(self, 'model_metadata'):
                        print(f"DEBUG: model_metadata: {self.model_metadata}")
                    
                    # Generate intelligent class label for positive class (index 1)
                    if (hasattr(self, 'model_metadata') and self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        target_column_name = self.model_metadata.get('target_column', 
                                             self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                        print(f"DEBUG: target_column_name extracted: {target_column_name}")
                        target_name = self._get_intelligent_class_label(1, target_column_name)
                        print(f"DEBUG: Generated intelligent target_name: {target_name}")
                    else:
                        target_name = self._get_target_name(1)
                        print(f"DEBUG: Generated fallback target_name: {target_name}")
                    
                    print("*"*100)
                    print("target_name",target_name)
                    print("*"*100)
                    # Single plot for binary classification
                    return self._create_single_waterfall_plot(
                        shap_values_to_plot, expected_value, sample_index, 
                        figsize, save_plot, max_display, target_name
                    )
                else:
                    # Multi-class - use first class
                    shap_values_to_plot = self.shap_values[0]
                    expected_value = self.expected_value[0] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    print(f"DEBUG: Multi-class classification detected")
                    print(f"DEBUG: Has model_metadata: {hasattr(self, 'model_metadata')}")
                    if hasattr(self, 'model_metadata'):
                        print(f"DEBUG: model_metadata: {self.model_metadata}")
                    
                    # Generate intelligent class label for first class (index 0)
                    if (hasattr(self, 'model_metadata') and self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        target_column_name = self.model_metadata.get('target_column', 
                                             self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                        print(f"DEBUG: target_column_name extracted: {target_column_name}")
                        target_name = self._get_intelligent_class_label(0, target_column_name)
                        print(f"DEBUG: Generated intelligent target_name: {target_name}")
                    else:
                        target_name = self._get_target_name(0)
                        print(f"DEBUG: Generated fallback target_name: {target_name}")
                    # print("*"*100)
                    # print("target_name",target_name)
                    # print("*"*100)
                    # Single plot for multi-class classification
                    return self._create_single_waterfall_plot(
                        shap_values_to_plot, expected_value, sample_index, 
                        figsize, save_plot, max_display, target_name
                    )
                
            elif self.shap_values.ndim == 3:
                # Check if this is binary classification: shape (n_samples, n_features, 2)
                n_samples, n_features, n_targets = self.shap_values.shape
                # print("*"*100)
                # print("ndm",3)
                # print("*"*100)
                if (n_targets == 2 and 
                    hasattr(self, 'model_metadata') and 
                    self.model_metadata and 
                    self.model_metadata.get('task_type') == 'classification'):
                    print(f"DEBUG: Binary classification with 3D SHAP values detected")
                    print("*"*100)
                    print("Binary classification")
                    print("*"*100)
                    # Binary classification - use positive class (index 1)
                    shap_values_to_plot = self.shap_values[:, :, 1]  # Select positive class
                    expected_value = self.expected_value[1] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    # Generate intelligent class label for positive class (index 1)
                    target_column_name = self.model_metadata.get('target_column', 
                                         self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                    print(f"DEBUG: target_column_name extracted: {target_column_name}")
                    target_name = self._get_intelligent_class_label(1, target_column_name)
                    print(f"DEBUG: Generated intelligent target_name: {target_name}")
                    
                    # Single plot for binary classification
                    return self._create_single_waterfall_plot(
                        shap_values_to_plot, expected_value, sample_index, 
                        figsize, save_plot, max_display, target_name
                    )
                    
                # Multi-target regression case: shape (n_samples, n_features, n_targets)
                
                if target_index is not None:
                    # print("*"*100)
                    # print("target_index is not None")
                    # print("*"*100)
                    # Plot specific target
                    if target_index >= n_targets:
                        logger.error(f"Target index {target_index} out of range (max: {n_targets-1})")
                        return "Target index out of range"
                    
                    target_shap = self.shap_values[sample_index, :, target_index]
                    target_expected = self.expected_value[target_index] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    # target_name = self._get_target_name(target_index)
                    target_column_name = self.model_metadata.get('target_column', 
                                            self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                    target_name = self._get_intelligent_class_label(target_index, target_column_name)
                    # print("*"*100)
                    # print("target_name",target_name)
                    # print("*"*100)

                    return self._create_single_waterfall_plot(
                        target_shap.reshape(1, -1), target_expected, 0,
                        figsize, save_plot, max_display, target_name
                    )
                else:
                    # Check if this is multi-class classification with 3D SHAP values
                    if (hasattr(self, 'model_metadata') and 
                        self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        print(f"DEBUG: Multi-class classification with 3D SHAP values detected - {n_targets} classes")
                        
                        # Create plots for all targets (classes)
                        plot_paths = []
                        for target_idx in range(min(n_targets, 10)):  # Limit to first 6 targets
                            target_shap = self.shap_values[sample_index, :, target_idx]
                            target_expected = self.expected_value[target_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                            
                            # Generate intelligent class label for each class
                            target_column_name = self.model_metadata.get('target_column', 
                                                 self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                            target_name = self._get_intelligent_class_label(target_idx, target_column_name)
                            # print("*"*100)
                            # print("plot_target_name",target_name)
                            # print("*"*100)
                            # print(f"DEBUG: Generated intelligent target_name for class {target_idx}: {target_name}")
                            
                            plot_path = self._create_single_waterfall_plot(
                                target_shap.reshape(1, -1), target_expected, 0,
                                figsize, save_plot, max_display, target_name, sample_index
                            )
                            plot_paths.append(plot_path)
                        
                        return "; ".join(plot_paths)
                    else:
                        # Multi-target regression case
                        plot_paths = []
                        for target_idx in range(min(n_targets, 10)):  # Limit to first 6 targets
                            target_shap = self.shap_values[sample_index, :, target_idx]
                            target_expected = self.expected_value[target_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                            target_name = self._get_target_name(target_idx)
                            
                            plot_path = self._create_single_waterfall_plot(
                                target_shap.reshape(1, -1), target_expected, 0,
                                figsize, save_plot, max_display, target_name, sample_index
                            )
                            plot_paths.append(plot_path)
                        
                        return "; ".join(plot_paths)
            else:
                # Single-target regression case
                return self._create_single_waterfall_plot(
                    self.shap_values, self.expected_value, sample_index,
                    figsize, save_plot, max_display
                )
                
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {str(e)}")
            plt.close()
            return f"Error: {str(e)}"
    
    def _create_single_waterfall_plot(
        self,
        shap_values: np.ndarray,
        expected_value: float,
        sample_index: int,
        figsize: Tuple[int, int],
        save_plot: bool,
        max_display: int,
        target_name: str = "",
        original_sample_index: Optional[int] = None
    ) -> str:
        """Create a single waterfall plot for one target."""
        try:
            print(f"DEBUG: _create_single_waterfall_plot called with:")
            print(f"DEBUG:   target_name: '{target_name}'")
            print(f"DEBUG:   sample_index: {sample_index}")
            print(f"DEBUG:   original_sample_index: {original_sample_index}")
            print(f"DEBUG:   save_plot: {save_plot}")
            
            plt.figure(figsize=figsize)
            
            if sample_index < len(shap_values):
                sample_shap = shap_values[sample_index]
                sample_data = self.X_data[original_sample_index or sample_index] if hasattr(self, 'X_data') and sample_index < len(self.X_data) else None
                
                # Use SHAP's waterfall plot
                shap.plots.waterfall(
                    shap.Explanation(
                        values=sample_shap,
                        base_values=expected_value,
                        data=sample_data,
                        feature_names=self.feature_names
                    ),
                    max_display=max_display,
                    show=False
                )
                
                # Add target name to title if provided
                if target_name:
                    plt.title(f"SHAP Waterfall Plot - Sample {original_sample_index or sample_index} - {target_name}")
                    print(f"DEBUG: Set plot title with target_name: {target_name}")
            else:
                logger.error(f"Sample index {sample_index} out of range")
                return "Index out of range"
            
            if save_plot:
                target_suffix = f"_{target_name}" if target_name else ""
                filename = f"waterfall_plot_sample_{original_sample_index or sample_index}{target_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                print(f"DEBUG: Generated filename: {filename}")
                print(f"DEBUG: target_suffix used: '{target_suffix}'")
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Waterfall plot saved to: {filepath}")
                return str(filepath)
            else:
                plt.show()
                return "displayed"
                
        except Exception as e:
            logger.error(f"Error creating single waterfall plot: {str(e)}")
            plt.close()
            return f"Error: {str(e)}"
    
    def create_force_plot(
        self,
        sample_index: int = 0,
        figsize: Tuple[int, int] = (20, 3),
        save_plot: bool = True,
        matplotlib: bool = True,
        target_index: Optional[int] = None
    ) -> str:
        """
        Create SHAP force plot for a specific sample.
        For multi-target models, creates plots for each target.
        
        Args:
            sample_index: Index of sample to visualize
            figsize: Figure size for the plot
            save_plot: Whether to save the plot to disk
            matplotlib: Whether to use matplotlib backend
            target_index: For multi-target models, which target to plot (None = all targets)
            
        Returns:
            Path to saved plot file or "displayed"
        """

        if not SHAP_AVAILABLE:
            return "SHAP not available"
        
        if self.shap_values is None:
            logger.error("No SHAP values available. Run analysis first.")
            return "No SHAP values"
        
        try:
            logger.info(f"Creating force plot for sample {sample_index}...")
            
            # Handle multi-output case
            if isinstance(self.shap_values, list):
                # Classification case
                if len(self.shap_values) == 2:
                    # Binary classification - use positive class
                    shap_values_to_plot = self.shap_values[1]
                    expected_value = self.expected_value[1] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    print(f"DEBUG FORCE: Binary classification detected")
                    
                    # Generate intelligent class label for positive class (index 1)
                    if (hasattr(self, 'model_metadata') and self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        target_column_name = self.model_metadata.get('target_column', 
                                             self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                        target_name = self._get_intelligent_class_label(1, target_column_name)
                        print(f"DEBUG FORCE: Generated intelligent target_name: {target_name}")
                    else:
                        target_name = self._get_target_name(1)
                        print(f"DEBUG FORCE: Generated fallback target_name: {target_name}")
                    
                    # Single plot for binary classification
                    return self._create_single_force_plot(
                        shap_values_to_plot, expected_value, sample_index,
                        figsize, save_plot, matplotlib, target_name
                    )
                else:
                    # Multi-class - use first class
                    shap_values_to_plot = self.shap_values[0]
                    expected_value = self.expected_value[0] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    # Generate intelligent class label for first class (index 0)
                    if (hasattr(self, 'model_metadata') and self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        target_column_name = self.model_metadata.get('target_column', 
                                             self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                        target_name = self._get_intelligent_class_label(0, target_column_name)
                    else:
                        target_name = self._get_target_name(0)
                    
                    # Single plot for multi-class classification
                    return self._create_single_force_plot(
                        shap_values_to_plot, expected_value, sample_index,
                        figsize, save_plot, matplotlib, target_name
                    )
                
            elif self.shap_values.ndim == 3:
                # Check if this is binary classification: shape (n_samples, n_features, 2)
                n_samples, n_features, n_targets = self.shap_values.shape
                
                if (n_targets == 2 and 
                    hasattr(self, 'model_metadata') and 
                    self.model_metadata and 
                    self.model_metadata.get('task_type') == 'classification'):
                    print(f"DEBUG FORCE: Binary classification with 3D SHAP values detected")
                    
                    # Binary classification - use positive class (index 1)
                    shap_values_to_plot = self.shap_values[:, :, 1]  # Select positive class
                    expected_value = self.expected_value[1] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    # Generate intelligent class label for positive class (index 1)
                    target_column_name = self.model_metadata.get('target_column', 
                                         self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                    target_name = self._get_intelligent_class_label(1, target_column_name)
                    print("*"*100)
                    print("target_name",target_name)
                    print("*"*100)
                    print(f"DEBUG FORCE: Generated intelligent target_name: {target_name}")
                    
                    # Single plot for binary classification
                    return self._create_single_force_plot(
                        shap_values_to_plot, expected_value, sample_index,
                        figsize, save_plot, matplotlib, target_name
                    )
                    
                # Multi-target regression case: shape (n_samples, n_features, n_targets)
                
                if target_index is not None:
                    # Plot specific target
                    if target_index >= n_targets:
                        logger.error(f"Target index {target_index} out of range (max: {n_targets-1})")
                        return "Target index out of range"
                    
                    target_shap = self.shap_values[sample_index, :, target_index]
                    target_expected = self.expected_value[target_index] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    # Generate intelligent class label for positive class (index 1)
                    target_column_name = self.model_metadata.get('target_column', 
                                         self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                    print(f"DEBUG: target_column_name extracted: {target_column_name}")
                    target_name = self._get_intelligent_class_label(target_index, target_column_name)
                    print("*"*100)
                    print("target_name",target_name)
                    print("*"*100)
                    print(f"DEBUG: Generated intelligent target_name: {target_name}")
                    
                    return self._create_single_force_plot(
                        target_shap.reshape(1, -1), target_expected, 0,
                        figsize, save_plot, matplotlib, target_name, sample_index
                    )
                else:
                    # Check if this is multi-class classification with 3D SHAP values
                    if (hasattr(self, 'model_metadata') and 
                        self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        print(f"DEBUG FORCE: Multi-class classification with 3D SHAP values detected - {n_targets} classes")
                        
                        # Create plots for all targets (classes)
                        plot_paths = []
                        for target_idx in range(min(n_targets, 6)):  # Limit to first 6 targets
                            target_shap = self.shap_values[sample_index, :, target_idx]
                            target_expected = self.expected_value[target_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                            
                            # Generate intelligent class label for each class
                            target_column_name = self.model_metadata.get('target_column', 
                                                 self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                            target_name = self._get_intelligent_class_label(target_idx, target_column_name)
                            print(f"DEBUG FORCE: Generated intelligent target_name for class {target_idx}: {target_name}")
                            
                            plot_path = self._create_single_force_plot(
                                target_shap.reshape(1, -1), target_expected, 0,
                                figsize, save_plot, matplotlib, target_name, sample_index
                            )
                            plot_paths.append(plot_path)
                        
                        return "; ".join(plot_paths)
                    else:
                        # Multi-target regression case
                        plot_paths = []
                        for target_idx in range(min(n_targets, 6)):  # Limit to first 6 targets
                            target_shap = self.shap_values[sample_index, :, target_idx]
                            target_expected = self.expected_value[target_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                            target_name = self._get_target_name(target_idx)
                            
                            plot_path = self._create_single_force_plot(
                                target_shap.reshape(1, -1), target_expected, 0,
                                figsize, save_plot, matplotlib, target_name, sample_index
                            )
                            plot_paths.append(plot_path)
                        
                        return "; ".join(plot_paths)
            else:
                # Single-target regression case
                return self._create_single_force_plot(
                    self.shap_values, self.expected_value, sample_index,
                    figsize, save_plot, matplotlib
                )
                
        except Exception as e:
            logger.error(f"Error creating force plot: {str(e)}")
            if matplotlib:
                plt.close()
            return f"Error: {str(e)}"
    
    def _create_single_force_plot(
        self,
        shap_values: np.ndarray,
        expected_value: float,
        sample_index: int,
        figsize: Tuple[int, int],
        save_plot: bool,
        matplotlib: bool,
        target_name: str = "",
        original_sample_index: Optional[int] = None
    ) -> str:
        """Create a single force plot for one target."""
        try:
            print(f"DEBUG FORCE: _create_single_force_plot called with:")
            print(f"DEBUG FORCE:   target_name: '{target_name}'")
            print(f"DEBUG FORCE:   sample_index: {sample_index}")
            print(f"DEBUG FORCE:   matplotlib: {matplotlib}")
            
            if sample_index < len(shap_values):
                sample_shap = shap_values[sample_index]
                sample_data = self.X_data[original_sample_index or sample_index] if hasattr(self, 'X_data') and sample_index < len(self.X_data) else None
                
                if matplotlib:
                    # Use matplotlib backend for better control
                    plt.figure(figsize=figsize)
                    
                    # Create force plot using matplotlib
                    shap.plots.force(
                        expected_value,
                        sample_shap,
                        sample_data,
                        feature_names=self.feature_names,
                        matplotlib=True,
                        show=False
                    )
                    
                    # Add target name to title if provided
                    if target_name:
                        print("*"*100)
                        print("title target_name",target_name)
                        print("*"*100)
                        plt.title(f"SHAP Force Plot - Sample {original_sample_index or sample_index} - {target_name}")
                        print(f"DEBUG FORCE: Set matplotlib plot title with target_name: {target_name}")
                    
                    if save_plot:
                        target_suffix = f"_{target_name}" if target_name else ""
                        filename = f"force_plot_sample_{original_sample_index or sample_index}{target_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        print(f"DEBUG FORCE: Generated matplotlib filename: {filename}")
                        filepath = self.output_dir / filename
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        logger.info(f"Force plot saved to: {filepath}")
                        return str(filepath)
                    else:
                        plt.show()
                        return "displayed"
                else:
                    # Use default SHAP visualization (returns HTML)
                    force_plot = shap.force_plot(
                        expected_value,
                        sample_shap,
                        sample_data,
                        feature_names=self.feature_names
                    )
                    
                    if save_plot:
                        target_suffix = f"_{target_name}" if target_name else ""
                        filename = f"force_plot_sample_{original_sample_index or sample_index}{target_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        filepath = self.output_dir / filename
                        shap.save_html(str(filepath), force_plot)
                        logger.info(f"Force plot saved to: {filepath}")
                        return str(filepath)
                    else:
                        return "displayed (HTML)"
            else:
                logger.error(f"Sample index {sample_index} out of range")
                return "Index out of range"
                
        except Exception as e:
            logger.error(f"Error creating single force plot: {str(e)}")
            if matplotlib:
                plt.close()
            return f"Error: {str(e)}"
    
    def create_decision_plot(
        self,
        sample_indices: Union[int, List[int]] = [0],
        figsize: Tuple[int, int] = (12, 8),
        save_plot: bool = True,
        max_display: int = 15,
        target_index: Optional[int] = None
    ) -> str:
        """
        Create SHAP decision plot for one or more samples.
        For multi-target models, creates plots for each target.
        
        Args:
            sample_indices: Index or list of indices of samples to visualize
            figsize: Figure size for the plot
            save_plot: Whether to save the plot to disk
            max_display: Maximum number of features to display
            target_index: For multi-target models, which target to plot (None = all targets)
            
        Returns:
            Path to saved plot file or "displayed"
        """
        if not SHAP_AVAILABLE:
            return "SHAP not available"
        
        if self.shap_values is None:
            logger.error("No SHAP values available. Run analysis first.")
            return "No SHAP values"
        
        try:
            # Ensure sample_indices is a list
            if isinstance(sample_indices, int):
                sample_indices = [sample_indices]
            
            logger.info(f"Creating decision plot for samples {sample_indices}...")
            
            # Handle multi-output case
            if isinstance(self.shap_values, list):
                # Classification case
                if len(self.shap_values) == 2:
                    # Binary classification - use positive class
                    shap_values_to_plot = self.shap_values[1]
                    expected_value = self.expected_value[1] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    # Generate intelligent class label for positive class (index 1)
                    if (hasattr(self, 'model_metadata') and self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        target_column_name = self.model_metadata.get('target_column', 
                                             self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                        target_name = self._get_intelligent_class_label(1, target_column_name)
                    else:
                        target_name = self._get_target_name(1)
                else:
                    # Multi-class - use first class
                    shap_values_to_plot = self.shap_values[0]
                    expected_value = self.expected_value[0] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    # Generate intelligent class label for first class (index 0)
                    if (hasattr(self, 'model_metadata') and self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        target_column_name = self.model_metadata.get('target_column', 
                                             self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                        target_name = self._get_intelligent_class_label(0, target_column_name)
                    else:
                        target_name = self._get_target_name(0)
                
                # Single plot for classification
                return self._create_single_decision_plot(
                    shap_values_to_plot, expected_value, sample_indices,
                    figsize, save_plot, max_display, target_name
                )
                
            elif self.shap_values.ndim == 3:
                # Check if this is binary classification: shape (n_samples, n_features, 2)
                n_samples, n_features, n_targets = self.shap_values.shape
                
                if (n_targets == 2 and 
                    hasattr(self, 'model_metadata') and 
                    self.model_metadata and 
                    self.model_metadata.get('task_type') == 'classification'):
                    print(f"DEBUG DECISION: Binary classification with 3D SHAP values detected")
                    
                    # Binary classification - use positive class (index 1)
                    shap_values_to_plot = self.shap_values[:, :, 1]  # Select positive class
                    expected_value = self.expected_value[1] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    
                    # Generate intelligent class label for positive class (index 1)
                    target_column_name = self.model_metadata.get('target_column', 
                                         self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                    target_name = self._get_intelligent_class_label(1, target_column_name)
                    print(f"DEBUG DECISION: Generated intelligent target_name: {target_name}")
                    
                    # Single plot for binary classification
                    return self._create_single_decision_plot(
                        shap_values_to_plot, expected_value, sample_indices,
                        figsize, save_plot, max_display, target_name
                    )
                    
                # Multi-target regression case: shape (n_samples, n_features, n_targets)
                
                if target_index is not None:
                    # Plot specific target
                    if target_index >= n_targets:
                        logger.error(f"Target index {target_index} out of range (max: {n_targets-1})")
                        return "Target index out of range"
                    
                    # Extract SHAP values for the specific target
                    target_shap = self.shap_values[:, :, target_index]
                    target_expected = self.expected_value[target_index] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                    # Generate intelligent class label for positive class (index 1)
                    target_column_name = self.model_metadata.get('target_column', 
                                         self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                    print(f"DEBUG: target_column_name extracted: {target_column_name}")
                    target_name = self._get_intelligent_class_label(target_index, target_column_name)
                    print(f"DEBUG: Generated intelligent target_name: {target_name}")
                    
                    return self._create_single_decision_plot(
                        target_shap, target_expected, sample_indices,
                        figsize, save_plot, max_display, target_name
                    )
                else:
                    # Check if this is multi-class classification with 3D SHAP values
                    if (hasattr(self, 'model_metadata') and 
                        self.model_metadata and 
                        self.model_metadata.get('task_type') == 'classification'):
                        print(f"DEBUG DECISION: Multi-class classification with 3D SHAP values detected - {n_targets} classes")
                        
                        # Create plots for all targets (classes)
                        plot_paths = []
                        for target_idx in range(min(n_targets, 6)):  # Limit to first 6 targets
                            target_shap = self.shap_values[:, :, target_idx]
                            target_expected = self.expected_value[target_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                            
                            # Generate intelligent class label for each class
                            target_column_name = self.model_metadata.get('target_column', 
                                                 self.model_metadata.get('target_name', [None])[0] if isinstance(self.model_metadata.get('target_name', []), list) else None)
                            target_name = self._get_intelligent_class_label(target_idx, target_column_name)
                            print(f"DEBUG DECISION: Generated intelligent target_name for class {target_idx}: {target_name}")
                            
                            plot_path = self._create_single_decision_plot(
                                target_shap, target_expected, sample_indices,
                                figsize, save_plot, max_display, target_name
                            )
                            plot_paths.append(plot_path)
                        
                        return "; ".join(plot_paths)
                    else:
                        # Multi-target regression case
                        plot_paths = []
                        for target_idx in range(min(n_targets, 6)):  # Limit to first 6 targets
                            target_shap = self.shap_values[:, :, target_idx]
                            target_expected = self.expected_value[target_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                            target_name = self._get_target_name(target_idx)
                            
                            plot_path = self._create_single_decision_plot(
                                target_shap, target_expected, sample_indices,
                                figsize, save_plot, max_display, target_name
                            )
                            plot_paths.append(plot_path)
                        
                        return "; ".join(plot_paths)
            else:
                # Single-target regression case
                return self._create_single_decision_plot(
                    self.shap_values, self.expected_value, sample_indices,
                    figsize, save_plot, max_display
                )
                
        except Exception as e:
            logger.error(f"Error creating decision plot: {str(e)}")
            plt.close()
            return f"Error: {str(e)}"
    
    def _create_single_decision_plot(
        self,
        shap_values: np.ndarray,
        expected_value: float,
        sample_indices: List[int],
        figsize: Tuple[int, int],
        save_plot: bool,
        max_display: int,
        target_name: str = ""
    ) -> str:
        """Create a single decision plot for one target."""
        try:
            # Validate sample indices
            valid_indices = [i for i in sample_indices if i < len(shap_values)]
            if not valid_indices:
                logger.error("No valid sample indices provided")
                return "No valid indices"
            
            # Extract SHAP values for selected samples
            selected_shap = shap_values[valid_indices]
            
            # Create the decision plot
            plt.figure(figsize=figsize)
            
            shap.decision_plot(
                expected_value,
                selected_shap,
                feature_names=self.feature_names,
                feature_display_range=slice(-max_display, None),
                show=False
            )
            
            # Add title
            if len(valid_indices) == 1:
                title = f'Decision Plot for Sample {valid_indices[0]}'
            else:
                title = f'Decision Plot for Samples {valid_indices}'
            
            if target_name:
                title += f' - {target_name}'
            
            plt.title(title)
            
            if save_plot:
                indices_str = "_".join(map(str, valid_indices))
                target_suffix = f"_{target_name}" if target_name else ""
                filename = f"decision_plot_samples_{indices_str}{target_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Decision plot saved to: {filepath}")
                return str(filepath)
            else:
                plt.show()
                return "displayed"
                
        except Exception as e:
            logger.error(f"Error creating single decision plot: {str(e)}")
            plt.close()
            return f"Error: {str(e)}"
    
    def create_all_plots(
        self,
        sample_index: int = 0,
        additional_samples: List[int] = [],
        figsize: Tuple[int, int] = (12, 8),
        save_plots: bool = True,
        target_index: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Create all available plots for local feature importance analysis.
        For multi-target models, creates plots for each target.
        
        Args:
            sample_index: Primary sample index for single-sample plots
            additional_samples: Additional sample indices for multi-sample plots
            figsize: Figure size for plots
            save_plots: Whether to save plots to disk
            target_index: For multi-target models, which target to plot (None = all targets)
            
        Returns:
            Dictionary with paths to all generated plots
        """
        plot_paths = {}
        
        try:
            logger.info(f"Creating all local importance plots...")
            
            # Get all samples to analyze, but filter out indices that are out of bounds
            all_samples = [sample_index] + additional_samples
            
            # Check if we have valid sample indices based on SHAP values shape
            if self.shap_values is not None and not isinstance(self.shap_values, list):
                n_samples = self.shap_values.shape[0]
                # Filter out any sample indices that are out of bounds
                valid_samples = [idx for idx in all_samples if 0 <= idx < n_samples]
                if len(valid_samples) < len(all_samples):
                    logger.warning(f"Some sample indices are out of bounds. SHAP values shape: {self.shap_values.shape}, requested indices: {all_samples}, using valid indices: {valid_samples}")
                    all_samples = valid_samples
            
            # Check if this is a multi-target model
            is_multi_target = (
                not isinstance(self.shap_values, list) and 
                self.shap_values is not None and 
                self.shap_values.ndim == 3
            )
            
            if is_multi_target and target_index is None:
                # For multi-target, create plots for all targets
                n_targets = self.shap_values.shape[2]
                target_indices = list(range(n_targets))  # Create plots for all targets
            else:
                target_indices = [target_index] if target_index is not None else [None]
            
            # Create plots for each target (or single target if not multi-target)
            for target_idx in target_indices:
                target_suffix = f"_target_{target_idx}" if target_idx is not None else ""
                
                # Create individual plots for each sample
                for i, sample_idx in enumerate(all_samples):
                    sample_suffix = f"_sample_{sample_idx}" if len(all_samples) > 1 else ""
                    
                    # Create waterfall plot for this sample
                    waterfall_path = self.create_waterfall_plot(
                        sample_index=sample_idx,
                        figsize=figsize,
                        save_plot=save_plots,
                        target_index=target_idx
                    )
                    if waterfall_path and waterfall_path not in ["SHAP not available", "No SHAP values", "Index out of range"]:
                        if ";" in waterfall_path:  # Multiple plots returned
                            for j, path in enumerate(waterfall_path.split(";")):
                                plot_paths[f'waterfall_target_{j}{sample_suffix}'] = path.strip()
                        else:
                            plot_paths[f'waterfall{target_suffix}{sample_suffix}'] = waterfall_path
                    
                    # Create force plot for this sample
                    print("*"*100)
                    print("target_idx",target_idx)
                    print("*"*100)
                    force_path = self.create_force_plot(
                        sample_index=sample_idx,
                        figsize=(20, 3),
                        save_plot=save_plots,
                        matplotlib=True,
                        target_index=target_idx
                    )
                    if force_path and force_path not in ["SHAP not available", "No SHAP values", "Index out of range"]:
                        if ";" in force_path:  # Multiple plots returned
                            for j, path in enumerate(force_path.split(";")):
                                plot_paths[f'force_target_{j}{sample_suffix}'] = path.strip()
                        else:
                            plot_paths[f'force{target_suffix}{sample_suffix}'] = force_path
                
                # Create decision plot for this target
                decision_path = self.create_decision_plot(
                    sample_indices=all_samples if len(all_samples) > 1 else sample_index,
                    figsize=figsize,
                    save_plot=save_plots,
                    target_index=target_idx
                )
                if decision_path and decision_path not in ["SHAP not available", "No SHAP values", "No valid indices"]:
                    if ";" in decision_path:  # Multiple plots returned
                        for j, path in enumerate(decision_path.split(";")):
                            plot_paths[f'decision_target_{j}'] = path.strip()
                    else:
                        plot_paths[f'decision{target_suffix}'] = decision_path
            
            logger.info(f"Created {len(plot_paths)} local importance plots for {len(all_samples)} samples")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating local importance plots: {str(e)}")
            return plot_paths
    
    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        plot_paths: Dict[str, str] = None,
        format_type: str = "html"
    ) -> str:
        """
        Generate a comprehensive local feature importance report.
        
        Args:
            analysis_results: Results from analyze_sample_importance or analyze_batch_importance
            plot_paths: Dictionary of plot paths from create_all_plots
            format_type: Report format ("html" or "json")
            
        Returns:
            Path to generated report file
        """
        try:
            logger.info(f"Generating {format_type.upper()} local importance report...")
            
            if format_type.lower() == "html":
                return self._generate_html_report(analysis_results, plot_paths)
            elif format_type.lower() == "json":
                return self._generate_json_report(analysis_results, plot_paths)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _prepare_sample_data(
        self, 
        sample_data: Union[np.ndarray, pd.DataFrame, Dict[str, float], List[float]], 
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare and validate single sample data."""
        if isinstance(sample_data, dict):
            if feature_names is None:
                feature_names = list(sample_data.keys())
            sample_array = np.array([[sample_data[name] for name in feature_names]])
        elif isinstance(sample_data, list):
            sample_array = np.array([sample_data])
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(sample_data))]
        elif isinstance(sample_data, pd.DataFrame):
            if feature_names is None:
                feature_names = sample_data.columns.tolist()
            sample_array = sample_data.values
            if sample_array.ndim == 1:
                sample_array = sample_array.reshape(1, -1)
        else:
            sample_array = np.array(sample_data)
            if sample_array.ndim == 1:
                sample_array = sample_array.reshape(1, -1)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(sample_array.shape[1])]
        
        return sample_array, feature_names
    
    def _prepare_batch_data(
        self, 
        batch_data: Union[np.ndarray, pd.DataFrame, List[Dict[str, float]]], 
        feature_names: Optional[List[str]] = None,
        max_samples: int = 10
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare and validate batch data."""
        if isinstance(batch_data, list) and len(batch_data) > 0 and isinstance(batch_data[0], dict):
            if feature_names is None:
                feature_names = list(batch_data[0].keys())
            batch_array = np.array([[sample[name] for name in feature_names] for sample in batch_data[:max_samples]])
        elif isinstance(batch_data, pd.DataFrame):
            if feature_names is None:
                feature_names = batch_data.columns.tolist()
            batch_array = batch_data.head(max_samples).values
        else:
            batch_array = np.array(batch_data)
            if batch_array.shape[0] > max_samples:
                batch_array = batch_array[:max_samples]
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(batch_array.shape[1])]
        
        return batch_array, feature_names
    
    def _prepare_background_data(
        self, 
        background_data: Union[np.ndarray, pd.DataFrame], 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare and validate background data."""
        if isinstance(background_data, pd.DataFrame):
            background_array = background_data.values
        else:
            background_array = np.array(background_data)
        
        return background_array, feature_names
    
    def _get_model_prediction(
        self, 
        model: Union[RandomForestClassifier, RandomForestRegressor], 
        X: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Get model prediction for the given data."""
        if hasattr(model, "predict_proba"):
            # Classification - return probabilities for positive class
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]  # Positive class probability
            else:
                return proba[:, 0]  # First class probability
        else:
            # Regression - return raw predictions
            return model.predict(X)
    
    def _get_top_features_across_batch(self, sample_results: List[Dict]) -> List[Dict[str, Any]]:
        """Get top features across all samples in a batch."""
        feature_importance_sum = {}
        feature_count = {}
        
        for sample in sample_results:
            for contrib in sample['feature_contributions']:
                feature = contrib['feature']
                abs_shap = abs(contrib['shap_value'])
                
                if feature not in feature_importance_sum:
                    feature_importance_sum[feature] = 0
                    feature_count[feature] = 0
                
                feature_importance_sum[feature] += abs_shap
                feature_count[feature] += 1
        
        # Calculate average importance
        avg_importance = [
            {
                'feature': feature,
                'avg_abs_shap': feature_importance_sum[feature] / feature_count[feature],
                'frequency': feature_count[feature]
            }
            for feature in feature_importance_sum
        ]
        
        # Sort by average importance
        avg_importance.sort(key=lambda x: x['avg_abs_shap'], reverse=True)
        
        return avg_importance[:10]  # Top 10 features
    
    def _generate_html_report(self, analysis_results: Dict[str, Any], plot_paths: Dict[str, str] = None) -> str:
        """Generate HTML format report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"local_importance_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        # Determine if single sample or batch analysis
        is_batch = 'batch_size' in analysis_results
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Local Feature Importance Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
                .positive {{ color: #d32f2f; }}
                .negative {{ color: #1976d2; }}
                .sample-header {{ background-color: #e3f2fd; padding: 10px; margin: 15px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Local Feature Importance Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Analysis Type: {'Batch Analysis' if is_batch else 'Single Sample Analysis'}</p>
            </div>
        """
        
        if is_batch:
            html_content += self._add_batch_analysis_section(analysis_results)
        else:
            html_content += self._add_single_sample_section(analysis_results)
        
        # Add plots section
        if plot_paths:
            html_content += self._add_local_plots_section(plot_paths)
        
        html_content += "</body></html>"
        
        # Save HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {filepath}")
        return str(filepath)
    
    def _generate_json_report(self, analysis_results: Dict[str, Any], plot_paths: Dict[str, str] = None) -> str:
        """Generate JSON format report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"local_importance_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'analysis_type': 'batch' if 'batch_size' in analysis_results else 'single_sample'
            },
            'analysis_results': analysis_results,
            'plot_paths': plot_paths or {}
        }
        
        # Save JSON file
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved to: {filepath}")
        return str(filepath)
    
    def _add_single_sample_section(self, results: Dict[str, Any]) -> str:
        """Add single sample analysis section to HTML report."""
        sample_idx = results.get('sample_index', 0)
        
        # Check if this is a multi-target result
        if results.get('is_multi_target', False):
            html = f"""
            <div class="section">
                <h2>Single Sample Analysis - Multi-Target Model</h2>
                <div class="metric">Sample Index: {sample_idx}</div>
                
                <div style="background-color: #fff3e0; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <h4>🎯 Multi-Target Analysis</h4>
                    <p>This sample has been analyzed for multiple target variables. Results are shown separately for each target.</p>
                </div>
            """
            
            # Show overall feature importance for this sample
            if 'overall_feature_importance' in results:
                html += """
                <h3>Overall Feature Importance (Across All Targets)</h3>
                <table>
                    <tr><th>Rank</th><th>Feature</th><th>Total Abs SHAP</th><th>Avg SHAP</th></tr>
                """
                for contrib in results['overall_feature_importance'][:10]:
                    html += f"""
                        <tr>
                            <td>{contrib['overall_rank']}</td>
                            <td>{contrib['feature']}</td>
                            <td>{contrib['total_abs_shap']:.4f}</td>
                            <td>{contrib.get('avg_shap', 0):.4f}</td>
                        </tr>
                    """
                html += "</table>"
            
            # Show target-specific results
            if 'target_results' in results:
                html += "<h3>Target-Specific Results</h3>"
                for target_name, target_result in results['target_results'].items():
                    target_pred = target_result.get('prediction', 0)
                    target_expected = target_result.get('expected_value', 0)
                    target_contribs = target_result.get('feature_contributions', [])
                    
                    html += f"""
                    <div style="border: 2px solid #2196f3; border-radius: 8px; margin: 15px 0; padding: 15px;">
                        <h4 style="color: #2196f3; margin-top: 0;">Target: {target_name}</h4>
                        <div class="metric">Expected Value: {target_expected:.4f}</div>
                        <div class="metric">Prediction: {target_pred:.4f}</div>
                        <div class="metric">Features Analyzed: {len(target_contribs)}</div>
                        
                        <h5>Top Feature Contributions (SHAP Values)</h5>
                        <table>
                            <tr>
                                <th>Rank</th>
                                <th>Feature</th>
                                <th>Feature Value</th>
                                <th>SHAP Value</th>
                                <th>Impact</th>
                            </tr>
                    """
                    
                    for contrib in target_contribs[:10]:  # Top 10 features per target
                        shap_val = contrib['shap_value']
                        impact_class = 'positive' if shap_val > 0 else 'negative'
                        impact_text = 'Increases' if shap_val > 0 else 'Decreases'
                        
                        html += f"""
                            <tr>
                                <td>{contrib['contribution_rank']}</td>
                                <td>{contrib['feature']}</td>
                                <td>{contrib['value']:.4f}</td>
                                <td class="{impact_class}">{shap_val:+.4f}</td>
                                <td class="{impact_class}">{impact_text}</td>
                            </tr>
                        """
                    
                    html += "</table></div>"
            
            html += "</div>"
            return html
        else:
            # Single target analysis - original logic
            prediction = results.get('prediction', 0)
            expected_value = results.get('expected_value', 0)
            contributions = results.get('feature_contributions', [])
            
            # Ensure values are converted to scalars for formatting
            prediction_scalar = float(prediction) if not isinstance(prediction, str) else 0.0
            expected_value_scalar = float(expected_value) if not isinstance(expected_value, str) else 0.0
            
            html = f"""
            <div class="section">
                <h2>Single Sample Analysis</h2>
                
                <div class="metric">Sample Index: {sample_idx}</div>
                <div class="metric">Expected Value: {expected_value_scalar:.4f}</div>
                <div class="metric">Prediction: {prediction_scalar:.4f}</div>
                <div class="metric">Features Analyzed: {len(contributions)}</div>
                
                <h3>Feature Contributions (SHAP Values)</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Feature Value</th>
                        <th>SHAP Value</th>
                        <th>Impact</th>
                    </tr>
            """
            
            for contrib in contributions[:15]:  # Top 15 features
                shap_val = contrib['shap_value']
                impact_class = 'positive' if shap_val > 0 else 'negative'
                impact_text = 'Increases' if shap_val > 0 else 'Decreases'
                
                # Format value based on its type
                value_display = contrib['value']
                if isinstance(value_display, (int, float)):
                    value_display = f"{value_display:.4f}"
                else:
                    value_display = str(value_display)
                
                html += f"""
                    <tr>
                        <td>{contrib['contribution_rank']}</td>
                        <td>{contrib['feature']}</td>
                        <td>{value_display}</td>
                        <td class="{impact_class}">{shap_val:+.4f}</td>
                        <td class="{impact_class}">{impact_text}</td>
                    </tr>
                """
            
            html += "</table></div>"
            return html
    
    def _add_batch_analysis_section(self, results: Dict[str, Any]) -> str:
        """Add batch analysis section to HTML report."""
        batch_size = results.get('batch_size', 0)
        batch_summary = results.get('batch_summary', {})
        sample_results = results.get('sample_results', [])
        
        # Ensure values are converted to scalars for formatting
        mean_pred = float(batch_summary.get('mean_prediction', 0))
        pred_std = float(batch_summary.get('prediction_std', 0))
        
        html = f"""
        <div class="section">
            <h2>Batch Analysis Summary</h2>
            
            <div class="metric">Batch Size: {batch_size}</div>
            <div class="metric">Mean Prediction: {mean_pred:.4f}</div>
            <div class="metric">Prediction Std: {pred_std:.4f}</div>
            
            <h3>Top Features Across Batch</h3>
            <table>
                <tr><th>Rank</th><th>Feature</th><th>Avg Abs SHAP</th><th>Frequency</th></tr>
        """
        
        top_features = batch_summary.get('top_features_across_batch', [])
        for i, feature in enumerate(top_features, 1):
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feature['feature']}</td>
                    <td>{feature['avg_abs_shap']:.4f}</td>
                    <td>{feature['frequency']}</td>
                </tr>
            """
        
        html += "</table>"
        
        # Add individual sample summaries
        html += "<h3>Individual Sample Results</h3>"
        for i, sample in enumerate(sample_results[:5]):  # Show first 5 samples
            # Handle both single-target and multi-target sample results
            if sample.get('is_multi_target', False):
                # Multi-target sample result
                html += f"""
                <div class="sample-header">
                    <h4>Sample {sample['sample_index']} - Multi-Target Predictions</h4>
                </div>
                """
                
                # Show overall feature importance for this sample
                if 'overall_feature_importance' in sample:
                    html += """
                    <h5>Overall Feature Importance (Across All Targets)</h5>
                    <table>
                        <tr><th>Feature</th><th>Total Abs SHAP</th><th>Overall Rank</th></tr>
                    """
                    for contrib in sample['overall_feature_importance'][:5]:
                        html += f"""
                            <tr>
                                <td>{contrib['feature']}</td>
                                <td>{contrib['total_abs_shap']:.4f}</td>
                                <td>{contrib['overall_rank']}</td>
                            </tr>
                        """
                    html += "</table>"
                
                # Show target-specific results
                if 'target_results' in sample:
                    html += "<h5>Target-Specific Results</h5>"
                    for target_name, target_result in sample['target_results'].items():
                        target_pred = target_result.get('prediction', 0)
                        html += f"""
                        <div class="target-section">
                            <h6>Target: {target_name} - Prediction: {target_pred:.4f}</h6>
                            <table>
                                <tr><th>Feature</th><th>Value</th><th>SHAP Value</th></tr>
                        """
                        
                        # Show top 3 features for each target
                        for contrib in target_result.get('feature_contributions', [])[:3]:
                            shap_val = contrib['shap_value']
                            impact_class = 'positive' if shap_val > 0 else 'negative'
                            
                            # Format value for display
                            value_display = contrib['value']
                            if isinstance(value_display, (int, float)):
                                value_display = f"{value_display:.4f}"
                            else:
                                value_display = str(value_display)
                            
                            html += f"""
                                <tr>
                                    <td>{contrib['feature']}</td>
                                    <td>{value_display}</td>
                                    <td class="{impact_class}">{shap_val:+.4f}</td>
                                </tr>
                            """
                        html += "</table></div>"
            else:
                # Single-target sample result
                sample_pred = float(sample.get('prediction', 0))
                html += f"""
                <div class="sample-header">
                    <h4>Sample {sample['sample_index']} - Prediction: {sample_pred:.4f}</h4>
                </div>
                <table>
                    <tr><th>Feature</th><th>Value</th><th>SHAP Value</th></tr>
                """
                
                # Show top 5 features for each sample
                for contrib in sample.get('feature_contributions', [])[:5]:
                    shap_val = contrib['shap_value']
                    impact_class = 'positive' if shap_val > 0 else 'negative'
                    
                    # Format value for display
                    value_display = contrib['value']
                    if isinstance(value_display, (int, float)):
                        value_display = f"{value_display:.4f}"
                    else:
                        value_display = str(value_display)
                    
                    html += f"""
                        <tr>
                            <td>{contrib['feature']}</td>
                            <td>{value_display}</td>
                            <td class="{impact_class}">{shap_val:+.4f}</td>
                        </tr>
                    """
                
                html += "</table>"
        
        html += "</div>"
        return html
    
    def _add_local_plots_section(self, plot_paths: Dict[str, str]) -> str:
        """Add local plots section to HTML report with detailed explanations."""
        html = '''
        <div class="section">
            <h2>📊 Local Feature Importance Visualizations</h2>
            <p>The following charts demonstrate SHAP-based local feature importance analysis for specific samples, revealing how each feature contributes to the model's prediction for those samples.</p>
        '''
        
        # Check if we have multi-target plots
        has_multi_target = any('_target_' in plot_type for plot_type in plot_paths.keys())
        
        if has_multi_target:
            html += '''
            <div style="background-color: #fff3e0; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h4>🎯 Multi-Target Analysis</h4>
                <p>This model has multiple output targets. Charts are generated separately for each target to show target-specific feature contributions.</p>
            </div>
            '''
        
        # Define plot descriptions with mathematical explanations
        plot_descriptions = {
            'waterfall': {
                'title': '🌊 SHAP Waterfall Plot (Feature Contribution Accumulation)',
                'description': '''
                <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Mathematical Foundation</h4>
                    <p><strong>Mathematical Expression:</strong> f(x) = E[f(X)] + Σφᵢ</p>
                    <ul>
                        <li><strong>E[f(X)]</strong>: Expected value of the model on background dataset (baseline)</li>
                        <li><strong>φᵢ</strong>: Shapley value contribution of feature i for this sample</li>
                        <li><strong>f(x)</strong>: Final prediction value for this sample</li>
                        <li><strong>Accumulation Principle</strong>: Starting from baseline, progressively adding each feature's contribution</li>
                    </ul>
                    
                    <h4>🎨 Chart Interpretation</h4>
                    <ul>
                        <li><strong>Starting Point</strong>: E[f(X)] represents average prediction without any feature information</li>
                        <li><strong>Positive Contributions</strong>: Red bars indicate features that increase the prediction</li>
                        <li><strong>Negative Contributions</strong>: Blue bars indicate features that decrease the prediction</li>
                        <li><strong>Final Prediction</strong>: Result after accumulating all contributions</li>
                    </ul>
                    
                    <h4>🔍 Analytical Value</h4>
                    <ul>
                        <li><strong>Causal Chain</strong>: Visualizes complete reasoning path from baseline to prediction</li>
                        <li><strong>Contribution Quantification</strong>: Shows precise numerical contribution of each feature</li>
                        <li><strong>Decision Explanation</strong>: Provides clear logical explanation for model predictions</li>
                        <li><strong>Anomaly Diagnosis</strong>: Identifies key features causing unusual predictions</li>
                    </ul>
                </div>
                '''
            },
            'force': {
                'title': '⚡ SHAP Force Plot (Feature Force Visualization)',
                'description': '''
                <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Physics Analogy</h4>
                    <p><strong>Force Model:</strong> Prediction = Baseline + Σ(Feature Force × Direction)</p>
                    <ul>
                        <li><strong>Baseline</strong>: System equilibrium point (no external forces)</li>
                        <li><strong>Push Features</strong>: Positive SHAP values, pushing prediction rightward (increase)</li>
                        <li><strong>Pull Features</strong>: Negative SHAP values, pulling prediction leftward (decrease)</li>
                        <li><strong>Force Balance</strong>: Vector sum of all forces determines final prediction position</li>
                    </ul>
                    
                    <h4>🎨 Chart Elements</h4>
                    <ul>
                        <li><strong>Arrow Direction</strong>: Points toward prediction increase (→) or decrease (←)</li>
                        <li><strong>Arrow Length</strong>: Absolute magnitude of SHAP value, indicating force strength</li>
                        <li><strong>Color Coding</strong>: Red = positive contribution, Blue = negative contribution</li>
                        <li><strong>Feature Values</strong>: Actual feature values displayed on arrows</li>
                    </ul>
                    
                    <h4>🎯 Use Cases</h4>
                    <ul>
                        <li><strong>Intuitive Explanation</strong>: Visualization format understandable by non-technical users</li>
                        <li><strong>Key Feature Identification</strong>: Quickly find features with greatest impact</li>
                        <li><strong>Feature Interactions</strong>: Observe cooperative/opposing effects between features</li>
                        <li><strong>Model Debugging</strong>: Discover unreasonable feature dependencies</li>
                    </ul>
                </div>
                '''
            },
            'decision': {
                'title': '🎯 SHAP Decision Plot (Sample Reasoning Path)',
                'description': '''
                <div style="background-color: #f3e5f5; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Decision Tree Analogy</h4>
                    <p><strong>Cumulative Decision Process:</strong> Starting from expected value, making sequential feature-based adjustments</p>
                    <ul>
                        <li><strong>Y-axis</strong>: Feature list, sorted by importance</li>
                        <li><strong>X-axis</strong>: Model output value (prediction result)</li>
                        <li><strong>Decision Path</strong>: Trajectory from baseline to final prediction</li>
                        <li><strong>Nodes</strong>: Decision points for each feature</li>
                    </ul>
                    
                    <h4>🔬 Mathematical Expression</h4>
                    <p><strong>Recurrence Formula:</strong> Outputᵢ = Outputᵢ₋₁ + φᵢ</p>
                    <ul>
                        <li><strong>Output₀</strong>: E[f(X)], expected value (starting point)</li>
                        <li><strong>φᵢ</strong>: Marginal contribution of the i-th feature</li>
                        <li><strong>Output_final</strong>: f(x), final prediction value</li>
                        <li><strong>Trajectory Line</strong>: Path connecting all decision points</li>
                    </ul>
                    
                    <h4>📈 Chart Advantages</h4>
                    <ul>
                        <li><strong>Sequential Impact</strong>: Shows cumulative effect of features in order of importance</li>
                        <li><strong>Marginal Effects</strong>: Decision adjustment magnitude at each step</li>
                        <li><strong>Decision Trajectory</strong>: Complete reasoning visualization</li>
                        <li><strong>Anomaly Detection</strong>: Identifies abnormally large single-step adjustments</li>
                    </ul>
                    
                    <h4>🔍 Multi-Sample Comparison (when applicable)</h4>
                    <p>When multiple samples are analyzed simultaneously:</p>
                    <ul>
                        <li><strong>Common Starting Point</strong>: All samples begin from the same expected value</li>
                        <li><strong>Parallel Trajectories</strong>: Each sample has independent decision path</li>
                        <li><strong>Divergence Points</strong>: Where different samples diverge on certain features</li>
                        <li><strong>Endpoint Differences</strong>: Source of final prediction differences</li>
                    </ul>
                    
                    <h4>💡 Analytical Insights</h4>
                    <ul>
                        <li><strong>Model Consistency</strong>: Verify reasoning consistency across different samples</li>
                        <li><strong>Feature Stability</strong>: Identify features performing consistently across scenarios</li>
                        <li><strong>Group Patterns</strong>: Discover sample clusters and their feature patterns</li>
                        <li><strong>Business Analysis</strong>: Understand decision logic for different customers/situations</li>
                    </ul>
                </div>
                '''
            }
        }
        
        # Group plots by target if multi-target
        if has_multi_target:
            # Group plots by target
            target_plots = {}
            single_plots = {}
            
            for plot_type, path in plot_paths.items():
                if '_target_' in plot_type:
                    # Extract target info from plot_type
                    parts = plot_type.split('_target_')
                    base_type = parts[0]
                    target_info = parts[1]  # e.g., "0_Target_A"
                    
                    if target_info not in target_plots:
                        target_plots[target_info] = {}
                    target_plots[target_info][base_type] = path
                else:
                    single_plots[plot_type] = path
            
            # Display single plots first (if any)
            for plot_type, path in single_plots.items():
                if path != "displayed" and Path(path).exists():
                    relative_path = Path(path).name
                    plot_info = plot_descriptions.get(plot_type, {
                        'title': plot_type.replace('_', ' ').title(),
                        'description': '<p>Local feature importance visualization chart</p>'
                    })
                    
                    html += f'''
                    <div class="plot">
                        <h3>{plot_info["title"]}</h3>
                        {plot_info["description"]}
                        <div style="text-align: center; margin: 20px 0; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background-color: white;">
                            <img src="{relative_path}" alt="{plot_info["title"]}" style="max-width: 100%; height: auto; border-radius: 4px;">
                            <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666; font-style: italic;">
                                Chart file: {relative_path}
                            </p>
                        </div>
                    </div>
                    '''
            
            # Display target-specific plots
            for target_info, target_plots_dict in target_plots.items():
                target_name = target_info.split('_', 1)[1] if '_' in target_info else target_info
                html += f'''
                <div style="border: 2px solid #2196f3; border-radius: 8px; margin: 20px 0; padding: 15px;">
                    <h3 style="color: #2196f3; margin-top: 0;">🎯 Target: {target_name}</h3>
                '''
                
                for base_type, path in target_plots_dict.items():
                    if path != "displayed" and Path(path).exists():
                        relative_path = Path(path).name
                        plot_info = plot_descriptions.get(base_type, {
                            'title': base_type.replace('_', ' ').title(),
                            'description': '<p>Local feature importance visualization chart</p>'
                        })
                        
                        # Modify title to include target info
                        modified_title = f"{plot_info['title']} - Target: {target_name}"
                        
                        html += f'''
                        <div class="plot">
                            <h4>{modified_title}</h4>
                            {plot_info["description"]}
                            <div style="text-align: center; margin: 20px 0; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background-color: white;">
                                <img src="{relative_path}" alt="{modified_title}" style="max-width: 100%; height: auto; border-radius: 4px;">
                                <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666; font-style: italic;">
                                    Chart file: {relative_path}
                                </p>
                            </div>
                        </div>
                        '''
                
                html += '</div>'
        else:
            # Single target - original logic
            for plot_type, path in plot_paths.items():
                if path != "displayed" and Path(path).exists():
                    relative_path = Path(path).name
                    
                    # Extract base plot type for description lookup (handle sample-specific names)
                    base_plot_type = plot_type
                    if '_sample_' in plot_type:
                        # Extract base type from names like "waterfall_sample_0", "force_sample_1"
                        base_plot_type = plot_type.split('_sample_')[0]
                        # Remove "_plot" suffix if present (e.g., "waterfall_plot" -> "waterfall")
                        if base_plot_type.endswith('_plot'):
                            base_plot_type = base_plot_type[:-5]
                    else:
                        # Handle cases like "waterfall_plot_sample_0_20250625_001221"
                        for base_type in ['waterfall', 'force', 'decision']:
                            if plot_type.startswith(base_type):
                                base_plot_type = base_type
                                break
                    
                    plot_info = plot_descriptions.get(base_plot_type, {
                        'title': plot_type.replace('_', ' ').title(),
                        'description': '<p>Local feature importance visualization chart</p>'
                    })
                    
                    # Create a more descriptive title for sample-specific plots
                    if '_sample_' in plot_type:
                        sample_info = plot_type.split('_sample_')[1].split('_')[0]  # Extract sample number
                        display_title = f"{plot_info['title']} Sample {sample_info}"
                    else:
                        display_title = plot_info["title"]
                    
                    html += f'''
                    <div class="plot">
                        <h3>{display_title}</h3>
                        {plot_info["description"]}
                        <div style="text-align: center; margin: 20px 0; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background-color: white;">
                            <img src="{relative_path}" alt="{display_title}" style="max-width: 100%; height: auto; border-radius: 4px;">
                            <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666; font-style: italic;">
                                Chart file: {relative_path}
                            </p>
                        </div>
                    </div>
                    '''
        
        html += "</div>"
        return html


# Convenience functions for easy usage
def analyze_local_importance(
    model: Union[RandomForestClassifier, RandomForestRegressor],
    sample_data: Union[np.ndarray, pd.DataFrame, Dict[str, float], List[float], List[Dict[str, float]]],
    background_data: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    output_dir: str = "local_feature_analysis",
    generate_plots: bool = True,
    generate_report: bool = True,
    model_metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze local feature importance for samples using SHAP.
    
    This is a convenience function that provides a simple interface for local feature importance analysis.
    For multi-target regression models, it will analyze each target separately and provide both
    individual target results and overall feature importance rankings.
    
    Args:
        model: Trained RandomForest model
        sample_data: Sample data to analyze (various formats supported)
        background_data: Background data for SHAP explainer
        feature_names: Names of features
        output_dir: Directory to save results
        generate_plots: Whether to generate visualization plots
        generate_report: Whether to generate analysis report
        model_metadata: Model metadata including target information for multi-target handling
        **kwargs: Additional arguments passed to the analyzer
        
    Returns:
        Dictionary with local importance analysis results
    """
    try:
        print(f"DEBUG MAIN: analyze_local_importance called with model_metadata: {model_metadata}")
        
        # Initialize analyzer
        analyzer = LocalFeatureImportanceAnalyzer(output_dir=output_dir)
        
        # Determine analysis type based on sample data
        if isinstance(sample_data, (dict, list)) and not isinstance(sample_data, np.ndarray):
            if isinstance(sample_data, dict) or (isinstance(sample_data, list) and len(sample_data) > 0 and isinstance(sample_data[0], dict)):
                # Single sample or batch of samples as dict(s)
                if isinstance(sample_data, dict):
                    # Single sample
                    results = analyzer.analyze_sample_importance(
                        model=model,
                        sample_data=sample_data,
                        background_data=background_data,
                        feature_names=feature_names,
                        sample_index=0,
                        model_metadata=model_metadata
                    )
                else:
                    # Batch of samples
                    results = analyzer.analyze_batch_importance(
                        model=model,
                        batch_data=sample_data,
                        background_data=background_data,
                        feature_names=feature_names,
                        max_samples=kwargs.get('max_samples', 10),
                        model_metadata=model_metadata
                    )
            else:
                # List of values - single sample
                results = analyzer.analyze_sample_importance(
                    model=model,
                    sample_data=sample_data,
                    background_data=background_data,
                    feature_names=feature_names,
                    sample_index=0,
                    model_metadata=model_metadata
                )
        else:
            # NumPy array or DataFrame
            sample_array = sample_data.values if hasattr(sample_data, 'values') else np.array(sample_data)
            
            if sample_array.ndim == 1:
                sample_array = sample_array.reshape(1, -1)
            
            if sample_array.shape[0] == 1:
                # Single sample
                results = analyzer.analyze_sample_importance(
                    model=model,
                    sample_data=sample_array,
                    background_data=background_data,
                    feature_names=feature_names,
                    sample_index=0,
                    model_metadata=model_metadata
                )
            else:
                # Batch analysis
                results = analyzer.analyze_batch_importance(
                    model=model,
                    batch_data=sample_array,
                    background_data=background_data,
                    feature_names=feature_names,
                    max_samples=kwargs.get('max_samples', sample_array.shape[0]),
                    model_metadata=model_metadata
                )
        
        # Generate plots if requested
        if generate_plots:
            if results.get('is_multi_target', False):
                # For multi-target results, generate plots for each target
                plot_paths = {}
                target_info = results.get('target_info', {})
                
                for target_index, target_name in target_info.items():
                    try:
                        target_plot_paths = analyzer.create_all_plots(
                            sample_index=0,
                            save_plots=True,
                            target_index=int(target_index)
                        )
                        # Add target suffix to plot names
                        for plot_type, path in target_plot_paths.items():
                            plot_key = f"{plot_type}_target_{target_index}_{target_name}"
                            plot_paths[plot_key] = path
                    except Exception as e:
                        logger.warning(f"Failed to generate plots for target {target_name}: {e}")
                        
                results['plot_paths'] = plot_paths
            else:
                # Single target results
                plot_paths = analyzer.create_all_plots(
                    sample_index=0,
                    save_plots=True
                )
                results['plot_paths'] = plot_paths
        
        # Generate report if requested
        if generate_report:
            plot_paths = results.get('plot_paths', {})
            report_path = analyzer.generate_report(
                analysis_results=results,
                plot_paths=plot_paths,
                format_type="html"
            )
            results['report_path'] = report_path
        
        return results
        
    except Exception as e:
        logger.error(f"Error in local importance analysis: {str(e)}")
        raise 