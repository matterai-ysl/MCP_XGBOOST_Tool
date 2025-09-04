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
    import xgboost as xgb
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
    
    def __init__(self, output_dir: str = "local_feature_analysis",model_metadata: Optional[Dict[str, Any]] = None):
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
        self.model_metadata = model_metadata
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
    def _get_intelligent_class_label(self, class_index: int, target_column_name: Optional[str] = None) -> str:
        if self.model_metadata:
            class_to_label = self.model_metadata.get('label_mapping', {}).get('class_to_label', {})
            if isinstance(class_to_label, dict):
                class_to_label = {k: str(v) for k, v in class_to_label.items()}


                return (class_to_label.get(str(int(class_index))) or 
                        class_to_label.get(int(class_index)) or 
                        f'Class_{int(class_index)}')
        else:
            return f'Class_{class_index}'
    
    def analyze_sample_importance(
        self,
        model: Union[xgb.XGBClassifier, xgb.XGBRegressor],
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
            model: Trained XGBoost model
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
            
            # Store model metadata for intelligent labeling
            if model_metadata:
                self.model_metadata = model_metadata
                self.target_names = model_metadata.get('target_name', [])
                if isinstance(self.target_names, str):
                    self.target_names = [self.target_names]
                self.target_dimension = model_metadata.get('target_dimension', 1)
            
            # Determine task type and configuration
            task_config = self._determine_task_configuration(model, model_metadata)
            logger.info(f"Task configuration: {task_config}")
            
            # Create SHAP explainer
            if self.shap_explainer is None:
                logger.info("Creating SHAP TreeExplainer...")
                self.shap_explainer = shap.TreeExplainer(model)
                self.expected_value = self.shap_explainer.expected_value
            
            # Calculate SHAP values for the sample
            logger.info("Calculating SHAP values for sample...")
            sample_shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Get model prediction
            prediction = self._get_model_prediction(model, X_sample)
            
            # Prepare original sample data
            original_sample = original_sample_data if original_sample_data is not None else None
            
            # Process based on task type
            if task_config['task_type'] == 'classification':
                return self._process_classification_sample(
                    sample_shap_values, prediction, X_sample, X_background,
                    feature_names, model, task_config, original_sample, sample_index
                )
            else:  # regression
                return self._process_regression_sample(
                    sample_shap_values, prediction, X_sample, X_background,
                    feature_names, model, task_config, original_sample, sample_index
                )
            
        except Exception as e:
            logger.error(f"Error in local sample importance analysis: {str(e)}")
            raise
    
    def analyze_batch_importance(
        self,
        model: Union[xgb.XGBClassifier, xgb.XGBRegressor],
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
            model: Trained XGBoost model
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
            
            # Determine task type and configuration from model and metadata
            task_config = self._determine_task_configuration(model, model_metadata)
            logger.info(f"Task configuration: {task_config}")
            
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
            
            # Prepare original batch samples
            original_samples = self._prepare_original_batch_samples(original_batch_data, X_batch.shape[0])
            
            # Process based on task type
            if task_config['task_type'] == 'classification':
                return self._process_classification_batch(
                    batch_shap_values, predictions, X_batch, X_background, 
                    feature_names, model, task_config, original_samples,
                    class_predictions=predictions
                )
            else:  # regression
                return self._process_regression_batch(
                    batch_shap_values, predictions, X_batch, X_background, 
                    feature_names, model, task_config, original_samples
                )

        except Exception as e:
            logger.error(f"Error in batch importance analysis: {str(e)}")
            raise
    
    def _determine_task_configuration(
        self, 
        model: Union[xgb.XGBClassifier, xgb.XGBRegressor], 
        model_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine task configuration from model type and metadata (unified approach).
        
        Args:
            model: XGBoost model
            model_metadata: Optional metadata from training
            
        Returns:
            Dictionary with task configuration including:
            - task_type: 'classification' or 'regression'
            - is_multi_target: bool (for regression)
            - n_classes: int (for classification)
            - n_targets: int (for regression)
            - target_names: list
        """
        config = {}
        
        # Determine task type from model
        if isinstance(model, xgb.XGBClassifier):
            config['task_type'] = 'classification'
            config['is_multi_target'] = False
            config['n_targets'] = 1
            
            # Determine number of classes
            if hasattr(model, 'classes_') and model.classes_ is not None:
                config['n_classes'] = len(model.classes_)
                config['is_binary_classification'] = config['n_classes'] == 2
            else:
                # Default to binary classification if classes_ not available
                config['n_classes'] = 2
                config['is_binary_classification'] = True
                
        else:  # XGBRegressor
            config['task_type'] = 'regression'
            config['n_classes'] = 1
            config['is_binary_classification'] = False
            
            # Determine if multi-target regression - support multiple field names
            target_dimension = 1
            target_names = []
            
            if model_metadata:
                # Support multiple field name variants
                target_dimension = (
                    model_metadata.get('target_dimension', 1) or
                    model_metadata.get('n_targets', 1) or
                    model_metadata.get('num_targets', 1) or
                    1
                )
                
                # Get target names from metadata
                target_names = model_metadata.get('target_names', [])
                if not target_names:
                    target_names = model_metadata.get('target_name', [])
                if isinstance(target_names, str):
                    target_names = [target_names]
                    
            config['n_targets'] = target_dimension
            config['is_multi_target'] = target_dimension > 1
            config['target_names'] = target_names if target_names else [f"Target_{i}" for i in range(target_dimension)]
            
        self.task_config = config
        return config
    
    def _prepare_original_batch_samples(
        self, 
        original_batch_data: Optional[Union[np.ndarray, pd.DataFrame, List[Dict[str, float]]]], 
        batch_size: int
    ) -> List[Any]:
        """
        Prepare original batch samples for use in analysis.
        
        Args:
            original_batch_data: Original batch data before preprocessing
            batch_size: Size of the batch
            
        Returns:
            List of original samples
        """
        original_samples = []

        if original_batch_data is not None:
            if isinstance(original_batch_data, pd.DataFrame):
                original_samples = [original_batch_data.iloc[i:i+1] for i in range(min(len(original_batch_data), batch_size))]
            elif isinstance(original_batch_data, list):
                original_samples = original_batch_data[:batch_size]
            elif isinstance(original_batch_data, np.ndarray):
                original_samples = [original_batch_data[i:i+1] for i in range(batch_size)]
            else:
                original_samples = [None] * batch_size
        else:
            original_samples = [None] * batch_size
            
        return original_samples
    
    def _process_classification_batch(
        self,
        batch_shap_values: Union[np.ndarray, List[np.ndarray]],
        predictions: Union[float, np.ndarray],
        X_batch: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        model: xgb.XGBClassifier,
        task_config: Dict[str, Any],
        original_samples: List[Any],
        class_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process batch SHAP analysis for classification tasks (unified approach).
        
        Args:
            batch_shap_values: SHAP values from explainer
            predictions: Model predictions
            X_batch: Batch data
            X_background: Background data
            feature_names: Feature names
            model: XGBoost classifier
            task_config: Task configuration
            original_samples: Original sample data
            class_predictions: probabilities of each class
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Processing classification batch: {task_config['n_classes']} classes, {X_batch.shape[0]} samples")
        
        # Store SHAP values
        self.shap_values = batch_shap_values
        
        # Extract SHAP values to analyze using unified approach
        # 选取预测概率最大的类进行分析
        analysis_predictions = np.argmax(class_predictions, axis=1)
        shap_values_to_analyze, expected_values, analysis_info = self._extract_classification_shap_values(
            batch_shap_values, analysis_predictions, task_config
        )
        
        # Create sample results
        sample_results = []
        for i in range(X_batch.shape[0]):
            # For each sample, get the probability of its predicted class
            predicted_class_idx = int(analysis_predictions[i])
            sample_prediction_probability = float(class_predictions[i, predicted_class_idx])
            predict_label = self._get_intelligent_class_label(predicted_class_idx)
            sample_result = self._create_single_target_results(
                i, expected_values[i],
                sample_prediction_probability,
                        shap_values_to_analyze[i], 
                        X_batch[i:i+1], feature_names, model, X_background,
                original_sample_data=original_samples[i],
                class_prediction_label=predict_label,
                is_batch=True
                    )
            sample_results.append(sample_result)
                
        # Calculate original scale statistics
        # Extract the predicted class probabilities for statistics
        sample_prediction_probabilities = np.array([
            class_predictions[i, int(analysis_predictions[i])] for i in range(X_batch.shape[0])
        ])
        
        original_scale_predictions, original_scale_expected_value = self._transform_to_original_scale(
            sample_prediction_probabilities, expected_values
        )

        # Create results
        results = {
            'batch_size': X_batch.shape[0],
            'n_classes': task_config['n_classes'],
            'sample_results': sample_results,
            'analysis_metadata': {
                'n_features': len(feature_names),
                'model_type': type(model).__name__,
                'background_samples': X_background.shape[0],
                'analysis_based_on_preprocessed_data': True,
                'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.',
            }
        }
        
        # Store analysis timestamp
        
        logger.info(f"Classification batch analysis completed for {X_batch.shape[0]} samples")
        return results
    
    def _extract_classification_shap_values(
        self,
        batch_shap_values: Union[np.ndarray, List[np.ndarray]],
        predictions: Union[float, np.ndarray],
        task_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Union[float, np.ndarray], Optional[Dict[str, Any]]]:
        """
        Extract SHAP values for classification analysis based on predicted classes.
        
        Args:
            batch_shap_values: Raw SHAP values from explainer
            predictions: Model predictions (class indices from argmax)
            task_config: Task configuration
            
        Returns:
            Tuple of:
            - shap_values_to_analyze: SHAP values for each sample's predicted class
            - expected_values: Expected values for each sample's predicted class
            - analysis_info: Analysis metadata
        """
        # Convert list format to numpy array for unified processing
        if isinstance(batch_shap_values, list):
            # Convert list to 3D array: (n_samples, n_features, n_classes)
            batch_shap_values = np.array(batch_shap_values).transpose(1, 2, 0)
            logger.info(f"Converted list SHAP values to 3D array: {batch_shap_values.shape}")
        
        # Handle 3D numpy array (n_samples, n_features, n_classes)
        if isinstance(batch_shap_values, np.ndarray) and batch_shap_values.ndim == 3:
            n_samples, n_features, n_classes = batch_shap_values.shape
            logger.info(f"Processing 3D SHAP values: {n_samples} samples, {n_features} features, {n_classes} classes")
            
            # Convert predictions to numpy array for easier processing
            predictions_array = np.array(predictions) if not isinstance(predictions, np.ndarray) else predictions
            
            # For each sample, extract SHAP values for its predicted class
            shap_values_to_analyze = np.zeros((n_samples, n_features))
            expected_values = np.zeros(n_samples)
            
            for i in range(n_samples):
                predicted_class = int(predictions_array[i])
                # Ensure predicted class is within valid range
                predicted_class = max(0, min(predicted_class, n_classes - 1))
                
                # Extract SHAP values for this sample's predicted class
                shap_values_to_analyze[i] = batch_shap_values[i, :, predicted_class]
                
                # Extract expected value for this class (no averaging)
                if isinstance(self.expected_value, np.ndarray) and len(self.expected_value) > predicted_class:
                    expected_values[i] = self.expected_value[predicted_class]
                else:
                    expected_values[i] = float(self.expected_value)
            
            # Create detailed analysis info
            unique_classes, class_counts = np.unique(predictions_array, return_counts=True)
            # prediction_stats = {
            #     'total_samples': len(predictions_array),
            #     'predicted_classes': predictions_array.tolist(),
            #     'unique_predicted_classes': unique_classes.tolist(),
            #     'class_counts': dict(zip(unique_classes.tolist(), class_counts.tolist())),
            #     'analysis_approach': 'individual_class_per_sample',
            #     'expected_values_per_class': {
            #         str(cls): float(self.expected_value[cls]) if isinstance(self.expected_value, np.ndarray) else float(self.expected_value)
            #         for cls in unique_classes
            #     }
            # }
            
            logger.info(f"Analyzing individual predicted classes for each sample: {dict(zip(unique_classes, class_counts))}")
            
            # analysis_info = {
            #     'analysis_strategy': 'individual_predicted_class_analysis',
            #     'total_classes': n_classes,
            #     'prediction_stats': prediction_stats,
            #     'analysis_note': f'Classification: each sample analyzed for its own predicted class with corresponding expected value'
            # }
            analysis_info = {'analysis_note': f'Classification: each sample analyzed for its own predicted class with corresponding expected value'}
        else:
            # Fallback for single output or unexpected formats
            shap_values_to_analyze = batch_shap_values
            expected_values = float(self.expected_value)
            analysis_info = {
                'analysis_strategy': 'single_output',
                'analysis_note': 'Single output classification analysis'
            }
        
        return shap_values_to_analyze, expected_values, analysis_info
    
    def _process_regression_batch(
        self,
        batch_shap_values: np.ndarray,
        predictions: Union[float, np.ndarray],
        X_batch: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        model: xgb.XGBRegressor,
        task_config: Dict[str, Any],
        original_samples: List[Any]
    ) -> Dict[str, Any]:
        """
        Process batch SHAP analysis for regression tasks (single and multi-target).
        
        Args:
            batch_shap_values: SHAP values from explainer
            predictions: Model predictions
            X_batch: Batch data
            X_background: Background data
            feature_names: Feature names
            model: XGBoost regressor
            task_config: Task configuration
            original_samples: Original sample data
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Processing regression batch: {task_config['n_targets']} targets, {X_batch.shape[0]} samples")
        
        # Store SHAP values
        self.shap_values = batch_shap_values
        
        if task_config['is_multi_target']:
            return self._process_multi_target_regression(
                batch_shap_values, predictions, X_batch, X_background,
                feature_names, model, task_config, original_samples
            )
        else:
            return self._process_single_target_regression(
                batch_shap_values, predictions, X_batch, X_background,
                feature_names, model, task_config, original_samples
            )
    
    def _process_single_target_regression(
        self,
        batch_shap_values: np.ndarray,
        predictions: Union[float, np.ndarray],
        X_batch: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        model: xgb.XGBRegressor,
        task_config: Dict[str, Any],
        original_samples: List[Any]
    ) -> Dict[str, Any]:
        """
        Process single-target regression batch.
        
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Processing single-target regression: {X_batch.shape[0]} samples")
        
        # For single-target regression, SHAP values should be 2D: (n_samples, n_features)
        shap_values_to_analyze = batch_shap_values
        expected_value = float(self.expected_value)
        
        # Create sample results
        sample_results = []
        for i in range(X_batch.shape[0]):
            sample_result = self._create_single_target_results(
                i, expected_value,
                float(predictions[i]) if hasattr(predictions, '__len__') else float(predictions),
                shap_values_to_analyze[i],
                X_batch[i:i+1], feature_names, model, X_background,
                target_name=task_config['target_names'][0] if task_config['target_names'] else None,
                original_sample_data=original_samples[i]
            )
            sample_results.append(sample_result)
        
        # Calculate original scale statistics
        original_scale_predictions, original_scale_expected_value = self._transform_to_original_scale(
            predictions, expected_value
        )
        
        # Create results
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
        
        logger.info(f"Single-target regression batch analysis completed for {X_batch.shape[0]} samples")
        return results
    
    def _transform_to_original_scale(
        self, 
        predictions: Union[float, np.ndarray], 
        expected_values: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Transform predictions and expected values to original scale if preprocessor is available.
        Now supports per-sample expected values for classification tasks.
        
        Args:
            predictions: Model predictions (probabilities for classification)
            expected_values: Expected values (can be per-sample for classification)
            
        Returns:
            Tuple of transformed predictions and expected values
        """
        if not hasattr(self, 'preprocessor') or self.preprocessor is None:
            return predictions, expected_values
            
        try:
            # For classification, transform each prediction independently
            if isinstance(predictions, np.ndarray) and predictions.ndim == 1:
                original_scale_predictions = np.array([
                    self.preprocessor.inverse_transform_prediction(pred)
                    for pred in predictions
                ])
                
                # Transform each expected value independently
                if isinstance(expected_values, np.ndarray) and expected_values.ndim == 1:
                    original_scale_expected_values = np.array([
                        self.preprocessor.inverse_transform_prediction(exp_val)
                        for exp_val in expected_values
                    ])
                else:
                    # Single expected value for all samples
                    original_scale_expected_values = self.preprocessor.inverse_transform_prediction(expected_values)
                    
            else:
                # Single prediction
                original_scale_predictions = self.preprocessor.inverse_transform_prediction(predictions)
                original_scale_expected_values = self.preprocessor.inverse_transform_prediction(expected_values)
                
        except Exception as e:
            logger.warning(f"Could not transform to original scale: {str(e)}")
            return predictions, expected_values
            
        return original_scale_predictions, original_scale_expected_values
    
    def _process_multi_target_regression(
        self,
        batch_shap_values: np.ndarray,
        predictions: Union[float, np.ndarray],
        X_batch: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        model: xgb.XGBRegressor,
        task_config: Dict[str, Any],
        original_samples: List[Any]
    ) -> Dict[str, Any]:
        """
        Process multi-target regression batch.
        
        Returns:
            Analysis results dictionary
        """
        n_targets = task_config['n_targets']
        target_names = task_config['target_names']
        
        logger.info(f"Processing multi-target regression: {n_targets} targets, {X_batch.shape[0]} samples")
        logger.info(f"Batch SHAP values shape: {batch_shap_values.shape}")
                
        # Determine SHAP format for multi-target
        shap_format = self._determine_multitarget_shap_format(batch_shap_values, len(feature_names), n_targets)
        logger.info(f"SHAP format detected: {shap_format}")
                
        # Initialize tracking structures
        sample_results = []
        target_batch_summaries = {}
        overall_batch_importance = {}
                
                # Initialize target summaries
        for target_idx in range(n_targets):
            target_name = target_names[target_idx] if target_idx < len(target_names) else f"Target_{target_idx + 1}"
            target_batch_summaries[target_name] = {
                        'predictions': [],
                        'sample_results': []
                    }
                
                # Process each sample
            for i in range(X_batch.shape[0]):
                    sample_target_results = {}
                    sample_overall_importance = {}
                    
            for target_idx in range(n_targets):
                target_name = target_names[target_idx] if target_idx < len(target_names) else f"Target_{target_idx + 1}"
                        
                        # Extract SHAP values for this target and sample
                target_shap = self._extract_multitarget_shap_values(
                    batch_shap_values, i, target_idx, shap_format, len(feature_names)
                )
                
                # Extract expected value and prediction for this target
                target_expected = self._extract_multitarget_expected_value(target_idx)
                target_prediction = self._extract_multitarget_prediction(predictions, i, target_idx)
                        
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
                'target_dimension': n_targets,
                'target_names': target_names,
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
            'target_dimension': n_targets,
            'target_names': target_names,
                    'sample_results': sample_results,
                    'target_batch_summaries': target_batch_summaries,
                    'batch_overall_importance': batch_overall_importance,
                    'analysis_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'n_features': len(feature_names),
                'n_targets': n_targets,
                        'model_type': type(model).__name__,
                        'background_samples': X_background.shape[0],
                'shap_format': shap_format,
                        'analysis_based_on_preprocessed_data': True,
                        'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.'
                    }
                }
                
        # Store analysis timestamp
        self.analysis_timestamp = results['analysis_metadata']['timestamp']
        
        logger.info(f"Multi-target regression batch analysis completed for {X_batch.shape[0]} samples")
        logger.info(f"Top 3 batch features: {', '.join([c['feature'] for c in batch_overall_importance[:3]])}")
                
        return results
    
    def _determine_multitarget_shap_format(
        self, 
        batch_shap_values: np.ndarray, 
        n_features: int, 
        n_targets: int
    ) -> str:
        """
        Determine the format of multi-target SHAP values.
        
        Returns:
            Format string: 'samples_features_targets' or 'samples_targets_features'
        """
        if batch_shap_values.shape[1] == n_features and batch_shap_values.shape[2] == n_targets:
            return "samples_features_targets"
        elif batch_shap_values.shape[1] == n_targets and batch_shap_values.shape[2] == n_features:
            return "samples_targets_features"
        else:
            logger.error(f"Unexpected SHAP shape: {batch_shap_values.shape} for {n_features} features and {n_targets} targets")
            raise ValueError(f"Unexpected SHAP values shape: {batch_shap_values.shape}")
    
    def _extract_multitarget_shap_values(
        self, 
        batch_shap_values: np.ndarray, 
        sample_idx: int, 
        target_idx: int, 
        shap_format: str, 
        n_features: int
    ) -> np.ndarray:
        """
        Extract SHAP values for a specific sample and target in multi-target regression.
        
        Returns:
            1D SHAP values for the specified sample and target
        """
        if batch_shap_values.ndim == 3:
            if shap_format == "samples_features_targets":
                return batch_shap_values[sample_idx, :, target_idx]
            else:  # samples_targets_features
                return batch_shap_values[sample_idx, target_idx, :]
        elif batch_shap_values.ndim == 2:
            # Fallback for single target
            return batch_shap_values[sample_idx, :]
        else:
            logger.warning(f"Unexpected batch SHAP values shape: {batch_shap_values.shape}")
            return batch_shap_values.flatten()[:n_features]
    
    def _extract_multitarget_expected_value(self, target_idx: int) -> float:
        """
        Extract expected value for a specific target in multi-target regression.
        
        Returns:
            Expected value for the target
        """
        if isinstance(self.expected_value, np.ndarray) and len(self.expected_value) > target_idx:
            return float(self.expected_value[target_idx])
        else:
            return float(self.expected_value)
    
    def _extract_multitarget_prediction(
        self, 
        predictions: Union[float, np.ndarray], 
        sample_idx: int, 
        target_idx: int
    ) -> float:
        """
        Extract prediction for a specific sample and target in multi-target regression.
        
        Returns:
            Prediction value for the specified sample and target
        """
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            return float(predictions[sample_idx, target_idx])
        elif isinstance(predictions, np.ndarray) and len(predictions) > sample_idx:
            if hasattr(predictions[sample_idx], '__len__') and len(predictions[sample_idx]) > target_idx:
                return float(predictions[sample_idx][target_idx])
            else:
                return float(predictions[sample_idx])
        else:
            return float(predictions)
    
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
        original_sample_data: Optional[Union[np.ndarray, pd.DataFrame, Dict[str, float], List[float]]] = None,
        class_prediction_label: Optional[int] = None,
        is_batch: bool = False
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
            class_prediction: Class label for classification tasks
            
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
        
        if is_batch:
            results = {
            'sample_index': sample_index or 0,
            'target_name': target_name,
            'base_value': base_value_display,
            'base_value_original_scale': original_base_value_display,
                'prediction': prediction
            }
            #分类任务显示预测的类别标签
            if class_prediction_label:
                results['prediction_class_label'] = class_prediction_label
            else:
            #回归任务显示预测的原始值
                results['prediction_original_scale'] = original_prediction

            results['feature_contributions'] = feature_contributions

        else:
            results = {
                'sample_index': sample_index or 0,
                'target_name': target_name,
                'base_value': base_value_display,
                'base_value_original_scale': original_base_value_display,
                'prediction': prediction
            }
            if class_prediction_label:
                results['prediction_class_label'] = class_prediction_label
            else:
                results['prediction_original_scale'] = original_prediction

            results['feature_contributions'] = feature_contributions

            results['analysis_metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'n_features': len(feature_names),
                'model_type': type(model).__name__,
                'background_samples': X_background.shape[0],
                'has_original_scale_values': hasattr(self, 'preprocessor') and self.preprocessor is not None,
                'analysis_based_on_preprocessed_data': True,
                'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.'
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
        Create SHAP waterfall plot for a specific sample and target.
        Simplified version that relies on external loops for multi-target/multi-sample handling.
        
        Args:
            sample_index: Index of sample to visualize
            figsize: Figure size for the plot
            save_plot: Whether to save the plot to disk
            max_display: Maximum number of features to display
            target_index: Target index for multi-target models (None for single-target)
            
        Returns:
            Path to saved plot file or error message
        """
        print(".... create_waterfall_plot",target_index)

        if not SHAP_AVAILABLE:
            return "SHAP not available"
        
        if self.shap_values is None:
            logger.error("No SHAP values available. Run analysis first.")
            return "No SHAP values"
            
        if not hasattr(self, 'task_config') or self.task_config is None:
            logger.error("No task configuration available.")
            return "No task config"
        
        try:
            logger.info(f"Creating waterfall plot for sample {sample_index}, target {target_index}...")
            
            # Use task_config to determine how to extract SHAP values
            if self.task_config['task_type'] == 'classification':
                shap_values, expected_value = self._extract_classification_waterfall_data(sample_index, target_index)
                target_name = self._get_classification_target_name(target_index)
            else:  # regression
                print(">>> 是回归")
                print(self.task_config)
                shap_values, expected_value = self._extract_regression_waterfall_data(sample_index, target_index)
                target_name = self._get_regression_target_name(target_index)
            print(".......",)
            return self._create_single_waterfall_plot(
                shap_values, expected_value, sample_index, 
                figsize, save_plot, max_display, target_name, target_index
            )
            
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {str(e)}")
            return f"Error: {str(e)}"
    
    def _extract_classification_waterfall_data(self, sample_index: int, target_index: Optional[int]) -> Tuple[np.ndarray, float]:
        """Extract SHAP values and expected value for classification waterfall plot."""
        if isinstance(self.shap_values, list):
            # List format: binary/multiclass classification
            if self.task_config.get('is_binary_classification', False):
                # Binary classification - use positive class (index 1)
                class_idx = 1
            else:
                # Multiclass - use first class or specified target
                class_idx = target_index if target_index is not None else 0
            
            shap_values = self.shap_values[class_idx][sample_index:sample_index+1]
            expected_value = self.expected_value[class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
            
        else:
            # Array format: check dimensionality
            if self.shap_values.ndim == 3:
                # 3D array format: (n_samples, n_features, n_classes)
                if self.task_config.get('is_binary_classification', False):
                    # Binary classification - use positive class (index 1)
                    class_idx = 1
                else:
                    # Multiclass - use specified target or first class
                    class_idx = target_index if target_index is not None else 0
                
                shap_values = self.shap_values[sample_index:sample_index+1, :, class_idx]
                expected_value = self.expected_value[class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
            
            elif self.shap_values.ndim == 2:
                # 2D array format: (n_samples, n_features) - for binary classification with single output
                shap_values = self.shap_values[sample_index:sample_index+1]
                expected_value = self.expected_value if not isinstance(self.expected_value, np.ndarray) else self.expected_value[0]
            
            else:
                raise ValueError(f"Unsupported SHAP values dimensionality: {self.shap_values.ndim}")
        
        return shap_values, expected_value
    
    def _extract_regression_waterfall_data(self, sample_index: int, target_index: Optional[int]) -> Tuple[np.ndarray, float]:
        """Extract SHAP values and expected value for regression waterfall plot."""
        if self.task_config.get('is_multi_target', False):
            # Multi-target regression
            if target_index is None:
                target_index = 0  # Default to first target if not specified
                
            if self.shap_values.ndim == 3:
                # Format: (n_samples, n_features, n_targets)
                shap_values = self.shap_values[sample_index:sample_index+1, :, target_index]
            elif self.shap_values.ndim == 2:
                # Format: (n_features, n_targets) - single sample case
                shap_values = self.shap_values[:, target_index:target_index+1].T  # Shape: (1, n_features)
            else:
                raise ValueError(f"Unsupported SHAP values dimensionality for multi-target: {self.shap_values.ndim}")
                
            expected_value = self.expected_value[target_index] if isinstance(self.expected_value, np.ndarray) else self.expected_value
        else:
            # Single-target regression
            if self.shap_values.ndim == 2:
                shap_values = self.shap_values[sample_index:sample_index+1]
            elif self.shap_values.ndim == 1:
                # Single sample case
                shap_values = self.shap_values.reshape(1, -1)
            else:
                raise ValueError(f"Unsupported SHAP values dimensionality for single-target: {self.shap_values.ndim}")
                
            expected_value = self.expected_value if not isinstance(self.expected_value, np.ndarray) else self.expected_value[0]
        
        return shap_values, expected_value
    
    def _get_classification_target_name(self, target_index: Optional[int]) -> str:
        """Get target name for classification task."""
        if self.task_config.get('is_binary_classification', False):
            class_idx = 1  # Positive class for binary classification
        else:
            class_idx = target_index if target_index is not None else 0
        
        if hasattr(self, 'model_metadata') and self.model_metadata:
            target_column_name = self.model_metadata.get('target_column')
            return self._get_intelligent_class_label(class_idx, target_column_name)
        else:
            return self._get_target_name(class_idx)
    
    def _get_regression_target_name(self, target_index: Optional[int]) -> str:
        """Get target name for regression task."""
        print(">>> _get_regression_target_name")
        print("target_index",target_index)
        if self.task_config.get('is_multi_target', False):
            idx = target_index if target_index is not None else 0
            target_names = self.task_config.get('target_names', [])
            return target_names[idx] if idx < len(target_names) else f"Target_{idx}"
        else:
            print(">>>>>>>>> 单目标")
            target_names = self.task_config.get('target_names')[0]
            print(">>> target_names",target_names)
            return target_names
    
    def _create_single_waterfall_plot(
        self,
        shap_values: np.ndarray,
        expected_value: float,
        sample_index: int,
        figsize: Tuple[int, int],
        save_plot: bool,
        max_display: int,
        target_name: str = "",
        target_index: Optional[int] = None,
        original_sample_index: Optional[int] = None
    ) -> str:
        """Create a single waterfall plot for one target."""
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        try:
            print(f"DEBUG: _create_single_waterfall_plot called with:")
            print(f"DEBUG:   target_name: '{target_name}'")
            print(f"DEBUG:   sample_index: {sample_index}")
            print(f"DEBUG:   original_sample_index: {original_sample_index}")
            print(f"DEBUG:   save_plot: {save_plot}")
            
            plt.figure(figsize=figsize)
            print(">>> shap_values",shap_values)
            
            sample_shap = shap_values[0]
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
                plt.title(f"SHAP Waterfall Plot - Sample {sample_index} - {target_name}")
                print(f"DEBUG: Set plot title with target_name: {target_name}")

            
            if save_plot:
                target_suffix = f"_{target_name}" if target_name else ""
                filename = f"waterfall_plot_sample_{sample_index}{target_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
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
        
        Args:
            sample_index: Index of sample to visualize
            figsize: Figure size for the plot
            save_plot: Whether to save the plot to disk
            matplotlib: Whether to use matplotlib backend
            target_index: For multi-target models, which target to plot
            
        Returns:
            Path to saved plot file or "displayed"
        """

        if not SHAP_AVAILABLE:
            return "SHAP not available"
        
        if self.shap_values is None:
            logger.error("No SHAP values available. Run analysis first.")
            return "No SHAP values"
            
        if not hasattr(self, 'task_config') or self.task_config is None:
            logger.error("No task configuration available.")
            return "No task config"
        
        try:
            logger.info(f"Creating force plot for sample {sample_index}, target {target_index}...")
            
            # Use task_config to determine how to extract SHAP values
            if self.task_config['task_type'] == 'classification':
                shap_values, expected_value = self._extract_classification_force_data(sample_index, target_index)
                target_name = self._get_classification_target_name(target_index)
            else:  # regression
                shap_values, expected_value = self._extract_regression_force_data(sample_index, target_index)
                target_name = self._get_regression_target_name(target_index)
            
            return self._create_single_force_plot(
                shap_values, expected_value, sample_index,
                figsize, save_plot, matplotlib, target_name, target_index
                )
                
        except Exception as e:
            logger.error(f"Error creating force plot: {str(e)}")
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
        target_index: Optional[int] = None,
        original_sample_index: Optional[int] = None
    ) -> str:
        """Create a single force plot for one target."""
        try:
            print(f"DEBUG: _create_single_force_plot called with:")
            print(f"DEBUG:   target_name: '{target_name}'")
            print(f"DEBUG:   sample_index: {sample_index}")
            print(f"DEBUG:   save_plot: {save_plot}")
            
            # Simplified SHAP value extraction
            sample_shap = shap_values[0]
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
                    plt.title(f"SHAP Force Plot - Sample {sample_index} - {target_name}")
                    print(f"DEBUG: Set matplotlib plot title with target_name: {target_name}")
                    
                    if save_plot:
                        target_suffix = f"_{target_name}" if target_name else ""
                        filename = f"force_plot_sample_{sample_index}{target_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        print(f"DEBUG: Generated matplotlib filename: {filename}")
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
                        filename = f"force_plot_sample_{sample_index}{target_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        filepath = self.output_dir / filename
                        shap.save_html(str(filepath), force_plot)
                        logger.info(f"Force plot saved to: {filepath}")
                        return str(filepath)
                    else:
                        return "displayed (HTML)"
                
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
        
        Args:
            sample_indices: Index or list of indices of samples to visualize
            figsize: Figure size for the plot
            save_plot: Whether to save the plot to disk
            max_display: Maximum number of features to display
            target_index: For multi-target models, which target to plot
            
        Returns:
            Path to saved plot file or "displayed"
        """
        print(">>> 开始创建决策图")
        if not SHAP_AVAILABLE:
            return "SHAP not available"
        
        if self.shap_values is None:
            return "No SHAP values"
        
        try:
            # Ensure sample_indices is a list
            if isinstance(sample_indices, int):
                sample_indices = [sample_indices]
            
            print(f"DEBUG: create_decision_plot called with:")
            print(f"DEBUG:   sample_indices: {sample_indices}")
            print(f"DEBUG:   target_index: {target_index}")
            
            # Extract data based on task type
            if self.task_config.get('task_type') == 'classification':
                print("分类任务")
                shap_values, expected_value, target_name = self._extract_classification_decision_data(sample_indices, target_index)
            else:
                shap_values, expected_value, target_name = self._extract_regression_decision_data(sample_indices, target_index)
            
            # Create single decision plot
            return self._create_single_decision_plot(
                shap_values, expected_value, sample_indices,
                    figsize, save_plot, max_display, target_name
                )
                
        except Exception as e:
            print(f"DEBUG: Error in create_decision_plot: {str(e)}")
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
            print(f"DEBUG: _create_single_decision_plot called with:")
            print(f"DEBUG:   target_name: '{target_name}'")
            print(f"DEBUG:   sample_indices: {sample_indices}")
            print(f"DEBUG:   save_plot: {save_plot}")
            
            # Validate sample indices
            valid_indices = [i for i in sample_indices if 0 <= i < len(shap_values)]
            if not valid_indices:
                return "No valid indices"
            
            # Extract SHAP values for selected samples
            selected_shap = shap_values[valid_indices]
            
            # Create the decision plot
            plt.figure(figsize=figsize)
            
            # Prepare colors for multiple samples
            sample_colors = None
            if len(valid_indices) > 1:
                print(">>> 多种样品对比")
                # Use tab10 colormap for up to 10 samples, then cycle through
                colors = plt.cm.tab10(np.linspace(0, 1, 10))
                sample_colors = [colors[i % 10] for i in range(len(valid_indices))]
            
            # Create the decision plot
            shap.decision_plot(
                expected_value,
                selected_shap,
                feature_names=self.feature_names,
                feature_display_range=slice(-max_display, None),
                show=False
            )
            
            # For multiple samples, manually set colors after plotting
            if sample_colors is not None:
                ax = plt.gca()
                lines = ax.get_lines()
                print(f"DEBUG: Found {len(lines)} lines for {len(valid_indices)} samples")
                
                # Filter lines to find the actual decision paths
                # Decision lines typically have specific characteristics
                decision_lines = []
                for line in lines:
                    # Decision lines usually have more than just a few points and are not dotted
                    xdata, ydata = line.get_data()
                    if len(xdata) > 2 and line.get_linestyle() in ['-', 'solid']:
                        decision_lines.append(line)
                
                print(f"DEBUG: Found {len(decision_lines)} decision lines")
                
                # Apply colors to decision lines
                if len(decision_lines) >= len(valid_indices):
                    # Take the last N decision lines (usually the main sample paths)
                    main_lines = decision_lines[-len(valid_indices):]
                    for i, line in enumerate(main_lines):
                        line.set_color(sample_colors[i])
                        line.set_linewidth(2.5)
                        print(f"DEBUG: Set decision line {i} to color {sample_colors[i]} for sample {valid_indices[i]}")
                elif len(decision_lines) > 0:
                    # If we have fewer decision lines than samples, apply colors to all available lines
                    for i, line in enumerate(decision_lines):
                        if i < len(sample_colors):
                            line.set_color(sample_colors[i])
                            line.set_linewidth(2.5)
                            print(f"DEBUG: Set line {i} to color {sample_colors[i]}")
                
                # Force redraw to ensure colors are applied
                plt.draw()
            
            # Add title and legend
            if len(valid_indices) == 1:
                title = f'Decision Plot for Sample {valid_indices[0]}'
            else:
                title = f'Decision Plot for Samples {valid_indices}'
                
                # Add legend for multiple samples (only if colors were applied)
                if sample_colors is not None:
                    import matplotlib.patches as mpatches
                    legend_elements = []
                    for i, sample_idx in enumerate(valid_indices):
                        color = sample_colors[i]
                        legend_elements.append(
                            mpatches.Patch(color=color, label=f'Sample {sample_idx}')
                        )
                    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            if target_name:
                title += f' - {target_name}'
            
            plt.title(title)
            
            if save_plot:
                indices_str = "_".join(map(str, valid_indices))
                target_suffix = f"_{target_name}" if target_name else ""
                filename = f"decision_plot_samples_{indices_str}{target_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print("保存成功",filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return "displayed"
                
        except Exception as e:
            print(f"DEBUG: Error in _create_single_decision_plot: {str(e)}")
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
                target_indices = [target_index] if target_index is not None else [0]
            
            print(">>> target_indices",target_indices)
            print(">>> target_index",target_index)
            print("all_samples",all_samples)
            # Create plots for each target (or single target if not multi-target)
            for target_idx in target_indices:
                print(".......", target_idx)
                target_suffix = f"_target_{target_idx}" if target_idx is not None else ""
                
                # Create individual plots for each sample
                for i, sample_idx in enumerate(all_samples):
                    sample_suffix = f"_sample_{sample_idx}" if len(all_samples) > 1 else ""
                    print("*"*100)
                    print("sample_idx",sample_idx)
                    print("*"*100)
                    
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
                
                    # Create decision plot for this sample
                    decision_path = self.create_decision_plot(
                        sample_indices=sample_idx,
                        figsize=figsize,
                        save_plot=save_plots,
                        target_index=target_idx
                    )
                    if decision_path and decision_path not in ["SHAP not available", "No SHAP values", "No valid indices"]:
                        if ";" in decision_path:  # Multiple plots returned
                            for j, path in enumerate(decision_path.split(";")):
                                plot_paths[f'decision_target_{j}{sample_suffix}'] = path.strip()
                        else:
                            plot_paths[f'decision{target_suffix}{sample_suffix}'] = decision_path
                
                # Create multi-sample decision plot for this target (if multiple samples)
                if len(all_samples) > 1:
                    decision_path_multi = self.create_decision_plot(
                        sample_indices=all_samples,
                        figsize=figsize,
                        save_plot=save_plots,
                        target_index=target_idx
                    )
                    if decision_path_multi and decision_path_multi not in ["SHAP not available", "No SHAP values", "No valid indices"]:
                        if ";" in decision_path_multi:  # Multiple plots returned
                            for j, path in enumerate(decision_path_multi.split(";")):
                                plot_paths[f'decision_multi_target_{j}'] = path.strip()
                        else:
                            plot_paths[f'decision_multi{target_suffix}'] = decision_path_multi
            
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
        model: Union[xgb.XGBClassifier, xgb.XGBRegressor], 
        X: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Get model prediction for the given data."""
        if hasattr(model, "predict_proba"):
            # Classification - return probabilities for positive class
            proba = model.predict_proba(X)
            return proba
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
            prediction_class_label = results.get('prediction_class_label')
            prediction_original_scale = results.get('prediction_original_scale')
            
            # Ensure values are converted to scalars for formatting
            prediction_scalar = float(prediction) if not isinstance(prediction, str) else 0.0
            expected_value_scalar = float(expected_value) if not isinstance(expected_value, str) else 0.0
            
            html = f"""
            <div class="section">
                <h2>Single Sample Analysis</h2>
                
                <div class="metric">Sample Index: {sample_idx}</div>
                <div class="metric">Expected Value: {expected_value_scalar:.4f}</div>
                <div class="metric">Prediction: {prediction_scalar:.4f}</div>"""
            
            # Add classification label or regression original scale value
            if prediction_class_label:
                html += f"""
                <div class="metric" style="background-color: #e8f5e8; border: 2px solid #4caf50;">
                    <strong>Predicted Class: {prediction_class_label}</strong>
                </div>"""
            elif prediction_original_scale is not None:
                html += f"""
                <div class="metric" style="background-color: #fff3e0; border: 2px solid #ff9800;">
                    <strong>Prediction (Original Scale): {float(prediction_original_scale):.4f}</strong>
                </div>"""
            
            html += f"""
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
        sample_results = results.get('sample_results', [])
        
        html = ""
        # Add individual sample summaries
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
                prediction_class_label = sample.get('prediction_class_label')
                prediction_original_scale = sample.get('prediction_original_scale')
                
                # Build the sample header with prediction information
                header_text = f"Sample {sample['sample_index']} - Prediction: {sample_pred:.4f}"
                
                # Add classification label or original scale value
                if prediction_class_label:
                    header_text += f" (Class: {prediction_class_label})"
                elif prediction_original_scale is not None:
                    header_text += f" (Original Prediction Scale: {float(prediction_original_scale):.4f})"
                
                html += f"""
                <div class="sample-header">
                    <h4>{header_text}</h4>
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
            
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196f3;">
                <h4>🎯 Why SHAP Outputs Multiple Visualization Types</h4>
                <p><strong>SHAP (SHapley Additive exPlanations)</strong> generates different types of visualizations because each type reveals different aspects of feature contributions:</p>
                <ul>
                    <li><strong>Waterfall Plot</strong>: Shows cumulative contribution path from baseline to final prediction</li>
                    <li><strong>Force Plot</strong>: Displays push-pull effects of features in a horizontal layout</li>
                    <li><strong>Decision Plot</strong>: Reveals feature interaction patterns and decision boundaries</li>
                </ul>
                <p><strong>For Classification Tasks</strong>: Each visualization considers how features contribute to predicting the <em>specific output class</em> for each sample, rather than all classes simultaneously. This provides precise, class-specific insights.</p>
                <p><strong>Mathematical Foundation</strong>: Each plot represents the same Shapley decomposition: <code>f(x) = E[f(X)] + Σφᵢ</code>, but visualizes the additive contributions in different ways to reveal different patterns.</p>
            </div>
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

    def _extract_classification_force_data(self, sample_index: int, target_index: Optional[int]) -> Tuple[np.ndarray, float]:
        """Extract SHAP values and expected value for classification force plot."""
        if isinstance(self.shap_values, list):
            # List format: binary/multiclass classification
            if self.task_config.get('is_binary_classification', False):
                # Binary classification - use positive class (index 1)
                class_idx = 1
            else:
                # Multiclass - use first class or specified target
                class_idx = target_index if target_index is not None else 0
            
            shap_values = self.shap_values[class_idx][sample_index:sample_index+1]
            expected_value = self.expected_value[class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
            
        else:
            # Array format: check dimensionality
            if self.shap_values.ndim == 3:
                # 3D array format: (n_samples, n_features, n_classes)
                if self.task_config.get('is_binary_classification', False):
                    # Binary classification - use positive class (index 1)
                    class_idx = 1
                else:
                    # Multiclass - use specified target or first class
                    class_idx = target_index if target_index is not None else 0
                
                shap_values = self.shap_values[sample_index:sample_index+1, :, class_idx]
                expected_value = self.expected_value[class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
            
            elif self.shap_values.ndim == 2:
                # 2D array format: (n_samples, n_features) - for binary classification with single output
                shap_values = self.shap_values[sample_index:sample_index+1]
                expected_value = self.expected_value if not isinstance(self.expected_value, np.ndarray) else self.expected_value[0]
            
            else:
                raise ValueError(f"Unsupported SHAP values dimensionality: {self.shap_values.ndim}")
        
        return shap_values, expected_value
    
    def _extract_regression_force_data(self, sample_index: int, target_index: Optional[int]) -> Tuple[np.ndarray, float]:
        """Extract SHAP values and expected value for regression force plot."""
        if self.task_config.get('is_multi_target', False):
            # Multi-target regression
            if target_index is None:
                target_index = 0  # Default to first target if not specified
                
            if self.shap_values.ndim == 3:
                # Format: (n_samples, n_features, n_targets)
                shap_values = self.shap_values[sample_index:sample_index+1, :, target_index]
            elif self.shap_values.ndim == 2:
                # Format: (n_features, n_targets) - single sample case
                shap_values = self.shap_values[:, target_index:target_index+1].T  # Shape: (1, n_features)
            else:
                raise ValueError(f"Unsupported SHAP values dimensionality for multi-target: {self.shap_values.ndim}")
                
            expected_value = self.expected_value[target_index] if isinstance(self.expected_value, np.ndarray) else self.expected_value
        else:
            # Single-target regression
            if self.shap_values.ndim == 3:
                # Multi-target regression
                idx = target_index if target_index is not None else 0
                shap_values = self.shap_values[sample_index:sample_index+1, :, idx]
                expected_value = self.expected_value[idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
            else:
                # Single-target regression
                shap_values = self.shap_values[sample_index:sample_index+1]
                expected_value = self.expected_value
        
        return shap_values, expected_value
    
    def _extract_classification_decision_data(self, sample_indices: List[int], target_index: Optional[int]) -> Tuple[np.ndarray, float, str]:
        """Extract SHAP values, expected value, and target name for classification decision plot."""
        if isinstance(self.shap_values, list):
            # List format: binary/multiclass classification
            if self.task_config.get('is_binary_classification', False):
                # Binary classification - use positive class (index 1)
                class_idx = 1
            else:
                # Multiclass - use first class or specified target
                class_idx = target_index if target_index is not None else 0
            
            shap_values = self.shap_values[class_idx]
            expected_value = self.expected_value[class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
            target_name = self._get_classification_target_name(class_idx)
            
        else:
            # Array format (could be 2D or 3D)
            if self.shap_values.ndim == 3:
                # 3D classification array: (n_samples, n_features, n_classes)
                if self.task_config.get('is_binary_classification', False):
                    # Binary classification - use positive class (index 1)
                    class_idx = 1
                else:
                    # Multiclass - use first class or specified target
                    class_idx = target_index if target_index is not None else 0
                    
                shap_values = self.shap_values[:, :, class_idx]
                expected_value = self.expected_value[class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                target_name = self._get_classification_target_name(class_idx)
            else:
                # 2D classification array: (n_samples, n_features) - binary classification
                shap_values = self.shap_values
                expected_value = self.expected_value
                target_name = self._get_classification_target_name(None)
        
        return shap_values, expected_value, target_name
    
    def _extract_regression_decision_data(self, sample_indices: List[int], target_index: Optional[int]) -> Tuple[np.ndarray, float, str]:
        """Extract SHAP values, expected value, and target name for regression decision plot."""
        if self.shap_values.ndim == 3:
            # Multi-target regression
            idx = target_index if target_index is not None else 0
            shap_values = self.shap_values[:, :, idx]
            expected_value = self.expected_value[idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
            target_name = self._get_regression_target_name(target_index)
        else:
            # Single-target regression
            shap_values = self.shap_values
            expected_value = self.expected_value
            target_name = self._get_regression_target_name(0)
        
        return shap_values, expected_value, target_name
    
    def _process_classification_sample(
        self,
        sample_shap_values: Union[np.ndarray, List[np.ndarray]],
        prediction: Union[float, np.ndarray],
        X_sample: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        model: xgb.XGBClassifier,
        task_config: Dict[str, Any],
        original_sample: Any,
        sample_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process single sample SHAP analysis for classification tasks.
        
        Args:
            sample_shap_values: SHAP values from explainer
            prediction: Model prediction (probabilities)
            X_sample: Sample data
            X_background: Background data
            feature_names: Feature names
            model: XGBoost classifier
            task_config: Task configuration
            original_sample: Original sample data
            sample_index: Sample index
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Processing classification sample: {task_config['n_classes']} classes")
        
        # Store SHAP values
        self.shap_values = sample_shap_values
        
        # Extract predicted class and probabilities
        if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
            class_probabilities = prediction[0]
            predicted_class_idx = int(np.argmax(class_probabilities))
            predicted_probability = float(class_probabilities[predicted_class_idx])
        else:
            # Handle single prediction case
            predicted_class_idx = 0
            predicted_probability = float(prediction) if isinstance(prediction, (int, float)) else float(prediction[0])
            class_probabilities = np.array([predicted_probability])
        
        # Extract SHAP values for the predicted class
        if isinstance(sample_shap_values, list):
            # Multi-class - use predicted class SHAP values
            target_shap_values = sample_shap_values[predicted_class_idx][0]
            expected_value = self.expected_value[predicted_class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
            target_class_idx = predicted_class_idx
        else:
            # Array format
            if sample_shap_values.ndim == 3:
                # 3D array: (n_samples, n_features, n_classes) - use predicted class SHAP values
                target_shap_values = sample_shap_values[0, :, predicted_class_idx]
                expected_value = self.expected_value[predicted_class_idx] if isinstance(self.expected_value, np.ndarray) else self.expected_value
                target_class_idx = predicted_class_idx
            else:
                # 2D array: (n_samples, n_features) - single output or binary classification
                target_shap_values = sample_shap_values[0]
                expected_value = self.expected_value if not isinstance(self.expected_value, np.ndarray) else self.expected_value[0]
                target_class_idx = predicted_class_idx
        
        # Generate intelligent class label
        target_name = self._get_intelligent_class_label(target_class_idx)
        
        # Create results using the unified method
        results = self._create_single_target_results(
            sample_index, expected_value, predicted_probability,
            target_shap_values, X_sample, feature_names, model, X_background,
            target_name=target_name,
            original_sample_data=original_sample,
            class_prediction_label=target_name
        )
        
        # Store analysis timestamp
        results['analysis_metadata']['timestamp'] = datetime.now().isoformat()
        self.analysis_timestamp = results['analysis_metadata']['timestamp']
        
        logger.info(f"Classification sample analysis completed for class: {target_name}")
        return results
    
    def _process_regression_sample(
        self,
        sample_shap_values: np.ndarray,
        prediction: Union[float, np.ndarray],
        X_sample: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        model: xgb.XGBRegressor,
        task_config: Dict[str, Any],
        original_sample: Any,
        sample_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process single sample SHAP analysis for regression tasks.
        
        Args:
            sample_shap_values: SHAP values from explainer
            prediction: Model prediction
            X_sample: Sample data
            X_background: Background data
            feature_names: Feature names
            model: XGBoost regressor
            task_config: Task configuration
            original_sample: Original sample data
            sample_index: Sample index
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Processing regression sample: {task_config['n_targets']} targets")
        
        # Store SHAP values
        self.shap_values = sample_shap_values
        
        if task_config['is_multi_target']:
            return self._process_multi_target_regression_sample(
                sample_shap_values, prediction, X_sample, X_background,
                feature_names, model, task_config, original_sample, sample_index
            )
        else:
            return self._process_single_target_regression_sample(
                sample_shap_values, prediction, X_sample, X_background,
                feature_names, model, task_config, original_sample, sample_index
            )
    
    def _process_single_target_regression_sample(
        self,
        sample_shap_values: np.ndarray,
        prediction: Union[float, np.ndarray],
        X_sample: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        model: xgb.XGBRegressor,
        task_config: Dict[str, Any],
        original_sample: Any,
        sample_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process single-target regression sample.
        
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Processing single-target regression sample")
        
        # Extract SHAP values for the sample
        if sample_shap_values.ndim > 1:
            target_shap_values = sample_shap_values[0]
        else:
            target_shap_values = sample_shap_values
        
        # Extract prediction value
        prediction_value = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
        
        # Extract expected value
        expected_value = float(self.expected_value) if not isinstance(self.expected_value, np.ndarray) else float(self.expected_value[0])
        
        # Get target name
        target_name = task_config['target_names'][0] if task_config['target_names'] else "Target"
        
        # Create results using the unified method
        results = self._create_single_target_results(
            sample_index, expected_value, prediction_value,
            target_shap_values, X_sample, feature_names, model, X_background,
            target_name=target_name,
            original_sample_data=original_sample
        )
        
        # Store analysis timestamp
        results['analysis_metadata']['timestamp'] = datetime.now().isoformat()
        self.analysis_timestamp = results['analysis_metadata']['timestamp']
        
        logger.info(f"Single-target regression sample analysis completed")
        return results
    
    def _process_multi_target_regression_sample(
        self,
        sample_shap_values: np.ndarray,
        prediction: Union[float, np.ndarray],
        X_sample: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        model: xgb.XGBRegressor,
        task_config: Dict[str, Any],
        original_sample: Any,
        sample_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process multi-target regression sample.
        
        Returns:
            Analysis results dictionary
        """
        n_targets = task_config['n_targets']
        target_names = task_config['target_names']
        
        logger.info(f"Processing multi-target regression sample: {n_targets} targets")
        
        # Determine SHAP format for multi-target
        shap_format = self._determine_multitarget_shap_format(
            sample_shap_values.reshape(1, *sample_shap_values.shape), 
            len(feature_names), n_targets
        )
        
        # Initialize tracking structures
        target_results = {}
        overall_feature_importance = {}
        
        # Process each target
        for target_idx in range(n_targets):
            target_name = target_names[target_idx] if target_idx < len(target_names) else f"Target_{target_idx + 1}"
            
            # Extract SHAP values for this target
            target_shap = self._extract_multitarget_shap_values(
                sample_shap_values.reshape(1, *sample_shap_values.shape),
                0, target_idx, shap_format, len(feature_names)
            )
            
            # Extract expected value and prediction for this target
            target_expected = self._extract_multitarget_expected_value(target_idx)
            target_prediction = self._extract_multitarget_prediction(prediction, 0, target_idx)
            
            # Create result for this target
            target_result = self._create_single_target_results(
                sample_index, target_expected, target_prediction,
                target_shap, X_sample, feature_names, model, X_background,
                target_name=target_name,
                original_sample_data=original_sample
            )
            
            target_results[target_name] = target_result
            
            # Accumulate overall feature importance
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
            'target_dimension': n_targets,
            'target_names': target_names,
            'target_results': target_results,
            'overall_feature_importance': overall_contributions,
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_features': len(feature_names),
                'n_targets': n_targets,
                'model_type': type(model).__name__,
                'background_samples': X_background.shape[0],
                'analysis_based_on_preprocessed_data': True,
                'data_preprocessing_note': 'SHAP analysis is performed on preprocessed (scaled/normalized) feature values. Original values are provided for reference.'
            }
        }
        
        # Store analysis timestamp
        self.analysis_timestamp = results['analysis_metadata']['timestamp']
        
        logger.info(f"Multi-target regression sample analysis completed")
        logger.info(f"Top 3 overall contributing features: {', '.join([c['feature'] for c in overall_contributions[:3]])}")
        
        return results
