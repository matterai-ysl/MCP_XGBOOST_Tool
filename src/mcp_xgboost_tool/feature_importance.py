"""
Feature Importance Analysis Module

This module provides comprehensive feature importance analysis capabilities including:
- Basic feature importance from XGBoost models
- Permutation importance analysis  
- SHAP value analysis
- Visualization tools
- Comprehensive reporting

Author: MCP-XGBoost-Tool
Version: 1.0.0
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import sys

# Optional SHAP import with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available. SHAP analysis will be disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib backend for non-interactive environments
plt.switch_backend('Agg')

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for XGBoost models.
    
    Provides multiple approaches to analyze feature importance:
    - Basic XGBoost feature importances (gain, weight, cover)
    - Permutation importance
    - SHAP values (if available)
    - Visualization and reporting capabilities
    """
    
    def __init__(self, output_dir: str = "feature_analysis"):
        """
        Initialize the FeatureImportanceAnalyzer.
        
        Args:
            output_dir: Directory to save analysis results and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis results storage
        self.basic_importance = None
        self.permutation_importance_results = None
        self.shap_values = None
        self.shap_explainer = None
        self.shap_results = None
        self.shap_feature_importance = None
        
        # Model and data storage
        self.model = None
        self.X_data = None
        self.y_data = None
        self.feature_names = None
        self.task_type = None
        
        logger.info(f"FeatureImportanceAnalyzer initialized with output directory: {self.output_dir}")
    
    def analyze_basic_importance(
        self,
        model: Union[xgb.XGBClassifier, xgb.XGBRegressor],
        feature_names: Optional[List[str]] = None,
        importance_type: str = "gain"
    ) -> Dict[str, Any]:
        """
        Analyze basic feature importance using XGBoost's built-in feature importance.
        
        Args:
            model: Trained XGBoost model
            feature_names: Names of features (if None, uses generic names)
            importance_type: Type of importance to extract ('gain', 'weight', 'cover')
            
        Returns:
            Dictionary with feature importance analysis results
        """
        try:
            logger.info("Starting basic feature importance analysis...")
            
            # Store model reference
            self.model = model
            
            # Get feature importances from XGBoost model
            # XGBoost supports different importance types: 'gain', 'weight', 'cover'
            importance_dict = model.get_booster().get_score(importance_type=importance_type)
            
            # Handle case where model might not have feature names set
            if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                model_feature_names = model.feature_names_in_.tolist()
            else:
                model_feature_names = list(importance_dict.keys()) if importance_dict else []
            
            # Create importance array, ensuring all features are represented
            n_features = len(model_feature_names) if model_feature_names else model.n_features_in_
            importances = np.zeros(n_features)
            
            for i, fname in enumerate(model_feature_names):
                if fname in importance_dict:
                    importances[i] = importance_dict[fname]
            
            # Generate feature names if not provided
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            elif len(feature_names) != n_features:
                raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't match number of features ({n_features})")
            
            self.feature_names = feature_names
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Calculate relative importances (percentages)
            total_importance = importance_df['importance'].sum()
            importance_df['importance_percent'] = (importance_df['importance'] / total_importance) * 100
            
            # Get top features
            top_features = importance_df.head(10)
            
            # Store results
            self.basic_importance = {
                'importance_scores': importance_df.to_dict('records'),
                'top_features': top_features.to_dict('records'),
                'feature_ranking': importance_df['feature'].tolist(),
                'total_features': n_features,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Basic importance analysis completed for {n_features} features")
            logger.info(f"Top 3 features: {', '.join(top_features['feature'].head(3).tolist())}")
            
            return self.basic_importance
            
        except Exception as e:
            logger.error(f"Error in basic importance analysis: {str(e)}")
            raise
    
    def analyze_permutation_importance(
        self,
        model: Union[xgb.XGBClassifier, xgb.XGBRegressor],
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        random_state: int = 42,
        target_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze permutation importance to measure feature impact on model performance.
        
        Args:
            model: Trained XGBoost model
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            n_repeats: Number of permutations to perform
            random_state: Random state for reproducibility
            target_index: For multi-target models, specify which target to analyze
            
        Returns:
            Dictionary with permutation importance results
        """
        try:
            logger.info("Starting permutation importance analysis...")
            
            # Store data references
            self.X_data = X
            self.y_data = y
            
            # Convert DataFrame to numpy if needed
            if isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = X.columns.tolist()
                X_array = X.values
            else:
                X_array = X
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Determine task type
            if hasattr(model, 'predict_proba'):
                self.task_type = 'classification'
                scoring = 'accuracy'
                model_wrapper = model
            else:
                self.task_type = 'regression'
                scoring = 'neg_mean_squared_error'
                
                # For multi-target regression, create a wrapper that returns only the specified target
                if target_index is not None:
                    logger.info(f"Creating model wrapper for target index: {target_index}")
                    
                    class SingleTargetWrapper:
                        def __init__(self, model, target_idx):
                            self.model = model
                            self.target_idx = target_idx
                        
                        def predict(self, X):
                            predictions = self.model.predict(X)
                            if predictions.ndim > 1 and predictions.shape[1] > 1:
                                # Multi-target case, return only the specified target
                                return predictions[:, self.target_idx]
                            else:
                                # Single target case, return as is
                                return predictions
                        
                        def __getattr__(self, name):
                            # Delegate all other attributes to the original model
                            return getattr(self.model, name)
                    
                    model_wrapper = SingleTargetWrapper(model, target_index)
                else:
                    model_wrapper = model
            
            # Perform permutation importance
            perm_importance = permutation_importance(
                model_wrapper, X_array, y,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring=scoring,
                n_jobs=1
            )
            
            # Create results dataframe
            perm_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            # Calculate confidence intervals (mean ± 2*std)
            perm_df['importance_lower'] = perm_df['importance_mean'] - 2 * perm_df['importance_std']
            perm_df['importance_upper'] = perm_df['importance_mean'] + 2 * perm_df['importance_std']
            
            # Get top features
            top_features = perm_df.head(10)
            
            # Store results
            self.permutation_importance_results = {
                'importance_scores': perm_df.to_dict('records'),
                'top_features': top_features.to_dict('records'),
                'feature_ranking': perm_df['feature'].tolist(),
                'n_repeats': n_repeats,
                'scoring_metric': scoring,
                'task_type': self.task_type,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Permutation importance analysis completed with {n_repeats} repeats")
            logger.info(f"Top 3 features: {', '.join(top_features['feature'].head(3).tolist())}")
            
            return self.permutation_importance_results
            
        except Exception as e:
            logger.error(f"Error in permutation importance analysis: {str(e)}")
            raise
    
    def analyze_shap_importance(
        self,
        model: Union[xgb.XGBClassifier, xgb.XGBRegressor],
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        max_samples: int = 100,
        create_dependency_plots: bool = True,
        create_interaction_plots: bool = True,
        dependency_features: Optional[List[str]] = None,
        interaction_features: Optional[List[Tuple[str, str]]] = None,
        model_id: Optional[str] = None,
        raw_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze SHAP values for model interpretability with enhanced dependency and interaction plots.
        
        Args:
            model: Trained XGBoost model
            X: Feature matrix (used as background data)
            feature_names: List of feature names (optional)
            max_samples: Maximum number of samples to use for SHAP analysis
            create_dependency_plots: Whether to create SHAP dependency plots for all features
            create_interaction_plots: Whether to create SHAP interaction plots for all feature pairs
            dependency_features: List of specific features for dependency plots (None = all features)
            interaction_features: List of specific feature pairs for interaction plots (None = all pairs)
            model_id: Model ID for locating raw data file
            raw_data_path: Direct path to raw data CSV file (overrides model_id)
            
        Returns:
            Dictionary containing SHAP analysis results including plot paths
        """
        try:
            logger.info("Starting enhanced SHAP analysis with comprehensive feature interactions...")
            
            # Convert to numpy array if pandas DataFrame
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                if feature_names is None:
                    feature_names = X.columns.tolist()
            else:
                X_array = X
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
            
            # Load raw display data if available
            X_display = None
            if raw_data_path or model_id:
                X_display = self._load_raw_display_data(raw_data_path, model_id, feature_names)
            print("X_display",X_display)
            print("raw_data_path",raw_data_path)
            print("model_id",model_id)
            # If no raw data available, use processed data for display
            if X_display is None:
                X_display = X_array
                logger.info("Using processed data for display (raw data not available)")
            else:
                logger.info(f"Using raw unprocessed data for display from: {raw_data_path or f'model {model_id}/raw_data.csv'}")
            
            # Sample data for SHAP analysis
            n_samples = min(max_samples, X_array.shape[0])
            if n_samples < X_array.shape[0]:
                logger.info(f"Sampling {n_samples} out of {X_array.shape[0]} samples for SHAP analysis")
                sample_indices = np.random.choice(X_array.shape[0], n_samples, replace=False)
                X_sample = X_array[sample_indices]
                X_display_sample = X_display[sample_indices] if X_display is not None else X_sample
            else:
                X_sample = X_array
                X_display_sample = X_display if X_display is not None else X_sample
            
            logger.info(f"Computing SHAP values for {X_sample.shape[0]} samples...")
            
            # Create SHAP explainer and compute values
            self.shap_explainer = shap.TreeExplainer(model)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Store display data for plotting
            self.X_data = X_array
            self.X_shap_data = X_sample
            self.X_display_data = X_display_sample  # Store raw display data
            self.feature_names = feature_names
            
            logger.info(f"SHAP values computed. Shape: {self.shap_values.shape if not isinstance(self.shap_values, list) else [sv.shape for sv in self.shap_values]}")
            
            # Interaction plots will be computed within the _create_shap_interaction_plots method
            
            # Handle multi-output case (classification)
            if isinstance(self.shap_values, list):
                # For multi-class classification, use the first class or average
                if len(self.shap_values) == 2:
                    # Binary classification - use positive class
                    shap_values_to_analyze = self.shap_values[1]
                else:
                    # Multi-class - use average absolute values across classes
                    shap_values_to_analyze = np.mean([np.abs(sv) for sv in self.shap_values], axis=0)
            else:
                # Check if this is a 3D array (samples, features, classes)
                if self.shap_values.ndim == 3:
                    if self.shap_values.shape[2] == 2:
                        # Binary classification - use positive class (index 1)
                        shap_values_to_analyze = self.shap_values[:, :, 1]
                    else:
                        # Multi-class - use average absolute values across classes
                        shap_values_to_analyze = np.mean(np.abs(self.shap_values), axis=2)
                else:
                    # Regression case or single output
                    shap_values_to_analyze = self.shap_values
            
            # Calculate feature importance from SHAP values
            feature_importance = np.mean(np.abs(shap_values_to_analyze), axis=0)
            self.shap_feature_importance = feature_importance
            # Debug: Print shapes and data types
            logger.debug(f"shap_values_to_analyze shape: {shap_values_to_analyze.shape}")
            logger.debug(f"feature_importance shape: {feature_importance.shape}")
            logger.debug(f"feature_importance dtype: {feature_importance.dtype}")
            logger.debug(f"feature_names length: {len(feature_names)}")
            
            # Ensure feature_importance is 1D
            if feature_importance.ndim > 1:
                logger.warning(f"feature_importance has {feature_importance.ndim} dimensions, flattening to 1D")
                feature_importance = feature_importance.flatten()
            
            # Ensure lengths match
            if len(feature_importance) != len(feature_names):
                logger.error(f"Length mismatch: feature_importance={len(feature_importance)}, feature_names={len(feature_names)}")
                raise ValueError(f"Length mismatch: feature_importance has {len(feature_importance)} elements, but feature_names has {len(feature_names)} elements")
            
            # Create results dataframe
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': feature_importance
            }).sort_values('shap_importance', ascending=False)
            
            # Calculate relative importances (percentages)
            total_importance = shap_df['shap_importance'].sum()
            shap_df['importance_percent'] = (shap_df['shap_importance'] / total_importance) * 100
            
            # Get top features
            top_features = shap_df.head(10)
            
            # Create dependency plots if requested
            dependency_plot_paths = []
            if create_dependency_plots:
                dependency_plot_paths = self._create_shap_dependency_plots(
                    X_sample, X_display_sample, shap_values_to_analyze, feature_names, dependency_features
                )
            
            # Create interaction plots if requested
            interaction_plot_paths = []
            if create_interaction_plots:
                interaction_plot_paths = self._create_shap_interaction_plots(
                    model, X_sample, X_display_sample, feature_names, interaction_features
                )
            
            # Store results
            shap_results = {
                'importance_scores': shap_df.to_dict('records'),
                'top_features': top_features.to_dict('records'),
                'feature_ranking': shap_df['feature'].tolist(),
                'n_samples_analyzed': X_sample.shape[0],
                'shap_values_shape': list(shap_values_to_analyze.shape),
                'analysis_timestamp': datetime.now().isoformat(),
                'shap_values': self.shap_values,
                'dependency_plots_created': create_dependency_plots,
                'interaction_plots_created': create_interaction_plots,
                'dependency_plot_paths': dependency_plot_paths,
                'interaction_plot_paths': interaction_plot_paths,
                'shap_interaction_values': create_interaction_plots,
                'raw_data_used': X_display is not None,
                'raw_data_source': raw_data_path or (f'model {model_id}/raw_data.csv' if model_id else None)
            }
            
            logger.info(f"SHAP analysis completed for {X_sample.shape[0]} samples")
            logger.info(f"Top 3 features: {', '.join(top_features['feature'].head(3).tolist())}")
            
            if create_dependency_plots:
                logger.info(f"Created {len(dependency_plot_paths)} dependency plots")
            if create_interaction_plots:
                logger.info(f"Created {len(interaction_plot_paths)} interaction plots")
            
            # Store results for report generation
            self.shap_results = shap_results
            
            return shap_results
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {str(e)}")
            raise
    
    def create_visualization(
        self,
        analysis_type: str = "all",
        figsize: Tuple[int, int] = (15, 10),
        save_plots: bool = True
    ) -> Dict[str, str]:
        """
        Create visualizations for feature importance analysis.
        
        Args:
            analysis_type: Type of analysis to visualize ("basic", "permutation", "shap", "shap_dependency", "shap_interaction", "all")
            figsize: Figure size for plots
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary with paths to saved plot files
        """
        try:
            logger.info(f"Creating visualizations for {analysis_type} analysis...")
            
            plot_paths = {}
            
            if analysis_type in ["basic", "all"] and self.basic_importance:
                plot_paths['basic'] = self._create_basic_importance_plot(figsize, save_plots)
            
            if analysis_type in ["permutation", "all"] and self.permutation_importance_results:
                plot_paths['permutation'] = self._create_permutation_importance_plot(figsize, save_plots)
            
            if analysis_type in ["shap", "all"] and self.shap_values is not None:
                shap_result = self._create_shap_plots(figsize, save_plots)
                if save_plots and "," in shap_result:
                    # Multiple SHAP plots returned
                    shap_paths = shap_result.split(",")
                    plot_paths['shap_summary'] = shap_paths[0]
                    plot_paths['shap_summary_bar'] = shap_paths[1]
                else:
                    plot_paths['shap'] = shap_result
            
            # Add SHAP dependency plots
            if analysis_type in ["shap_dependency", "all"] and self.shap_results and self.shap_results.get('dependency_plots_created', False):
                dependency_plot_paths = self.shap_results.get('dependency_plot_paths', [])
                if dependency_plot_paths:
                    plot_paths['shap_dependency'] = dependency_plot_paths
            
            # Add SHAP interaction plots
            if analysis_type in ["shap_interaction", "all"] and self.shap_results and self.shap_results.get('interaction_plots_created', False):
                interaction_plot_paths = self.shap_results.get('interaction_plot_paths', [])
                if interaction_plot_paths:
                    plot_paths['shap_interaction'] = interaction_plot_paths
            
            if analysis_type == "all" and len(plot_paths) > 1:
                plot_paths['comparison'] = self._create_comparison_plot(figsize, save_plots)
            
            logger.info(f"Created {len(plot_paths)} visualization(s)")
            # Cache generated plot paths to avoid duplicate generation in reports
            if save_plots:
                self._plots_generated = plot_paths
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def _create_basic_importance_plot(self, figsize: Tuple[int, int], save: bool) -> str:
        """Create basic feature importance plot with publication styling."""
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get top 15 features for visualization
        top_features = pd.DataFrame(self.basic_importance['importance_scores']).head(15)
        
        # Create horizontal bar plot with enhanced styling
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])

        # Enhanced styling for publication
        ax.set_xlabel('Feature importance', fontsize=32)
        ax.set_title('Basic feature importance (XGBoost)', 
                    fontsize=24,  pad=20)
        ax.tick_params(axis='both', labelsize=24)

        ax.invert_yaxis()
        # 加粗坐标轴边框（spines）
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_linewidth(3)  # 设置线宽
        # 隐藏上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add value labels with better formatting
        for i, v in enumerate(top_features['importance']):
            ax.text(v + max(top_features['importance']) * 0.01, i, f'{v:.3f}', 
                   va='center', fontsize=24)
        
        plt.tight_layout()
        
        if save:
            filename = f"basic_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close("all")
            return str(filepath)
        else:
            plt.show()
            return "displayed"
    
    def _create_permutation_importance_plot(self, figsize: Tuple[int, int], save: bool) -> str:
        """Create permutation importance plot with error bars and publication styling."""
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get top 15 features for visualization
        top_features = pd.DataFrame(self.permutation_importance_results['importance_scores']).head(15)
        
        # Create horizontal bar plot with error bars and enhanced styling
        bars = ax.barh(range(len(top_features)), top_features['importance_mean'], 
                      xerr=top_features['importance_std'], color='lightcoral', alpha=0.8,
                      capsize=3, edgecolor='black', linewidth=0.5, ecolor='darkred')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        
        # Enhanced styling for publication
        ax.set_xlabel('Permutation importance', fontsize=32)
        ax.set_title('Permutation feature importance', 
                    fontsize=24, pad=20)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.invert_yaxis()
        # 加粗坐标轴边框（spines）
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_linewidth(3)  # 设置线宽
        # 隐藏上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add value labels with better formatting
        for i, (mean, std) in enumerate(zip(top_features['importance_mean'], top_features['importance_std'])):
            ax.text(mean + std + max(top_features['importance_mean']) * 0.01, i, 
                   f'{mean:.3f}±{std:.3f}', va='center', fontsize=24)
        
        plt.tight_layout()
        
        if save:
            filename = f"permutation_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close("all")
            return str(filepath)
        else:
            plt.show()
            return "displayed"
    
    def _create_shap_plots(self, figsize: Tuple[int, int], save: bool) -> str:
        """Create SHAP-related plots."""
        if not SHAP_AVAILABLE:
            return "SHAP not available"
        
        plot_paths = []
        
        # Handle multi-output case for both plots
        if isinstance(self.shap_values, list):
            shap_values_to_plot = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
        else:
            # Check if this is a 3D array (samples, features, classes)
            if self.shap_values.ndim == 3:
                if self.shap_values.shape[2] == 2:
                    # Binary classification - use positive class (index 1)
                    shap_values_to_plot = self.shap_values[:, :, 1]
                else:
                    # Multi-class - use first class for visualization
                    shap_values_to_plot = self.shap_values[:, :, 0]
            else:
                # Regression case or single output
                shap_values_to_plot = self.shap_values
        
        # Use the sampled X data that matches SHAP values dimensions
        if hasattr(self, 'X_shap_data'):
            X_for_plot = self.X_shap_data
        else:
            X_for_plot = self.X_data[:shap_values_to_plot.shape[0]] if self.X_data is not None else None
        
        # Create SHAP summary plot (scatter plot) with publication styling
        plt.figure(figsize=figsize,dpi=300)
        plt.rcParams['font.family'] = 'Arial'
        shap.summary_plot(shap_values_to_plot, X_for_plot, 
                         feature_names=self.feature_names, show=False)
        
        # Enhanced styling for publication - get current axes
        ax = plt.gca()
        ax.set_title('SHAP summary plot\n(Feature impact on model output)', 
                    fontsize=16,  pad=20)
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        
        # Improve colorbar if present
        fig = plt.gcf()
        if len(fig.axes) > 1:  # Colorbar exists
            cbar = fig.axes[1]
            cbar.tick_params(labelsize=16)
            cbar.set_ylabel('Feature value', fontsize=16)
        
        plt.tight_layout()
        
        if save:
            filename = f"shap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_paths.append(str(filepath))
            logger.info(f"SHAP summary plot saved to: {filepath}")
        else:
            plt.show()
        
        # Create SHAP summary bar plot with publication styling
        plt.figure(figsize=figsize,dpi=300)
        shap.summary_plot(shap_values_to_plot, X_for_plot, 
                         feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        
        # Enhanced styling for publication - get current axes
        ax = plt.gca()
        ax.set_title('SHAP feature importance\n(Mean absolute SHAP values)', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Mean |SHAP value|', fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save:
            filename = f"shap_summary_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close("all")
            plot_paths.append(str(filepath))
            logger.info(f"SHAP summary bar plot saved to: {filepath}")
        else:
            plt.show()
        
        if save:
            # Return paths as a comma-separated string for backwards compatibility
            return ",".join(plot_paths)
        else:
            return "displayed"
    
    def _create_shap_dependency_plots(
        self, 
        X_sample: np.ndarray, 
        X_display_sample: np.ndarray, 
        shap_values: np.ndarray, 
        feature_names: List[str], 
        dependency_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        Create SHAP dependency plots for specified features using raw display data.
        For each feature, create dependency plots with every other feature as the interaction feature.
        
        Args:
            X_sample: Sample data used for SHAP analysis (processed)
            X_display_sample: Raw display data corresponding to X_sample
            shap_values: SHAP values array
            feature_names: List of feature names
            dependency_features: List of features to create dependency plots for
            
        Returns:
            List of file paths for created dependency plots
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not available. Cannot create dependency plots.")
            return []
        
        try:
            logger.info("Creating SHAP dependency plots with raw display data...")
            
            plot_paths = []
            
            # If no specific features specified, use all features
            if dependency_features is None:
                dependency_features = feature_names
                logger.info(f"Creating dependency plots for all {len(feature_names)} features")
            
            logger.info(f"Creating dependency plots for {len(dependency_features)} features with all possible interactions")
            
            # Create dependency plot for each feature pair using nested for loops
            for i, main_feature in enumerate(dependency_features):
                if main_feature not in feature_names:
                    logger.warning(f"Feature '{main_feature}' not found in feature names. Skipping.")
                    continue
                
                main_feature_idx = feature_names.index(main_feature)
                
                # For each main feature, create plots with all other features as interaction features
                for interaction_feature in feature_names:
                    # Skip self-interaction
                    if interaction_feature == main_feature:
                        continue
                        
                    interaction_idx = feature_names.index(interaction_feature)
                    
                    logger.info(f"Creating dependency plot for {main_feature} with interaction feature {interaction_feature}")
                    
                    try:
                        # Create figure with publication-ready styling
                        plt.figure(figsize=(12, 8),dpi=300)
                        plt.rcParams['font.family'] = 'Arial'
                        shap.dependence_plot(
                            main_feature_idx, 
                            shap_values, 
                            X_display_sample,  # Use raw display data for X-axis
                            feature_names=feature_names,
                            interaction_index=interaction_idx,  # Explicitly specify interaction feature
                            show=False
                        )
                        
                        # Enhanced styling for publication - get current axes
                        ax = plt.gca()
                        fig = plt.gcf()
                        ax.set_title(f'SHAP dependency plot: {main_feature} vs {interaction_feature}\n(Raw data)', 
                                   fontsize=16, pad=20)
                        ax.set_xlabel(ax.get_xlabel(), fontsize=24)
                        ax.set_ylabel('SHAP value', fontsize=24)
                        
                        # Improve tick labels
                        ax.tick_params(axis='both', labelsize=16)
                        
                        # Add grid for better readability
                        ax.grid(True, alpha=0.3, linestyle='--')
                        # 加粗坐标轴边框（spines）
                        for spine in ['top', 'right', 'bottom', 'left']:
                            ax.spines[spine].set_linewidth(1.5)  # 设置线宽
                        # 隐藏上边框和右边框
                        ax.spines['top'].set_visible(True)
                        ax.spines['right'].set_visible(True)
                        # Improve colorbar if present
                        if len(fig.axes) > 1:  # Colorbar exists
                            cbar = fig.axes[1]
                            cbar.tick_params(labelsize=16)
                            cbar.set_ylabel(cbar.get_ylabel(), fontsize=16)
                        
                        plt.tight_layout()
                        
                        # Save plot with high quality
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"shap_dependency_{main_feature.replace(' ', '_').replace('/', '_')}_vs_{interaction_feature.replace(' ', '_').replace('/', '_')}_{timestamp}.png"
                        filepath = self.output_dir / filename
                        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close("all")
                        
                        plot_paths.append(str(filepath))
                        logger.info(f"SHAP dependency plot saved to: {filepath}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create dependency plot for {main_feature} vs {interaction_feature}: {str(e)}")
                        plt.close()
                        continue
            
            logger.info(f"Created {len(plot_paths)} SHAP dependency plots using raw display data")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating SHAP dependency plots: {str(e)}")
            return []

    def _create_shap_interaction_plots(
        self, 
        model, 
        X_sample: np.ndarray, 
        X_display_sample: np.ndarray, 
        feature_names: List[str], 
        interaction_features: Optional[List[Tuple[str, str]]] = None
    ) -> List[str]:
        """
        Create both SHAP interaction dependence plots and heatmaps for feature pairs.
        For each feature pair, creates:
        1. An interaction dependence plot showing how the two features interact
        2. A 2D heatmap showing the interaction values distribution
        
        Args:
            model: Trained model
            X_sample: Sample data used for SHAP analysis (processed)
            X_display_sample: Raw display data corresponding to X_sample
            feature_names: List of feature names
            interaction_features: List of feature pairs to create interaction plots for
        
        Returns:
            List of file paths for created plots (includes both dependence plots and heatmaps)
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP package not available. Skipping interaction plots.")
            return []
        
        plot_paths = []
        try:
            logger.info("Creating SHAP interaction plots (publication-ready)...")
            
            # Calculate SHAP interaction values
            explainer = shap.TreeExplainer(model)
            shap_interaction_values = explainer.shap_interaction_values(X_sample)
            
            if isinstance(shap_interaction_values, list):
                shap_interaction_values = shap_interaction_values[0]
            
            # Generate feature pairs if not provided
            if interaction_features is None:
                interaction_features = []
                for i in range(len(feature_names)):
                    for j in range(len(feature_names)):
                        if i != j:
                            interaction_features.append((feature_names[i], feature_names[j]))
                logger.info(f"Generated {len(interaction_features)} feature pairs for interaction plots")
            
            # Ensure display_features is accepted – convert to DataFrame for better axis labels
            display_features = None
            try:
                import pandas as pd
                display_features = pd.DataFrame(X_display_sample, columns=feature_names)
            except Exception:
                display_features = X_display_sample
            
            # Generate both dependence plot and heatmap for each pair
            logger.info("Creating SHAP interaction plots and heatmaps...")
            for idx, (feature1, feature2) in enumerate(interaction_features):
                if feature1 not in feature_names or feature2 not in feature_names:
                    continue
                
                feature1_idx = feature_names.index(feature1)
                feature2_idx = feature_names.index(feature2)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                try:
                    # 1. Create Dependence Plot
                    plt.figure(figsize=(12, 8),dpi=300)
                    plt.rcParams['font.family'] = 'Arial'
                    shap.dependence_plot(
                        (feature1_idx, feature2_idx),
                        shap_interaction_values,
                        X_display_sample,
                        feature_names=feature_names,
                        display_features=display_features,
                        show=False
                    )
                    
                    # Enhanced styling for publication - get current axes
                    ax = plt.gca()
                    fig = plt.gcf()
                    ax.set_title(f'SHAP interaction dependence: {feature1} vs {feature2}\n(Raw data)', 
                               fontsize=16, pad=20)
                    ax.set_xlabel(f'{feature1} value', fontsize=24)
                    ax.set_ylabel(f'SHAP interaction value', fontsize=24)
                    ax.tick_params(axis='both', labelsize=16)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Improve colorbar if present
                    if len(fig.axes) > 1:
                        cbar = fig.axes[1]
                        cbar.tick_params(labelsize=16)
                        cbar.set_ylabel(f'{feature2} value', fontsize=16)
                    
                    plt.tight_layout()
                    
                    # Save dependence plot
                    dependence_plot_path = os.path.join(
                        self.output_dir,
                        f'shap_interaction_dependence_{feature1}_vs_{feature2}_{timestamp}.png'
                    )
                    plt.savefig(dependence_plot_path, dpi=300, bbox_inches='tight')
                    plt.close("all")
                    plot_paths.append(dependence_plot_path)
                    
                    # 2. Create Interaction Heatmap
                    from matplotlib import colors, cm
                    plt.figure(figsize=(10, 8), dpi=300)
                    plt.rcParams['font.family'] = 'Arial'

                    interaction_values = shap_interaction_values[:, feature1_idx, feature2_idx]

                    # Create 2D histogram/heatmap
                    feature1_values = X_display_sample[:, feature1_idx]
                    feature2_values = X_display_sample[:, feature2_idx]

                    heatmap, xedges, yedges = np.histogram2d(
                        feature1_values, 
                        feature2_values, 
                        bins=20,
                        weights=interaction_values
                    )

                    # Normalize by count to get average interaction value per bin
                    count, _, _ = np.histogram2d(feature1_values, feature2_values, bins=20)
                    heatmap = np.divide(heatmap, count, out=np.zeros_like(heatmap), where=count != 0)

                    ax = plt.gca()
                    from matplotlib.colors import LinearSegmentedColormap
                    # --- 计算对称 colorbar 范围 ---
                    finite_values = heatmap[~np.isnan(heatmap) & (heatmap != 0)]
                    if len(finite_values) > 0:
                        abs_max = max(abs(finite_values.min()), abs(finite_values.max()))
                    else:
                        abs_max = 0.5  # 默认 fallback 值

                    vmin = -abs_max
                    vmax = abs_max

                    # 自定义 colormap：蓝 -> 白 -> 红
                    #    可根据需要修改颜色渐变
                    colors_neg = plt.cm.Blues(np.linspace(1, 0.2, 100))   # 蓝色系（负值）
                    colors_pos = plt.cm.Reds(np.linspace(0.2, 1, 100))    # 红色系（正值）
                    white = np.array([1, 1, 1, 1])                        # 白色 [R, G, B, A]
                    full_colors = np.vstack((colors_neg, white, colors_pos))
                    custom_cmap = colors.ListedColormap(full_colors)
                    norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
                    
                    im = ax.imshow(
                        heatmap.T,
                        origin='lower',
                        aspect='auto',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        cmap=custom_cmap,
                        norm=norm
                    )

                    # 添加 colorbar 并设置字体大小
                    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
                    cbar.set_label('Average interaction value', fontsize=20)
                    cbar.ax.tick_params(labelsize=16)

                    # 设置其他样式
                    ax.tick_params(axis='both', labelsize=24)
                    ax.set_title(f'SHAP interaction heatmap\n{feature1} vs {feature2}', fontsize=24, pad=20)
                    ax.set_xlabel(f'{feature1} value', fontsize=32)
                    ax.set_ylabel(f'{feature2} value', fontsize=32)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    # 设置四条边框线宽为 1.5
                    for spine in ['top', 'bottom', 'left', 'right']:
                        ax.spines[spine].set_linewidth(1.5)
                    plt.tight_layout()

                    # Save heatmap
                    heatmap_path = os.path.join(
                        self.output_dir,
                        f'shap_interaction_heatmap_{feature1}_vs_{feature2}_{timestamp}.png'
                    )
                    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close("all")
                    plot_paths.append(heatmap_path)
                    
                    logger.info(f"[{idx + 1}/{len(interaction_features)}] Created plots for {feature1} vs {feature2}")
                
                except Exception as e:
                    logger.error(f"Error creating interaction plots for {feature1} vs {feature2}: {str(e)}")
                    continue
            
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error in create_shap_interaction_plots: {str(e)}")
            return []
    
    def _create_comparison_plot(self, figsize: Tuple[int, int], save: bool) -> str:
        """Create comparison plot of different importance methods with publication styling."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for comparison
        comparison_data = []
        
        if self.basic_importance:
            basic_df = pd.DataFrame(self.basic_importance['importance_scores'])
            comparison_data.append(('Basic', basic_df.set_index('feature')['importance']))
        
        if self.permutation_importance_results:
            perm_df = pd.DataFrame(self.permutation_importance_results['importance_scores'])
            comparison_data.append(('Permutation', perm_df.set_index('feature')['importance_mean']))
        
        if len(comparison_data) >= 2:
            # Find common features
            common_features = set(comparison_data[0][1].index)
            for _, data in comparison_data[1:]:
                common_features = common_features.intersection(set(data.index))
            
            # Get top 10 common features based on first method
            top_common = list(comparison_data[0][1].loc[list(common_features)].nlargest(10).index)
            
            # Create grouped bar plot with enhanced styling
            x = np.arange(len(top_common))
            width = 0.35
            colors = ['steelblue', 'lightcoral', 'lightgreen', 'gold']
            
            for i, (method, data) in enumerate(comparison_data):
                values = [data.loc[feature] for feature in top_common]
                ax.bar(x + i * width, values, width, label=method, alpha=0.8, 
                      color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
            
            # Enhanced styling for publication
            ax.set_xlabel('Features', fontsize=24)
            ax.set_ylabel('Importance score', fontsize=24)
            ax.set_title('Feature importance comparison', 
                        fontsize=24, pad=20)
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(top_common, rotation=45, ha='right')
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            # 加粗坐标轴边框（spines）
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_linewidth(1.5)  # 设置线宽
            # 隐藏上边框和右边框
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)

            ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True)
            plt.tight_layout()
        
        if save:
            filename = f"importance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close("all")
            return str(filepath)
        else:
            plt.show()
            return "displayed"
    
    def generate_report(
        self,
        include_plots: bool = True,
        format_type: str = "html"
    ) -> str:
        """
        Generate comprehensive feature importance analysis report.
        
        Args:
            include_plots: Whether to include plots in the report
            format_type: Report format ("html" or "json")
            
        Returns:
            Path to generated report file
        """
        try:
            logger.info(f"Generating {format_type.upper()} analysis report...")
            
            if format_type.lower() == "html":
                return self._generate_html_report(include_plots)
            elif format_type.lower() == "json":
                return self._generate_json_report()
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_html_report(self, include_plots: bool) -> str:
        """Generate HTML format report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"feature_importance_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        # Generate plots if requested
        plot_paths = {}
        if include_plots:
            # Reuse cached plots if they exist to avoid duplicate files
            if hasattr(self, "_plots_generated") and self._plots_generated:
                plot_paths = self._plots_generated
            else:
                plot_paths = self.create_visualization("all", save_plots=True)
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Importance Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Feature Importance Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add basic importance analysis
        if self.basic_importance:
            html_content += self._add_basic_importance_section()
        
        # Add permutation importance analysis
        if self.permutation_importance_results:
            html_content += self._add_permutation_importance_section()
        
        # Add SHAP analysis
        if self.shap_results is not None:
            html_content += self._add_shap_analysis_section()
        
        # Add plots
        if include_plots and plot_paths:
            html_content += self._add_plots_section(plot_paths)
        
        html_content += "</body></html>"
        
        # Save HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {filepath}")
        return str(filepath)
    
    def _generate_json_report(self) -> str:
        """Generate JSON format report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"feature_importance_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'total_features': len(self.feature_names) if self.feature_names else 0
            },
            'basic_importance': self.basic_importance,
            'permutation_importance': self.permutation_importance_results,
            'shap_analysis': {'available': self.shap_values is not None}
        }
        
        # Save JSON file
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved to: {filepath}")
        return str(filepath)
    
    def _add_basic_importance_section(self) -> str:
        """Add basic importance section to HTML report."""
        top_features = self.basic_importance['top_features'][:10]
        
        html = f"""
        <div class="section">
            <h2>Basic Feature Importance Analysis</h2>
            <p>XGBoost built-in feature importance based on gain metrics.</p>
            
            <div class="metric">Total Features: {self.basic_importance['total_features']}</div>
            <div class="metric">Analysis Date: {self.basic_importance['analysis_timestamp'][:10]}</div>
            
            <h3>Top 10 Features</h3>
            <table>
                <tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Percentage</th></tr>
        """
        
        for i, feature in enumerate(top_features, 1):
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feature['feature']}</td>
                    <td>{feature['importance']:.4f}</td>
                    <td>{feature['importance_percent']:.2f}%</td>
                </tr>
            """
        
        html += "</table></div>"
        return html
    
    def _add_permutation_importance_section(self) -> str:
        """Add permutation importance section to HTML report."""
        top_features = self.permutation_importance_results['top_features'][:10]
        
        html = f"""
        <div class="section">
            <h2>Permutation Importance Analysis</h2>
            <p>Feature importance based on the decrease in model performance when feature values are randomly shuffled.</p>
            
            <div class="metric">Scoring Metric: {self.permutation_importance_results['scoring_metric']}</div>
            <div class="metric">Repeats: {self.permutation_importance_results['n_repeats']}</div>
            <div class="metric">Task Type: {self.permutation_importance_results['task_type']}</div>
            
            <h3>Top 10 Features</h3>
            <table>
                <tr><th>Rank</th><th>Feature</th><th>Mean Importance</th><th>Std Dev</th><th>95% CI</th></tr>
        """
        
        for i, feature in enumerate(top_features, 1):
            ci_lower = feature['importance_lower']
            ci_upper = feature['importance_upper']
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feature['feature']}</td>
                    <td>{feature['importance_mean']:.4f}</td>
                    <td>{feature['importance_std']:.4f}</td>
                    <td>[{ci_lower:.4f}, {ci_upper:.4f}]</td>
                </tr>
            """
        
        html += "</table></div>"
        return html
    
    def _add_shap_analysis_section(self) -> str:
        """Add SHAP analysis section to HTML report."""
        if not SHAP_AVAILABLE:
            return """
            <div class="section">
                <h2>SHAP Analysis</h2>
                <p style="color: red;">SHAP library not available. This analysis was skipped.</p>
            </div>
            """
        
        if self.shap_results is None:
            return """
            <div class="section">
                <h2>SHAP Analysis</h2>
                <p style="color: orange;">SHAP analysis not performed. Please run analyze_shap_importance() first.</p>
            </div>
            """
        
        top_features = self.shap_results['top_features'][:10]
        
        html = f"""
        <div class="section">
            <h2>SHAP Feature Importance Analysis</h2>
            <p>SHAP (SHapley Additive exPlanations) values provide detailed explanations of model predictions based on game theory.</p>
            
            <div class="metric">Samples Analyzed: {self.shap_results['n_samples_analyzed']}</div>
            <div class="metric">SHAP Values Shape: {self.shap_results['shap_values_shape']}</div>
            <div class="metric">Analysis Date: {self.shap_results['analysis_timestamp'][:10]}</div>
            <div class="metric">Dependency Plots: {'✅ Created' if self.shap_results.get('dependency_plots_created', False) else '❌ Not Created'}</div>
            <div class="metric">Interaction Plots: {'✅ Created' if self.shap_results.get('interaction_plots_created', False) else '❌ Not Created'}</div>
            
            <h3>Top 10 Features (Mean Absolute SHAP Values)</h3>
            <table>
                <tr><th>Rank</th><th>Feature</th><th>SHAP Importance</th><th>Percentage</th></tr>
        """
        
        for i, feature in enumerate(top_features, 1):
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feature['feature']}</td>
                    <td>{feature['shap_importance']:.4f}</td>
                    <td>{feature['importance_percent']:.2f}%</td>
                </tr>
            """
        
        html += """
            </table>
        """
        
        # Add dependency plots section - removed to avoid duplication; handled in overall plots section
        # Add interaction plots section - removed to avoid duplication
        
        html += """
            <div style="background-color: #e1f5fe; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📖 SHAP Interpretation Guide</h4>
                <ul>
                    <li><strong>SHAP Importance</strong>: Mean absolute SHAP value across all samples - indicates average feature contribution magnitude</li>
                    <li><strong>Percentage</strong>: Relative importance compared to all features (sums to 100%)</li>
                    <li><strong>Mathematical Foundation</strong>: Based on Shapley values from cooperative game theory</li>
                    <li><strong>Advantages</strong>: Model-agnostic, theoretically grounded, provides both local and global explanations</li>
                    <li><strong>Dependency Plots</strong>: Show how feature values affect SHAP values, revealing non-linear relationships</li>
                    <li><strong>Interaction Plots</strong>: Reveal how pairs of features work together to influence predictions</li>
                </ul>
            </div>
        </div>
        """
        return html
    
    def _add_plots_section(self, plot_paths: Dict[str, str]) -> str:
        """Add plots section to HTML report with detailed explanations."""
        html = """
        <div class="section">
            <h2>📊 Visualization Analysis</h2>
            <p>The following charts provide multi-perspective feature importance analysis to help understand the contribution mechanisms of each feature to model predictions.</p>
        """
        
        # Define plot descriptions with mathematical explanations
        plot_descriptions = {
            'basic': {
                'title': '🌳 Basic Feature Importance (Gini Importance)',
                'description': """
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Algorithm Principles</h4>
                    <p><strong>Calculation Formula:</strong> For feature i, importance = Σ(w_j × Δ_impurity_j)</p>
                    <ul>
                        <li><strong>w_j</strong>: Sample weight proportion of node j</li>
                        <li><strong>Δ_impurity_j</strong>: Impurity reduction before and after splitting at node j</li>
                        <li><strong>Gini Impurity</strong>: I_G(t) = 1 - Σ(p_i²), where p_i is the probability of class i</li>
                    </ul>
                    
                    <h4>🎯 Key Interpretations</h4>
                    <ul>
                        <li><strong>Numerical Meaning</strong>: Higher values indicate greater contribution in decision tree splits</li>
                        <li><strong>Normalization</strong>: Sum of all feature importances equals 1.0</li>
                        <li><strong>Bias</strong>: Tends to favor features with larger value ranges or more categories</li>
                        <li><strong>Use Cases</strong>: Quick identification of potentially important features, but should be validated with other methods</li>
                    </ul>
                    
                    <h4>⚠️ Important Notes</h4>
                    <p>Basic importance may overestimate the importance of continuous features and high-cardinality categorical features. It is recommended to combine with permutation importance analysis.</p>
                </div>
                """
            },
            'permutation': {
                'title': '🔄 Permutation Feature Importance (Model Performance Validation)',
                'description': """
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Algorithm Principles</h4>
                    <p><strong>Calculation Steps:</strong></p>
                    <ol>
                        <li>Record baseline model performance: S_baseline</li>
                        <li>Randomly shuffle feature i values: X'_i = shuffle(X_i)</li>
                        <li>Calculate performance after shuffling: S_permuted</li>
                        <li>Importance = S_baseline - S_permuted</li>
                        <li>Repeat multiple times and take average: E[ΔS] ± σ(ΔS)</li>
                    </ol>
                    
                    <h4>📊 Statistical Interpretation</h4>
                    <ul>
                        <li><strong>Expected Value E[ΔS]</strong>: Average contribution of feature to model performance</li>
                        <li><strong>Standard Deviation σ(ΔS)</strong>: Uncertainty in importance estimation</li>
                        <li><strong>95% Confidence Interval</strong>: [E[ΔS] ± 1.96×σ(ΔS)]</li>
                        <li><strong>Negative Values</strong>: Feature may introduce noise, performance improves after removal</li>
                    </ul>
                    
                    <h4>🎯 Key Advantages</h4>
                    <ul>
                        <li><strong>Model Agnostic</strong>: Applicable to any machine learning model</li>
                        <li><strong>True Reflection</strong>: Directly measures feature impact on prediction performance</li>
                        <li><strong>Unbiased Estimation</strong>: Not affected by feature type and value range</li>
                        <li><strong>Statistical Reliability</strong>: Provides confidence intervals, quantifies uncertainty</li>
                    </ul>
                </div>
                """
            },
            'shap_summary': {
                'title': '🔍 SHAP Value Distribution Plot (Shapley Value Analysis)',
                'description': """
                <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Theoretical Foundation</h4>
                    <p><strong>Shapley Value Formula:</strong></p>
                    <p>φ_i = Σ_{S⊆N\\{i}} [|S|!(n-|S|-1)!/n!] × [f(S∪{i}) - f(S)]</p>
                    <ul>
                        <li><strong>φ_i</strong>: Shapley value of feature i</li>
                        <li><strong>S</strong>: Feature subset not containing feature i</li>
                        <li><strong>f(S)</strong>: Model output using feature set S</li>
                        <li><strong>Marginal Contribution</strong>: f(S∪{i}) - f(S)</li>
                    </ul>
                    
                    <h4>🎨 Chart Interpretation</h4>
                    <ul>
                        <li><strong>X-axis</strong>: SHAP value magnitude (degree of impact on prediction)</li>
                        <li><strong>Color Coding</strong>: Feature value magnitude (red=high value, blue=low value)</li>
                        <li><strong>Point Distribution</strong>: Each point represents a sample's feature contribution</li>
                        <li><strong>Overlap Density</strong>: Color intensity indicates sample density</li>
                    </ul>
                    
                    <h4>🔬 In-depth Analysis</h4>
                    <ul>
                        <li><strong>Non-linear Relationships</strong>: Reveals complex interactions between features and targets</li>
                        <li><strong>Threshold Effects</strong>: Identifies critical threshold points for feature values</li>
                        <li><strong>Directionality</strong>: Positive values increase prediction, negative values decrease prediction</li>
                        <li><strong>Consistency</strong>: Satisfies efficiency, symmetry, and dummy axioms</li>
                    </ul>
                </div>
                """
            },
            'shap_summary_bar': {
                'title': '📊 SHAP Importance Bar Chart (Mean Absolute Contribution)',
                'description': """
                <div style="background-color: #f3e5f5; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Calculation Method</h4>
                    <p><strong>Mean Absolute SHAP Value:</strong> E[|φ_i|] = (1/n) × Σ|φ_{i,j}|</p>
                    <ul>
                        <li><strong>φ_{i,j}</strong>: SHAP value of feature i in sample j</li>
                        <li><strong>Absolute Value</strong>: Eliminates cancellation effects of positive and negative impacts</li>
                        <li><strong>Expected Value</strong>: Average contribution level across all samples</li>
                    </ul>
                    
                    <h4>📈 Ranking Logic</h4>
                    <ul>
                        <li><strong>Descending Order</strong>: From most important to least important</li>
                        <li><strong>Overall Impact</strong>: Reflects average influence of features on all predictions</li>
                        <li><strong>Stability</strong>: Robust estimation unaffected by extreme values</li>
                    </ul>
                    
                    <h4>🎯 Application Value</h4>
                    <ul>
                        <li><strong>Feature Selection</strong>: Identify low-contribution features that can be removed</li>
                        <li><strong>Model Simplification</strong>: Retain core features to improve interpretability</li>
                        <li><strong>Business Understanding</strong>: Quantify business importance of various factors</li>
                    </ul>
                </div>
                """
            },
            'shap_dependency': {
                'title': '🔗 SHAP Dependency Plots (Feature-Value Relationship Analysis)',
                'description': """
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Analysis Purpose</h4>
                    <p>SHAP dependency plots reveal how individual feature values affect model predictions and highlight interaction effects with other features.</p>
                    
                    <h4>🎨 Chart Interpretation</h4>
                    <ul>
                        <li><strong>X-axis</strong>: Feature value range (original feature values)</li>
                        <li><strong>Y-axis</strong>: SHAP value for this feature (contribution to prediction)</li>
                        <li><strong>Color Coding</strong>: Values of the most interacting feature (reveals interaction patterns)</li>
                        <li><strong>Scatter Points</strong>: Each point represents one sample's feature value and SHAP contribution</li>
                    </ul>
                    
                    <h4>🔍 Key Insights</h4>
                    <ul>
                        <li><strong>Non-linear Relationships</strong>: Curved patterns indicate non-linear feature effects</li>
                        <li><strong>Threshold Effects</strong>: Sharp transitions reveal critical value thresholds</li>
                        <li><strong>Interaction Effects</strong>: Color gradients show how other features modulate the relationship</li>
                        <li><strong>Outlier Detection</strong>: Points far from main patterns may indicate data quality issues</li>
                    </ul>
                    
                    <h4>💡 Practical Applications</h4>
                    <ul>
                        <li><strong>Feature Engineering</strong>: Identify optimal binning or transformation strategies</li>
                        <li><strong>Business Rules</strong>: Discover actionable thresholds for decision making</li>
                        <li><strong>Model Validation</strong>: Ensure model behavior aligns with domain knowledge</li>
                    </ul>
                </div>
                """
            },
            'shap_interaction': {
                'title': '🔄 SHAP Interaction Plots (Feature Synergy Analysis)',
                'description': """
                <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📖 Theoretical Foundation</h4>
                    <p><strong>SHAP Interaction Values:</strong> φ_{i,j} = Σ_{S⊆N\\{i,j}} [|S|!(n-|S|-2)!/(n-1)!] × [f(S∪{i,j}) - f(S∪{i}) - f(S∪{j}) + f(S)]</p>
                    <ul>
                        <li><strong>φ_{i,j}</strong>: Interaction effect between features i and j</li>
                        <li><strong>Symmetric Property</strong>: φ_{i,j} = φ_{j,i}</li>
                        <li><strong>Additive Decomposition</strong>: Total SHAP = Main effects + Interaction effects</li>
                    </ul>
                    
                    <h4>🎨 Chart Interpretation</h4>
                    <ul>
                        <li><strong>X-axis</strong>: Primary feature value</li>
                        <li><strong>Y-axis</strong>: SHAP interaction value (synergistic contribution)</li>
                        <li><strong>Color Coding</strong>: Secondary feature value (interaction partner)</li>
                        <li><strong>Scatter Patterns</strong>: Reveal how feature combinations affect predictions</li>
                    </ul>
                    
                    <h4>🔬 Advanced Analysis</h4>
                    <ul>
                        <li><strong>Synergistic Effects</strong>: Positive interactions enhance combined feature impact</li>
                        <li><strong>Antagonistic Effects</strong>: Negative interactions reduce combined feature impact</li>
                        <li><strong>Conditional Dependencies</strong>: How one feature's effect depends on another's value</li>
                        <li><strong>Complex Patterns</strong>: Non-linear interaction surfaces in high-dimensional space</li>
                    </ul>
                    
                    <h4>💼 Business Applications</h4>
                    <ul>
                        <li><strong>Strategy Development</strong>: Understand how different factors work together</li>
                        <li><strong>Resource Allocation</strong>: Optimize combinations of interventions</li>
                        <li><strong>Risk Assessment</strong>: Identify dangerous feature combinations</li>
                        <li><strong>Product Design</strong>: Leverage feature synergies for better outcomes</li>
                    </ul>
                </div>
                """
            },
            'shap': {
                'title': '🎯 SHAP Comprehensive Analysis Chart',
                'description': """
                <div style="background-color: #e1f5fe; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📊 Multi-dimensional Analysis</h4>
                    <p>SHAP analysis provides a complete perspective from individual to global levels:</p>
                    
                    <h4>🔍 Individual Level</h4>
                    <ul>
                        <li><strong>Single Sample Explanation</strong>: Feature contribution decomposition for each prediction</li>
                        <li><strong>Additivity</strong>: Base value + Σφ_i = Predicted value</li>
                        <li><strong>Local Accuracy</strong>: Perfect explanation of each individual prediction</li>
                    </ul>
                    
                    <h4>🌐 Global Level</h4>
                    <ul>
                        <li><strong>Feature Importance</strong>: E[|φ_i|] reflects global importance</li>
                        <li><strong>Interaction Effects</strong>: Identifies synergistic/competitive relationships between features</li>
                        <li><strong>Pattern Discovery</strong>: Reveals hidden patterns in data</li>
                    </ul>
                    
                    <h4>⚖️ Theoretical Guarantees</h4>
                    <ul>
                        <li><strong>Uniqueness</strong>: Unique solution satisfying Shapley axioms</li>
                        <li><strong>Fairness</strong>: Fair allocation of marginal contributions</li>
                        <li><strong>Interpretability</strong>: Mathematically rigorous attribution method</li>
                    </ul>
                </div>
                """
            },
            'comparison': {
                'title': '⚖️ Feature Importance Method Comparison',
                'description': """
                <div style="background-color: #fafafa; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>📊 Methodological Comparison</h4>
                    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                        <tr style="background-color: #f5f5f5;">
                            <th style="border: 1px solid #ddd; padding: 8px;">Method</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Theoretical Foundation</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Computational Cost</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Applicability</th>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;"><strong>Gini Importance</strong></td>
                            <td style="border: 1px solid #ddd; padding: 8px;">Impurity Reduction</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">Very Low</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">XGBoost Only</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;"><strong>Permutation Importance</strong></td>
                            <td style="border: 1px solid #ddd; padding: 8px;">Performance Degradation</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">Medium</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">Any Model</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;"><strong>SHAP Values</strong></td>
                            <td style="border: 1px solid #ddd; padding: 8px;">Game Theory</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">High</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">Any Model</td>
                        </tr>
                    </table>
                    
                    <h4>🎯 Consistency Analysis</h4>
                    <ul>
                        <li><strong>High Consistency</strong>: Similar rankings across three methods indicate reliable feature importance</li>
                        <li><strong>Partial Differences</strong>: Normal phenomenon reflecting different emphases of methods</li>
                        <li><strong>Significant Divergence</strong>: Requires in-depth investigation, may indicate data quality issues</li>
                    </ul>
                    
                    <h4>💡 Selection Recommendations</h4>
                    <ul>
                        <li><strong>Quick Screening</strong>: Use Gini importance</li>
                        <li><strong>Reliable Validation</strong>: Use permutation importance</li>
                        <li><strong>Deep Interpretation</strong>: Use SHAP analysis</li>
                        <li><strong>Comprehensive Decision</strong>: Combine consistent conclusions from all three methods</li>
                    </ul>
                </div>
                """
            }
        }
        
        for plot_type, path in plot_paths.items():
            if plot_type in ['shap_dependency', 'shap_interaction']:
                # Handle list of paths for dependency and interaction plots
                if isinstance(path, list) and path:
                    plot_info = plot_descriptions.get(plot_type, {
                        'title': plot_type.replace('_', ' ').title(),
                        'description': '<p>SHAP可视化图表</p>'
                    })
                    
                    html += f"""
                    <div class="plot">
                        <h3>{plot_info['title']}</h3>
                        {plot_info['description']}
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0;">
                    """
                    
                    for plot_path in path:
                        if Path(plot_path).exists():
                            relative_path = Path(plot_path).name
                            html += f"""
                            <div style="border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background-color: white;">
                                <img src="{relative_path}" alt="{plot_info['title']}" style="width: 100%; height: auto; border-radius: 4px;">
                                <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666; font-style: italic;">
                                    Chart File: {relative_path}
                                </p>
                            </div>
                            """
                    
                    html += """
                        </div>
                    </div>
                    """
            elif path != "displayed" and Path(path).exists():
                relative_path = Path(path).name
                plot_info = plot_descriptions.get(plot_type, {
                    'title': plot_type.replace('_', ' ').title(),
                    'description': '<p>特征重要性可视化图表</p>'
                })
                
                html += f"""
                <div class="plot">
                    <h3>{plot_info['title']}</h3>
                    {plot_info['description']}
                    <div style="text-align: center; margin: 20px 0; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; background-color: white;">
                        <img src="{relative_path}" alt="{plot_info['title']}" style="max-width: 100%; height: auto; border-radius: 4px;">
                        <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666; font-style: italic;">
                            Chart File: {relative_path}
                        </p>
                    </div>
                </div>
                """
        
        html += "</div>"
        return html

    def create_multi_target_visualizations(
        self,
        importance_matrix: Dict[str, Dict[str, float]],
        figsize: Tuple[int, int] = (12, 8),
        save_plots: bool = True,
        top_n: int = 15
    ) -> Dict[str, str]:
        """
        Create individual feature importance plots for each target in multi-target regression.
        
        Args:
            importance_matrix: Dictionary with target-feature importance values 
                              Format: {target: {feature: importance_value}}
            figsize: Figure size for each plot
            save_plots: Whether to save plots to files
            top_n: Number of top features to display per target
            
        Returns:
            Dictionary mapping target names to plot file paths
        """
        try:
            logger.info(f"Creating multi-target visualizations for {len(importance_matrix)} targets...")
            
            plot_paths = {}
            
            for target_name, feature_importance in importance_matrix.items():
                # Convert to DataFrame and sort by importance
                df = pd.DataFrame([
                    {'feature': feature, 'importance': importance}
                    for feature, importance in feature_importance.items()
                ]).sort_values('importance', ascending=False).head(top_n)
                
                # Create plot for this target
                plt.figure(figsize=figsize)
                
                # Create horizontal bar plot
                bars = plt.barh(range(len(df)), df['importance'], color='skyblue', alpha=0.8)
                plt.yticks(range(len(df)), df['feature'])
                plt.xlabel('SHAP Feature Importance')
                plt.title(f'Feature Importance for Target: {target_name}')
                plt.gca().invert_yaxis()
                
                # Add value labels on bars
                for i, (importance, feature) in enumerate(zip(df['importance'], df['feature'])):
                    plt.text(importance + max(df['importance']) * 0.01, i, 
                            f'{importance:.4f}', va='center', fontsize=9)
                
                # Add grid for better readability
                plt.grid(axis='x', alpha=0.3, linestyle='--')
                plt.tight_layout()
                
                if save_plots:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"multi_target_importance_{target_name}_{timestamp}.png"
                    filepath = self.output_dir / filename
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_paths[target_name] = str(filepath)
                    logger.info(f"Plot for target {target_name} saved to: {filepath}")
                else:
                    plt.show()
                    plot_paths[target_name] = "displayed"
            
            logger.info(f"Multi-target visualizations created for {len(plot_paths)} targets")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating multi-target visualizations: {str(e)}")
            raise

    def generate_multi_target_html_report(
        self,
        importance_matrix: Dict[str, Dict[str, float]],
        plot_paths: Optional[Dict[str, str]] = None,
        analysis_method: str = "SHAP",
        include_plots: bool = True
    ) -> str:
        """
        Generate HTML report for multi-target feature importance analysis.
        
        Args:
            importance_matrix: Dictionary with target-feature importance values
            plot_paths: Dictionary mapping target names to plot file paths
            analysis_method: Method used for analysis (e.g., "SHAP")
            include_plots: Whether to include plots in the report
            
        Returns:
            Path to generated HTML report file
        """
        try:
            logger.info("Generating multi-target HTML report...")
            
            # Start HTML content
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Target Feature Importance Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; margin-top: 25px; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .target-section {{ margin: 30px 0; padding: 20px; border: 1px solid #bdc3c7; border-radius: 8px; }}
        .feature-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .feature-table th, .feature-table td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; }}
        .feature-table th {{ background-color: #3498db; color: white; }}
        .feature-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .plot-container {{ text-align: center; margin: 20px 0; }}
        .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        .method-badge {{ background-color: #e74c3c; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Multi-Target Feature Importance Analysis Report</h1>
        
        <div class="summary">
            <h3>📊 Analysis Summary</h3>
            <p><strong>Analysis Method:</strong> <span class="method-badge">{analysis_method}</span></p>
            <p><strong>Number of Targets:</strong> {len(importance_matrix)}</p>
            <p><strong>Generated on:</strong> <span class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
            <p><strong>Targets Analyzed:</strong> {', '.join(importance_matrix.keys())}</p>
        </div>
"""
            
            # Add section for each target
            for target_name, feature_importance in importance_matrix.items():
                html_content += f"""
        <div class="target-section">
            <h2>🔍 Target: {target_name}</h2>
            
            <h3>Top Features by Importance</h3>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature Name</th>
                        <th>{analysis_method} Importance</th>
                        <th>Relative Importance (%)</th>
                    </tr>
                </thead>
                <tbody>
"""
                # Sort features by importance and add to table
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                total_importance = sum(feature_importance.values())
                
                for rank, (feature, importance) in enumerate(sorted_features, 1):
                    relative_importance = (importance / total_importance) * 100 if total_importance > 0 else 0
                    html_content += f"""
                    <tr>
                        <td>{rank}</td>
                        <td><strong>{feature}</strong></td>
                        <td>{importance:.4f}</td>
                        <td>{relative_importance:.2f}%</td>
                    </tr>
"""
                
                html_content += """
                </tbody>
            </table>
"""
                
                # Add plot if available
                if include_plots and plot_paths and target_name in plot_paths:
                    plot_path = Path(plot_paths[target_name])
                    if plot_path.exists():
                        relative_plot_path = plot_path.relative_to(self.output_dir)
                        html_content += f"""
            <div class="plot-container">
                <h3>📈 Visualization</h3>
                <img src="{relative_plot_path}" alt="Feature importance plot for {target_name}" />
            </div>
"""
                
                html_content += """
        </div>
"""
            
            # Close HTML
            html_content += """
        <div class="summary">
            <h3>📝 Report Notes</h3>
            <p>• This report shows feature importance analysis for multi-target regression using {analysis_method} method.</p>
            <p>• Higher importance values indicate features that have more influence on the target prediction.</p>
            <p>• Rankings are calculated independently for each target.</p>
            <p>• Relative importance percentages show each feature's contribution relative to all features for that target.</p>
        </div>
    </div>
</body>
</html>
""".format(analysis_method=analysis_method)
            
            # Save HTML report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"multi_target_importance_report_{timestamp}.html"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Multi-target HTML report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating multi-target HTML report: {str(e)}")
            raise

    def _load_raw_display_data(self, raw_data_path: Optional[str], model_id: Optional[str], feature_names: List[str]) -> Optional[np.ndarray]:
        """
        Load raw unprocessed data for display purposes.
        
        Args:
            raw_data_path: Direct path to raw data CSV file
            model_id: Model ID for locating raw data in trained_models directory
            feature_names: List of expected feature names
            
        Returns:
            Raw data array or None if not found
        """
        try:
            # Determine the path to raw data
            if raw_data_path:
                data_path = Path(raw_data_path)
            elif model_id:
                # Look in trained_models directory
                data_path = Path("trained_models") / model_id / "raw_data.csv"
                print("data_path",data_path)
            else:
                return None
            
            # Check if file exists
            if not data_path.exists():
                logger.warning(f"Raw data file not found: {data_path}")
                return None
            
            # Load the raw data
            logger.info(f"Loading raw display data from: {data_path}")
            raw_df = pd.read_csv(data_path)
            
            # Extract feature columns (exclude target column if present)
            available_columns = raw_df.columns.tolist()
            
            # Try to match feature names
            matched_features = []
            for feature in feature_names:
                if feature in available_columns:
                    matched_features.append(feature)
                else:
                    logger.warning(f"Feature '{feature}' not found in raw data columns: {available_columns}")
            
            if not matched_features:
                logger.warning("No matching features found in raw data")
                return None
            
            # Extract the matching feature columns
            raw_features = raw_df[matched_features].values
            
            # If some features are missing, pad with zeros or handle appropriately
            if len(matched_features) < len(feature_names):
                logger.warning(f"Only {len(matched_features)}/{len(feature_names)} features found in raw data")
                # Create full array with missing features as zeros
                full_array = np.zeros((raw_features.shape[0], len(feature_names)))
                for i, feature in enumerate(feature_names):
                    if feature in matched_features:
                        feature_idx = matched_features.index(feature)
                        full_array[:, i] = raw_features[:, feature_idx]
                raw_features = full_array
            
            logger.info(f"Successfully loaded raw display data: {raw_features.shape}")
            return raw_features
            
        except Exception as e:
            logger.error(f"Error loading raw display data: {str(e)}")
            return None


