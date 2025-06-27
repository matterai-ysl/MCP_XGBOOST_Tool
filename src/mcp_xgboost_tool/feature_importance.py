"""
Feature Importance Analysis Module

This module provides comprehensive feature importance analysis capabilities including:
- Basic feature importance from RandomForest
- Permutation importance analysis  
- SHAP value analysis
- Visualization tools
- Comprehensive reporting

Author: MCP-RandomForest-Tool
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error

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
    Comprehensive feature importance analysis for RandomForest models.
    
    Provides multiple approaches to analyze feature importance:
    - Basic RandomForest feature importances
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
        
        # Model and data storage
        self.model = None
        self.X_data = None
        self.y_data = None
        self.feature_names = None
        self.task_type = None
        
        logger.info(f"FeatureImportanceAnalyzer initialized with output directory: {self.output_dir}")
    
    def analyze_basic_importance(
        self,
        model: Union[RandomForestClassifier, RandomForestRegressor],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze basic feature importance using RandomForest's built-in feature_importances_.
        
        Args:
            model: Trained RandomForest model
            feature_names: Names of features (if None, uses generic names)
            
        Returns:
            Dictionary with feature importance analysis results
        """
        try:
            logger.info("Starting basic feature importance analysis...")
            
            # Store model reference
            self.model = model
            
            # Get feature importances
            importances = model.feature_importances_
            n_features = len(importances)
            
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
        model: Union[RandomForestClassifier, RandomForestRegressor],
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Analyze permutation importance to measure feature impact on model performance.
        
        Args:
            model: Trained RandomForest model
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            n_repeats: Number of permutations to perform
            random_state: Random state for reproducibility
            
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
            else:
                self.task_type = 'regression'
                scoring = 'neg_mean_squared_error'
            
            # Perform permutation importance
            perm_importance = permutation_importance(
                model, X_array, y,
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
        model: Union[RandomForestClassifier, RandomForestRegressor],
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze SHAP values for model interpretability.
        
        Args:
            model: Trained RandomForest model
            X: Feature matrix (used as background data)
            feature_names: Names of features
            max_samples: Maximum number of samples to use for SHAP analysis
            
        Returns:
            Dictionary with SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not available. Skipping SHAP analysis.")
            return {'error': 'SHAP library not available'}
        
        try:
            logger.info("Starting SHAP importance analysis...")
            
            # Convert DataFrame to numpy if needed
            if isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = X.columns.tolist()
                X_array = X.values
            else:
                X_array = X
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Store feature names for visualization
            self.feature_names = feature_names
            
            # Limit samples for computational efficiency
            if X_array.shape[0] > max_samples:
                sample_indices = np.random.choice(X_array.shape[0], max_samples, replace=False)
                X_sample = X_array[sample_indices]
                logger.info(f"Using {max_samples} samples for SHAP analysis")
            else:
                X_sample = X_array
                logger.info(f"Using all {X_array.shape[0]} samples for SHAP analysis")
            
            # Store the sampled data for visualization
            self.X_shap_data = X_sample
            
            # Create SHAP explainer
            self.shap_explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            
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
            
            # Store results
            shap_results = {
                'importance_scores': shap_df.to_dict('records'),
                'top_features': top_features.to_dict('records'),
                'feature_ranking': shap_df['feature'].tolist(),
                'n_samples_analyzed': X_sample.shape[0],
                'shap_values_shape': list(shap_values_to_analyze.shape),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"SHAP analysis completed for {X_sample.shape[0]} samples")
            logger.info(f"Top 3 features: {', '.join(top_features['feature'].head(3).tolist())}")
            
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
            analysis_type: Type of analysis to visualize ("basic", "permutation", "shap", "all")
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
            
            if analysis_type == "all" and len(plot_paths) > 1:
                plot_paths['comparison'] = self._create_comparison_plot(figsize, save_plots)
            
            logger.info(f"Created {len(plot_paths)} visualization(s)")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def _create_basic_importance_plot(self, figsize: Tuple[int, int], save: bool) -> str:
        """Create basic feature importance plot."""
        plt.figure(figsize=figsize)
        
        # Get top 15 features for visualization
        top_features = pd.DataFrame(self.basic_importance['importance_scores']).head(15)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Basic Feature Importance (RandomForest)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save:
            filename = f"basic_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return "displayed"
    
    def _create_permutation_importance_plot(self, figsize: Tuple[int, int], save: bool) -> str:
        """Create permutation importance plot with error bars."""
        plt.figure(figsize=figsize)
        
        # Get top 15 features for visualization
        top_features = pd.DataFrame(self.permutation_importance_results['importance_scores']).head(15)
        
        # Create horizontal bar plot with error bars
        plt.barh(range(len(top_features)), top_features['importance_mean'], 
                xerr=top_features['importance_std'], color='lightcoral', capsize=3)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title('Permutation Feature Importance (with std dev)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(top_features['importance_mean'], top_features['importance_std'])):
            plt.text(mean + std + 0.001, i, f'{mean:.3f}±{std:.3f}', va='center')
        
        plt.tight_layout()
        
        if save:
            filename = f"permutation_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
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
        
        # Create SHAP summary plot (scatter plot)
        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values_to_plot, X_for_plot, 
                         feature_names=self.feature_names, show=False)
        
        if save:
            filename = f"shap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(str(filepath))
            logger.info(f"SHAP summary plot saved to: {filepath}")
        else:
            plt.show()
        
        # Create SHAP summary bar plot
        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values_to_plot, X_for_plot, 
                         feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        
        if save:
            filename = f"shap_summary_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(str(filepath))
            logger.info(f"SHAP summary bar plot saved to: {filepath}")
        else:
            plt.show()
        
        if save:
            # Return paths as a comma-separated string for backwards compatibility
            return ",".join(plot_paths)
        else:
            return "displayed"
    
    def _create_comparison_plot(self, figsize: Tuple[int, int], save: bool) -> str:
        """Create comparison plot of different importance methods."""
        plt.figure(figsize=figsize)
        
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
            
            # Create grouped bar plot
            x = np.arange(len(top_common))
            width = 0.35
            
            for i, (method, data) in enumerate(comparison_data):
                values = [data.loc[feature] for feature in top_common]
                plt.bar(x + i * width, values, width, label=method, alpha=0.8)
            
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.title('Feature Importance Comparison')
            plt.xticks(x + width/2, top_common, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
        
        if save:
            filename = f"importance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
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
        if self.shap_values is not None:
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
            <p>RandomForest built-in feature importance based on impurity decrease.</p>
            
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
        
        html = """
        <div class="section">
            <h2>SHAP Analysis</h2>
            <p>SHAP (SHapley Additive exPlanations) values provide detailed explanations of model predictions.</p>
            <p>SHAP analysis completed successfully. See visualizations below for detailed insights.</p>
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
                            <td style="border: 1px solid #ddd; padding: 8px;">Random Forest Only</td>
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
            if path != "displayed" and Path(path).exists():
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


