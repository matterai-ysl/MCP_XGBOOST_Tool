"""
MCP Server for Random Forest Regression Tool

This module implements the MCP server using FastMCP and provides 8 core tool functions:
1. train_random_forest - Multi-target regression model training with comprehensive scoring metrics
2. train_classification_forest - Classification model training  
3. predict_from_file - Batch prediction from file
4. predict_from_values - Real-time prediction from values
5. analyze_feature_importance - Feature importance analysis
6. list_models - List all trained models
7. get_model_info - Get model detailed information
8. delete_model - Delete trained model
"""

import logging
import traceback
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import json,uuid
import numpy as np
import asyncio
from datetime import datetime
import zipfile
import os

# FastMCP import
from mcp.server import FastMCP

# Internal modules
from .training import TrainingEngine
from .prediction import PredictionEngine
from .feature_importance import  FeatureImportanceAnalyzer
from .model_manager import ModelManager
from .data_validator import DataValidator
from starlette.requests import Request
from starlette.responses import PlainTextResponse
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server with detailed instructions
mcp = FastMCP(
    name="Random Forest Regression Tool",
    instructions="""
    This is a comprehensive Machine Learning server providing Random Forest regression capabilities.
    
    Available tools:
    1. train_random_forest - Train a Random Forest regression model (supports multi-target regression)
    2. train_classification_forest - Train a classification Random Forest model
    3. predict_from_file - Make batch predictions from a data file
    4. predict_from_values - Make real-time predictions from feature values
    5. analyze_feature_importance - Analyze feature importance of a trained model
    6. list_models - List all available trained models
    7. get_model_info - Get detailed information about a specific model
    8. delete_model - Delete a trained model
    
    Use these tools to build complete ML workflows from training to deployment.
    The main train_random_forest function now supports multi-target regression with various scoring metrics.
    """
)

# Initialize engines
root_dir = Path("./trained_models")
training_engine = TrainingEngine("trained_models")
prediction_engine = PredictionEngine("trained_models")
model_manager = ModelManager("trained_models")
base_url = "http://localhost:8080"
# @mcp.custom_route("/hello/{name}", methods=["GET"])
# async def simple_hello(request: Request) -> PlainTextResponse:
#     name = request.path_params["name"]
#     return PlainTextResponse(f"Hello, {name}!")

@mcp.tool()
async def train_random_forest(
    data_source: str,
    target_dimension: int = 1,
    optimize_hyperparameters: bool = True,
    n_trials: int = 50,
    cv_folds: int = 5,
    scoring_metric: str = "MAE",
    validate_data: bool = True,
    save_model: bool = True,
    apply_preprocessing: bool = True,
    scaling_method: str = "standard"
) -> Dict[str, Any]:
    """
    Train a Random Forest regression model supporting multi-target regression.
    
    Args:
        data_source: Path to training data file (CSV, Excel, etc.)
        target_dimension: Number of target columns for multi-target regression (positive integer)
        optimize_hyperparameters: Whether to run hyperparameter optimization
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        scoring_metric: Scoring metric for regression optimization. Supported metrics:
                       - 'MSE' (default): Mean Squared Error
                       - 'MAE': Mean Absolute Error
                       - 'RMSE': Root Mean Squared Error
                       - 'R2': R-squared score
                       - 'MAPE': Mean Absolute Percentage Error
                       - 'explained_variance': Explained Variance Score
                       - 'max_error': Maximum Residual Error
                       - 'MAD': Median Absolute Deviation
        validate_data: Whether to validate data quality
        save_model: Whether to save the trained model
        apply_preprocessing: Whether to apply data preprocessing
        scaling_method: Scaling method ('standard', 'minmax', 'robust', 'quantile', 'power')
        
    Returns:
        Training results including model performance and metadata
    """
    try:
        logger.info(f"Training Random Forest regression model from: {data_source}")

        if scoring_metric == 'MSE':
            scoring_metric = 'neg_mean_squared_error'
        elif scoring_metric == 'MAE':
            scoring_metric = 'neg_mean_absolute_error'
        elif scoring_metric == 'RMSE':
            scoring_metric = 'neg_root_mean_squared_error'
        elif scoring_metric == 'R2':
            scoring_metric = 'r2'
        elif scoring_metric == 'MAPE':
            scoring_metric = 'neg_mean_absolute_percentage_error'
        elif scoring_metric == 'explained_variance':
            scoring_metric = 'explained_variance'
        elif scoring_metric == 'max_error':
            scoring_metric = 'max_error'
        elif scoring_metric == 'MAD':
            scoring_metric = 'neg_median_absolute_error'
        else:
            raise ValueError(f"Invalid scoring metric: {scoring_metric}. Supported metrics: MSE, MAE, RMSE, R2, MAPE, explained_variance, max_error, MAD")


        
        # Load and validate data for multi-target regression
        from .data_utils import DataProcessor
        data_processor = DataProcessor()
        df = data_processor.load_data(data_source)
        
        # Validate target_dimension parameter
        if target_dimension <= 0:
            raise ValueError(f"Target dimension must be a positive integer, got: {target_dimension}")
        
        if target_dimension > len(df.columns):
            raise ValueError(f"Target dimension {target_dimension} exceeds number of columns {len(df.columns)}")
        
        # Get target columns (last N columns based on target_dimension)
        target_columns = df.columns[-target_dimension:].tolist()
        
        logger.info(f"Multi-target regression with {target_dimension} target(s): {target_columns}")
        
        model_id = str(uuid.uuid4())
        
        # Run training in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: training_engine.train_random_forest(
                data_source=data_source,
                target_column=target_columns if target_dimension > 1 else target_columns[0],
                model_name= model_id,
                optimize_hyperparameters=optimize_hyperparameters,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring_metric=scoring_metric,
                validate_data=validate_data,
                save_model=save_model,
                apply_preprocessing=apply_preprocessing,
                scaling_method=scaling_method
            )
        )
        
        logger.info("Random Forest regression training completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Regression training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def train_classification_forest(
    data_source: str,
    target_dimension: int = -1,
    optimize_hyperparameters: bool = True,
    n_trials: int = 50,
    cv_folds: int = 5,
    scoring_metric: str = "f1_weighted",
    validate_data: bool = True,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Train a Random Forest classification model.
    
    Args:
        data_source: Path to training data file (CSV, Excel, etc.)
        target_dimension: Index of target column (default: -1 for last column)
        optimize_hyperparameters: Whether to run hyperparameter optimization
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        scoring_metric: Scoring metric for optimization (default: f1_weighted)
        validate_data: Whether to validate data quality
        save_model: Whether to save the trained model
        
    Returns:
        Training results including model performance and metadata
    """
    try:
        logger.info(f"Training Classification Random Forest from: {data_source}")
        
        # Convert target_dimension to target_column for internal processing
        from .data_utils import DataProcessor
        data_processor = DataProcessor()
        df = data_processor.load_data(data_source)
        
        # Get target column name from dimension
        if target_dimension == -1:
            target_column = df.columns[-1]
        else:
            if target_dimension >= len(df.columns) or target_dimension < 0:
                raise ValueError(f"Target dimension {target_dimension} is out of range. Data has {len(df.columns)} columns.")
            target_column = df.columns[target_dimension]
        
        model_name = str(uuid.uuid4())
        
        # Run training in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: training_engine.train_classification_forest(
                data_source=data_source,
                target_column=target_column,
                model_name=model_name,
                optimize_hyperparameters=optimize_hyperparameters,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring_metric=scoring_metric,
                validate_data=validate_data,
                save_model=save_model
            )
        )
        
        logger.info("Classification Random Forest training completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Classification training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def predict_from_file(
    model_id: str,
    data_source: str,
    output_path: str = None,
    include_confidence: bool = True,
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Make batch predictions from a data file.
    
    Args:
        model_id: Unique identifier for the trained model
        data_source: Path to prediction data file (CSV, Excel, etc.)
        output_path: Path to save prediction results (if None, uses default)
        include_confidence: Whether to include confidence scores
        generate_report: Whether to generate detailed report
        
    Returns:
        Prediction results and analysis
    """
    try:
        logger.info(f"Making batch predictions with model {model_id} from file: {data_source}")
        
        # Run prediction in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: prediction_engine.predict_from_file(
                model_id=model_id,
                data_source=data_source,
                output_path=output_path,
                include_confidence=include_confidence,
                generate_report=generate_report
            )
        )
        
        logger.info("Batch prediction completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Batch prediction failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def predict_from_values(
    model_id: str,
    feature_values: Union[List[float], List[List[float]], Dict[str, float], List[Dict[str, float]]],
    feature_names: List[str] = None,
    include_confidence: bool = True,
    save_intermediate_files: bool = True,
    generate_report: bool = True,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Make real-time predictions from feature values with CSV export and reporting.
    
    Supports both single and batch predictions:
    - Single: [1, 2, 3] or {'feature1': 1, 'feature2': 2}
    - Batch: [[1, 2, 3], [4, 5, 6]] or [{'feature1': 1}, {'feature1': 4}]
    
    Args:
        model_id: Unique identifier for the trained model
        feature_values: Feature values in various formats (single or batch)
        feature_names: Names of features (required if feature_values is a list of lists),if not provided, the feature names will be inferred from the model metadata.
        include_confidence: Whether to include confidence scores
        save_intermediate_files: Whether to save CSV files with processed features, predictions, and confidence scores
        generate_report: Whether to generate detailed experiment report
        output_path: Custom output path for prediction files (optional)
        
    Returns:
        Prediction results, CSV file paths, and analysis
    """
    try:
        logger.info(f"Making real-time prediction with model {model_id}")
        
        # Run prediction in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: prediction_engine.predict_from_values(
                model_id=model_id,
                feature_values=feature_values,
                feature_names=feature_names,
                include_confidence=include_confidence,
                save_intermediate_files=save_intermediate_files,
                generate_report=generate_report,
                output_path=output_path
            )
        )
        
        logger.info("Real-time prediction completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Real-time prediction failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def analyze_global_feature_importance(
    model_id: str,
    data_source: str = None,
    analysis_types: List[str] = ["basic"],
    generate_plots: bool = True,
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Analyze feature importance of a trained model.
    
    Args:
        model_id: Unique identifier for the trained model
        data_source: Path to data file for permutation analysis (if None, uses saved training data)
        analysis_types: Types of analysis to perform ["basic", "permutation", "shap"], default is ["basic"]
        generate_plots: Whether to generate visualization plots
        generate_report: Whether to generate analysis report
    Returns:
        Feature importance analysis results
    """
    try:
        logger.info(f"Analyzing feature importance for model {model_id}")
        
        # Load model info first
        model_info = model_manager.get_model_info(model_id)
        
        def run_analysis():
            # Load model
            model = model_manager.load_model(model_id)
            
            # Set up output directory for feature analysis
            model_dir = Path("trained_models") / model_id
            task_id = str(uuid.uuid4())
            output_dir = model_dir / "feature_analysis" / "global" / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Feature analysis output directory: {output_dir}")
            
            # Prepare data
            X, y, feature_names = None, None, None
            
            if data_source:
                # Use provided data source
                from .data_utils import DataProcessor
                data_processor = DataProcessor()
                df = data_processor.load_data(data_source)
                logger.info(f"Using provided data source: {data_source}")
                use_preprocessing = True  # Need to preprocess external data
            else:
                # Try to use saved processed data first
                processed_data_path = model_dir / "processed_data" / "processed_data.csv"
                if processed_data_path.exists():
                    import pandas as pd
                    df = pd.read_csv(processed_data_path)
                    logger.info(f"Using saved processed training data: {processed_data_path}")
                    use_preprocessing = False  # Data already processed
                else:
                    # Fallback to raw data
                    raw_data_path = model_dir / "raw_data.csv"
                    if raw_data_path.exists():
                        import pandas as pd
                        df = pd.read_csv(raw_data_path)
                        logger.info(f"Using saved raw training data: {raw_data_path}")
                        use_preprocessing = True  # Need to preprocess raw data
                    else:
                        raise FileNotFoundError(f"No training data found. Looked for: {processed_data_path} and {raw_data_path}")
            
            # Get target column information from model metadata
            target_column = model_info.get('target_column')
            if not target_column:
                target_name = model_info.get('target_name', [])
                if target_name:
                    target_column = target_name[0] if isinstance(target_name, list) else target_name
                else:
                    # Default to last column
                    target_column = df.columns[-1]
                    logger.warning(f"No target column found in metadata, using last column: {target_column}")
            
            # Separate features and target
            # Handle both single target column and multi-target cases
            if isinstance(target_column, list):
                # Multi-target case - use first target for feature importance analysis
                target_col = target_column[0]
                logger.info(f"Multi-target model detected, using first target for analysis: {target_col}")
            else:
                target_col = target_column
            
            if target_col in df.columns:
                X_raw = df.drop(columns=[target_col])
                y_raw = df[target_col].values
                logger.info(f"Data loaded - Features shape: {X_raw.shape}, Target shape: {y_raw.shape}")
            else:
                raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {list(df.columns)}")
            
            # Apply preprocessing if needed
            if use_preprocessing:
                preprocessing_pipeline_path = model_dir / "preprocessing_pipeline.pkl"
                if preprocessing_pipeline_path.exists():
                    logger.info("Loading and applying preprocessing pipeline...")
                    try:
                        # Load the preprocessing pipeline
                        import joblib
                        from .data_preprocessing import DataPreprocessor
                        
                        preprocessor = DataPreprocessor()
                        preprocessor.load_pipeline(str(preprocessing_pipeline_path))
                        
                        # Apply feature preprocessing
                        X = preprocessor.transform_features(X_raw)
                        y = preprocessor.transform_target(y_raw)
                        
                        # Get processed feature names
                        feature_names = preprocessor.get_feature_names()
                        if not feature_names or len(feature_names) != X.shape[1]:
                            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                            logger.warning(f"Using generic feature names, processed features: {X.shape[1]}")
                        
                        logger.info(f"Preprocessing applied - Features shape: {X.shape}, Target shape: {y.shape}")
                        logger.info(f"Processed feature names: {feature_names}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to apply preprocessing pipeline: {e}")
                        logger.info("Falling back to raw data (this may cause permutation importance to fail)")
                        X = X_raw.values if hasattr(X_raw, 'values') else X_raw
                        y = y_raw
                        feature_names = X_raw.columns.tolist() if hasattr(X_raw, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
                else:
                    logger.info("No preprocessing pipeline found, using raw data")
                    X = X_raw.values if hasattr(X_raw, 'values') else X_raw
                    y = y_raw
                    feature_names = X_raw.columns.tolist() if hasattr(X_raw, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
            else:
                # Data is already preprocessed
                logger.info("Using preprocessed data directly")
                X = X_raw.values if hasattr(X_raw, 'values') else X_raw
                y = y_raw
                feature_names = X_raw.columns.tolist() if hasattr(X_raw, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
            
            # Ensure data types are correct for analysis
            if hasattr(X, 'values'):
                X = X.values
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not isinstance(y, np.ndarray):
                y = np.array(y)
                
            logger.info(f"Final data for analysis - X shape: {X.shape}, y shape: {y.shape}")
            logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
            
            # Check for any remaining issues with the data
            if np.any(pd.isna(X)):
                logger.warning("Found NaN values in features, this may cause analysis to fail")
            if np.any(pd.isna(y)):
                logger.warning("Found NaN values in target, this may cause analysis to fail")
            
            # Initialize analyzer with correct output directory
            from .feature_importance import FeatureImportanceAnalyzer
            analyzer = FeatureImportanceAnalyzer(str(output_dir))
            
            results = {}
            
            # Basic importance analysis
            if "basic" in analysis_types:
                logger.info("Performing basic feature importance analysis...")
                try:
                    results['basic_importance'] = analyzer.analyze_basic_importance(
                        model, feature_names
                    )
                    logger.info("✓ Basic importance analysis completed successfully")
                except Exception as e:
                    logger.error(f"Basic importance analysis failed: {e}")
                    results['basic_error'] = str(e)
            
            # Permutation importance analysis
            if "permutation" in analysis_types:
                logger.info("Performing permutation feature importance analysis...")
                try:
                    results['permutation_importance'] = analyzer.analyze_permutation_importance(
                        model, X, y, feature_names
                    )
                    logger.info("✓ Permutation importance analysis completed successfully")
                except Exception as e:
                    logger.error(f"Permutation importance analysis failed: {e}")
                    logger.error(f"Error details: {traceback.format_exc()}")
                    results['permutation_error'] = str(e)
            
            # SHAP analysis (if available and requested)
            if "shap" in analysis_types:
                logger.info("Performing SHAP feature importance analysis...")
                try:
                    results['shap_importance'] = analyzer.analyze_shap_importance(
                        model, X, feature_names
                    )
                    logger.info("✓ SHAP importance analysis completed successfully")
                except Exception as e:
                    logger.warning(f"SHAP analysis failed: {e}")
                    results['shap_error'] = str(e)
            
            # Generate visualizations
            if generate_plots:
                logger.info("Generating feature importance plots...")
                try:
                    plot_paths = analyzer.create_visualization(save_plots=True)
                    # results['plot_paths'] = plot_paths
                    logger.info("✓ Visualizations generated successfully")
                except Exception as e:
                    logger.warning(f"Visualization generation failed: {e}")
                    results['visualization_error'] = str(e)
            
            # Generate report
            if generate_report:
                logger.info("Generating feature importance report...")
                try:
                    report_path = analyzer.generate_report(include_plots=generate_plots)
                    results['report_summary'] = f"You can find the html golbal feature importance report summary in {base_url}/static/{Path(report_path).relative_to(root_dir).as_posix()}"
                    logger.info("✓ Report generated successfully")
                except Exception as e:
                    logger.warning(f"Report generation failed: {e}")
                    results['report_error'] = str(e)
            
            # results.update({
                # 'model_id': model_id,
                # 'analysis_types': analysis_types,
                # 'data_source': data_source if data_source else str(model_dir / "raw_data.csv"),
            #     'feature_count': len(feature_names) if feature_names else 0,
            #     'data_shape': f"{X.shape[0]} samples, {X.shape[1]} features" if X is not None else "Unknown",
            # })
            
            # Create analysis archive
            try:
                archive_path = _create_analysis_archive(
                    output_dir=output_dir,
                    analysis_type="global",
                    task_id=task_id,
                    model_id=model_id
                )
                if archive_path:
                    results['dowland_archive_path'] = f"You can download the global feature importance analysis details in {base_url}/download/file/{Path(archive_path).relative_to(root_dir.parent).as_posix()}"

                    logger.info(f"Global analysis results archived: {archive_path}")
            except Exception as e:
                logger.warning(f"Failed to archive global analysis results: {e}")
            
            logger.info(f"Feature importance analysis completed. Results saved to: {output_dir}")

            # 简化返回结果 - 只保留每个分析类型的 importance_scores
            simplified_results = {
                'model_id': model_id,
                'analysis_types': analysis_types,
                'data_source': data_source if data_source else str(model_dir / "raw_data.csv"),
                'feature_count': len(feature_names) if feature_names else 0,
                'data_shape': f"{X.shape[0]} samples, {X.shape[1]} features" if X is not None else "Unknown",
            }
            
            # 添加基础重要性分析结果
            if "basic" in analysis_types and results.get('basic_importance'):
                simplified_results['basic_importance_scores'] = results['basic_importance'].get('importance_scores', [])
                logger.info("✓ Basic importance scores extracted")
            
            # 添加排列重要性分析结果  
            if "permutation" in analysis_types and results.get('permutation_importance'):
                simplified_results['permutation_importance_scores'] = results['permutation_importance'].get('importance_scores', [])
                logger.info("✓ Permutation importance scores extracted")
            
            # 添加SHAP重要性分析结果
            if "shap" in analysis_types and results.get('shap_importance'):
                simplified_results['shap_importance_scores'] = results['shap_importance'].get('importance_scores', [])
                logger.info("✓ SHAP importance scores extracted")
            
            # 保留报告链接（如果存在）
            if 'report_summary' in results:
                simplified_results['report_summary'] = results['report_summary']
                
            # 保留归档路径（如果存在）
            # if 'archive_path' in results:
            #     simplified_results['archive_path'] = results['archive_path']
            if 'dowland_archive_path' in results:
                simplified_results['download_archive_path'] = results['dowland_archive_path']
            
            logger.info(f"Results simplified - kept only importance_scores for {len([k for k in simplified_results.keys() if 'importance_scores' in k])} analysis types")
            return simplified_results
        
        # Run analysis in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(None, run_analysis)
        
        logger.info("Feature importance analysis completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Feature importance analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def list_models() -> List[Dict[str, Any]]:
    """
    List all available trained models.
    
    Returns:
        List of model information including IDs, names, and metadata
    """
    try:
        logger.info("Listing all available models")
        
        # Run in executor to avoid blocking
        models = await asyncio.get_event_loop().run_in_executor(
            None, model_manager.list_models
        )
        
        logger.info(f"Found {len(models)} available models")
        return models
        
    except Exception as e:
        error_msg = f"Failed to list models: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Unique identifier for the model
        
    Returns:
        Detailed model information including performance metrics and metadata
    """
    try:
        logger.info(f"Getting information for model: {model_id}")
        
        # Run in executor to avoid blocking
        model_info = await asyncio.get_event_loop().run_in_executor(
            None, model_manager.get_model_info, model_id
        )
        
        logger.info("Model information retrieved successfully")
        return model_info
        
    except Exception as e:
        error_msg = f"Failed to get model info: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def delete_model(model_id: str) -> Dict[str, Any]:
    """
    Delete a trained model.
    
    Args:
        model_id: Unique identifier for the model to delete
        
    Returns:
        Deletion status and information
    """
    try:
        logger.info(f"Deleting model: {model_id}")
        
        # Run in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None, model_manager.delete_model, model_id
        )
        
        if result.get('success', False):
            logger.info("Model deleted successfully")
        else:
            logger.warning(f"Model deletion failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to delete model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

# Add a health check tool

@mcp.tool()
async def analyze_local_feature_importance(
    model_id: str,
    sample_data: Union[List[float], List[List[float]], Dict[str, float], List[Dict[str, float]]] = None,
    data_source: str = None,
    plot_types: List[str] = ["waterfall", "force", "decision"],
    generate_plots: bool = True,
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Analyze local feature importance for individual samples using SHAP.
    
    Args:
        model_id: Unique identifier for the trained model
        sample_data: Sample data to analyze (if None, uses data from data_source)
        data_source: Path to data file - if provided, analyzes ALL samples in the file
        plot_types: Types of plots to generate ["waterfall", "force", "decision"]
        generate_plots: Whether to generate visualization plots
        generate_report: Whether to generate analysis report
    Returns:
        Local feature importance analysis results including:
        - raw_shap_data: Contains base_value, SHAP values for each feature, feature names, sample data
        - feature_contributions: Detailed feature-level analysis with values and rankings
        - plot_paths: Paths to generated plots (if generate_plots=True)
        - report_path: Path to HTML report (if generate_report=True)
    """
    try:
        logger.info(f"Analyzing local feature importance for model {model_id}")
        print("--------------------------------") 
        print(f"Analyzing local feature importance for model {model_id}")
        
        # Load model info first
        model_info = model_manager.get_model_info(model_id)
        
        def run_analysis():
            # Load model
            model = model_manager.load_model(model_id)
            # Set up output directory for local feature analysis
            model_dir = Path("trained_models") / model_id
            task_id = str(uuid.uuid4())
            output_dir = model_dir / "feature_analysis" / "local" / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local feature analysis output directory: {output_dir}")
            # Prepare background data (use saved training data)
            X_background, y_background, feature_names = None, None, None
            # Try to load processed data first, then raw data
            processed_data_path = model_dir / "processed_data" / "processed_data.csv"
            if processed_data_path.exists():
                import pandas as pd
                df_background = pd.read_csv(processed_data_path)
                logger.info(f"Using saved processed training data as background: {processed_data_path}")
                use_preprocessing = False  # Data already processed
            else:
                # Fallback to raw data
                raw_data_path = model_dir / "raw_data.csv"
                if raw_data_path.exists():
                    import pandas as pd
                    df_background = pd.read_csv(raw_data_path)
                    logger.info(f"Using saved raw training data as background: {raw_data_path}")
                    use_preprocessing = True  # Need to preprocess raw data
                else:
                    raise FileNotFoundError(f"No training data found for background. Looked for: {processed_data_path} and {raw_data_path}")
            
            # Get target column information
            target_column = model_info.get('target_column')
            if not target_column:
                target_name = model_info.get('target_name', [])
                if target_name:
                    target_column = target_name[0] if isinstance(target_name, list) else target_name
                else:
                    target_column = df_background.columns[-1]
                    logger.warning(f"No target column found in metadata, using last column: {target_column}")
            
            # Separate features and target from background data
            # Handle both single target (string) and multi-target (list) cases
            if isinstance(target_column, list):
                # Multi-target case
                missing_columns = [col for col in target_column if col not in df_background.columns]
                if missing_columns:
                    raise ValueError(f"Target columns {missing_columns} not found in background data")
                X_background_raw = df_background.drop(columns=target_column)
                y_background = df_background[target_column].values
                logger.info(f"Background data loaded (multi-target) - Features shape: {X_background_raw.shape}, Target columns: {target_column}")
            else:
                # Single target case
                if target_column in df_background.columns:
                    X_background_raw = df_background.drop(columns=[target_column])
                    y_background = df_background[target_column].values
                    logger.info(f"Background data loaded (single-target) - Features shape: {X_background_raw.shape}")
                else:
                    raise ValueError(f"Target column '{target_column}' not found in background data")
            
            # Prepare sample data for analysis
            sample_data_to_analyze = None
            original_sample_data = None  # Store original values for feature_contributions
            
            if sample_data is not None:
                # Use provided sample data
                logger.info("Using provided sample data")
                sample_data_to_analyze = sample_data
                # Store original sample data for later use
                original_sample_data = sample_data
                
                # If using provided data, we need the feature names for preprocessing
                if use_preprocessing:
                    feature_names = X_background_raw.columns.tolist()
                    # Convert sample data to match background data format
                    if isinstance(sample_data, dict):
                        # Single sample as dictionary
                        sample_df = pd.DataFrame([sample_data])
                    elif isinstance(sample_data, list) and len(sample_data) > 0:
                        if isinstance(sample_data[0], dict):
                            # Batch of samples as list of dictionaries
                            sample_df = pd.DataFrame(sample_data)
                        elif isinstance(sample_data[0], list):
                            # Batch of samples as list of lists
                            sample_df = pd.DataFrame(sample_data, columns=feature_names)
                        else:
                            # Single sample as list
                            sample_df = pd.DataFrame([{feature_names[i]: val for i, val in enumerate(sample_data)}])
                    else:
                        raise ValueError("Invalid sample_data format")
                    
                    # Ensure sample_df has the same columns as background
                    missing_cols = set(feature_names) - set(sample_df.columns)
                    if missing_cols:
                        raise ValueError(f"Sample data missing required features: {missing_cols}")
                    
                    sample_df = sample_df[feature_names]  # Reorder columns
                    sample_data_to_analyze = sample_df
                
            elif data_source is not None:
                # Load ALL sample data from file
                logger.info(f"Loading ALL samples from data source: {data_source}")
                from .data_utils import DataProcessor
                data_processor = DataProcessor()
                df_samples = data_processor.load_data(data_source)
                
                logger.info(f"Analyzing all {len(df_samples)} samples from file")
                
                # Remove target column if present
                if isinstance(target_column, list):
                    # Multi-target case - remove all target columns that exist
                    columns_to_remove = [col for col in target_column if col in df_samples.columns]
                    if columns_to_remove:
                        sample_data_to_analyze = df_samples.drop(columns=columns_to_remove)
                        logger.info(f"Removed target columns from samples: {columns_to_remove}")
                    else:
                        sample_data_to_analyze = df_samples
                else:
                    # Single target case
                    if target_column in df_samples.columns:
                        sample_data_to_analyze = df_samples.drop(columns=[target_column])
                        logger.info(f"Removed target column from samples: {target_column}")
                    else:
                        sample_data_to_analyze = df_samples
                
                # Store original sample data for feature_contributions
                original_sample_data = sample_data_to_analyze.copy()
            
            else:
                # Use a few samples from background data as default
                logger.info("No sample data provided, using first 3 samples from background data")
                sample_data_to_analyze = X_background_raw.head(3)
                original_sample_data = sample_data_to_analyze.copy()
                use_preprocessing = False  # Same source as background
            
            # Apply preprocessing if needed
            if use_preprocessing:
                preprocessing_pipeline_path = model_dir / "preprocessing_pipeline.pkl"
                if preprocessing_pipeline_path.exists():
                    logger.info("Applying preprocessing to background and sample data...")
                    try:
                        import joblib
                        from .data_preprocessing import DataPreprocessor
                        
                        preprocessor = DataPreprocessor()
                        preprocessor.load_pipeline(str(preprocessing_pipeline_path))
                        
                        # Apply feature preprocessing to background data
                        X_background = preprocessor.transform_features(X_background_raw)
                        
                        # Apply preprocessing to sample data
                        if isinstance(sample_data_to_analyze, pd.DataFrame):
                            X_sample = preprocessor.transform_features(sample_data_to_analyze)
                        else:
                            # Convert to DataFrame first
                            sample_df = pd.DataFrame(sample_data_to_analyze, columns=X_background_raw.columns)
                            X_sample = preprocessor.transform_features(sample_df)
                        
                        # Get processed feature names
                        feature_names = preprocessor.get_feature_names()
                        if not feature_names or len(feature_names) != X_background.shape[1]:
                            feature_names = [f"feature_{i}" for i in range(X_background.shape[1])]
                        
                        logger.info(f"Preprocessing applied - Background: {X_background.shape}, Sample: {X_sample.shape}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to apply preprocessing: {e}")
                        logger.info("Using raw data")
                        X_background = X_background_raw.values
                        X_sample = sample_data_to_analyze.values if hasattr(sample_data_to_analyze, 'values') else np.array(sample_data_to_analyze)
                        feature_names = X_background_raw.columns.tolist()
                else:
                    logger.info("No preprocessing pipeline found, using raw data")
                    X_background = X_background_raw.values
                    X_sample = sample_data_to_analyze.values if hasattr(sample_data_to_analyze, 'values') else np.array(sample_data_to_analyze)
                    feature_names = X_background_raw.columns.tolist()
            else:
                # Data is already processed or from same source
                X_background = X_background_raw.values if hasattr(X_background_raw, 'values') else X_background_raw
                
                # Handle sample data conversion to DataFrame if needed
                if not hasattr(sample_data_to_analyze, 'values'):
                    # Convert raw sample data to DataFrame if it's not already
                    feature_names_for_df = X_background_raw.columns.tolist() if hasattr(X_background_raw, 'columns') else [f"feature_{i}" for i in range(X_background.shape[1])]
                    
                    if isinstance(sample_data_to_analyze, list) and len(sample_data_to_analyze) > 0:
                        if isinstance(sample_data_to_analyze[0], dict):
                            # List of dictionaries
                            sample_data_to_analyze = pd.DataFrame(sample_data_to_analyze)
                        elif isinstance(sample_data_to_analyze[0], list):
                            # List of lists
                            sample_data_to_analyze = pd.DataFrame(sample_data_to_analyze, columns=feature_names_for_df)
                        else:
                            # Single sample as list
                            sample_data_to_analyze = pd.DataFrame([{feature_names_for_df[i]: val for i, val in enumerate(sample_data_to_analyze)}])
                    elif isinstance(sample_data_to_analyze, dict):
                        # Single sample as dictionary
                        sample_data_to_analyze = pd.DataFrame([sample_data_to_analyze])
                
                X_sample = sample_data_to_analyze.values if hasattr(sample_data_to_analyze, 'values') else np.array(sample_data_to_analyze)
                feature_names = X_background_raw.columns.tolist() if hasattr(X_background_raw, 'columns') else [f"feature_{i}" for i in range(X_background.shape[1])]
            
            # Ensure correct data types
            if not isinstance(X_background, np.ndarray):
                X_background = np.array(X_background)
            if not isinstance(X_sample, np.ndarray):
                X_sample = np.array(X_sample)
            
            # Ensure sample data has correct dimensions
            if X_sample.ndim == 1:
                X_sample = X_sample.reshape(1, -1)
            
            logger.info(f"Final data shapes - Background: {X_background.shape}, Sample: {X_sample.shape}")
            
            # Load preprocessing pipeline for target inverse transformation
            preprocessor = None
            try:
                preprocessing_path = model_dir / "preprocessing_pipeline.pkl"
                if preprocessing_path.exists():
                    from .data_preprocessing import DataPreprocessor
                    preprocessor = DataPreprocessor()
                    preprocessor.load_pipeline(str(preprocessing_path))
                    logger.info("Loaded preprocessing pipeline for target inverse transformation")
                else:
                    logger.warning("Preprocessing pipeline not found - predictions will remain in scaled form")
            except Exception as e:
                logger.warning(f"Failed to load preprocessing pipeline: {e} - predictions will remain in scaled form")

            # Run the analysis
            from .local_feature_importance import LocalFeatureImportanceAnalyzer
            analyzer = LocalFeatureImportanceAnalyzer(output_dir=str(output_dir))
            
            # Pre-set target names if available in model metadata
            if model_info and 'target_name' in model_info:
                target_names = model_info.get('target_name', [])
                if isinstance(target_names, str):
                    target_names = [target_names]
                analyzer.target_names = target_names
                analyzer.target_dimension = model_info.get('target_dimension', 1)
                logger.info(f"Pre-set target names in analyzer: {target_names}")
            
            # Set preprocessor in analyzer for target inverse transformation
            analyzer.preprocessor = preprocessor
            
            # Determine analysis type based on sample data
            logger.info(f"Starting analysis - Sample shape: {X_sample.shape}, Is multi-target: {model_info.get('target_dimension', 1) > 1}")
            
            if isinstance(X_sample, np.ndarray) and X_sample.shape[0] == 1:
                # Single sample analysis
                logger.info("Running single sample analysis...")
                analysis_results = analyzer.analyze_sample_importance(
                    model=model,
                    sample_data=X_sample,
                    background_data=X_background,
                    feature_names=feature_names,
                    sample_index=0,
                    model_metadata=model_info,
                    original_sample_data=original_sample_data
                )
            else:
                # Batch analysis
                logger.info("Running batch analysis...")
                analysis_results = analyzer.analyze_batch_importance(
                    model=model,
                    batch_data=X_sample,
                    background_data=X_background,
                    feature_names=feature_names,
                    max_samples=50,
                    model_metadata=model_info,
                    original_batch_data=original_sample_data
                )
            
            # Generate plots (optional)
            if generate_plots:
                # Create plots for first few samples and aggregate analysis
                sample_indices_to_plot = list(range(min(5, X_sample.shape[0])))  # Plot first 5 samples max
                plot_paths = analyzer.create_all_plots(
                    sample_index=0,
                    additional_samples=sample_indices_to_plot[1:],
                    save_plots=True
                )
                analysis_results['plot_paths'] = plot_paths
                logger.info(f"✓ Generated {len(plot_paths)} plots for local analysis")
            
            # Generate report (optional)
            if generate_report:
                plot_paths = analysis_results.get('plot_paths', {})
                logger.info(f"Generating report with {len(plot_paths)} plots")
                report_path = analyzer.generate_report(
                    analysis_results=analysis_results,
                    plot_paths=plot_paths,
                    format_type="html"
                )
                analysis_results['html_report'] = f"You can find the html local feature importance report summary in {base_url}/static/{Path(report_path).relative_to(root_dir.as_posix())}"
            
            # Add raw SHAP data for direct access
            if hasattr(analyzer, 'shap_values') and analyzer.shap_values is not None:
                raw_shap_data = {
                    'base_value': analyzer.expected_value,
                    'shap_values': analyzer.shap_values,
                    'feature_names': feature_names,
                    'sample_data': X_sample.tolist(),
                    'data_shape': {
                        'n_samples': X_sample.shape[0],
                        'n_features': X_sample.shape[1]
                    }
                }
                
                # Handle different SHAP value formats
                if isinstance(analyzer.shap_values, list):
                    # Classification case - convert to lists for JSON serialization
                    raw_shap_data['shap_values'] = [sv.tolist() for sv in analyzer.shap_values]
                    raw_shap_data['base_value'] = analyzer.expected_value.tolist() if isinstance(analyzer.expected_value, np.ndarray) else analyzer.expected_value
                    raw_shap_data['analysis_type'] = 'classification'
                    raw_shap_data['n_classes'] = len(analyzer.shap_values)
                elif isinstance(analyzer.shap_values, np.ndarray):
                    # Regression case
                    raw_shap_data['shap_values'] = analyzer.shap_values.tolist()
                    raw_shap_data['base_value'] = analyzer.expected_value.tolist() if isinstance(analyzer.expected_value, np.ndarray) else analyzer.expected_value
                    if analyzer.shap_values.ndim == 3:
                        raw_shap_data['analysis_type'] = 'multi_target_regression'
                        raw_shap_data['n_targets'] = analyzer.shap_values.shape[2]
                    else:
                        raw_shap_data['analysis_type'] = 'single_target_regression'
                
                # analysis_results['raw_shap_data'] = raw_shap_data
                # analysis_results['sample_data'] = X_sample.tolist()
                analysis_results['analysis_type'] = raw_shap_data['analysis_type']
            
            # Add analysis summary
            # analysis_results['analysis_summary'] = {
            #     'model_id': model_id,
            #     'n_samples_analyzed': X_sample.shape[0],
            #     'n_features': X_sample.shape[1],
            #     'n_background_samples': X_background.shape[0],
            #     'analysis_type': 'single_sample' if X_sample.shape[0] == 1 else 'batch',
                # 'output_directory': str(output_dir),
                # 'data_source_used': data_source if data_source else 'provided_sample_data' if sample_data else 'background_data_subset',
            # }
            
            # Create analysis archive
            try:
                archive_path = _create_analysis_archive(
                    output_dir=output_dir,
                    analysis_type="local",
                    task_id=task_id,
                    model_id=model_id
                )
                if archive_path:
                    print("*"*100)
                    print("archive_path",archive_path)
                    analysis_results['download_archive_path'] = f"You can download the local feature importance analysis details in {base_url}/download/file/{Path(archive_path).relative_to(root_dir.parent).as_posix()}"
                    # analysis_results['task_id'] = task_id
                    logger.info(f"Local analysis results archived: {archive_path}")
            except Exception as e:
                logger.warning(f"Failed to archive local analysis results: {e}")
            
            logger.info(f"Local feature importance analysis completed for {X_sample.shape[0]} samples")

            #简化返回结果
            if  "plot_paths" in analysis_results.keys():
                del analysis_results['plot_paths']                


            return analysis_results
        
        # Run analysis in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_analysis)
        
        return {
            "status": "success",
            "message": f"Local feature importance analysis completed for model {model_id}",
            "results": result
        }
        
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        logger.error(f"Error in local feature importance analysis: {str(e)}")
        logger.error(f"Full traceback:\n{full_traceback}")
        return {
            "status": "error",
            "message": f"Failed to analyze local feature importance: {str(e)}",
            "error_details": str(e),
            "traceback": full_traceback
        }

def _create_analysis_archive(output_dir: Path, analysis_type: str, task_id: str, model_id: str) -> str:

    """创建分析结果的ZIP压缩包"""
    try:
        # 创建archives目录
        archives_dir = Path("trained_models") / model_id / "feature_analysis" / "archives"
        archives_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建ZIP文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{analysis_type}_analysis_{task_id}_{timestamp}.zip"
        zip_path = archives_dir / zip_filename
        
        # 创建ZIP压缩包
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)
        
        return str(zip_path)
    except Exception as e:
        logger.error(f"Failed to create analysis archive: {e}")
        return None
