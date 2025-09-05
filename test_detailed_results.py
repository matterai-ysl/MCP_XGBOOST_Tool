#!/usr/bin/env python3
"""
Test Detailed Training Results

This script tests the complete training workflow with detailed result retrieval
to demonstrate that all training information is preserved and accessible.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import json
from src.mcp_xgboost_tool.training_queue import get_queue_manager, initialize_queue_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_test_data():
    """Create test dataset for training."""
    np.random.seed(42)
    n_samples = 200
    n_features = 8
    
    # Create synthetic dataset with known relationships
    X = np.random.randn(n_samples, n_features)
    # Create target with clear feature relationships
    y = (X[:, 0] * 2 + X[:, 1] * -1.5 + X[:, 2] * 0.8 + 
         X[:, 3] * 0.3 + np.random.randn(n_samples) * 0.1)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    test_file = "test_detailed_results.csv"
    df.to_csv(test_file, index=False)
    logger.info(f"Created test data: {test_file}")
    return test_file

async def test_complete_training_workflow():
    """Test the complete training workflow with detailed results."""
    logger.info("Testing complete training workflow with detailed results...")
    
    try:
        # Initialize queue manager
        await initialize_queue_manager()
        queue_manager = get_queue_manager()
        
        # Create test data
        test_file = await create_test_data()
        
        # Submit training task with optimization
        logger.info("Submitting training task...")
        task_id = await queue_manager.submit_task(
            task_type="regression",
            params={
                'data_source': test_file,
                'target_column': 'target',
                'task_type': 'regression',
                'optimize_hyperparameters': True,
                'n_trials': 15,
                'cv_folds': 5,
                'scoring_metric': 'neg_mean_absolute_error',
                'save_model': True,
                'validate_data': False,
                'apply_preprocessing': False
            },
            user_id="test_detailed_user"
        )
        
        logger.info(f"Task submitted: {task_id}")
        
        # Monitor progress
        logger.info("Monitoring task progress...")
        max_wait_time = 180  # 3 minutes
        wait_time = 0
        
        while wait_time < max_wait_time:
            task_status = await queue_manager.get_task_status(task_id)
            
            if not task_status:
                raise Exception(f"Task {task_id} not found")
            
            status = task_status["status"]
            logger.info(f"Task status: {status} (waited {wait_time}s)")
            
            if status == "completed":
                logger.info("✅ Task completed successfully!")
                break
            elif status == "failed":
                error_msg = task_status.get("error_message", "Unknown error")
                raise Exception(f"Task failed: {error_msg}")
            
            await asyncio.sleep(5)
            wait_time += 5
        
        if wait_time >= max_wait_time:
            logger.warning("Task did not complete within timeout")
            return False
        
        # Get detailed results
        logger.info("Retrieving detailed training results...")
        final_task_status = await queue_manager.get_task_status(task_id)
        
        if final_task_status["status"] == "completed" and final_task_status.get("result"):
            result = final_task_status["result"]
            
            # Display comprehensive results
            logger.info("=" * 60)
            logger.info("📊 TRAINING RESULTS SUMMARY")
            logger.info("=" * 60)
            
            logger.info(f"✅ Model ID: {result.get('model_id')}")
            logger.info(f"📁 Model Directory: {result.get('model_directory')}")
            logger.info(f"⏱️ Training Time: {result.get('training_time_seconds', 0):.2f} seconds")
            logger.info(f"🎯 Task Type: {result.get('task_type')}")
            logger.info(f"📈 Performance: {result.get('performance_summary', 'N/A')}")
            
            # Feature importance
            feature_importance = result.get('feature_importance', [])
            if feature_importance:
                logger.info(f"🔍 Feature Count: {len(feature_importance)}")
                logger.info("🌟 Top 3 Important Features:")
                for i, feat in enumerate(feature_importance[:3]):
                    logger.info(f"   {i+1}. {feat.get('feature', 'Unknown')}: {feat.get('importance', 0):.4f}")
            
            # Cross-validation results
            cv_results = result.get('cross_validation_results', {})
            if cv_results:
                test_scores = cv_results.get('test_scores', {})
                logger.info("📊 Cross-Validation Scores:")
                for metric, scores in test_scores.items():
                    if isinstance(scores, dict) and 'mean' in scores:
                        mean_score = scores['mean']
                        std_score = scores.get('std', 0)
                        logger.info(f"   {metric}: {mean_score:.4f} ± {std_score:.4f}")
            
            # Hyperparameter optimization
            opt_results = result.get('optimization_results', {})
            if opt_results:
                logger.info("🎛️ Hyperparameter Optimization:")
                logger.info(f"   Best Score: {opt_results.get('best_score', 'N/A')}")
                logger.info(f"   Trials: {opt_results.get('n_trials', 'N/A')}")
                best_params = opt_results.get('best_params', {})
                if best_params:
                    logger.info("   Best Parameters:")
                    for param, value in list(best_params.items())[:5]:  # Show top 5
                        if isinstance(value, float):
                            logger.info(f"     {param}: {value:.6f}")
                        else:
                            logger.info(f"     {param}: {value}")
            
            # Model metadata
            metadata = result.get('metadata', {})
            if metadata:
                data_shape = metadata.get('data_shape', {})
                logger.info("📋 Dataset Information:")
                logger.info(f"   Samples: {data_shape.get('n_samples', 'N/A')}")
                logger.info(f"   Features: {data_shape.get('n_features', 'N/A')}")
                logger.info(f"   Preprocessing: {metadata.get('preprocessing_applied', 'N/A')}")
            
            logger.info("=" * 60)
            logger.info("🎉 All training information successfully retrieved!")
            
            # Test JSON serialization (ensure results can be transmitted)
            try:
                json_result = json.dumps(result, indent=2, default=str)
                logger.info(f"📦 Results JSON size: {len(json_result):,} characters")
                logger.info("✅ Results are JSON serializable")
            except Exception as e:
                logger.warning(f"⚠️ JSON serialization issue: {e}")
            
            return True
        else:
            logger.error("❌ No detailed results available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        queue_manager = get_queue_manager()
        if queue_manager.is_running:
            await queue_manager.stop()

async def main():
    """Main test function."""
    logger.info("Starting detailed training results test...")
    
    success = await test_complete_training_workflow()
    
    if success:
        logger.info("🎉 Test passed! All training information is preserved and accessible.")
    else:
        logger.error("❌ Test failed! Training information may not be properly preserved.")

if __name__ == "__main__":
    asyncio.run(main())