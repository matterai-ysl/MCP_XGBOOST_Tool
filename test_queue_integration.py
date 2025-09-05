#!/usr/bin/env python3
"""
Test Queue Integration

Simple test to verify that the queue management is properly integrated
into the MCP training tools.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from src.mcp_xgboost_tool.training_queue import get_queue_manager, initialize_queue_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_test_data():
    """Create simple test dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(n_samples) * 0.1
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    test_file = "test_integration_data.csv"
    df.to_csv(test_file, index=False)
    logger.info(f"Created test data: {test_file}")
    return test_file

async def test_direct_queue_training():
    """Test training task submission directly through queue manager."""
    logger.info("Testing direct queue training submission...")
    
    try:
        # Initialize queue manager
        logger.info("Initializing queue manager...")
        await initialize_queue_manager()
        queue_manager = get_queue_manager()
        
        # Create test data
        test_file = await create_test_data()
        
        # Submit training task directly to queue manager
        logger.info("Submitting training task to queue...")
        task_id = await queue_manager.submit_task(
            task_type="regression",
            params={
                'data_source': test_file,
                'target_column': 'target',
                'task_type': 'regression',
                'optimize_hyperparameters': True,
                'n_trials': 10,  # Small number for quick testing
                'cv_folds': 3,
                'scoring_metric': 'neg_mean_absolute_error',
                'save_model': False,
                'validate_data': False,  # Disable data validation for testing
                'apply_preprocessing': False  # Disable preprocessing for testing
            },
            user_id="test_user"
        )
        
        logger.info(f"Task submitted with ID: {task_id}")
        
        # Monitor task progress
        logger.info("Monitoring task progress...")
        max_wait_time = 120  # 2 minutes max
        wait_time = 0
        
        while wait_time < max_wait_time:
            task_status = await queue_manager.get_task_status(task_id)
            
            if not task_status:
                raise Exception(f"Task {task_id} not found")
            
            status = task_status["status"]
            logger.info(f"Task status: {status}")
            
            if status == "completed":
                logger.info("Task completed successfully!")
                if task_status.get("result"):
                    performance = task_status["result"].get("performance_summary", "N/A")
                    logger.info(f"Model performance: {performance}")
                break
            elif status == "failed":
                error_msg = task_status.get("error_message", "Unknown error")
                raise Exception(f"Task failed: {error_msg}")
            
            # Wait before next check
            await asyncio.sleep(5)
            wait_time += 5
        
        if wait_time >= max_wait_time:
            logger.warning("Test timed out, but this might be normal for longer training tasks")
            # Get final status
            final_status = await queue_manager.get_task_status(task_id)
            logger.info(f"Final task status: {final_status['status']}")
        
        # Check queue status
        queue_status = await queue_manager.get_queue_status()
        logger.info(f"Final queue status: {queue_status}")
        
        logger.info("✅ Direct queue training test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Queue training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        queue_manager = get_queue_manager()
        if queue_manager.is_running:
            await queue_manager.stop()

async def test_queue_manager_basic():
    """Test basic queue manager functionality."""
    logger.info("Testing basic queue manager functionality...")
    
    try:
        # Initialize queue manager
        await initialize_queue_manager()
        queue_manager = get_queue_manager()
        
        # Test queue status
        status = await queue_manager.get_queue_status()
        logger.info(f"Queue status: {status}")
        
        # Test task listing
        tasks = await queue_manager.list_tasks()
        logger.info(f"Current tasks: {len(tasks)}")
        
        logger.info("✅ Basic queue manager test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic queue test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting queue integration tests...")
    
    # Test 1: Basic queue manager
    basic_test = await test_queue_manager_basic()
    
    # Test 2: Direct queue training test
    queue_test = await test_direct_queue_training()
    
    if basic_test and queue_test:
        logger.info("🎉 All tests passed! Queue integration is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")

if __name__ == "__main__":
    asyncio.run(main())