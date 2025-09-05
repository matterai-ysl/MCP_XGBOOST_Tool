#!/usr/bin/env python3
"""
Test Academic Report Generation
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from src.mcp_xgboost_tool.training_queue import get_queue_manager, initialize_queue_manager

async def test_academic_report():
    """Test academic report generation in async training."""
    # Create test data 
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(40),
        'feature2': np.random.randn(40) * 2 + 5,
        'target': np.random.choice(['Yes', 'No'], 40)
    }
    df = pd.DataFrame(data)
    df.to_csv('test_academic.csv', index=False)
    
    # Initialize queue
    await initialize_queue_manager()
    queue_manager = get_queue_manager()
    
    print("Testing academic report generation...")
    
    # Submit task with model saving enabled
    task_id = await queue_manager.submit_task(
        task_type="classification",
        params={
            'data_source': 'test_academic.csv',
            'target_column': 'target',
            'task_type': 'classification',
            'optimize_hyperparameters': False,
            'save_model': True,
            'validate_data': False,
            'apply_preprocessing': True,
            'cv_folds': 3
        },
        user_id="academic_test"
    )
    
    print(f"Task submitted: {task_id}")
    
    # Wait for completion
    max_wait = 60
    wait_time = 0
    
    while wait_time < max_wait:
        status = await queue_manager.get_task_status(task_id)
        print(f"Status: {status['status']} (waited {wait_time}s)")
        
        if status["status"] == "completed":
            result = status.get("result", {})
            print("✅ Task completed successfully!")
            
            model_directory = Path(result.get("model_directory", ""))
            
            if model_directory:
                # Check for academic report in model directory root (academic_report.md)
                academic_report_path = model_directory / "academic_report.md"
                if academic_report_path.exists():
                    print(f"✅ Academic report found: {academic_report_path}")
                    return True
                else:
                    # Also check reports subdirectory for any academic files
                    reports_dir = model_directory / "reports"
                    if reports_dir.exists():
                        academic_files = list(reports_dir.glob("*academic*"))
                        if academic_files:
                            print(f"✅ Academic report files found in reports: {[f.name for f in academic_files]}")
                            return True
                        else:
                            print("❌ No academic report files found")
                            print(f"Model directory contents: {list(model_directory.glob('*'))}")
                            print(f"Reports directory contents: {list(reports_dir.glob('*'))}")
                            return False
                    else:
                        print("❌ Academic report not found in model directory")
                        print(f"Model directory contents: {list(model_directory.glob('*'))}")
                        return False
            else:
                print("❌ Model directory not found in results")
                return False
                
        elif status["status"] == "failed":
            print(f"❌ Task failed: {status.get('error_message', 'Unknown error')}")
            return False
        
        await asyncio.sleep(3)
        wait_time += 3
    
    print("❌ Task timed out")
    return False

async def main():
    """Main test function."""
    success = await test_academic_report()
    
    # Clean up
    queue_manager = get_queue_manager()
    if queue_manager.is_running:
        await queue_manager.stop()
    
    if success:
        print("🎉 Academic report test passed!")
    else:
        print("❌ Academic report test failed!")
    
    # Clean up test file
    import os
    try:
        os.remove('test_academic.csv')
    except:
        pass

if __name__ == "__main__":
    asyncio.run(main())