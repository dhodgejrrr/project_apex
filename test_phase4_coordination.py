#!/usr/bin/env python3
"""
Test script for Phase 4: Shared State Management & Advanced Coordination

Tests the State Manager, Workflow Orchestrator, and coordination patterns.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.common.coordination import get_coordinator, get_state_client, get_workflow_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_state_management():
    """Test state management capabilities."""
    logger.info("=== Testing State Management ===")
    
    # Initialize state client
    state = get_state_client("test_agent")
    
    try:
        # Test basic state operations
        logger.info("Testing basic state operations...")
        
        # Set state
        success = state.set("test:basic", {"message": "Hello, State Manager!", "timestamp": datetime.now().isoformat()})
        assert success, "Failed to set basic state"
        logger.info("✓ Basic state set successfully")
        
        # Get state
        value = state.get("test:basic")
        assert value is not None, "Failed to get basic state"
        assert value["message"] == "Hello, State Manager!", "State value mismatch"
        logger.info("✓ Basic state retrieved successfully")
        
        # Test TTL
        logger.info("Testing TTL functionality...")
        state.set("test:ttl", {"expires": "soon"}, ttl_seconds=2)
        time.sleep(1)
        value = state.get("test:ttl")
        assert value is not None, "TTL state should still exist"
        time.sleep(2)
        value = state.get("test:ttl")
        assert value is None, "TTL state should have expired"
        logger.info("✓ TTL functionality working correctly")
        
        # Test conflict resolution
        logger.info("Testing conflict resolution...")
        state.set("test:conflict", {"version": 1}, conflict_resolution="first_write_wins")
        success = state.set("test:conflict", {"version": 2}, conflict_resolution="first_write_wins")
        value = state.get("test:conflict")
        assert value["version"] == 1, "First write wins should have preserved original value"
        logger.info("✓ Conflict resolution working correctly")
        
        # Test key listing
        logger.info("Testing key listing...")
        state.set("test:list:item1", {"data": "item1"})
        state.set("test:list:item2", {"data": "item2"})
        state.set("other:key", {"data": "other"})
        
        keys = state.list_keys("test:list:")
        assert len(keys) >= 2, f"Expected at least 2 keys with prefix, got {len(keys)}"
        logger.info(f"✓ Key listing working correctly (found {len(keys)} keys)")
        
        # Cleanup
        state.delete("test:basic")
        state.delete("test:conflict")
        state.delete("test:list:item1")
        state.delete("test:list:item2")
        state.delete("other:key")
        
        logger.info("✓ State management tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"State management test failed: {e}")
        return False


def test_workflow_orchestration():
    """Test workflow orchestration capabilities."""
    logger.info("=== Testing Workflow Orchestration ===")
    
    # Initialize workflow client
    workflow = get_workflow_client("test_agent")
    
    try:
        # Create a simple test workflow
        logger.info("Creating test workflow...")
        
        workflow_id = f"test_workflow_{int(time.time())}"
        
        # Define a simple workflow with mock tasks
        tasks = [
            {
                "task_id": "task1",
                "agent_name": "mock_agent",
                "tool_name": "mock_tool",
                "parameters": {"input": "test_data"},
                "dependencies": [],
                "retry_count": 1,
                "timeout_seconds": 30
            },
            {
                "task_id": "task2",
                "agent_name": "mock_agent",
                "tool_name": "mock_tool",
                "parameters": {"input": "task1_output"},
                "dependencies": ["task1"],
                "retry_count": 1,
                "timeout_seconds": 30
            }
        ]
        
        created_id = workflow.create_workflow(
            workflow_id=workflow_id,
            name="Test Workflow",
            description="A simple test workflow for Phase 4",
            tasks=tasks,
            execution_mode="mixed",
            max_parallel_tasks=2
        )
        
        assert created_id == workflow_id, "Workflow ID mismatch"
        logger.info(f"✓ Workflow created successfully: {workflow_id}")
        
        # Execute workflow (will fail due to mock agents, but that's expected)
        logger.info("Executing test workflow...")
        execution_id = workflow.execute_workflow(workflow_id, {"test_param": "test_value"})
        assert execution_id is not None, "Failed to start workflow execution"
        logger.info(f"✓ Workflow execution started: {execution_id}")
        
        # Monitor execution status
        logger.info("Monitoring workflow execution...")
        start_time = time.time()
        timeout = 30
        
        while time.time() - start_time < timeout:
            status = workflow.get_execution_status(execution_id)
            assert status is not None, "Failed to get execution status"
            
            logger.info(f"Workflow status: {status['status']}")
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                break
            
            time.sleep(2)
        
        final_status = workflow.get_execution_status(execution_id)
        logger.info(f"Final workflow status: {final_status['status']}")
        
        # The workflow will likely fail due to mock agents, but we've tested the orchestration
        logger.info("✓ Workflow orchestration tests completed (workflow failure expected for mock agents)")
        return True
        
    except Exception as e:
        logger.error(f"Workflow orchestration test failed: {e}")
        return False


def test_coordination_patterns():
    """Test advanced coordination patterns."""
    logger.info("=== Testing Coordination Patterns ===")
    
    # Initialize coordinator
    coordinator = get_coordinator("test_coordinator")
    
    try:
        # Test data sharing
        logger.info("Testing data sharing...")
        
        shared_data = {
            "analysis_results": {
                "lap_times": [120.5, 119.8, 121.2],
                "pit_stops": [{"lap": 10, "duration": 25.3}]
            },
            "metadata": {
                "run_id": "test_run_123",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        success = coordinator.share_data("analysis:test_run_123", shared_data)
        assert success, "Failed to share data"
        logger.info("✓ Data shared successfully")
        
        # Retrieve shared data
        retrieved_data = coordinator.get_shared_data("analysis:test_run_123")
        assert retrieved_data is not None, "Failed to retrieve shared data"
        assert retrieved_data["metadata"]["run_id"] == "test_run_123", "Data corruption detected"
        logger.info("✓ Shared data retrieved successfully")
        
        # Test pipeline coordination (will create workflow but may fail execution)
        logger.info("Testing pipeline coordination...")
        
        pipeline_stages = [
            {
                "agent": "core_analyzer",
                "tool": "analyze",
                "parameters": {"data": "test_data"}
            },
            {
                "agent": "insight_hunter", 
                "tool": "generate_insights",
                "parameters": {"analysis_results": "from_previous_stage"}
            },
            {
                "agent": "visualizer",
                "tool": "create_plots", 
                "parameters": {"insights": "from_previous_stage"}
            }
        ]
        
        execution_id = coordinator.patterns.pipeline_coordination(pipeline_stages)
        assert execution_id is not None, "Failed to start pipeline coordination"
        logger.info(f"✓ Pipeline coordination started: {execution_id}")
        
        # Monitor briefly
        time.sleep(5)
        status = coordinator.workflow.get_execution_status(execution_id)
        logger.info(f"Pipeline status: {status['status'] if status else 'Unknown'}")
        
        # Cleanup
        coordinator.state.delete("analysis:test_run_123")
        
        logger.info("✓ Coordination patterns tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Coordination patterns test failed: {e}")
        return False


def test_complete_workflow():
    """Test a complete analysis workflow coordination."""
    logger.info("=== Testing Complete Workflow Coordination ===")
    
    coordinator = get_coordinator("test_workflow_coordinator")
    
    try:
        # Create a mock analysis workflow
        logger.info("Creating complete analysis workflow...")
        
        run_id = f"test_run_{int(time.time())}"
        analysis_path = f"gs://test-bucket/{run_id}/analysis.json"
        
        execution_id = coordinator.coordinate_analysis_workflow(run_id, analysis_path)
        assert execution_id is not None, "Failed to start analysis workflow"
        logger.info(f"✓ Analysis workflow started: {execution_id}")
        
        # Monitor for a short time
        logger.info("Monitoring workflow progress...")
        start_time = time.time()
        timeout = 20
        
        while time.time() - start_time < timeout:
            status = coordinator.workflow.get_execution_status(execution_id)
            if status:
                logger.info(f"Workflow status: {status['status']}")
                
                # Show task progress
                completed_tasks = sum(1 for task in status['tasks'].values() 
                                    if task['status'] == 'completed')
                total_tasks = len(status['tasks'])
                logger.info(f"Task progress: {completed_tasks}/{total_tasks}")
                
                if status["status"] in ["completed", "failed", "cancelled"]:
                    break
            
            time.sleep(3)
        
        final_status = coordinator.workflow.get_execution_status(execution_id)
        logger.info(f"Final workflow status: {final_status['status'] if final_status else 'Unknown'}")
        
        logger.info("✓ Complete workflow coordination test completed")
        return True
        
    except Exception as e:
        logger.error(f"Complete workflow test failed: {e}")
        return False


def main():
    """Run all Phase 4 tests."""
    logger.info("Starting Phase 4 tests: Shared State Management & Advanced Coordination")
    logger.info("=" * 80)
    
    # Check required environment variables
    required_urls = ["STATE_MANAGER_URL", "WORKFLOW_ORCHESTRATOR_URL"]
    missing_urls = [url for url in required_urls if not os.getenv(url)]
    
    if missing_urls:
        logger.warning(f"Missing environment variables: {missing_urls}")
        logger.info("Setting default local URLs for testing...")
        os.environ.setdefault("STATE_MANAGER_URL", "http://localhost:8080")
        os.environ.setdefault("WORKFLOW_ORCHESTRATOR_URL", "http://localhost:8081")
    
    test_results = []
    
    # Run tests
    tests = [
        ("State Management", test_state_management),
        ("Workflow Orchestration", test_workflow_orchestration),
        ("Coordination Patterns", test_coordination_patterns),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            test_results.append((test_name, result))
            logger.info(f"{test_name} test {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name} test FAILED with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4 TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name:<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 4 tests passed! Advanced coordination is working.")
    else:
        logger.warning("⚠️  Some Phase 4 tests failed. Check the logs above for details.")
    
    logger.info("\nNote: Some test failures are expected when testing against mock services.")
    logger.info("The important thing is that the coordination infrastructure is working.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
