#!/usr/bin/env python3
"""
Example workflow demonstrating Phase 4 coordination capabilities.

This script shows how to use the State Manager, Workflow Orchestrator,
and coordination patterns for complex multi-agent workflows.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.common.coordination import get_coordinator, get_state_client, get_workflow_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def example_data_sharing():
    """Demonstrate data sharing between agents."""
    logger.info("=== Example: Data Sharing ===")
    
    # Initialize state client
    state = get_state_client("example_agent")
    
    # Share analysis data
    analysis_data = {
        "run_id": "example_run_2025",
        "lap_times": [120.5, 119.8, 121.2, 120.1, 119.9],
        "pit_stops": [
            {"lap": 10, "duration": 25.3, "tire_change": True},
            {"lap": 20, "duration": 24.8, "tire_change": True}
        ],
        "weather": {"temperature": 25, "humidity": 60, "wind_speed": 5},
        "track": "Road America",
        "session": "Race",
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info("Sharing analysis data...")
    success = state.set("analysis:example_run_2025", analysis_data, ttl_seconds=3600)
    if success:
        logger.info("✓ Analysis data shared successfully")
    else:
        logger.error("✗ Failed to share analysis data")
        return
    
    # Another agent retrieves the data
    logger.info("Retrieving shared data...")
    retrieved_data = state.get("analysis:example_run_2025")
    if retrieved_data:
        logger.info(f"✓ Retrieved data for run: {retrieved_data['run_id']}")
        logger.info(f"  Track: {retrieved_data['track']}")
        logger.info(f"  Lap times: {len(retrieved_data['lap_times'])} laps")
        logger.info(f"  Pit stops: {len(retrieved_data['pit_stops'])} stops")
    else:
        logger.error("✗ Failed to retrieve shared data")
        return
    
    # Share insights based on the analysis
    insights_data = {
        "run_id": "example_run_2025",
        "insights": [
            "Consistent lap times indicate good tire management",
            "Pit stop timing was optimal for track position",
            "Weather conditions were favorable for performance"
        ],
        "recommendations": [
            "Continue current tire strategy",
            "Maintain aggressive pit timing",
            "Focus on consistency over single-lap pace"
        ],
        "generated_by": "insight_hunter",
        "timestamp": datetime.now().isoformat()
    }
    
    state.set("insights:example_run_2025", insights_data, ttl_seconds=3600)
    logger.info("✓ Insights data shared")
    
    # List all keys for this run
    keys = state.list_keys("example_run_2025")
    logger.info(f"✓ Found {len(keys)} data entries for this run")
    
    logger.info("Data sharing example completed successfully!\n")


def example_simple_workflow():
    """Demonstrate a simple workflow creation and execution."""
    logger.info("=== Example: Simple Workflow ===")
    
    # Initialize workflow client
    workflow = get_workflow_client("example_orchestrator")
    
    # Define a simple analysis workflow
    workflow_id = f"example_analysis_{int(time.time())}"
    
    tasks = [
        {
            "task_id": "data_preparation",
            "agent_name": "core_analyzer",
            "tool_name": "prepare_data",
            "parameters": {
                "run_id": "example_run_2025",
                "data_source": "gs://example-bucket/raw_data.csv"
            },
            "dependencies": [],
            "retry_count": 2,
            "timeout_seconds": 60
        },
        {
            "task_id": "analysis",
            "agent_name": "core_analyzer", 
            "tool_name": "analyze_performance",
            "parameters": {
                "run_id": "example_run_2025",
                "analysis_type": "comprehensive"
            },
            "dependencies": ["data_preparation"],
            "retry_count": 2,
            "timeout_seconds": 120
        },
        {
            "task_id": "insights",
            "agent_name": "insight_hunter",
            "tool_name": "generate_insights",
            "parameters": {
                "analysis_path": "analysis:example_run_2025",
                "use_autonomous": True
            },
            "dependencies": ["analysis"],
            "retry_count": 1,
            "timeout_seconds": 90
        }
    ]
    
    logger.info("Creating workflow definition...")
    try:
        created_id = workflow.create_workflow(
            workflow_id=workflow_id,
            name="Example Analysis Workflow",
            description="Demonstrates basic workflow orchestration",
            tasks=tasks,
            execution_mode="mixed",
            max_parallel_tasks=2,
            timeout_seconds=600
        )
        logger.info(f"✓ Workflow created: {created_id}")
    except Exception as e:
        logger.error(f"✗ Failed to create workflow: {e}")
        return
    
    # Execute the workflow
    logger.info("Executing workflow...")
    try:
        execution_id = workflow.execute_workflow(workflow_id, {
            "priority": "high",
            "notify_on_completion": True
        })
        logger.info(f"✓ Workflow execution started: {execution_id}")
    except Exception as e:
        logger.error(f"✗ Failed to execute workflow: {e}")
        return
    
    # Monitor execution (briefly for demo)
    logger.info("Monitoring workflow execution...")
    for i in range(10):  # Monitor for ~30 seconds
        try:
            status = workflow.get_execution_status(execution_id)
            if status:
                logger.info(f"  Status: {status['status']}")
                
                # Show task progress
                task_statuses = {}
                for task_id, task in status['tasks'].items():
                    task_statuses[task['status']] = task_statuses.get(task['status'], 0) + 1
                
                status_summary = ", ".join([f"{status}: {count}" for status, count in task_statuses.items()])
                logger.info(f"  Tasks: {status_summary}")
                
                if status["status"] in ["completed", "failed", "cancelled"]:
                    logger.info(f"✓ Workflow {status['status']}")
                    break
            else:
                logger.warning("Could not get workflow status")
            
            time.sleep(3)
        except Exception as e:
            logger.warning(f"Error monitoring workflow: {e}")
            break
    
    logger.info("Simple workflow example completed!\n")


def example_coordination_patterns():
    """Demonstrate advanced coordination patterns."""
    logger.info("=== Example: Coordination Patterns ===")
    
    # Initialize coordinator
    coordinator = get_coordinator("example_coordinator")
    
    # Example 1: Pipeline Coordination
    logger.info("Creating pipeline coordination example...")
    
    pipeline_stages = [
        {
            "agent": "core_analyzer",
            "tool": "extract_telemetry",
            "parameters": {
                "source": "gs://example-bucket/telemetry.csv",
                "run_id": "example_run_2025"
            }
        },
        {
            "agent": "core_analyzer",
            "tool": "analyze_performance", 
            "parameters": {
                "run_id": "example_run_2025",
                "analysis_depth": "detailed"
            }
        },
        {
            "agent": "insight_hunter",
            "tool": "generate_insights",
            "parameters": {
                "analysis_source": "pipeline_stage_1",
                "insight_level": "strategic"
            }
        },
        {
            "agent": "visualizer",
            "tool": "create_summary_plots",
            "parameters": {
                "insights_source": "pipeline_stage_2",
                "plot_types": ["performance", "consistency"]
            }
        }
    ]
    
    try:
        execution_id = coordinator.patterns.pipeline_coordination(pipeline_stages)
        logger.info(f"✓ Pipeline coordination started: {execution_id}")
        
        # Brief monitoring
        time.sleep(5)
        status = coordinator.workflow.get_execution_status(execution_id)
        if status:
            logger.info(f"  Pipeline status: {status['status']}")
    except Exception as e:
        logger.warning(f"Pipeline coordination failed: {e}")
    
    # Example 2: Load Balancing
    logger.info("Creating load balancing example...")
    
    work_units = [
        {"tool": "analyze_sector", "parameters": {"sector": 1, "run_id": "example_run_2025"}},
        {"tool": "analyze_sector", "parameters": {"sector": 2, "run_id": "example_run_2025"}},
        {"tool": "analyze_sector", "parameters": {"sector": 3, "run_id": "example_run_2025"}},
        {"tool": "analyze_stint", "parameters": {"stint": 1, "run_id": "example_run_2025"}},
        {"tool": "analyze_stint", "parameters": {"stint": 2, "run_id": "example_run_2025"}},
    ]
    
    worker_agents = ["core_analyzer", "insight_hunter"]
    
    try:
        execution_id = coordinator.patterns.load_balancing_coordination(
            worker_agents, work_units, max_parallel=3
        )
        logger.info(f"✓ Load balancing coordination started: {execution_id}")
        
        # Brief monitoring
        time.sleep(5)
        status = coordinator.workflow.get_execution_status(execution_id)
        if status:
            logger.info(f"  Load balancing status: {status['status']}")
    except Exception as e:
        logger.warning(f"Load balancing coordination failed: {e}")
    
    logger.info("Coordination patterns example completed!\n")


def example_complete_analysis_coordination():
    """Demonstrate complete analysis workflow coordination."""
    logger.info("=== Example: Complete Analysis Coordination ===")
    
    coordinator = get_coordinator("analysis_coordinator")
    
    # Coordinate a complete analysis workflow
    run_id = "example_run_2025"
    analysis_path = "gs://example-bucket/example_run_2025/analysis.json"
    
    logger.info(f"Coordinating complete analysis for run: {run_id}")
    
    try:
        execution_id = coordinator.coordinate_analysis_workflow(run_id, analysis_path)
        logger.info(f"✓ Complete analysis workflow started: {execution_id}")
        
        # Monitor progress for a reasonable time
        logger.info("Monitoring complete analysis workflow...")
        start_time = time.time()
        timeout = 30  # Monitor for 30 seconds
        
        while time.time() - start_time < timeout:
            status = coordinator.workflow.get_execution_status(execution_id)
            if status:
                logger.info(f"  Workflow status: {status['status']}")
                
                # Show detailed task progress
                task_progress = {}
                for task_id, task in status['tasks'].items():
                    task_progress[task['status']] = task_progress.get(task['status'], 0) + 1
                
                progress_str = ", ".join([f"{s}: {c}" for s, c in task_progress.items()])
                logger.info(f"  Task progress: {progress_str}")
                
                if status["status"] in ["completed", "failed", "cancelled"]:
                    logger.info(f"✓ Complete analysis workflow {status['status']}")
                    break
            else:
                logger.warning("Could not get workflow status")
            
            time.sleep(5)
        
        if time.time() - start_time >= timeout:
            logger.info("Monitoring timeout reached - workflow may still be running")
        
    except Exception as e:
        logger.warning(f"Complete analysis coordination failed: {e}")
    
    logger.info("Complete analysis coordination example completed!\n")


def main():
    """Run all Phase 4 examples."""
    logger.info("Phase 4 Examples: Shared State Management & Advanced Coordination")
    logger.info("=" * 80)
    
    # Check environment setup
    required_urls = ["STATE_MANAGER_URL", "WORKFLOW_ORCHESTRATOR_URL"]
    for url_var in required_urls:
        url = os.getenv(url_var)
        if url:
            logger.info(f"✓ {url_var}: {url}")
        else:
            logger.warning(f"⚠ {url_var} not set - using default local URL")
    
    # Set defaults for local development
    os.environ.setdefault("STATE_MANAGER_URL", "http://localhost:8080")
    os.environ.setdefault("WORKFLOW_ORCHESTRATOR_URL", "http://localhost:8081")
    os.environ.setdefault("TOOL_REGISTRY_URL", "http://localhost:8082")
    
    logger.info("")
    
    # Run examples
    examples = [
        ("Data Sharing", example_data_sharing),
        ("Simple Workflow", example_simple_workflow),
        ("Coordination Patterns", example_coordination_patterns),
        ("Complete Analysis Coordination", example_complete_analysis_coordination)
    ]
    
    for example_name, example_func in examples:
        logger.info(f"Running {example_name} example...")
        try:
            example_func()
            logger.info(f"✓ {example_name} example completed successfully")
        except Exception as e:
            logger.error(f"✗ {example_name} example failed: {e}")
        
        logger.info("-" * 40)
    
    logger.info("All Phase 4 examples completed!")
    logger.info("\nNote: Some examples may show warnings or failures when connecting")
    logger.info("to services that aren't running. This is expected in demo mode.")
    logger.info("\nTo run with actual services:")
    logger.info("1. Start State Manager: python agents/state_manager/main.py")
    logger.info("2. Start Workflow Orchestrator: python agents/workflow_orchestrator/main.py") 
    logger.info("3. Start Tool Registry: python agents/tool_registry/main.py")
    logger.info("4. Start other agents as needed")


if __name__ == "__main__":
    main()
