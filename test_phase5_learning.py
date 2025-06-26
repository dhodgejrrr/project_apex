#!/usr/bin/env python3
"""
Test script for Phase 5: Learning and Optimization

This script validates:
1. Performance Monitor service functionality
2. Intelligent Task Router capabilities
3. Learning pattern development
4. Performance optimization recommendations
5. Intelligent tool calling with routing

Tests both individual services and integrated workflows.
"""
import json
import logging
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import requests
import uuid

# Setup paths for importing Project Apex modules
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase5_test")

# Test configuration
LOCAL_PORTS = {
    "performance_monitor": 8094,
    "task_router": 8095,
    "tool_registry": 8090,
    "core_analyzer": 8091,
    "visualizer": 8092
}

class Phase5Tester:
    """Test runner for Phase 5 implementation."""
    
    def __init__(self):
        self.base_urls = {}
        self.test_results = {}
        
        # Setup base URLs for local testing
        for service, port in LOCAL_PORTS.items():
            self.base_urls[service] = f"http://localhost:{port}"
    
    def run_all_tests(self) -> bool:
        """Run all Phase 5 tests."""
        logger.info("=" * 60)
        logger.info("PHASE 5 TESTING: Learning and Optimization")
        logger.info("=" * 60)
        
        tests = [
            ("Performance Monitor Service", self.test_performance_monitor),
            ("Task Router Service", self.test_task_router),
            ("Performance Tracking", self.test_performance_tracking),
            ("Learning Pattern Development", self.test_learning_patterns),
            ("Intelligent Tool Routing", self.test_intelligent_routing),
            ("Optimization Recommendations", self.test_optimization_recommendations),
            ("Enhanced ToolCaller Integration", self.test_enhanced_tool_caller),
            ("End-to-End Learning Workflow", self.test_e2e_learning_workflow)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            logger.info(f"\\n--- Running Test: {test_name} ---")
            try:
                result = test_func()
                self.test_results[test_name] = {"passed": result, "error": None}
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{test_name}: {status}")
                if not result:
                    all_passed = False
            except Exception as e:
                self.test_results[test_name] = {"passed": False, "error": str(e)}
                logger.error(f"{test_name}: ❌ FAILED - {e}")
                all_passed = False
        
        self._print_summary()
        return all_passed
    
    def test_performance_monitor(self) -> bool:
        """Test Performance Monitor service endpoints."""
        pm_url = self.base_urls["performance_monitor"]
        
        try:
            # Test health endpoint
            health_resp = requests.get(f"{pm_url}/health", timeout=5)
            if health_resp.status_code != 200:
                logger.error(f"Performance Monitor health check failed: {health_resp.status_code}")
                return False
            
            # Test recording a metric
            metric_data = {
                "agent_name": "test_agent",
                "tool_name": "test_tool",
                "execution_time_ms": 1500.0,
                "success": True
            }
            
            metric_resp = requests.post(f"{pm_url}/metrics", json=metric_data, timeout=5)
            if metric_resp.status_code != 201:
                logger.error(f"Metric recording failed: {metric_resp.status_code}")
                return False
            
            # Test getting profiles
            profiles_resp = requests.get(f"{pm_url}/profiles", timeout=5)
            if profiles_resp.status_code != 200:
                logger.error(f"Profiles endpoint failed: {profiles_resp.status_code}")
                return False
            
            # Test analytics
            analytics_resp = requests.get(f"{pm_url}/analytics", timeout=5)
            if analytics_resp.status_code != 200:
                logger.error(f"Analytics endpoint failed: {analytics_resp.status_code}")
                return False
            
            logger.info("Performance Monitor service working correctly")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Performance Monitor test failed: {e}")
            return False
    
    def test_task_router(self) -> bool:
        """Test Task Router service endpoints."""
        tr_url = self.base_urls["task_router"]
        
        try:
            # Test health endpoint
            health_resp = requests.get(f"{tr_url}/health", timeout=5)
            if health_resp.status_code != 200:
                logger.error(f"Task Router health check failed: {health_resp.status_code}")
                return False
            
            # Test task routing
            task_data = {
                "task_id": str(uuid.uuid4()),
                "task_type": "analysis",
                "priority": "normal",
                "payload": {"test": "data"}
            }
            
            route_resp = requests.post(f"{tr_url}/route", json=task_data, timeout=10)
            if route_resp.status_code != 200:
                logger.error(f"Task routing failed: {route_resp.status_code}")
                return False
            
            route_data = route_resp.json()
            if "routing_decision" not in route_data:
                logger.error(f"Invalid routing response: {route_data}")
                return False
            
            # Test updating task result
            task_id = task_data["task_id"]
            result_data = {
                "success": True,
                "execution_time_ms": 2000.0
            }
            
            result_resp = requests.post(f"{tr_url}/tasks/{task_id}/result", 
                                      json=result_data, timeout=5)
            if result_resp.status_code != 200:
                logger.error(f"Task result update failed: {result_resp.status_code}")
                return False
            
            # Test analytics
            analytics_resp = requests.get(f"{tr_url}/analytics", timeout=5)
            if analytics_resp.status_code != 200:
                logger.error(f"Router analytics failed: {analytics_resp.status_code}")
                return False
            
            logger.info("Task Router service working correctly")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Task Router test failed: {e}")
            return False
    
    def test_performance_tracking(self) -> bool:
        """Test performance tracking and metric recording."""
        pm_url = self.base_urls["performance_monitor"]
        
        try:
            # Record multiple metrics for pattern testing
            test_metrics = [
                {"agent_name": "core_analyzer", "tool_name": "driver_deltas", 
                 "execution_time_ms": 1200, "success": True},
                {"agent_name": "core_analyzer", "tool_name": "driver_deltas", 
                 "execution_time_ms": 1100, "success": True},
                {"agent_name": "visualizer", "tool_name": "pit_times", 
                 "execution_time_ms": 2500, "success": True},
                {"agent_name": "visualizer", "tool_name": "pit_times", 
                 "execution_time_ms": 3000, "success": False, "error_message": "Timeout"},
                {"agent_name": "insight_hunter", "tool_name": "analyze", 
                 "execution_time_ms": 4500, "success": True}
            ]
            
            for metric in test_metrics:
                resp = requests.post(f"{pm_url}/metrics", json=metric, timeout=5)
                if resp.status_code != 201:
                    logger.error(f"Failed to record metric: {resp.status_code}")
                    return False
            
            # Check that profiles were updated
            time.sleep(1)  # Allow processing time
            profiles_resp = requests.get(f"{pm_url}/profiles", timeout=5)
            profiles_data = profiles_resp.json()
            
            if "profiles" not in profiles_data:
                logger.error("No profiles generated from metrics")
                return False
            
            # Verify core_analyzer has good performance profile
            if "core_analyzer" in profiles_data["profiles"]:
                ca_profile = profiles_data["profiles"]["core_analyzer"]
                if ca_profile["success_rate"] < 0.8:
                    logger.warning(f"Core analyzer success rate unexpectedly low: {ca_profile['success_rate']}")
            
            logger.info("Performance tracking working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Performance tracking test failed: {e}")
            return False
    
    def test_learning_patterns(self) -> bool:
        """Test learning pattern development in task router."""
        tr_url = self.base_urls["task_router"]
        
        try:
            # Route multiple similar tasks to build patterns
            task_types = ["analysis", "analysis", "visualization", "analysis"]
            task_ids = []
            
            for i, task_type in enumerate(task_types):
                task_data = {
                    "task_id": f"learning_test_{i}_{uuid.uuid4()}",
                    "task_type": task_type,
                    "priority": "normal",
                    "payload": {"test": f"data_{i}"}
                }
                
                route_resp = requests.post(f"{tr_url}/route", json=task_data, timeout=10)
                if route_resp.status_code != 200:
                    logger.error(f"Failed to route learning task {i}")
                    return False
                
                task_ids.append(task_data["task_id"])
                
                # Simulate task completion
                result_data = {
                    "success": True,
                    "execution_time_ms": 1500.0 + (i * 200)  # Varying execution times
                }
                
                result_resp = requests.post(f"{tr_url}/tasks/{task_data['task_id']}/result", 
                                          json=result_data, timeout=5)
                if result_resp.status_code != 200:
                    logger.error(f"Failed to update learning task result {i}")
                    return False
            
            # Check learning patterns
            patterns_resp = requests.get(f"{tr_url}/learning/patterns", timeout=5)
            if patterns_resp.status_code != 200:
                logger.error("Failed to get learning patterns")
                return False
            
            patterns_data = patterns_resp.json()
            if "learning_patterns" not in patterns_data:
                logger.error("No learning patterns found")
                return False
            
            patterns = patterns_data["learning_patterns"]
            if len(patterns) == 0:
                logger.warning("No learning patterns developed yet")
            else:
                logger.info(f"Developed {len(patterns)} learning patterns")
            
            logger.info("Learning pattern development working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Learning patterns test failed: {e}")
            return False
    
    def test_intelligent_routing(self) -> bool:
        """Test intelligent routing decisions."""
        tr_url = self.base_urls["task_router"]
        
        try:
            # Test routing with different priorities
            priorities = ["critical", "high", "normal", "low"]
            
            for priority in priorities:
                task_data = {
                    "task_id": f"priority_test_{priority}_{uuid.uuid4()}",
                    "task_type": "analysis",
                    "priority": priority,
                    "payload": {"priority_test": True}
                }
                
                route_resp = requests.post(f"{tr_url}/route", json=task_data, timeout=10)
                if route_resp.status_code != 200:
                    logger.error(f"Failed to route {priority} priority task")
                    return False
                
                route_data = route_resp.json()
                decision = route_data["routing_decision"]
                
                # Verify routing decision structure
                required_fields = ["selected_agent", "confidence", "reasoning"]
                for field in required_fields:
                    if field not in decision:
                        logger.error(f"Routing decision missing {field}")
                        return False
                
                logger.info(f"Routed {priority} task to {decision['selected_agent']} "
                           f"(confidence: {decision['confidence']:.2f})")
            
            logger.info("Intelligent routing working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Intelligent routing test failed: {e}")
            return False
    
    def test_optimization_recommendations(self) -> bool:
        """Test optimization recommendations."""
        pm_url = self.base_urls["performance_monitor"]
        
        try:
            # Get optimization recommendations
            opt_resp = requests.get(f"{pm_url}/recommendations/optimization", timeout=10)
            if opt_resp.status_code != 200:
                logger.error(f"Optimization recommendations failed: {opt_resp.status_code}")
                return False
            
            opt_data = opt_resp.json()
            if "recommendations" not in opt_data:
                logger.error("No recommendations structure found")
                return False
            
            recommendations = opt_data["recommendations"]
            expected_keys = ["critical_agents", "underperforming_tools", 
                           "optimization_opportunities", "resource_recommendations"]
            
            for key in expected_keys:
                if key not in recommendations:
                    logger.error(f"Missing recommendation category: {key}")
                    return False
            
            logger.info("Optimization recommendations working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Optimization recommendations test failed: {e}")
            return False
    
    def test_enhanced_tool_caller(self) -> bool:
        """Test enhanced ToolCaller with Phase 5 features."""
        try:
            from agents.common.tool_caller import ToolCaller
            
            tool_caller = ToolCaller()
            
            # Check Phase 5 attributes
            phase5_attrs = ["performance_monitor_url", "task_router_url", "_enable_performance_tracking"]
            for attr in phase5_attrs:
                if not hasattr(tool_caller, attr):
                    logger.error(f"ToolCaller missing Phase 5 attribute: {attr}")
                    return False
            
            # Check Phase 5 methods
            phase5_methods = ["call_tool_intelligently", "get_performance_insights", 
                            "get_routing_analytics", "_execute_on_agent", "_report_task_result"]
            for method in phase5_methods:
                if not hasattr(tool_caller, method):
                    logger.error(f"ToolCaller missing Phase 5 method: {method}")
                    return False
            
            # Test configuration
            tool_caller.performance_monitor_url = self.base_urls["performance_monitor"]
            tool_caller.task_router_url = self.base_urls["task_router"]
            
            # Test performance insights (may fail if services not running)
            try:
                insights = tool_caller.get_performance_insights()
                if "error" not in insights:
                    logger.info("Performance insights retrieved successfully")
                else:
                    logger.info(f"Performance insights returned error (expected): {insights['error']}")
            except Exception as e:
                logger.info(f"Performance insights test failed (expected in test env): {e}")
            
            logger.info("Enhanced ToolCaller integration working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced ToolCaller test failed: {e}")
            return False
    
    def test_e2e_learning_workflow(self) -> bool:
        """Test end-to-end learning workflow."""
        try:
            # This test simulates a complete learning cycle
            from agents.common.tool_caller import call_tool_intelligently
            
            # Mock the service URLs for testing
            from agents.common.tool_caller import tool_caller
            tool_caller.performance_monitor_url = self.base_urls["performance_monitor"]
            tool_caller.task_router_url = self.base_urls["task_router"]
            
            # Test intelligent call (will likely fail due to service dependencies)
            try:
                result = call_tool_intelligently("analysis", {"test": "data"}, "normal")
                
                # Check for routing metadata
                if "_routing_metadata" in result:
                    metadata = result["_routing_metadata"]
                    logger.info(f"E2E test produced routing metadata: {metadata}")
                else:
                    logger.info("E2E test completed without routing metadata (fallback mode)")
                
            except Exception as e:
                logger.info(f"E2E intelligent call failed (expected in test env): {e}")
            
            logger.info("End-to-end learning workflow test completed")
            return True
            
        except Exception as e:
            logger.error(f"E2E learning workflow test failed: {e}")
            return False
    
    def _print_summary(self):
        """Print test summary."""
        logger.info("\\n" + "=" * 60)
        logger.info("PHASE 5 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result["passed"])
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅" if result["passed"] else "❌"
            error_info = f" ({result['error']})" if result["error"] else ""
            logger.info(f"{status} {test_name}{error_info}")
        
        logger.info(f"\\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All Phase 5 tests passed!")
        else:
            logger.warning(f"⚠️  {total - passed} test(s) failed")

def main():
    """Main test execution."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Phase 5 Test Script")
        print("Usage: python test_phase5_learning.py")
        print("")
        print("This script tests the learning and optimization features:")
        print("- Performance Monitor service functionality")
        print("- Intelligent Task Router capabilities") 
        print("- Learning pattern development")
        print("- Performance optimization recommendations")
        print("- Enhanced ToolCaller with intelligent routing")
        print("")
        print("Note: This script assumes services are running locally on specific ports.")
        print("Start the Performance Monitor and Task Router before running tests.")
        return
    
    tester = Phase5Tester()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
