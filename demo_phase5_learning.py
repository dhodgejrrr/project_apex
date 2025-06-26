#!/usr/bin/env python3
"""
Phase 5 Learning and Optimization Demonstration

This script demonstrates the advanced learning and optimization capabilities
of Project Apex Phase 5, including intelligent task routing, performance
monitoring, and adaptive system behavior.
"""
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List
import requests

# Setup paths for importing Project Apex modules
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("phase5_demo")

# Demo configuration
DEMO_SERVICES = {
    "performance_monitor": "http://localhost:8094",
    "task_router": "http://localhost:8095",
    "tool_registry": "http://localhost:8090"
}

class Phase5Demo:
    """Demonstration of Phase 5 learning and optimization features."""
    
    def __init__(self):
        self.performance_monitor_url = DEMO_SERVICES["performance_monitor"]
        self.task_router_url = DEMO_SERVICES["task_router"]
        self.tool_registry_url = DEMO_SERVICES["tool_registry"]
        
    def run_demo(self):
        """Run the complete Phase 5 demonstration."""
        logger.info("🚀 Starting Phase 5 Learning and Optimization Demo")
        logger.info("=" * 60)
        
        try:
            # Step 1: Check service availability
            self._check_services()
            
            # Step 2: Demonstrate performance monitoring
            self._demo_performance_monitoring()
            
            # Step 3: Demonstrate intelligent task routing
            self._demo_intelligent_routing()
            
            # Step 4: Demonstrate learning pattern development
            self._demo_learning_patterns()
            
            # Step 5: Demonstrate optimization recommendations
            self._demo_optimization_recommendations()
            
            # Step 6: Show system analytics
            self._show_system_analytics()
            
            logger.info("✅ Phase 5 demonstration completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False
        
        return True
    
    def _check_services(self):
        """Check if required services are running."""
        logger.info("🔍 Checking service availability...")
        
        for service, url in DEMO_SERVICES.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {service} is running at {url}")
                else:
                    logger.warning(f"⚠️ {service} returned status {response.status_code}")
            except requests.RequestException:
                logger.warning(f"⚠️ {service} is not available at {url}")
    
    def _demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities."""
        logger.info("\\n📊 Demonstrating Performance Monitoring")
        logger.info("-" * 40)
        
        # Simulate recording various performance metrics
        demo_metrics = [
            {
                "agent_name": "core_analyzer",
                "tool_name": "driver_deltas",
                "execution_time_ms": 1200.0,
                "success": True,
                "scenario": "Fast analysis execution"
            },
            {
                "agent_name": "core_analyzer", 
                "tool_name": "trend_analysis",
                "execution_time_ms": 2500.0,
                "success": True,
                "scenario": "Standard trend analysis"
            },
            {
                "agent_name": "visualizer",
                "tool_name": "pit_times",
                "execution_time_ms": 3500.0,
                "success": True,
                "scenario": "Visualization generation"
            },
            {
                "agent_name": "visualizer",
                "tool_name": "consistency",
                "execution_time_ms": 8000.0,
                "success": False,
                "error_message": "Timeout during chart generation",
                "scenario": "Failed visualization (timeout)"
            },
            {
                "agent_name": "insight_hunter",
                "tool_name": "analyze", 
                "execution_time_ms": 4200.0,
                "success": True,
                "scenario": "Insight generation"
            }
        ]
        
        logger.info("Recording performance metrics...")
        for i, metric in enumerate(demo_metrics):
            try:
                scenario = metric.pop("scenario")  # Remove scenario from payload
                
                response = requests.post(f"{self.performance_monitor_url}/metrics", 
                                       json=metric, timeout=10)
                
                if response.status_code == 201:
                    status = "✅ SUCCESS" if metric["success"] else "❌ FAILED"
                    logger.info(f"  {i+1}. {scenario}: {status} "
                               f"({metric['execution_time_ms']:.0f}ms)")
                else:
                    logger.warning(f"  {i+1}. Failed to record metric: {response.status_code}")
                    
            except requests.RequestException as e:
                logger.warning(f"  {i+1}. Network error recording metric: {e}")
        
        # Get updated performance profiles
        try:
            response = requests.get(f"{self.performance_monitor_url}/profiles", timeout=10)
            if response.status_code == 200:
                profiles = response.json()["profiles"]
                logger.info(f"\\n📈 Generated profiles for {len(profiles)} agents:")
                
                for agent_name, profile in profiles.items():
                    logger.info(f"  • {agent_name}: {profile['performance_level']} "
                               f"({profile['success_rate']:.1%} success, "
                               f"{profile['avg_execution_time_ms']:.0f}ms avg)")
                    
                    if profile.get("recommendations"):
                        logger.info(f"    Recommendations: {', '.join(profile['recommendations'])}")
                        
        except requests.RequestException as e:
            logger.warning(f"Failed to get performance profiles: {e}")
    
    def _demo_intelligent_routing(self):
        """Demonstrate intelligent task routing."""
        logger.info("\\n🧠 Demonstrating Intelligent Task Routing")
        logger.info("-" * 40)
        
        # Test different types of tasks with various priorities
        demo_tasks = [
            {
                "task_type": "analysis",
                "priority": "critical",
                "description": "Critical race analysis for live event"
            },
            {
                "task_type": "visualization", 
                "priority": "high",
                "description": "High-priority chart generation"
            },
            {
                "task_type": "analysis",
                "priority": "normal", 
                "description": "Standard data analysis task"
            },
            {
                "task_type": "insight_generation",
                "priority": "low",
                "description": "Background insight analysis"
            }
        ]
        
        logger.info("Routing tasks with intelligent algorithm...")
        routed_tasks = []
        
        for i, task_config in enumerate(demo_tasks):
            try:
                task_data = {
                    "task_id": f"demo_task_{i}_{uuid.uuid4()}",
                    "task_type": task_config["task_type"],
                    "priority": task_config["priority"],
                    "payload": {
                        "description": task_config["description"],
                        "demo": True
                    }
                }
                
                response = requests.post(f"{self.task_router_url}/route", 
                                       json=task_data, timeout=10)
                
                if response.status_code == 200:
                    decision = response.json()["routing_decision"]
                    routed_tasks.append((task_data, decision))
                    
                    logger.info(f"  {i+1}. {task_config['description']}")
                    logger.info(f"     → Routed to: {decision['selected_agent']}")
                    logger.info(f"     → Confidence: {decision['confidence']:.2f}")
                    logger.info(f"     → Reasoning: {decision['reasoning']}")
                    
                    # Simulate task completion with realistic results
                    completion_time = self._simulate_task_execution(task_config, decision)
                    
                    # Report results back to router
                    result_data = {
                        "success": True,
                        "execution_time_ms": completion_time
                    }
                    
                    result_response = requests.post(
                        f"{self.task_router_url}/tasks/{task_data['task_id']}/result",
                        json=result_data, timeout=5)
                    
                    if result_response.status_code == 200:
                        logger.info(f"     → Completed in {completion_time:.0f}ms ✅")
                    
                else:
                    logger.warning(f"  {i+1}. Failed to route task: {response.status_code}")
                    
            except requests.RequestException as e:
                logger.warning(f"  {i+1}. Network error routing task: {e}")
        
        return routed_tasks
    
    def _simulate_task_execution(self, task_config: Dict, decision: Dict) -> float:
        """Simulate realistic task execution time based on agent and task type."""
        base_times = {
            "analysis": 2000,
            "visualization": 4000,
            "insight_generation": 5000
        }
        
        agent_modifiers = {
            "core_analyzer": 0.8,    # Fast analyzer
            "visualizer": 1.2,       # Slower visualization
            "insight_hunter": 1.5,   # Complex insights take time
            "publicist": 1.1,
            "historian": 1.3
        }
        
        priority_modifiers = {
            "critical": 0.7,  # Higher priority gets faster processing
            "high": 0.9,
            "normal": 1.0,
            "low": 1.3
        }
        
        base_time = base_times.get(task_config["task_type"], 3000)
        agent_mod = agent_modifiers.get(decision["selected_agent"], 1.0)
        priority_mod = priority_modifiers.get(task_config["priority"], 1.0)
        
        # Add some randomness
        import random
        randomness = random.uniform(0.8, 1.2)
        
        return base_time * agent_mod * priority_mod * randomness
    
    def _demo_learning_patterns(self):
        """Demonstrate learning pattern development."""
        logger.info("\\n🎓 Demonstrating Learning Pattern Development")
        logger.info("-" * 40)
        
        try:
            # Get current learning patterns
            response = requests.get(f"{self.task_router_url}/learning/patterns", timeout=10)
            
            if response.status_code == 200:
                patterns = response.json()["learning_patterns"]
                
                if patterns:
                    logger.info(f"📚 Current learning patterns ({len(patterns)} total):")
                    
                    for pattern_key, pattern_data in patterns.items():
                        task_type, agent = pattern_key.split(":", 1)
                        success_rate = pattern_data["success_rate"]
                        routing_count = pattern_data["routing_count"]
                        
                        logger.info(f"  • {task_type} → {agent}")
                        logger.info(f"    Success Rate: {success_rate:.1%} "
                                   f"({routing_count} executions)")
                        
                        # Interpret the pattern
                        if success_rate >= 0.9:
                            logger.info("    🌟 Excellent performance pattern")
                        elif success_rate >= 0.7:
                            logger.info("    ✅ Good performance pattern")
                        elif success_rate >= 0.5:
                            logger.info("    ⚠️ Average performance pattern")
                        else:
                            logger.info("    ❌ Poor performance pattern")
                else:
                    logger.info("📚 No learning patterns developed yet")
                    logger.info("    (Patterns develop after multiple task executions)")
                    
        except requests.RequestException as e:
            logger.warning(f"Failed to get learning patterns: {e}")
    
    def _demo_optimization_recommendations(self):
        """Demonstrate optimization recommendations."""
        logger.info("\\n🔧 Demonstrating Optimization Recommendations")
        logger.info("-" * 40)
        
        try:
            response = requests.get(f"{self.performance_monitor_url}/recommendations/optimization", 
                                  timeout=10)
            
            if response.status_code == 200:
                recommendations = response.json()["recommendations"]
                
                logger.info("🎯 System optimization recommendations:")
                
                # Critical agents
                if recommendations["critical_agents"]:
                    logger.info("  ⚠️ Critical Agents Needing Attention:")
                    for agent_info in recommendations["critical_agents"]:
                        logger.info(f"    • {agent_info['agent']}")
                        for issue in agent_info["issues"]:
                            logger.info(f"      - Issue: {issue}")
                        for rec in agent_info["recommendations"]:
                            logger.info(f"      - Recommendation: {rec}")
                else:
                    logger.info("  ✅ No critical agent issues detected")
                
                # Underperforming tools
                if recommendations["underperforming_tools"]:
                    logger.info("  📉 Underperforming Tools:")
                    for tool_info in recommendations["underperforming_tools"]:
                        logger.info(f"    • {tool_info['tool']}: "
                                   f"{tool_info['success_rate']:.1%} success rate "
                                   f"({tool_info['executions']} executions)")
                else:
                    logger.info("  ✅ No underperforming tools detected")
                
                # Optimization opportunities
                if recommendations["optimization_opportunities"]:
                    logger.info("  🚀 Optimization Opportunities:")
                    for opportunity in recommendations["optimization_opportunities"]:
                        logger.info(f"    • {opportunity}")
                else:
                    logger.info("  ✅ No immediate optimization opportunities identified")
                    
        except requests.RequestException as e:
            logger.warning(f"Failed to get optimization recommendations: {e}")
    
    def _show_system_analytics(self):
        """Show comprehensive system analytics."""
        logger.info("\\n📈 System Analytics Overview")
        logger.info("-" * 40)
        
        # Performance analytics
        try:
            perf_response = requests.get(f"{self.performance_monitor_url}/analytics", timeout=10)
            if perf_response.status_code == 200:
                perf_analytics = perf_response.json()["analytics"]
                
                overview = perf_analytics["system_overview"]
                logger.info("🔍 Performance Overview:")
                logger.info(f"  • Total Executions: {overview['total_executions']}")
                logger.info(f"  • System Success Rate: {overview['system_success_rate']:.1%}")
                logger.info(f"  • Average Execution Time: {overview['avg_execution_time_ms']:.0f}ms")
                logger.info(f"  • Monitored Agents: {overview['monitored_agents']}")
                
                if "performance_levels" in perf_analytics:
                    levels = perf_analytics["performance_levels"]
                    logger.info("  📊 Agent Performance Distribution:")
                    for level, count in levels.items():
                        if count > 0:
                            logger.info(f"    - {level.title()}: {count} agents")
                            
        except requests.RequestException as e:
            logger.warning(f"Failed to get performance analytics: {e}")
        
        # Routing analytics
        try:
            routing_response = requests.get(f"{self.task_router_url}/analytics", timeout=10)
            if routing_response.status_code == 200:
                routing_analytics = routing_response.json()["analytics"]
                
                if "routing_overview" in routing_analytics:
                    overview = routing_analytics["routing_overview"]
                    logger.info("\\n🧭 Routing Overview:")
                    logger.info(f"  • Total Routes: {overview['total_routes']}")
                    logger.info(f"  • Average Confidence: {overview['avg_confidence']:.2f}")
                    
                    if "agent_distribution" in overview:
                        logger.info("  📊 Agent Distribution:")
                        for agent, count in overview["agent_distribution"].items():
                            logger.info(f"    - {agent}: {count} tasks")
                            
        except requests.RequestException as e:
            logger.warning(f"Failed to get routing analytics: {e}")

def main():
    """Main demonstration execution."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Phase 5 Learning and Optimization Demo")
        print("Usage: python demo_phase5_learning.py")
        print("")
        print("This demo showcases Phase 5 capabilities:")
        print("- Performance monitoring and analysis")
        print("- Intelligent task routing with learning")
        print("- Optimization recommendations")
        print("- System analytics and insights")
        print("")
        print("Prerequisites:")
        print("- Start Performance Monitor: python agents/performance_monitor/main.py")
        print("- Start Task Router: python agents/task_router/main.py")
        print("- Optionally start Tool Registry for enhanced features")
        return
    
    logger.info("Phase 5 Learning and Optimization Demo")
    logger.info("Showcasing intelligent multi-agent coordination")
    
    demo = Phase5Demo()
    success = demo.run_demo()
    
    if success:
        logger.info("\\n🎉 Demo completed successfully!")
        logger.info("Phase 5 demonstrates advanced learning and optimization capabilities")
    else:
        logger.error("\\n❌ Demo encountered issues")
        logger.info("Ensure Performance Monitor and Task Router services are running")

if __name__ == "__main__":
    main()
