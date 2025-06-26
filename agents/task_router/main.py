"""
Intelligent Task Router Service for Project Apex Phase 5.

Routes tasks to optimal agents based on performance data, learning patterns,
and real-time system state. Provides adaptive load balancing and failure recovery.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import threading
from collections import defaultdict, deque
import random

from flask import Flask, request, jsonify, Response
import requests
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

# ---------------------------------------------------------------------------
# Configuration & Types
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")

class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"

@dataclass
class Task:
    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    attempt_count: int = 0
    max_attempts: int = 3
    estimated_duration_ms: Optional[float] = None
    deadline: Optional[datetime] = None

@dataclass
class AgentLoad:
    agent_name: str
    current_tasks: int
    max_capacity: int
    avg_response_time_ms: float
    health_status: str
    last_updated: datetime

@dataclass
class RoutingDecision:
    task_id: str
    selected_agent: str
    confidence: float
    reasoning: str
    alternative_agents: List[str]
    estimated_completion: datetime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("task_router")

# ---------------------------------------------------------------------------
# Intelligent Task Routing Engine
# ---------------------------------------------------------------------------
class TaskRouter:
    """Intelligent task routing with learning and optimization capabilities."""
    
    def __init__(self):
        self.task_queue: Dict[str, Task] = {}
        self.agent_loads: Dict[str, AgentLoad] = {}
        self.routing_history: List[RoutingDecision] = []
        self.performance_cache: Dict[str, Any] = {}
        self.learning_patterns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Service URLs
        self.performance_monitor_url = os.getenv("PERFORMANCE_MONITOR_URL", "")
        self.tool_registry_url = os.getenv("TOOL_REGISTRY_URL", "")
        
        # Routing parameters
        self.load_balance_weight = 0.4
        self.performance_weight = 0.4
        self.learning_weight = 0.2
        
        # Background thread for monitoring
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
    def start_monitoring(self):
        """Start background monitoring of agents and performance."""
        if self.monitoring_thread is None:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            LOGGER.info("Started background monitoring thread")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring_thread:
            self.shutdown_event.set()
            self.monitoring_thread.join(timeout=5)
            LOGGER.info("Stopped background monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                self._update_agent_loads()
                self._update_performance_cache()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                LOGGER.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_agent_loads(self):
        """Update current agent load information."""
        if not self.tool_registry_url:
            return
        
        try:
            # Get service topology from Tool Registry
            topology_resp = self._make_request("GET", f"{self.tool_registry_url}/topology")
            if topology_resp and "services" in topology_resp:
                for agent_name, service_info in topology_resp["services"].items():
                    # Estimate current load based on response time and health
                    response_time = service_info.get("response_time_ms", 5000)
                    health_status = service_info.get("status", "unknown")
                    
                    # Simple load estimation (would be enhanced with real metrics)
                    max_capacity = 10  # Configurable per agent
                    current_tasks = min(int(response_time / 1000), max_capacity)  # Rough estimation
                    
                    self.agent_loads[agent_name] = AgentLoad(
                        agent_name=agent_name,
                        current_tasks=current_tasks,
                        max_capacity=max_capacity,
                        avg_response_time_ms=response_time,
                        health_status=health_status,
                        last_updated=datetime.now()
                    )
                    
            LOGGER.debug(f"Updated load information for {len(self.agent_loads)} agents")
            
        except Exception as e:
            LOGGER.warning(f"Failed to update agent loads: {e}")
    
    def _update_performance_cache(self):
        """Update performance data cache."""
        if not self.performance_monitor_url:
            return
        
        try:
            # Get performance profiles
            profiles_resp = self._make_request("GET", f"{self.performance_monitor_url}/profiles")
            if profiles_resp and "profiles" in profiles_resp:
                self.performance_cache = profiles_resp["profiles"]
                LOGGER.debug(f"Updated performance cache with {len(self.performance_cache)} agent profiles")
                
        except Exception as e:
            LOGGER.warning(f"Failed to update performance cache: {e}")
    
    def _make_request(self, method: str, url: str, data: Dict = None) -> Optional[Dict]:
        """Make authenticated HTTP request."""
        try:
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                # Local mode
                if method == "GET":
                    resp = requests.get(url, timeout=10)
                else:
                    resp = requests.post(url, json=data, timeout=10)
            else:
                # Production mode with authentication
                auth_req = grequests.Request()
                id_token_credentials = id_token.fetch_id_token(auth_req, url)
                headers = {
                    "Authorization": f"Bearer {id_token_credentials}",
                    "Content-Type": "application/json",
                    "User-Agent": "project-apex-task-router/1.0",
                }
                
                if method == "GET":
                    resp = requests.get(url, headers=headers, timeout=10)
                else:
                    resp = requests.post(url, json=data, headers=headers, timeout=10)
            
            resp.raise_for_status()
            return resp.json()
            
        except Exception as e:
            LOGGER.warning(f"Request failed {method} {url}: {e}")
            return None
    
    def route_task(self, task: Task) -> RoutingDecision:
        """Route a task to the optimal agent using intelligent algorithms."""
        
        # Get task recommendation from performance monitor
        performance_rec = self._get_performance_recommendation(task)
        
        # Get available agents from tool registry
        available_agents = self._get_available_agents(task)
        
        # Apply intelligent routing algorithm
        decision = self._calculate_optimal_routing(task, performance_rec, available_agents)
        
        # Learn from this decision
        self._update_learning_patterns(task, decision)
        
        # Record the decision
        self.routing_history.append(decision)
        
        # Update task status
        task.assigned_agent = decision.selected_agent
        task.status = TaskStatus.ASSIGNED
        self.task_queue[task.task_id] = task
        
        LOGGER.info(f"Routed task {task.task_id} to {decision.selected_agent} "
                   f"(confidence: {decision.confidence:.2f})")
        
        return decision
    
    def _get_performance_recommendation(self, task: Task) -> Optional[Dict]:
        """Get performance-based recommendation from performance monitor."""
        if not self.performance_monitor_url:
            return None
        
        recommendation_data = {
            "task_type": task.task_type,
            "description": json.dumps(task.payload)
        }
        
        return self._make_request("POST", f"{self.performance_monitor_url}/recommendations/task", 
                                 recommendation_data)
    
    def _get_available_agents(self, task: Task) -> List[str]:
        """Get list of available agents that can handle the task."""
        available = []
        
        # Check agent loads and health
        for agent_name, load_info in self.agent_loads.items():
            if (load_info.health_status == "healthy" and 
                load_info.current_tasks < load_info.max_capacity):
                available.append(agent_name)
        
        # Fallback to all known agents if no load info available
        if not available and self.performance_cache:
            available = list(self.performance_cache.keys())
        
        # Final fallback to default agents
        if not available:
            available = ["core_analyzer", "visualizer", "insight_hunter", "publicist"]
        
        return available
    
    def _calculate_optimal_routing(self, task: Task, performance_rec: Optional[Dict], 
                                 available_agents: List[str]) -> RoutingDecision:
        """Calculate optimal routing decision using multiple factors."""
        
        agent_scores = {}
        
        for agent in available_agents:
            score = 0.0
            reasoning_parts = []
            
            # Factor 1: Performance-based recommendation
            if performance_rec and "recommendation" in performance_rec:
                rec_data = performance_rec["recommendation"]
                if rec_data["recommended_agent"] == agent:
                    score += self.performance_weight * rec_data["confidence"]
                    reasoning_parts.append(f"performance recommendation ({rec_data['confidence']:.2f})")
                elif agent in rec_data.get("alternative_agents", []):
                    score += self.performance_weight * 0.5
                    reasoning_parts.append("alternative performance option")
            
            # Factor 2: Load balancing
            if agent in self.agent_loads:
                load_info = self.agent_loads[agent]
                load_factor = 1.0 - (load_info.current_tasks / load_info.max_capacity)
                score += self.load_balance_weight * load_factor
                reasoning_parts.append(f"load factor ({load_factor:.2f})")
            else:
                score += self.load_balance_weight * 0.5  # Neutral if no load info
            
            # Factor 3: Learning patterns
            pattern_key = f"{task.task_type}:{agent}"
            if pattern_key in self.learning_patterns:
                learning_score = self.learning_patterns[pattern_key].get("success_rate", 0.5)
                score += self.learning_weight * learning_score
                reasoning_parts.append(f"learning pattern ({learning_score:.2f})")
            else:
                score += self.learning_weight * 0.5  # Neutral for new patterns
            
            # Priority boost for critical tasks
            if task.priority == TaskPriority.CRITICAL:
                if agent in self.performance_cache:
                    perf_profile = self.performance_cache[agent]
                    if perf_profile.get("performance_level") in ["excellent", "good"]:
                        score *= 1.2  # 20% boost for high-performing agents on critical tasks
                        reasoning_parts.append("critical task priority boost")
            
            agent_scores[agent] = (score, reasoning_parts)
        
        # Select the best agent
        if agent_scores:
            best_agent = max(agent_scores.keys(), key=lambda a: agent_scores[a][0])
            best_score, reasoning_parts = agent_scores[best_agent]
            
            # Get alternatives
            sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1][0], reverse=True)
            alternatives = [agent for agent, _ in sorted_agents[1:4]]
            
            # Estimate completion time
            estimated_completion = self._estimate_completion_time(task, best_agent)
            
            return RoutingDecision(
                task_id=task.task_id,
                selected_agent=best_agent,
                confidence=min(best_score, 1.0),
                reasoning="; ".join(reasoning_parts),
                alternative_agents=alternatives,
                estimated_completion=estimated_completion
            )
        else:
            # Fallback decision
            fallback_agent = available_agents[0] if available_agents else "core_analyzer"
            return RoutingDecision(
                task_id=task.task_id,
                selected_agent=fallback_agent,
                confidence=0.3,
                reasoning="fallback routing - no scoring data available",
                alternative_agents=[],
                estimated_completion=datetime.now() + timedelta(minutes=5)
            )
    
    def _estimate_completion_time(self, task: Task, agent: str) -> datetime:
        """Estimate task completion time."""
        base_time = datetime.now()
        
        # Default estimation
        estimated_duration = 60  # 1 minute default
        
        # Use performance data if available
        if agent in self.performance_cache:
            profile = self.performance_cache[agent]
            avg_time_ms = profile.get("avg_execution_time_ms", 60000)
            estimated_duration = max(avg_time_ms / 1000, 10)  # At least 10 seconds
        
        # Adjust for task priority
        if task.priority == TaskPriority.CRITICAL:
            estimated_duration *= 0.8  # Faster processing expected
        elif task.priority == TaskPriority.LOW:
            estimated_duration *= 1.5  # May take longer
        
        # Add some buffer time
        estimated_duration *= 1.2
        
        return base_time + timedelta(seconds=estimated_duration)
    
    def _update_learning_patterns(self, task: Task, decision: RoutingDecision):
        """Update learning patterns based on routing decision."""
        pattern_key = f"{task.task_type}:{decision.selected_agent}"
        
        # Initialize pattern if new
        if pattern_key not in self.learning_patterns:
            self.learning_patterns[pattern_key] = {
                "routing_count": 0,
                "success_rate": 0.5,  # Start neutral
                "avg_confidence": 0.0
            }
        
        # Update pattern
        pattern = self.learning_patterns[pattern_key]
        pattern["routing_count"] += 1
        
        # Update confidence tracking
        pattern["avg_confidence"] = (
            (pattern["avg_confidence"] * (pattern["routing_count"] - 1) + decision.confidence) /
            pattern["routing_count"]
        )
    
    def update_task_result(self, task_id: str, success: bool, execution_time_ms: float, 
                          error_message: str = None):
        """Update task result and learn from the outcome."""
        if task_id in self.task_queue:
            task = self.task_queue[task_id]
            
            # Update task status
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            
            # Update learning patterns
            if task.assigned_agent:
                pattern_key = f"{task.task_type}:{task.assigned_agent}"
                if pattern_key in self.learning_patterns:
                    pattern = self.learning_patterns[pattern_key]
                    
                    # Update success rate with weighted average
                    old_rate = pattern["success_rate"]
                    count = pattern["routing_count"]
                    new_rate = (old_rate * (count - 1) + (1.0 if success else 0.0)) / count
                    pattern["success_rate"] = new_rate
            
            # Send performance data to performance monitor
            if self.performance_monitor_url:
                metric_data = {
                    "agent_name": task.assigned_agent,
                    "tool_name": task.task_type,
                    "execution_time_ms": execution_time_ms,
                    "success": success,
                    "error_message": error_message
                }
                self._make_request("POST", f"{self.performance_monitor_url}/metrics", metric_data)
            
            LOGGER.info(f"Updated task {task_id} result: {'SUCCESS' if success else 'FAILED'} "
                       f"in {execution_time_ms:.1f}ms")
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get routing analytics and insights."""
        total_routes = len(self.routing_history)
        if total_routes == 0:
            return {"message": "No routing data available"}
        
        # Agent distribution
        agent_counts = defaultdict(int)
        confidence_sum = 0
        
        for decision in self.routing_history:
            agent_counts[decision.selected_agent] += 1
            confidence_sum += decision.confidence
        
        # Learning pattern insights
        pattern_insights = {}
        for pattern_key, pattern_data in self.learning_patterns.items():
            if pattern_data["routing_count"] >= 3:  # Only patterns with sufficient data
                pattern_insights[pattern_key] = {
                    "success_rate": pattern_data["success_rate"],
                    "routing_count": pattern_data["routing_count"],
                    "avg_confidence": pattern_data["avg_confidence"]
                }
        
        analytics = {
            "routing_overview": {
                "total_routes": total_routes,
                "avg_confidence": confidence_sum / total_routes,
                "agent_distribution": dict(agent_counts)
            },
            "learning_patterns": pattern_insights,
            "current_load": {
                agent: asdict(load) for agent, load in self.agent_loads.items()
            }
        }
        
        return analytics


# ---------------------------------------------------------------------------
# Global Router Instance
# ---------------------------------------------------------------------------
router = TaskRouter()

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "task_router"}), 200

@app.route("/route", methods=["POST"])
def route_task() -> Response:
    """Route a task to the optimal agent."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["task_id", "task_type", "payload"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create task object
        task = Task(
            task_id=data["task_id"],
            task_type=data["task_type"],
            priority=TaskPriority(data.get("priority", "normal")),
            payload=data["payload"],
            created_at=datetime.now(),
            max_attempts=data.get("max_attempts", 3)
        )
        
        if data.get("deadline"):
            task.deadline = datetime.fromisoformat(data["deadline"])
        
        # Route the task
        decision = router.route_task(task)
        
        return jsonify({"routing_decision": asdict(decision)}), 200
        
    except Exception as e:
        LOGGER.error(f"Error routing task: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tasks/<task_id>/result", methods=["POST"])
def update_task_result(task_id: str) -> Response:
    """Update the result of a task execution."""
    try:
        data = request.get_json()
        
        required_fields = ["success", "execution_time_ms"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        router.update_task_result(
            task_id=task_id,
            success=bool(data["success"]),
            execution_time_ms=float(data["execution_time_ms"]),
            error_message=data.get("error_message")
        )
        
        return jsonify({"message": "Task result updated successfully"}), 200
        
    except Exception as e:
        LOGGER.error(f"Error updating task result: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/analytics", methods=["GET"])
def get_analytics() -> Response:
    """Get routing analytics and insights."""
    try:
        analytics = router.get_routing_analytics()
        return jsonify({"analytics": analytics}), 200
        
    except Exception as e:
        LOGGER.error(f"Error generating analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/agents/load", methods=["GET"])
def get_agent_loads() -> Response:
    """Get current agent load information."""
    loads = {agent: asdict(load) for agent, load in router.agent_loads.items()}
    return jsonify({"agent_loads": loads}), 200

@app.route("/learning/patterns", methods=["GET"])
def get_learning_patterns() -> Response:
    """Get current learning patterns."""
    return jsonify({"learning_patterns": dict(router.learning_patterns)}), 200

if __name__ == "__main__":
    # Start background monitoring
    router.start_monitoring()
    
    try:
        LOGGER.info("Task Router initialized and monitoring started")
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
    finally:
        router.stop_monitoring()
