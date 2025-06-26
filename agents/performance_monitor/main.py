"""
Performance Monitor Service for Project Apex Phase 5.

Tracks agent performance, analyzes patterns, and provides recommendations
for optimization and intelligent task routing.
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
import statistics
from collections import defaultdict, deque

from flask import Flask, request, jsonify, Response
import requests
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

# ---------------------------------------------------------------------------
# Configuration & Types
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")

class PerformanceLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    agent_name: str
    tool_name: str
    execution_time_ms: float
    success: bool
    timestamp: datetime
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    error_message: Optional[str] = None
    user_satisfaction: Optional[float] = None  # 0-1 scale

@dataclass
class AgentPerformanceProfile:
    agent_name: str
    total_executions: int
    success_rate: float
    avg_execution_time_ms: float
    performance_level: PerformanceLevel
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    last_updated: datetime

@dataclass
class TaskRecommendation:
    task_type: str
    recommended_agent: str
    confidence: float
    reasoning: str
    alternative_agents: List[str]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("performance_monitor")

# ---------------------------------------------------------------------------
# Performance Analysis Engine
# ---------------------------------------------------------------------------
class PerformanceAnalyzer:
    """Analyzes agent performance and provides optimization recommendations."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.task_patterns: Dict[str, List[str]] = defaultdict(list)  # task_type -> successful agents
        
        # Performance thresholds (configurable)
        self.thresholds = {
            "excellent_time_ms": 1000,
            "good_time_ms": 3000,
            "average_time_ms": 10000,
            "poor_time_ms": 30000,
            "excellent_success_rate": 0.95,
            "good_success_rate": 0.85,
            "average_success_rate": 0.70,
            "poor_success_rate": 0.50
        }
    
    def record_execution(self, metric: PerformanceMetric):
        """Record a new performance metric."""
        self.metrics.append(metric)
        
        # Update task patterns for successful executions
        if metric.success:
            task_key = f"{metric.agent_name}.{metric.tool_name}"
            self.task_patterns[task_key].append(metric.agent_name)
        
        # Trigger profile update for the agent
        self._update_agent_profile(metric.agent_name)
        
        LOGGER.info(f"Recorded execution: {metric.agent_name}.{metric.tool_name} - "
                   f"{'SUCCESS' if metric.success else 'FAILED'} in {metric.execution_time_ms:.1f}ms")
    
    def _update_agent_profile(self, agent_name: str):
        """Update performance profile for a specific agent."""
        agent_metrics = [m for m in self.metrics if m.agent_name == agent_name]
        
        if not agent_metrics:
            return
        
        # Calculate performance statistics
        total_executions = len(agent_metrics)
        success_rate = sum(1 for m in agent_metrics if m.success) / total_executions
        
        successful_metrics = [m for m in agent_metrics if m.success]
        avg_time = statistics.mean([m.execution_time_ms for m in successful_metrics]) if successful_metrics else 0
        
        # Determine performance level
        perf_level = self._calculate_performance_level(success_rate, avg_time)
        
        # Generate insights
        strengths, weaknesses, recommendations = self._generate_insights(agent_name, agent_metrics)
        
        # Update profile
        self.agent_profiles[agent_name] = AgentPerformanceProfile(
            agent_name=agent_name,
            total_executions=total_executions,
            success_rate=success_rate,
            avg_execution_time_ms=avg_time,
            performance_level=perf_level,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            last_updated=datetime.now()
        )
    
    def _calculate_performance_level(self, success_rate: float, avg_time_ms: float) -> PerformanceLevel:
        """Calculate overall performance level based on success rate and timing."""
        if success_rate >= self.thresholds["excellent_success_rate"] and avg_time_ms <= self.thresholds["excellent_time_ms"]:
            return PerformanceLevel.EXCELLENT
        elif success_rate >= self.thresholds["good_success_rate"] and avg_time_ms <= self.thresholds["good_time_ms"]:
            return PerformanceLevel.GOOD
        elif success_rate >= self.thresholds["average_success_rate"] and avg_time_ms <= self.thresholds["average_time_ms"]:
            return PerformanceLevel.AVERAGE
        elif success_rate >= self.thresholds["poor_success_rate"]:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def _generate_insights(self, agent_name: str, metrics: List[PerformanceMetric]) -> Tuple[List[str], List[str], List[str]]:
        """Generate performance insights for an agent."""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze timing patterns
        successful_times = [m.execution_time_ms for m in metrics if m.success]
        failed_metrics = [m for m in metrics if not m.success]
        
        if successful_times:
            avg_time = statistics.mean(successful_times)
            if avg_time <= self.thresholds["excellent_time_ms"]:
                strengths.append("Consistently fast execution times")
            elif avg_time <= self.thresholds["good_time_ms"]:
                strengths.append("Good execution performance")
        
        # Analyze success patterns
        success_rate = sum(1 for m in metrics if m.success) / len(metrics)
        if success_rate >= self.thresholds["excellent_success_rate"]:
            strengths.append("Very high success rate")
        elif success_rate >= self.thresholds["good_success_rate"]:
            strengths.append("Good reliability")
        else:
            weaknesses.append("Below average success rate")
        
        # Analyze failure patterns
        if failed_metrics:
            error_patterns = defaultdict(int)
            for metric in failed_metrics:
                if metric.error_message:
                    # Categorize errors
                    if "timeout" in metric.error_message.lower():
                        error_patterns["timeout"] += 1
                    elif "memory" in metric.error_message.lower():
                        error_patterns["memory"] += 1
                    elif "connection" in metric.error_message.lower():
                        error_patterns["connection"] += 1
                    else:
                        error_patterns["other"] += 1
            
            # Generate recommendations based on error patterns
            for error_type, count in error_patterns.items():
                if count > len(failed_metrics) * 0.3:  # More than 30% of failures
                    if error_type == "timeout":
                        recommendations.append("Consider increasing timeout values or optimizing processing")
                        weaknesses.append("Frequent timeout errors")
                    elif error_type == "memory":
                        recommendations.append("Monitor memory usage and consider optimization")
                        weaknesses.append("Memory-related failures")
                    elif error_type == "connection":
                        recommendations.append("Improve network reliability and error handling")
                        weaknesses.append("Network connectivity issues")
        
        # Generate general recommendations
        if not strengths:
            recommendations.append("Performance monitoring shows areas for improvement")
        
        if avg_time > self.thresholds["average_time_ms"]:
            recommendations.append("Consider performance optimization to reduce execution time")
        
        return strengths, weaknesses, recommendations
    
    def get_task_recommendation(self, task_type: str, task_description: str = "") -> TaskRecommendation:
        """Get recommendation for which agent should handle a specific task."""
        
        # Find agents that have successfully handled similar tasks
        relevant_patterns = []
        for pattern_key, agents in self.task_patterns.items():
            if task_type.lower() in pattern_key.lower() or any(keyword in pattern_key.lower() for keyword in task_description.lower().split()):
                relevant_patterns.extend(agents)
        
        if not relevant_patterns:
            # Fallback to general performance profiles
            available_agents = list(self.agent_profiles.keys())
            if available_agents:
                best_agent = max(available_agents, key=lambda a: self.agent_profiles[a].success_rate)
                return TaskRecommendation(
                    task_type=task_type,
                    recommended_agent=best_agent,
                    confidence=0.5,
                    reasoning="Selected based on overall performance metrics",
                    alternative_agents=available_agents[:3]
                )
            else:
                return TaskRecommendation(
                    task_type=task_type,
                    recommended_agent="core_analyzer",  # Default fallback
                    confidence=0.3,
                    reasoning="No performance data available, using default",
                    alternative_agents=[]
                )
        
        # Analyze patterns to find best agent
        agent_scores = defaultdict(list)
        for agent in relevant_patterns:
            if agent in self.agent_profiles:
                profile = self.agent_profiles[agent]
                # Score based on success rate and performance level
                score = profile.success_rate
                if profile.performance_level == PerformanceLevel.EXCELLENT:
                    score += 0.2
                elif profile.performance_level == PerformanceLevel.GOOD:
                    score += 0.1
                agent_scores[agent].append(score)
        
        # Calculate average scores
        agent_avg_scores = {agent: statistics.mean(scores) for agent, scores in agent_scores.items()}
        
        if agent_avg_scores:
            best_agent = max(agent_avg_scores.keys(), key=lambda a: agent_avg_scores[a])
            confidence = min(agent_avg_scores[best_agent], 1.0)
            
            # Get alternatives
            sorted_agents = sorted(agent_avg_scores.items(), key=lambda x: x[1], reverse=True)
            alternatives = [agent for agent, _ in sorted_agents[1:4]]
            
            return TaskRecommendation(
                task_type=task_type,
                recommended_agent=best_agent,
                confidence=confidence,
                reasoning=f"Based on {len(relevant_patterns)} similar task executions",
                alternative_agents=alternatives
            )
        
        # Final fallback
        return TaskRecommendation(
            task_type=task_type,
            recommended_agent="core_analyzer",
            confidence=0.3,
            reasoning="No suitable patterns found, using default",
            alternative_agents=[]
        )
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get system-wide optimization recommendations."""
        recommendations = {
            "critical_agents": [],
            "underperforming_tools": [],
            "optimization_opportunities": [],
            "resource_recommendations": [],
            "coordination_improvements": []
        }
        
        # Identify critical agents
        for agent_name, profile in self.agent_profiles.items():
            if profile.performance_level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                recommendations["critical_agents"].append({
                    "agent": agent_name,
                    "issues": profile.weaknesses,
                    "recommendations": profile.recommendations
                })
        
        # Identify underperforming tools
        tool_metrics = defaultdict(list)
        for metric in self.metrics:
            tool_key = f"{metric.agent_name}.{metric.tool_name}"
            tool_metrics[tool_key].append(metric)
        
        for tool_key, metrics in tool_metrics.items():
            success_rate = sum(1 for m in metrics if m.success) / len(metrics)
            if success_rate < self.thresholds["average_success_rate"]:
                recommendations["underperforming_tools"].append({
                    "tool": tool_key,
                    "success_rate": success_rate,
                    "executions": len(metrics)
                })
        
        # System-wide optimization opportunities
        if len(self.metrics) > 50:  # Enough data for analysis
            avg_system_time = statistics.mean([m.execution_time_ms for m in self.metrics if m.success])
            if avg_system_time > self.thresholds["good_time_ms"]:
                recommendations["optimization_opportunities"].append(
                    "System-wide performance is below optimal - consider infrastructure scaling"
                )
        
        return recommendations


# ---------------------------------------------------------------------------
# Global Analyzer Instance
# ---------------------------------------------------------------------------
analyzer = PerformanceAnalyzer()

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "performance_monitor"}), 200

@app.route("/metrics", methods=["POST"])
def record_metric() -> Response:
    """Record a new performance metric."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["agent_name", "tool_name", "execution_time_ms", "success"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create metric object
        metric = PerformanceMetric(
            agent_name=data["agent_name"],
            tool_name=data["tool_name"],
            execution_time_ms=float(data["execution_time_ms"]),
            success=bool(data["success"]),
            timestamp=datetime.now(),
            input_size=data.get("input_size"),
            output_size=data.get("output_size"),
            error_message=data.get("error_message"),
            user_satisfaction=data.get("user_satisfaction")
        )
        
        # Record the metric
        analyzer.record_execution(metric)
        
        return jsonify({"message": "Metric recorded successfully"}), 201
        
    except Exception as e:
        LOGGER.error(f"Error recording metric: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/profiles", methods=["GET"])
def get_performance_profiles() -> Response:
    """Get performance profiles for all agents."""
    agent_filter = request.args.get("agent")
    
    if agent_filter:
        profile = analyzer.agent_profiles.get(agent_filter)
        if profile:
            return jsonify({"profile": asdict(profile)}), 200
        else:
            return jsonify({"error": "Agent not found"}), 404
    
    return jsonify({
        "profiles": {name: asdict(profile) for name, profile in analyzer.agent_profiles.items()},
        "total_agents": len(analyzer.agent_profiles)
    }), 200

@app.route("/recommendations/task", methods=["POST"])
def get_task_recommendation() -> Response:
    """Get recommendation for task assignment."""
    try:
        data = request.get_json()
        task_type = data.get("task_type", "")
        task_description = data.get("description", "")
        
        if not task_type:
            return jsonify({"error": "task_type is required"}), 400
        
        recommendation = analyzer.get_task_recommendation(task_type, task_description)
        
        return jsonify({"recommendation": asdict(recommendation)}), 200
        
    except Exception as e:
        LOGGER.error(f"Error generating task recommendation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/recommendations/optimization", methods=["GET"])
def get_optimization_recommendations() -> Response:
    """Get system-wide optimization recommendations."""
    try:
        recommendations = analyzer.get_optimization_recommendations()
        return jsonify({"recommendations": recommendations}), 200
        
    except Exception as e:
        LOGGER.error(f"Error generating optimization recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/analytics", methods=["GET"])
def get_analytics() -> Response:
    """Get performance analytics and statistics."""
    try:
        total_metrics = len(analyzer.metrics)
        if total_metrics == 0:
            return jsonify({"message": "No metrics available"}), 200
        
        # Calculate system-wide statistics
        success_rate = sum(1 for m in analyzer.metrics if m.success) / total_metrics
        successful_metrics = [m for m in analyzer.metrics if m.success]
        
        avg_execution_time = statistics.mean([m.execution_time_ms for m in successful_metrics]) if successful_metrics else 0
        
        # Agent distribution
        agent_counts = defaultdict(int)
        for metric in analyzer.metrics:
            agent_counts[metric.agent_name] += 1
        
        analytics = {
            "system_overview": {
                "total_executions": total_metrics,
                "system_success_rate": success_rate,
                "avg_execution_time_ms": avg_execution_time,
                "monitored_agents": len(analyzer.agent_profiles)
            },
            "agent_distribution": dict(agent_counts),
            "performance_levels": {
                level.value: sum(1 for p in analyzer.agent_profiles.values() if p.performance_level == level)
                for level in PerformanceLevel
            }
        }
        
        return jsonify({"analytics": analytics}), 200
        
    except Exception as e:
        LOGGER.error(f"Error generating analytics: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Initialize with some sample data for testing
    sample_metrics = [
        PerformanceMetric("core_analyzer", "driver_deltas", 1200, True, datetime.now()),
        PerformanceMetric("visualizer", "pit_times", 2500, True, datetime.now()),
        PerformanceMetric("insight_hunter", "analyze", 4200, True, datetime.now()),
        PerformanceMetric("core_analyzer", "trend_analysis", 800, True, datetime.now()),
        PerformanceMetric("publicist", "generate", 15000, False, datetime.now(), error_message="Timeout error"),
    ]
    
    for metric in sample_metrics:
        analyzer.record_execution(metric)
    
    LOGGER.info("Performance Monitor initialized with sample data")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
