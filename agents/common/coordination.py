"""
Advanced coordination utilities for Project Apex agents.

Provides state management, workflow orchestration, and advanced coordination
patterns for multi-agent collaboration.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import requests
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

LOGGER = logging.getLogger("apex.coordination")


class StateClient:
    """Client for interacting with the State Manager service."""
    
    def __init__(self, state_manager_url: Optional[str] = None, agent_id: str = "unknown"):
        self.state_manager_url = state_manager_url or os.getenv("STATE_MANAGER_URL", "")
        self.agent_id = agent_id
    
    def _make_request(self, endpoint: str, method: str = "GET", 
                     payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request to State Manager."""
        if not self.state_manager_url:
            raise ValueError("State Manager URL not configured")
        
        url = f"{self.state_manager_url}{endpoint}"
        headers = {"X-Agent-ID": self.agent_id}
        
        if os.getenv("PUBSUB_EMULATOR_HOST"):
            # Local mode
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=10)
            elif method == "PUT":
                resp = requests.put(url, json=payload, headers=headers, timeout=10)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=10)
            else:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
        else:
            # Production mode with authentication
            auth_req = grequests.Request()
            id_token_credentials = id_token.fetch_id_token(auth_req, url)
            headers.update({
                "Authorization": f"Bearer {id_token_credentials}",
                "Content-Type": "application/json",
                "User-Agent": "project-apex-coordination/1.0",
            })
            
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=10)
            elif method == "PUT":
                resp = requests.put(url, json=payload, headers=headers, timeout=10)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=10)
            else:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
        
        resp.raise_for_status()
        return resp.json()
    
    def get(self, key: str) -> Optional[Any]:
        """Get state value by key."""
        try:
            result = self._make_request(f"/state/{key}")
            return result.get("value")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            conflict_resolution: str = "last_write_wins",
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set state value."""
        payload = {
            "value": value,
            "conflict_resolution": conflict_resolution,
            "metadata": metadata or {}
        }
        if ttl_seconds:
            payload["ttl_seconds"] = ttl_seconds
        
        try:
            self._make_request(f"/state/{key}", "PUT", payload)
            return True
        except Exception as e:
            LOGGER.error(f"Failed to set state {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete state entry."""
        try:
            self._make_request(f"/state/{key}", "DELETE")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return False
            return False
        except Exception as e:
            LOGGER.error(f"Failed to delete state {key}: {e}")
            return False
    
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List state keys with optional prefix filter."""
        try:
            params = f"?prefix={prefix}" if prefix else ""
            result = self._make_request(f"/state{params}")
            return result.get("keys", [])
        except Exception as e:
            LOGGER.error(f"Failed to list keys: {e}")
            return []
    
    def subscribe(self, key_pattern: str, webhook_url: str, 
                 event_types: Optional[List[str]] = None) -> Optional[str]:
        """Subscribe to state change events."""
        payload = {
            "key_pattern": key_pattern,
            "webhook_url": webhook_url,
            "event_types": event_types or ["created", "updated", "deleted"]
        }
        
        try:
            result = self._make_request("/subscriptions", "POST", payload)
            return result.get("subscription_id")
        except Exception as e:
            LOGGER.error(f"Failed to subscribe: {e}")
            return None
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from state change events."""
        try:
            self._make_request(f"/subscriptions/{subscription_id}", "DELETE")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to unsubscribe: {e}")
            return False


class WorkflowClient:
    """Client for interacting with the Workflow Orchestrator service."""
    
    def __init__(self, orchestrator_url: Optional[str] = None, agent_id: str = "unknown"):
        self.orchestrator_url = orchestrator_url or os.getenv("WORKFLOW_ORCHESTRATOR_URL", "")
        self.agent_id = agent_id
    
    def _make_request(self, endpoint: str, method: str = "GET",
                     payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request to Workflow Orchestrator."""
        if not self.orchestrator_url:
            raise ValueError("Workflow Orchestrator URL not configured")
        
        url = f"{self.orchestrator_url}{endpoint}"
        headers = {"X-Agent-ID": self.agent_id}
        
        if os.getenv("PUBSUB_EMULATOR_HOST"):
            # Local mode
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=30)
            else:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
        else:
            # Production mode with authentication
            auth_req = grequests.Request()
            id_token_credentials = id_token.fetch_id_token(auth_req, url)
            headers.update({
                "Authorization": f"Bearer {id_token_credentials}",
                "Content-Type": "application/json",
                "User-Agent": "project-apex-coordination/1.0",
            })
            
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=30)
            else:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
        
        resp.raise_for_status()
        return resp.json()
    
    def create_workflow(self, workflow_id: str, name: str, description: str,
                       tasks: List[Dict[str, Any]], **kwargs) -> str:
        """Create a new workflow definition."""
        payload = {
            "workflow_id": workflow_id,
            "name": name,
            "description": description,
            "tasks": tasks,
            **kwargs
        }
        
        result = self._make_request("/workflows/definitions", "POST", payload)
        return result.get("workflow_id")
    
    def execute_workflow(self, workflow_id: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Execute a workflow."""
        payload = {"parameters": parameters or {}}
        result = self._make_request(f"/workflows/{workflow_id}/execute", "POST", payload)
        return result.get("execution_id")
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status."""
        try:
            return self._make_request(f"/executions/{execution_id}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution."""
        try:
            self._make_request(f"/executions/{execution_id}/cancel", "POST")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to cancel execution {execution_id}: {e}")
            return False
    
    def wait_for_completion(self, execution_id: str, timeout_seconds: int = 300,
                           poll_interval: int = 5) -> Dict[str, Any]:
        """Wait for workflow completion with polling."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            status = self.get_execution_status(execution_id)
            if not status:
                raise ValueError(f"Execution {execution_id} not found")
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Workflow {execution_id} did not complete within {timeout_seconds}s")


class CoordinationPatterns:
    """Advanced coordination patterns for multi-agent collaboration."""
    
    def __init__(self, state_client: StateClient, workflow_client: WorkflowClient):
        self.state = state_client
        self.workflow = workflow_client
    
    def master_slave_coordination(self, master_agent: str, slave_agents: List[str],
                                 task_distribution: Dict[str, Dict[str, Any]]) -> str:
        """Implement master-slave coordination pattern."""
        workflow_id = f"master_slave_{int(time.time())}"
        
        # Create coordination workflow
        tasks = []
        
        # Master task to coordinate
        tasks.append({
            "task_id": "master_coordination",
            "agent_name": master_agent,
            "tool_name": "coordinate",
            "parameters": {
                "slaves": slave_agents,
                "task_distribution": task_distribution
            },
            "dependencies": []
        })
        
        # Slave tasks
        for i, slave_agent in enumerate(slave_agents):
            task_id = f"slave_task_{i}"
            tasks.append({
                "task_id": task_id,
                "agent_name": slave_agent,
                "tool_name": "execute_assigned_task",
                "parameters": task_distribution.get(slave_agent, {}),
                "dependencies": ["master_coordination"]
            })
        
        # Create and execute workflow
        self.workflow.create_workflow(
            workflow_id=workflow_id,
            name="Master-Slave Coordination",
            description=f"Master {master_agent} coordinating {len(slave_agents)} slaves",
            tasks=tasks,
            execution_mode="mixed"
        )
        
        return self.workflow.execute_workflow(workflow_id)
    
    def consensus_decision_making(self, participating_agents: List[str],
                                decision_criteria: Dict[str, Any],
                                voting_threshold: float = 0.6) -> str:
        """Implement consensus-based decision making."""
        workflow_id = f"consensus_{int(time.time())}"
        
        # Store decision criteria in shared state
        decision_key = f"consensus:{workflow_id}:criteria"
        self.state.set(decision_key, decision_criteria)
        
        # Create voting workflow
        tasks = []
        
        # Individual voting tasks
        for i, agent in enumerate(participating_agents):
            task_id = f"vote_{i}"
            tasks.append({
                "task_id": task_id,
                "agent_name": agent,
                "tool_name": "cast_vote",
                "parameters": {
                    "decision_key": decision_key,
                    "voter_id": agent
                },
                "dependencies": []
            })
        
        # Consensus aggregation task
        tasks.append({
            "task_id": "aggregate_consensus",
            "agent_name": "arbiter",  # Use arbiter for neutral aggregation
            "tool_name": "aggregate_votes",
            "parameters": {
                "decision_key": decision_key,
                "voters": participating_agents,
                "threshold": voting_threshold
            },
            "dependencies": [f"vote_{i}" for i in range(len(participating_agents))]
        })
        
        # Create and execute workflow
        self.workflow.create_workflow(
            workflow_id=workflow_id,
            name="Consensus Decision Making",
            description=f"Consensus among {len(participating_agents)} agents",
            tasks=tasks,
            execution_mode="mixed"
        )
        
        return self.workflow.execute_workflow(workflow_id)
    
    def pipeline_coordination(self, pipeline_stages: List[Dict[str, Any]]) -> str:
        """Implement pipeline coordination with sequential stages."""
        workflow_id = f"pipeline_{int(time.time())}"
        
        tasks = []
        previous_task_id = None
        
        for i, stage in enumerate(pipeline_stages):
            task_id = f"stage_{i}"
            dependencies = [previous_task_id] if previous_task_id else []
            
            tasks.append({
                "task_id": task_id,
                "agent_name": stage["agent"],
                "tool_name": stage["tool"],
                "parameters": stage.get("parameters", {}),
                "dependencies": dependencies
            })
            
            previous_task_id = task_id
        
        # Create and execute workflow
        self.workflow.create_workflow(
            workflow_id=workflow_id,
            name="Pipeline Coordination",
            description=f"Sequential pipeline with {len(pipeline_stages)} stages",
            tasks=tasks,
            execution_mode="sequential"
        )
        
        return self.workflow.execute_workflow(workflow_id)
    
    def load_balancing_coordination(self, worker_agents: List[str],
                                  work_units: List[Dict[str, Any]],
                                  max_parallel: int = 3) -> str:
        """Implement load-balanced task distribution."""
        workflow_id = f"load_balanced_{int(time.time())}"
        
        tasks = []
        
        # Distribute work units among available agents
        for i, work_unit in enumerate(work_units):
            agent = worker_agents[i % len(worker_agents)]  # Round-robin distribution
            task_id = f"work_unit_{i}"
            
            tasks.append({
                "task_id": task_id,
                "agent_name": agent,
                "tool_name": work_unit.get("tool", "process_work"),
                "parameters": work_unit.get("parameters", {}),
                "dependencies": []
            })
        
        # Create and execute workflow
        self.workflow.create_workflow(
            workflow_id=workflow_id,
            name="Load-Balanced Coordination",
            description=f"Load balancing {len(work_units)} tasks across {len(worker_agents)} agents",
            tasks=tasks,
            execution_mode="parallel",
            max_parallel_tasks=max_parallel
        )
        
        return self.workflow.execute_workflow(workflow_id)


class AdvancedCoordinator:
    """High-level coordinator combining state management and workflow orchestration."""
    
    def __init__(self, agent_id: str = "coordinator"):
        self.agent_id = agent_id
        self.state = StateClient(agent_id=agent_id)
        self.workflow = WorkflowClient(agent_id=agent_id)
        self.patterns = CoordinationPatterns(self.state, self.workflow)
    
    def share_data(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Share data with other agents through state management."""
        return self.state.set(key, data, ttl_seconds)
    
    def get_shared_data(self, key: str) -> Any:
        """Get shared data from state management."""
        return self.state.get(key)
    
    def coordinate_analysis_workflow(self, run_id: str, analysis_path: str) -> str:
        """Coordinate a complete analysis workflow."""
        workflow_tasks = [
            {
                "task_id": "core_analysis",
                "agent_name": "core_analyzer",
                "tool_name": "analyze_full",
                "parameters": {"run_id": run_id, "analysis_path": analysis_path},
                "dependencies": []
            },
            {
                "task_id": "insight_generation",
                "agent_name": "insight_hunter",
                "tool_name": "generate_insights",
                "parameters": {"analysis_path": analysis_path, "use_autonomous": True},
                "dependencies": ["core_analysis"]
            },
            {
                "task_id": "visualization",
                "agent_name": "visualizer",
                "tool_name": "generate_all_plots",
                "parameters": {"analysis_path": analysis_path},
                "dependencies": ["core_analysis"]
            },
            {
                "task_id": "social_content",
                "agent_name": "publicist",
                "tool_name": "generate_posts",
                "parameters": {"analysis_path": analysis_path, "use_autonomous": True},
                "dependencies": ["insight_generation", "visualization"]
            }
        ]
        
        workflow_id = f"analysis_workflow_{run_id}"
        
        self.workflow.create_workflow(
            workflow_id=workflow_id,
            name="Complete Analysis Workflow",
            description=f"End-to-end analysis for run {run_id}",
            tasks=workflow_tasks,
            execution_mode="mixed",
            max_parallel_tasks=3
        )
        
        return self.workflow.execute_workflow(workflow_id)
    
    def monitor_workflow(self, execution_id: str, timeout_seconds: int = 600) -> Dict[str, Any]:
        """Monitor workflow execution with timeout."""
        return self.workflow.wait_for_completion(execution_id, timeout_seconds)


# Global instances for easy importing
def get_coordinator(agent_id: str = "unknown") -> AdvancedCoordinator:
    """Get a coordinator instance for the specified agent."""
    return AdvancedCoordinator(agent_id)


def get_state_client(agent_id: str = "unknown") -> StateClient:
    """Get a state client for the specified agent."""
    return StateClient(agent_id=agent_id)


def get_workflow_client(agent_id: str = "unknown") -> WorkflowClient:
    """Get a workflow client for the specified agent."""
    return WorkflowClient(agent_id=agent_id)
