"""
Workflow Orchestrator Service for Project Apex.

Manages complex multi-agent workflows with task delegation, dependency management,
parallel execution coordination, and error recovery.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio
import threading
from collections import defaultdict, deque

from flask import Flask, request, jsonify, Response
import requests
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

# ---------------------------------------------------------------------------
# Configuration & Types
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"

class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MIXED = "mixed"

@dataclass
class TaskDefinition:
    task_id: str
    agent_name: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskExecution:
    task_id: str
    workflow_id: str
    status: TaskStatus
    agent_name: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300

@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    description: str
    tasks: List[TaskDefinition]
    execution_mode: ExecutionMode = ExecutionMode.MIXED
    max_parallel_tasks: int = 5
    timeout_seconds: int = 3600
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    workflow_id: str
    execution_id: str
    name: str
    status: WorkflowStatus
    tasks: Dict[str, TaskExecution]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "unknown"
    execution_mode: ExecutionMode = ExecutionMode.MIXED
    max_parallel_tasks: int = 5
    current_parallel_count: int = 0
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("workflow_orchestrator")

# ---------------------------------------------------------------------------
# Workflow Orchestrator
# ---------------------------------------------------------------------------
class WorkflowOrchestrator:
    """Manages workflow execution with task dependencies and parallel coordination."""
    
    def __init__(self):
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        self.execution_queue: deque = deque()
        self.running_tasks: Dict[str, threading.Thread] = {}
        self.executor_running = False
        self.tool_registry_url = os.getenv("TOOL_REGISTRY_URL", "")
        self.state_manager_url = os.getenv("STATE_MANAGER_URL", "")
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        return str(uuid.uuid4())
    
    def _invoke_service(self, service_url: str, payload: Dict[str, Any] = None, 
                       method: str = "GET", timeout: int = 10) -> Dict[str, Any]:
        """Make authenticated request to a service."""
        if os.getenv("PUBSUB_EMULATOR_HOST"):
            # Local mode
            if method == "GET":
                resp = requests.get(service_url, timeout=timeout)
            else:
                resp = requests.post(service_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            # Production mode with authentication
            auth_req = grequests.Request()
            id_token_credentials = id_token.fetch_id_token(auth_req, service_url)
            
            headers = {
                "Authorization": f"Bearer {id_token_credentials}",
                "Content-Type": "application/json",
                "User-Agent": "project-apex-workflow-orchestrator/1.0",
            }
            
            if method == "GET":
                resp = requests.get(service_url, headers=headers, timeout=timeout)
            else:
                resp = requests.post(service_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
    
    def _get_available_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available tools from Tool Registry."""
        if not self.tool_registry_url:
            return {}
        
        try:
            registry_url = f"{self.tool_registry_url}/registry"
            registry_data = self._invoke_service(registry_url)
            
            tools_by_agent = {}
            for service_name, service_info in registry_data.get("services", {}).items():
                if service_info.get("status") == "healthy":
                    capabilities = service_info.get("capabilities", {})
                    tools_by_agent[service_name] = capabilities.get("tools", [])
            
            return tools_by_agent
            
        except Exception as e:
            LOGGER.warning(f"Failed to get tools from registry: {e}")
            return {}
    
    def _execute_task(self, execution: WorkflowExecution, task: TaskExecution):
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        execution.current_parallel_count += 1
        
        LOGGER.info(f"Starting task {task.task_id} for workflow {execution.workflow_id}")
        
        try:
            # Get agent base URL from Tool Registry or environment
            agent_url = self._get_agent_url(task.agent_name)
            if not agent_url:
                raise ValueError(f"Agent {task.agent_name} not available")
            
            # Build tool endpoint URL
            tool_url = f"{agent_url}/tools/{task.tool_name}"
            
            # Execute the task
            start_time = time.time()
            result = self._invoke_service(tool_url, task.parameters, "POST", task.timeout_seconds)
            execution_time = time.time() - start_time
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Store result in shared state if State Manager is available
            if self.state_manager_url:
                self._store_task_result(execution.workflow_id, task.task_id, result)
            
            LOGGER.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            error_msg = str(e)
            LOGGER.error(f"Task {task.task_id} failed: {error_msg}")
            
            # Handle retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRY
                task.error = f"Retry {task.retry_count}/{task.max_retries}: {error_msg}"
                # Re-queue for retry
                self.execution_queue.append((execution, task))
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = error_msg
        
        finally:
            execution.current_parallel_count -= 1
    
    def _get_agent_url(self, agent_name: str) -> Optional[str]:
        """Get agent URL from Tool Registry or environment."""
        # Try Tool Registry first
        if self.tool_registry_url:
            try:
                registry_url = f"{self.tool_registry_url}/services"
                services = self._invoke_service(registry_url)
                
                service_info = services.get("services", {}).get(agent_name)
                if service_info and service_info.get("status") == "healthy":
                    return service_info.get("url")
            except Exception as e:
                LOGGER.warning(f"Failed to get agent URL from registry: {e}")
        
        # Fallback to environment variables
        env_var = f"{agent_name.upper()}_URL"
        return os.getenv(env_var)
    
    def _store_task_result(self, workflow_id: str, task_id: str, result: Any):
        """Store task result in shared state."""
        try:
            state_key = f"workflow:{workflow_id}:task:{task_id}:result"
            state_url = f"{self.state_manager_url}/state/{state_key}"
            
            payload = {
                "value": result,
                "metadata": {
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "stored_at": datetime.now().isoformat()
                }
            }
            
            self._invoke_service(state_url, payload, "PUT")
            
        except Exception as e:
            LOGGER.warning(f"Failed to store task result in state: {e}")
    
    def _get_ready_tasks(self, execution: WorkflowExecution) -> List[TaskExecution]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        
        for task in execution.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = execution.tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready_tasks.append(task)
        
        return ready_tasks
    
    def _update_workflow_status(self, execution: WorkflowExecution):
        """Update workflow status based on task states."""
        task_statuses = [task.status for task in execution.tasks.values()]
        
        if all(status == TaskStatus.COMPLETED for status in task_statuses):
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            # Collect all task results
            execution.result = {
                task_id: task.result for task_id, task in execution.tasks.items()
                if task.result is not None
            }
            
            LOGGER.info(f"Workflow {execution.workflow_id} completed successfully")
            
        elif any(status == TaskStatus.FAILED for status in task_statuses):
            # Check if any failed tasks have exhausted retries
            failed_tasks = [task for task in execution.tasks.values() 
                          if task.status == TaskStatus.FAILED]
            
            if failed_tasks:
                execution.status = WorkflowStatus.FAILED
                execution.completed_at = datetime.now()
                execution.error = f"Tasks failed: {[t.task_id for t in failed_tasks]}"
                
                LOGGER.error(f"Workflow {execution.workflow_id} failed")
    
    def _execute_workflow_loop(self):
        """Main workflow execution loop."""
        while self.executor_running:
            try:
                # Process execution queue
                if self.execution_queue:
                    execution, task = self.execution_queue.popleft()
                    
                    # Check if we can run this task (parallel limit)
                    if (execution.current_parallel_count < execution.max_parallel_tasks and
                        task.status in [TaskStatus.PENDING, TaskStatus.RETRY]):
                        
                        # Start task in separate thread
                        thread = threading.Thread(
                            target=self._execute_task,
                            args=(execution, task),
                            name=f"task-{task.task_id}"
                        )
                        thread.start()
                        self.running_tasks[task.task_id] = thread
                    else:
                        # Put back in queue
                        self.execution_queue.append((execution, task))
                
                # Check for newly ready tasks in all running workflows
                for execution in self.workflow_executions.values():
                    if execution.status == WorkflowStatus.RUNNING:
                        ready_tasks = self._get_ready_tasks(execution)
                        
                        for task in ready_tasks:
                            if (execution.current_parallel_count < execution.max_parallel_tasks):
                                self.execution_queue.append((execution, task))
                        
                        # Update workflow status
                        self._update_workflow_status(execution)
                
                # Clean up completed threads
                completed_threads = []
                for task_id, thread in self.running_tasks.items():
                    if not thread.is_alive():
                        completed_threads.append(task_id)
                
                for task_id in completed_threads:
                    del self.running_tasks[task_id]
                
                time.sleep(1)  # Avoid busy waiting
                
            except Exception as e:
                LOGGER.error(f"Error in workflow execution loop: {e}")
                time.sleep(5)
    
    def start_executor(self):
        """Start the workflow execution engine."""
        if not self.executor_running:
            self.executor_running = True
            executor_thread = threading.Thread(
                target=self._execute_workflow_loop,
                name="workflow-executor",
                daemon=True
            )
            executor_thread.start()
            LOGGER.info("Workflow executor started")
    
    def stop_executor(self):
        """Stop the workflow execution engine."""
        self.executor_running = False
        LOGGER.info("Workflow executor stopped")
    
    def create_workflow_definition(self, definition: WorkflowDefinition) -> str:
        """Create a new workflow definition."""
        self.workflow_definitions[definition.workflow_id] = definition
        LOGGER.info(f"Created workflow definition: {definition.workflow_id}")
        return definition.workflow_id
    
    def execute_workflow(self, workflow_id: str, agent_id: str,
                        parameters: Optional[Dict[str, Any]] = None) -> str:
        """Start execution of a workflow."""
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow definition not found: {workflow_id}")
        
        definition = self.workflow_definitions[workflow_id]
        execution_id = self._generate_id()
        
        # Create task executions
        tasks = {}
        for task_def in definition.tasks:
            # Merge workflow parameters with task parameters
            task_params = {**task_def.parameters}
            if parameters:
                task_params.update(parameters)
            
            task_execution = TaskExecution(
                task_id=task_def.task_id,
                workflow_id=workflow_id,
                status=TaskStatus.PENDING,
                agent_name=task_def.agent_name,
                tool_name=task_def.tool_name,
                parameters=task_params,
                dependencies=task_def.dependencies,
                created_at=datetime.now(),
                max_retries=task_def.retry_count,
                timeout_seconds=task_def.timeout_seconds
            )
            tasks[task_def.task_id] = task_execution
        
        # Create workflow execution
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            name=definition.name,
            status=WorkflowStatus.RUNNING,
            tasks=tasks,
            created_at=datetime.now(),
            started_at=datetime.now(),
            created_by=agent_id,
            execution_mode=definition.execution_mode,
            max_parallel_tasks=definition.max_parallel_tasks
        )
        
        self.workflow_executions[execution_id] = execution
        
        # Queue initial tasks (those with no dependencies)
        for task in tasks.values():
            if not task.dependencies:
                self.execution_queue.append((execution, task))
        
        LOGGER.info(f"Started workflow execution: {execution_id}")
        return execution_id
    
    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status."""
        return self.workflow_executions.get(execution_id)
    
    def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow."""
        execution = self.workflow_executions.get(execution_id)
        if not execution:
            return False
        
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.now()
        
        # Cancel running tasks
        for task in execution.tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
        
        LOGGER.info(f"Cancelled workflow execution: {execution_id}")
        return True
    
    def list_workflows(self) -> Dict[str, Any]:
        """List all workflow definitions and executions."""
        return {
            "definitions": {wid: asdict(wdef) for wid, wdef in self.workflow_definitions.items()},
            "executions": {eid: self._serialize_execution(exe) 
                          for eid, exe in self.workflow_executions.items()}
        }
    
    def _serialize_execution(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Serialize workflow execution for JSON response."""
        data = asdict(execution)
        
        # Convert datetime objects
        data["created_at"] = execution.created_at.isoformat()
        if execution.started_at:
            data["started_at"] = execution.started_at.isoformat()
        if execution.completed_at:
            data["completed_at"] = execution.completed_at.isoformat()
        
        # Convert task datetimes
        for task_id, task_data in data["tasks"].items():
            task = execution.tasks[task_id]
            task_data["created_at"] = task.created_at.isoformat()
            if task.started_at:
                task_data["started_at"] = task.started_at.isoformat()
            if task.completed_at:
                task_data["completed_at"] = task.completed_at.isoformat()
        
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_definitions": len(self.workflow_definitions),
            "total_executions": len(self.workflow_executions),
            "running_executions": len([e for e in self.workflow_executions.values() 
                                     if e.status == WorkflowStatus.RUNNING]),
            "queued_tasks": len(self.execution_queue),
            "running_tasks": len(self.running_tasks),
            "executor_running": self.executor_running
        }

# ---------------------------------------------------------------------------
# Global Orchestrator Instance
# ---------------------------------------------------------------------------
orchestrator = WorkflowOrchestrator()

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "workflow_orchestrator"}), 200

@app.route("/workflows/definitions", methods=["POST"])
def create_workflow_definition() -> Response:
    """Create a new workflow definition."""
    try:
        payload = request.get_json()
        
        # Convert task definitions
        tasks = []
        for task_data in payload["tasks"]:
            task = TaskDefinition(**task_data)
            tasks.append(task)
        
        definition = WorkflowDefinition(
            workflow_id=payload["workflow_id"],
            name=payload["name"],
            description=payload["description"],
            tasks=tasks,
            execution_mode=ExecutionMode(payload.get("execution_mode", "mixed")),
            max_parallel_tasks=payload.get("max_parallel_tasks", 5),
            timeout_seconds=payload.get("timeout_seconds", 3600),
            metadata=payload.get("metadata", {})
        )
        
        workflow_id = orchestrator.create_workflow_definition(definition)
        
        return jsonify({"workflow_id": workflow_id}), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/workflows/<workflow_id>/execute", methods=["POST"])
def execute_workflow(workflow_id: str) -> Response:
    """Execute a workflow."""
    try:
        payload = request.get_json() or {}
        agent_id = request.headers.get("X-Agent-ID", "unknown")
        
        parameters = payload.get("parameters", {})
        
        execution_id = orchestrator.execute_workflow(workflow_id, agent_id, parameters)
        
        return jsonify({"execution_id": execution_id}), 201
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/executions/<execution_id>", methods=["GET"])
def get_execution_status(execution_id: str) -> Response:
    """Get workflow execution status."""
    execution = orchestrator.get_workflow_status(execution_id)
    if not execution:
        return jsonify({"error": "execution not found"}), 404
    
    return jsonify(orchestrator._serialize_execution(execution)), 200

@app.route("/executions/<execution_id>/cancel", methods=["POST"])
def cancel_execution(execution_id: str) -> Response:
    """Cancel a workflow execution."""
    success = orchestrator.cancel_workflow(execution_id)
    if not success:
        return jsonify({"error": "execution not found"}), 404
    
    return jsonify({"message": "cancelled"}), 200

@app.route("/workflows", methods=["GET"])
def list_workflows() -> Response:
    """List all workflows."""
    workflows = orchestrator.list_workflows()
    return jsonify(workflows), 200

@app.route("/statistics", methods=["GET"])
def get_statistics() -> Response:
    """Get orchestrator statistics."""
    stats = orchestrator.get_statistics()
    return jsonify(stats), 200

if __name__ == "__main__":
    # Start workflow executor
    orchestrator.start_executor()
    
    # Register with Tool Registry at startup
    try:
        from agents.common.tool_caller import register_agent_with_registry
        port = int(os.environ.get("PORT", 8080))
        base_url = os.getenv("WORKFLOW_ORCHESTRATOR_URL", f"http://localhost:{port}")
        register_agent_with_registry("workflow_orchestrator", base_url)
    except Exception as e:
        LOGGER.warning(f"Failed to register with Tool Registry: {e}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
