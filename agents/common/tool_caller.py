"""
Universal tool calling infrastructure for Project Apex agents.

Provides standardized methods for agents to discover and invoke each other's
capabilities with proper authentication and error handling.
"""
from __future__ import annotations

import json
import logging
import os
import time
import requests
from typing import Any, Dict, List, Optional
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

LOGGER = logging.getLogger("apex.tool_caller")


class ToolCaller:
    """Handles authenticated tool calls between Project Apex agents."""
    
    def __init__(self):
        self.tool_registry_url = os.getenv("TOOL_REGISTRY_URL", "")
        self.performance_monitor_url = os.getenv("PERFORMANCE_MONITOR_URL", "")
        self.task_router_url = os.getenv("TASK_ROUTER_URL", "")
        self.service_registry = self._build_service_registry()
        self._cached_registry = None
        self._cache_timeout = 300  # 5 minutes
        self._last_cache_update = 0
        
        # Performance tracking
        self._enable_performance_tracking = True
        self._task_counter = 0
    
    def _build_service_registry(self) -> Dict[str, str]:
        """Build registry of known services from environment variables."""
        return {
            "core_analyzer": os.getenv("CORE_ANALYZER_URL", ""),
            "visualizer": os.getenv("VISUALIZER_URL", ""),
            "insight_hunter": os.getenv("INSIGHT_HUNTER_URL", ""),
            "historian": os.getenv("HISTORIAN_URL", ""),
            "publicist": os.getenv("PUBLICIST_URL", ""),
            "scribe": os.getenv("SCRIBE_URL", ""),
            "arbiter": os.getenv("ARBITER_URL", ""),
            "tool_registry": self.tool_registry_url
        }
    
    def invoke_cloud_run(self, service_url: str, payload: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Invoke a Cloud Run service with proper authentication.
        
        Args:
            service_url: Full URL to the service endpoint
            payload: JSON payload to send
            timeout: Request timeout in seconds
            
        Returns:
            JSON response from the service
            
        Raises:
            requests.RequestException: If the request fails
        """
        # Check if running in local/emulated environment
        if os.getenv("PUBSUB_EMULATOR_HOST"):
            LOGGER.info(f"Invoking {service_url} via simple HTTP (local)...")
            
            resp = requests.post(service_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            # Production path: use authenticated request
            LOGGER.info(f"Invoking {service_url} via authenticated request (production)...")
            auth_req = grequests.Request()
            id_token_credentials = id_token.fetch_id_token(auth_req, service_url)

            headers = {
                "Authorization": f"Bearer {id_token_credentials}",
                "Content-Type": "application/json",
                "User-Agent": "project-apex-tool-caller/1.0",
            }

            resp = requests.post(service_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
    
    def discover_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """
        Discover available tools for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., 'core_analyzer', 'visualizer')
            
        Returns:
            Dictionary containing available tools and their specifications
        """
        # Try Tool Registry first
        registry_data = self._get_registry_data()
        if registry_data and "services" in registry_data:
            service_info = registry_data["services"].get(agent_name)
            if service_info and service_info.get("status") == "healthy":
                # Return cached capabilities from registry
                capabilities = service_info.get("capabilities", {})
                if capabilities:
                    LOGGER.info(f"Using cached capabilities for {agent_name} from Tool Registry")
                    return capabilities
        
        # Fallback to direct discovery
        base_url = self.service_registry.get(agent_name)
        if not base_url:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        capabilities_url = f"{base_url}/tools/capabilities"
        
        try:
            # Use GET request for capabilities endpoint
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                # Local environment - use simple HTTP GET
                LOGGER.info(f"Invoking {capabilities_url} via GET (local)...")
                response = requests.get(capabilities_url, timeout=300)
                response.raise_for_status()
                return response.json()
            else:
                # Production environment - use authenticated GET (implement if needed)
                # For now, fall back to empty capabilities
                LOGGER.warning(f"Authenticated GET not implemented for production, using empty capabilities")
                return {"tools": []}
        except Exception as e:
            LOGGER.error(f"Failed to discover capabilities for {agent_name}: {e}")
            return {"tools": [], "error": str(e)}
    
    def call_tool(self, agent_name: str, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific tool on an agent.
        
        Args:
            agent_name: Name of the agent
            tool_name: Name of the tool to call
            params: Parameters to pass to the tool
            
        Returns:
            Response from the tool
        """
        # First discover capabilities to get the correct endpoint
        capabilities = self.discover_capabilities(agent_name)
        
        # Find the tool in capabilities
        tool_info = None
        for tool in capabilities.get("tools", []):
            if tool["name"] == tool_name:
                tool_info = tool
                break
        
        if not tool_info:
            raise ValueError(f"Tool '{tool_name}' not found for agent '{agent_name}'")
        
        # Build full URL
        base_url = self.service_registry[agent_name]
        tool_url = f"{base_url}{tool_info['endpoint']}"
        
        # Make the tool call
        try:
            response = self.invoke_cloud_run(tool_url, params)
            LOGGER.info(f"Successfully called {agent_name}.{tool_name}")
            return response
        except Exception as e:
            LOGGER.error(f"Tool call failed {agent_name}.{tool_name}: {e}")
            raise
    
    def call_core_analyzer_tool(self, tool_name: str, run_id: str, **kwargs) -> Dict[str, Any]:
        """Convenience method for calling CoreAnalyzer tools."""
        params = {"run_id": run_id, **kwargs}
        return self.call_tool("core_analyzer", tool_name, params)
    
    def call_visualizer_tool(self, tool_name: str, analysis_path: str, **kwargs) -> Dict[str, Any]:
        """Convenience method for calling Visualizer tools."""
        params = {"analysis_path": analysis_path, **kwargs}
        return self.call_tool("visualizer", tool_name, params)
    
    def get_all_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities for all known agents."""
        # Try Tool Registry first for a comprehensive view
        registry_data = self._get_registry_data()
        if registry_data and "services" in registry_data:
            all_capabilities = {}
            for agent_name, service_info in registry_data["services"].items():
                if service_info.get("status") == "healthy":
                    capabilities = service_info.get("capabilities", {})
                    if capabilities:
                        all_capabilities[agent_name] = capabilities
                    else:
                        all_capabilities[agent_name] = {"error": "No capabilities cached"}
                else:
                    all_capabilities[agent_name] = {"error": f"Service unhealthy: {service_info.get('status')}"}
            
            if all_capabilities:
                LOGGER.info("Using Tool Registry for comprehensive capabilities overview")
                return all_capabilities
        
        # Fallback to direct discovery
        all_capabilities = {}
        for agent_name in self.service_registry:
            if self.service_registry[agent_name]:  # Only check if URL is configured
                try:
                    all_capabilities[agent_name] = self.discover_capabilities(agent_name)
                except Exception as e:
                    all_capabilities[agent_name] = {"error": str(e)}
        
        return all_capabilities
    
    def _get_registry_data(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Get registry data with caching."""
        import time
        
        current_time = time.time()
        
        # Check if we need to refresh the cache
        if (force_refresh or 
            self._cached_registry is None or 
            current_time - self._last_cache_update > self._cache_timeout):
            
            if not self.tool_registry_url:
                LOGGER.warning("Tool Registry URL not configured, falling back to static discovery")
                return None
            
            try:
                registry_url = f"{self.tool_registry_url}/registry"
                
                if os.getenv("PUBSUB_EMULATOR_HOST"):
                    resp = requests.get(registry_url, timeout=10)
                else:
                    auth_req = grequests.Request()
                    id_token_credentials = id_token.fetch_id_token(auth_req, registry_url)
                    headers = {
                        "Authorization": f"Bearer {id_token_credentials}",
                        "User-Agent": "project-apex-tool-caller/1.0",
                    }
                    resp = requests.get(registry_url, headers=headers, timeout=10)
                
                resp.raise_for_status()
                self._cached_registry = resp.json()
                self._last_cache_update = current_time
                LOGGER.info("Successfully refreshed tool registry cache")
                
            except Exception as e:
                LOGGER.warning(f"Failed to fetch tool registry: {e}")
                if self._cached_registry is None:
                    return None
        
        return self._cached_registry
    
    def register_with_tool_registry(self, agent_name: str, base_url: str) -> bool:
        """Register this agent with the central Tool Registry."""
        if not self.tool_registry_url:
            LOGGER.warning("Tool Registry URL not configured, skipping registration")
            return False
        
        try:
            register_url = f"{self.tool_registry_url}/services/register"
            payload = {
                "name": agent_name,
                "base_url": base_url
            }
            
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                resp = requests.post(register_url, json=payload, timeout=10)
            else:
                auth_req = grequests.Request()
                id_token_credentials = id_token.fetch_id_token(auth_req, register_url)
                headers = {
                    "Authorization": f"Bearer {id_token_credentials}",
                    "Content-Type": "application/json",
                    "User-Agent": "project-apex-tool-caller/1.0",
                }
                resp = requests.post(register_url, json=payload, headers=headers, timeout=10)
            
            resp.raise_for_status()
            LOGGER.info(f"Successfully registered {agent_name} with Tool Registry")
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to register {agent_name} with Tool Registry: {e}")
            return False
        
    def search_tools_by_capability(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search for tools across all agents by capability keywords."""
        if not self.tool_registry_url:
            LOGGER.warning("Tool Registry not available, cannot search tools")
            return []
        
        try:
            search_url = f"{self.tool_registry_url}/tools/search"
            payload = {"keywords": keywords}
            
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                resp = requests.post(search_url, json=payload, timeout=10)
            else:
                auth_req = grequests.Request()
                id_token_credentials = id_token.fetch_id_token(auth_req, search_url)
                headers = {
                    "Authorization": f"Bearer {id_token_credentials}",
                    "Content-Type": "application/json",
                    "User-Agent": "project-apex-tool-caller/1.0",
                }
                resp = requests.post(search_url, json=payload, headers=headers, timeout=10)
            
            resp.raise_for_status()
            result = resp.json()
            return result.get("matching_tools", [])
            
        except Exception as e:
            LOGGER.error(f"Failed to search tools: {e}")
            return []
    
    def get_system_topology(self) -> Dict[str, Any]:
        """Get system topology from Tool Registry."""
        if not self.tool_registry_url:
            return {"error": "Tool Registry not available"}
        
        try:
            topology_url = f"{self.tool_registry_url}/topology"
            
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                resp = requests.get(topology_url, timeout=10)
            else:
                auth_req = grequests.Request()
                id_token_credentials = id_token.fetch_id_token(auth_req, topology_url)
                headers = {
                    "Authorization": f"Bearer {id_token_credentials}",
                    "User-Agent": "project-apex-tool-caller/1.0",
                }
                resp = requests.get(topology_url, headers=headers, timeout=10)
            
            resp.raise_for_status()
            return resp.json()
            
        except Exception as e:
            LOGGER.error(f"Failed to get system topology: {e}")
            return {"error": str(e)}
    
    def call_tool_intelligently(self, task_type: str, params: Dict[str, Any], 
                                priority: str = "normal") -> Dict[str, Any]:
        """
        Call a tool using intelligent routing based on performance data and learning.
        
        Args:
            task_type: Type of task (e.g., 'analysis', 'visualization', 'insight_generation')
            params: Parameters for the task
            priority: Task priority ('critical', 'high', 'normal', 'low')
            
        Returns:
            Task execution result with routing metadata
        """
        if not self.task_router_url:
            LOGGER.warning("Task Router not configured, falling back to static routing")
            return self._fallback_intelligent_call(task_type, params)
        
        import uuid
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Step 1: Get intelligent routing recommendation
            routing_payload = {
                "task_id": task_id,
                "task_type": task_type,
                "priority": priority,
                "payload": params
            }
            
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                resp = requests.post(f"{self.task_router_url}/route", json=routing_payload, timeout=10)
            else:
                auth_req = grequests.Request()
                id_token_credentials = id_token.fetch_id_token(auth_req, self.task_router_url)
                headers = {
                    "Authorization": f"Bearer {id_token_credentials}",
                    "Content-Type": "application/json",
                    "User-Agent": "project-apex-tool-caller/1.0",
                }
                resp = requests.post(f"{self.task_router_url}/route", json=routing_payload, 
                                   headers=headers, timeout=10)
            
            resp.raise_for_status()
            routing_response = resp.json()
            routing_decision = routing_response["routing_decision"]
            
            selected_agent = routing_decision["selected_agent"]
            confidence = routing_decision["confidence"]
            
            LOGGER.info(f"Intelligent routing selected {selected_agent} for {task_type} "
                       f"(confidence: {confidence:.2f})")
            
            # Step 2: Execute the task on the selected agent
            execution_start = time.time()
            
            try:
                # Map task type to specific tool calls
                result = self._execute_on_agent(selected_agent, task_type, params)
                execution_time = (time.time() - execution_start) * 1000
                success = True
                error_message = None
                
            except Exception as e:
                execution_time = (time.time() - execution_start) * 1000
                success = False
                error_message = str(e)
                result = {"error": error_message, "agent": selected_agent}
                
                LOGGER.error(f"Task execution failed on {selected_agent}: {e}")
            
            # Step 3: Report results back to task router and performance monitor
            self._report_task_result(task_id, selected_agent, task_type, success, 
                                   execution_time, error_message)
            
            # Step 4: Add routing metadata to result
            result["_routing_metadata"] = {
                "task_id": task_id,
                "selected_agent": selected_agent,
                "confidence": confidence,
                "reasoning": routing_decision["reasoning"],
                "execution_time_ms": execution_time,
                "success": success
            }
            
            return result
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            LOGGER.error(f"Intelligent routing failed: {e}")
            
            # Fallback to traditional routing
            fallback_result = self._fallback_intelligent_call(task_type, params)
            fallback_result["_routing_metadata"] = {
                "task_id": task_id,
                "selected_agent": "fallback",
                "confidence": 0.3,
                "reasoning": f"Intelligent routing failed: {e}",
                "execution_time_ms": total_time,
                "success": "error" not in fallback_result
            }
            
            return fallback_result
    
    def _execute_on_agent(self, agent_name: str, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task on a specific agent based on task type."""
        
        # Map task types to agent-specific tool calls
        task_mappings = {
            "analysis": {
                "core_analyzer": ["driver_deltas", "trend_analysis", "stint_analysis"],
                "insight_hunter": ["analyze"]
            },
            "visualization": {
                "visualizer": ["pit_times", "consistency", "stint_falloff"]
            },
            "insight_generation": {
                "insight_hunter": ["analyze"]
            },
            "post_generation": {
                "publicist": ["generate"]
            },
            "report_generation": {
                "scribe": ["generate"]
            }
        }
        
        # Find appropriate tool for this agent and task type
        if task_type in task_mappings and agent_name in task_mappings[task_type]:
            available_tools = task_mappings[task_type][agent_name]
            tool_name = available_tools[0]  # Use first available tool
            
            return self.call_tool(agent_name, tool_name, params)
        else:
            # Generic fallback - try to discover capabilities
            capabilities = self.discover_capabilities(agent_name)
            available_tools = capabilities.get("tools", [])
            
            if available_tools:
                # Use first available tool
                tool_name = available_tools[0]["name"]
                return self.call_tool(agent_name, tool_name, params)
            else:
                raise ValueError(f"No suitable tools found for {task_type} on {agent_name}")
    
    def _fallback_intelligent_call(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback intelligent call when task router is unavailable."""
        
        # Simple task type to agent mapping
        agent_preferences = {
            "analysis": "core_analyzer",
            "visualization": "visualizer", 
            "insight_generation": "insight_hunter",
            "post_generation": "publicist",
            "report_generation": "scribe"
        }
        
        preferred_agent = agent_preferences.get(task_type, "core_analyzer")
        
        try:
            return self._execute_on_agent(preferred_agent, task_type, params)
        except Exception as e:
            LOGGER.error(f"Fallback execution failed: {e}")
            return {"error": str(e), "agent": preferred_agent}
    
    def _report_task_result(self, task_id: str, agent_name: str, task_type: str, 
                           success: bool, execution_time_ms: float, error_message: str = None):
        """Report task execution results to monitoring services."""
        
        # Report to task router
        if self.task_router_url:
            try:
                result_payload = {
                    "success": success,
                    "execution_time_ms": execution_time_ms,
                    "error_message": error_message
                }
                
                if os.getenv("PUBSUB_EMULATOR_HOST"):
                    requests.post(f"{self.task_router_url}/tasks/{task_id}/result", 
                                json=result_payload, timeout=5)
                else:
                    auth_req = grequests.Request()
                    id_token_credentials = id_token.fetch_id_token(auth_req, self.task_router_url)
                    headers = {
                        "Authorization": f"Bearer {id_token_credentials}",
                        "Content-Type": "application/json",
                        "User-Agent": "project-apex-tool-caller/1.0",
                    }
                    requests.post(f"{self.task_router_url}/tasks/{task_id}/result", 
                                json=result_payload, headers=headers, timeout=5)
                    
            except Exception as e:
                LOGGER.warning(f"Failed to report to task router: {e}")
        
        # Report to performance monitor
        if self.performance_monitor_url:
            try:
                metric_payload = {
                    "agent_name": agent_name,
                    "tool_name": task_type,
                    "execution_time_ms": execution_time_ms,
                    "success": success,
                    "error_message": error_message
                }
                
                if os.getenv("PUBSUB_EMULATOR_HOST"):
                    requests.post(f"{self.performance_monitor_url}/metrics", 
                                json=metric_payload, timeout=5)
                else:
                    auth_req = grequests.Request()
                    id_token_credentials = id_token.fetch_id_token(auth_req, self.performance_monitor_url)
                    headers = {
                        "Authorization": f"Bearer {id_token_credentials}",
                        "Content-Type": "application/json",
                        "User-Agent": "project-apex-tool-caller/1.0",
                    }
                    requests.post(f"{self.performance_monitor_url}/metrics", 
                                json=metric_payload, headers=headers, timeout=5)
                    
            except Exception as e:
                LOGGER.warning(f"Failed to report to performance monitor: {e}")
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights from performance monitor."""
        if not self.performance_monitor_url:
            return {"error": "Performance Monitor not configured"}
        
        try:
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                resp = requests.get(f"{self.performance_monitor_url}/analytics", timeout=10)
            else:
                auth_req = grequests.Request()
                id_token_credentials = id_token.fetch_id_token(auth_req, self.performance_monitor_url)
                headers = {
                    "Authorization": f"Bearer {id_token_credentials}",
                    "User-Agent": "project-apex-tool-caller/1.0",
                }
                resp = requests.get(f"{self.performance_monitor_url}/analytics", 
                                  headers=headers, timeout=10)
            
            resp.raise_for_status()
            return resp.json()
            
        except Exception as e:
            LOGGER.error(f"Failed to get performance insights: {e}")
            return {"error": str(e)}
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get routing analytics from task router."""
        if not self.task_router_url:
            return {"error": "Task Router not configured"}
        
        try:
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                resp = requests.get(f"{self.task_router_url}/analytics", timeout=10)
            else:
                auth_req = grequests.Request()
                id_token_credentials = id_token.fetch_id_token(auth_req, self.task_router_url)
                headers = {
                    "Authorization": f"Bearer {id_token_credentials}",
                    "User-Agent": "project-apex-tool-caller/1.0",
                }
                resp = requests.get(f"{self.task_router_url}/analytics", 
                                  headers=headers, timeout=10)
            
            resp.raise_for_status()
            return resp.json()
            
        except Exception as e:
            LOGGER.error(f"Failed to get routing analytics: {e}")
            return {"error": str(e)}


# Global instance for easy importing
tool_caller = ToolCaller()


def discover_agent_capabilities(agent_name: str) -> Dict[str, Any]:
    """Convenience function to discover capabilities of an agent."""
    return tool_caller.discover_capabilities(agent_name)


def call_agent_tool(agent_name: str, tool_name: str, **params) -> Dict[str, Any]:
    """Convenience function to call a tool on an agent."""
    return tool_caller.call_tool(agent_name, tool_name, params)


def search_tools_by_capability(keywords: List[str]) -> List[Dict[str, Any]]:
    """Convenience function to search for tools by capability keywords."""
    return tool_caller.search_tools_by_capability(keywords)


def get_system_topology() -> Dict[str, Any]:
    """Convenience function to get system topology."""
    return tool_caller.get_system_topology()


def register_agent_with_registry(agent_name: str, base_url: str) -> bool:
    """Convenience function to register an agent with the Tool Registry."""
    return tool_caller.register_with_tool_registry(agent_name, base_url)


def call_tool_intelligently(task_type: str, params: Dict[str, Any], priority: str = "normal") -> Dict[str, Any]:
    """Convenience function for intelligent tool calling."""
    return tool_caller.call_tool_intelligently(task_type, params, priority)


def get_performance_insights() -> Dict[str, Any]:
    """Convenience function to get performance insights."""
    return tool_caller.get_performance_insights()


def get_routing_analytics() -> Dict[str, Any]:
    """Convenience function to get routing analytics."""
    return tool_caller.get_routing_analytics()
