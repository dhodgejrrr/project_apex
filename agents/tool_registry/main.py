"""
Tool Registry Service for Project Apex.

Centralized service discovery and capability management for all agents.
Provides dynamic tool discovery, health monitoring, and service coordination.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from flask import Flask, request, jsonify, Response
import requests
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

# ---------------------------------------------------------------------------
# Configuration & Types
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceInfo:
    name: str
    base_url: str
    status: ServiceStatus
    last_check: datetime
    capabilities: Dict[str, Any]
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class ToolInfo:
    name: str
    agent: str
    endpoint: str
    description: str
    required_params: List[str]
    optional_params: List[str]
    category: str = "general"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("tool_registry")

# ---------------------------------------------------------------------------
# Service Registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """Manages service discovery and capability tracking for all agents."""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.tools: Dict[str, ToolInfo] = {}
        self.last_full_scan = None
        self._init_known_services()
    
    def _init_known_services(self):
        """Initialize known services from environment variables."""
        known_services = {
            "core_analyzer": os.getenv("CORE_ANALYZER_URL", ""),
            "visualizer": os.getenv("VISUALIZER_URL", ""),
            "insight_hunter": os.getenv("INSIGHT_HUNTER_URL", ""),
            "historian": os.getenv("HISTORIAN_URL", ""),
            "publicist": os.getenv("PUBLICIST_URL", ""),
            "scribe": os.getenv("SCRIBE_URL", ""),
            "arbiter": os.getenv("ARBITER_URL", ""),
            "ui_portal": os.getenv("UI_PORTAL_URL", "")
        }
        
        for name, url in known_services.items():
            if url:
                self.services[name] = ServiceInfo(
                    name=name,
                    base_url=url,
                    status=ServiceStatus.UNKNOWN,
                    last_check=datetime.now(),
                    capabilities={}
                )
                LOGGER.info(f"Registered known service: {name} at {url}")
    
    def _invoke_service(self, service_url: str, timeout: int = 10) -> Dict[str, Any]:
        """Make authenticated request to a service."""
        if os.getenv("PUBSUB_EMULATOR_HOST"):
            # Local mode
            resp = requests.get(service_url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            # Production mode with authentication
            auth_req = grequests.Request()
            id_token_credentials = id_token.fetch_id_token(auth_req, service_url.split('/')[0:3])
            
            headers = {
                "Authorization": f"Bearer {id_token_credentials}",
                "User-Agent": "project-apex-tool-registry/1.0",
            }
            
            resp = requests.get(service_url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
    
    def check_service_health(self, service_name: str) -> ServiceInfo:
        """Check health and update capabilities for a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service = self.services[service_name]
        start_time = time.time()
        
        try:
            # Try health endpoint first
            health_url = f"{service.base_url}/health"
            health_response = self._invoke_service(health_url, timeout=5)
            
            # Try capabilities endpoint
            capabilities_url = f"{service.base_url}/tools/capabilities"
            capabilities_response = self._invoke_service(capabilities_url, timeout=10)
            
            # Update service info
            response_time = (time.time() - start_time) * 1000
            service.status = ServiceStatus.HEALTHY
            service.last_check = datetime.now()
            service.capabilities = capabilities_response
            service.response_time_ms = response_time
            service.error_message = None
            
            # Update tools registry
            self._update_tools_from_capabilities(service_name, capabilities_response)
            
            LOGGER.info(f"Service {service_name} is healthy (response time: {response_time:.1f}ms)")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            service.status = ServiceStatus.UNHEALTHY
            service.last_check = datetime.now()
            service.response_time_ms = response_time
            service.error_message = str(e)
            
            LOGGER.warning(f"Service {service_name} health check failed: {e}")
        
        return service
    
    def _update_tools_from_capabilities(self, agent_name: str, capabilities: Dict[str, Any]):
        """Update tools registry from service capabilities response."""
        tools = capabilities.get("tools", [])
        
        # Remove old tools for this agent
        self.tools = {k: v for k, v in self.tools.items() if v.agent != agent_name}
        
        # Add new tools
        for tool_data in tools:
            tool_key = f"{agent_name}.{tool_data['name']}"
            self.tools[tool_key] = ToolInfo(
                name=tool_data["name"],
                agent=agent_name,
                endpoint=tool_data["endpoint"],
                description=tool_data.get("description", ""),
                required_params=tool_data.get("required_params", []),
                optional_params=tool_data.get("optional_params", []),
                category=tool_data.get("category", "general")
            )
    
    def scan_all_services(self) -> Dict[str, ServiceInfo]:
        """Perform health check on all registered services."""
        LOGGER.info("Starting full service scan...")
        
        results = {}
        for service_name in self.services:
            results[service_name] = self.check_service_health(service_name)
        
        self.last_full_scan = datetime.now()
        healthy_count = sum(1 for s in results.values() if s.status == ServiceStatus.HEALTHY)
        
        LOGGER.info(f"Service scan complete: {healthy_count}/{len(results)} services healthy")
        return results
    
    def get_tools_by_category(self, category: str = None) -> List[ToolInfo]:
        """Get tools filtered by category."""
        if category is None:
            return list(self.tools.values())
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def find_tools_for_capability(self, capability_keywords: List[str]) -> List[ToolInfo]:
        """Find tools that match capability keywords."""
        matching_tools = []
        
        for tool in self.tools.values():
            description_lower = tool.description.lower()
            name_lower = tool.name.lower()
            
            for keyword in capability_keywords:
                if keyword.lower() in description_lower or keyword.lower() in name_lower:
                    matching_tools.append(tool)
                    break
        
        return matching_tools
    
    def get_service_topology(self) -> Dict[str, Any]:
        """Get overview of service architecture and relationships."""
        topology = {
            "services": {},
            "tools": {},
            "categories": {},
            "health_summary": {}
        }
        
        # Service overview
        for name, service in self.services.items():
            topology["services"][name] = {
                "status": service.status.value,
                "url": service.base_url,
                "response_time_ms": service.response_time_ms,
                "tools_count": len([t for t in self.tools.values() if t.agent == name])
            }
        
        # Tools by agent
        for tool_key, tool in self.tools.items():
            if tool.agent not in topology["tools"]:
                topology["tools"][tool.agent] = []
            topology["tools"][tool.agent].append({
                "name": tool.name,
                "endpoint": tool.endpoint,
                "category": tool.category
            })
        
        # Tools by category
        for tool in self.tools.values():
            if tool.category not in topology["categories"]:
                topology["categories"][tool.category] = []
            topology["categories"][tool.category].append(f"{tool.agent}.{tool.name}")
        
        # Health summary
        statuses = [s.status for s in self.services.values()]
        topology["health_summary"] = {
            "total_services": len(self.services),
            "healthy": sum(1 for s in statuses if s == ServiceStatus.HEALTHY),
            "unhealthy": sum(1 for s in statuses if s == ServiceStatus.UNHEALTHY),
            "unknown": sum(1 for s in statuses if s == ServiceStatus.UNKNOWN),
            "last_scan": self.last_full_scan.isoformat() if self.last_full_scan else None
        }
        
        return topology

# ---------------------------------------------------------------------------
# Global Registry Instance
# ---------------------------------------------------------------------------
registry = ToolRegistry()

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "tool_registry"}), 200

@app.route("/registry", methods=["GET"])
def get_full_registry() -> Response:
    """Get complete service and tool registry."""
    return jsonify({
        "services": {name: asdict(service) for name, service in registry.services.items()},
        "tools": {key: asdict(tool) for key, tool in registry.tools.items()},
        "last_scan": registry.last_full_scan.isoformat() if registry.last_full_scan else None
    }), 200

@app.route("/services", methods=["GET"])
def get_services() -> Response:
    """Get all registered services with their status."""
    return jsonify({
        "services": {name: asdict(service) for name, service in registry.services.items()}
    }), 200

@app.route("/services/<service_name>/health", methods=["POST"])
def check_service_health(service_name: str) -> Response:
    """Trigger health check for a specific service."""
    try:
        service_info = registry.check_service_health(service_name)
        return jsonify(asdict(service_info)), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Health check failed: {str(e)}"}), 500

@app.route("/scan", methods=["POST"])
def scan_all_services() -> Response:
    """Perform full health scan of all services."""
    results = registry.scan_all_services()
    return jsonify({
        "scan_results": {name: asdict(service) for name, service in results.items()},
        "scan_time": datetime.now().isoformat()
    }), 200

@app.route("/tools", methods=["GET"])
def get_tools() -> Response:
    """Get all available tools across all agents."""
    category = request.args.get("category")
    tools = registry.get_tools_by_category(category)
    
    return jsonify({
        "tools": [asdict(tool) for tool in tools],
        "total_count": len(tools),
        "filtered_by_category": category
    }), 200

@app.route("/tools/search", methods=["POST"])
def search_tools() -> Response:
    """Search for tools by capability keywords."""
    try:
        payload = request.get_json()
        keywords = payload.get("keywords", [])
        
        if not keywords:
            return jsonify({"error": "keywords required"}), 400
        
        matching_tools = registry.find_tools_for_capability(keywords)
        
        return jsonify({
            "matching_tools": [asdict(tool) for tool in matching_tools],
            "keywords": keywords,
            "matches_found": len(matching_tools)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/topology", methods=["GET"])
def get_topology() -> Response:
    """Get system topology and architecture overview."""
    topology = registry.get_service_topology()
    return jsonify(topology), 200

@app.route("/services/register", methods=["POST"])
def register_service() -> Response:
    """Register a new service dynamically."""
    try:
        payload = request.get_json()
        service_name = payload["name"]
        base_url = payload["base_url"]
        
        # Create new service entry
        registry.services[service_name] = ServiceInfo(
            name=service_name,
            base_url=base_url,
            status=ServiceStatus.UNKNOWN,
            last_check=datetime.now(),
            capabilities={}
        )
        
        # Immediate health check
        service_info = registry.check_service_health(service_name)
        
        return jsonify({
            "message": f"Service {service_name} registered successfully",
            "service_info": asdict(service_info)
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Perform initial service scan on startup
    registry.scan_all_services()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
