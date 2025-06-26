#!/usr/bin/env python3
"""
Startup script for Project Apex agents.
Handles service registration with Tool Registry before starting the Flask app.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Add agents directory to Python path
sys.path.insert(0, '/app/agents')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("startup")

def register_with_tool_registry(agent_name: str, max_retries: int = 10, retry_delay: int = 5):
    """Register this agent with the Tool Registry service."""
    try:
        from agents.common.tool_caller import register_agent_with_registry
        
        port = int(os.environ.get("PORT", 8080))
        
        # Map agent names to their service URLs in docker-compose
        service_urls = {
            "core_analyzer": os.getenv("CORE_ANALYZER_URL", f"http://core-analyzer:{port}"),
            "visualizer": os.getenv("VISUALIZER_URL", f"http://visualizer:{port}"),
            "insight_hunter": os.getenv("INSIGHT_HUNTER_URL", f"http://insight-hunter:{port}"),
            "publicist": os.getenv("PUBLICIST_URL", f"http://publicist:{port}"),
            "historian": os.getenv("HISTORIAN_URL", f"http://historian:{port}"),
            "scribe": os.getenv("SCRIBE_URL", f"http://scribe:{port}"),
            "arbiter": os.getenv("ARBITER_URL", f"http://arbiter:{port}"),
            "adk_orchestrator": os.getenv("ADK_ORCHESTRATOR_URL", f"http://adk-orchestrator:{port}"),
        }
        
        base_url = service_urls.get(agent_name, f"http://{agent_name}:{port}")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to register {agent_name} with Tool Registry (attempt {attempt + 1}/{max_retries})")
                success = register_agent_with_registry(agent_name, base_url)
                if success:
                    logger.info(f"Successfully registered {agent_name} with Tool Registry")
                    return True
                else:
                    logger.warning(f"Registration returned False for {agent_name}")
            except Exception as e:
                logger.warning(f"Registration attempt {attempt + 1} failed for {agent_name}: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
        
        logger.error(f"Failed to register {agent_name} after {max_retries} attempts")
        return False
        
    except ImportError as e:
        logger.warning(f"Could not import registration function: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        return False

def start_gunicorn(agent_name: str):
    """Start the Gunicorn server for the agent."""
    port = os.environ.get("PORT", "8080")
    
    # Change to the agent's directory
    agent_dir = f"/app/agents/{agent_name}"
    if os.path.exists(agent_dir):
        os.chdir(agent_dir)
        logger.info(f"Changed directory to {agent_dir}")
    else:
        logger.warning(f"Agent directory {agent_dir} not found, staying in current directory")
    
    # Start Gunicorn
    cmd = [
        "gunicorn",
        "--bind", f":{port}",
        "--workers", "1",
        "--threads", "8", 
        "--timeout", "0",
        "main:app"
    ]
    
    logger.info(f"Starting Gunicorn for {agent_name}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Gunicorn failed for {agent_name}: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info(f"Shutting down {agent_name}")
        sys.exit(0)

def main():
    """Main startup function."""
    agent_name = os.environ.get("AGENT_NAME")
    if not agent_name:
        logger.error("AGENT_NAME environment variable not set")
        sys.exit(1)
    
    logger.info(f"Starting agent: {agent_name}")
    
    # Skip registration for core services that don't need it
    skip_registration = ["tool_registry", "performance_monitor", "task_router", "ui_portal"]
    
    if agent_name not in skip_registration:
        # Wait a bit for Tool Registry to be ready
        logger.info("Waiting for services to initialize...")
        time.sleep(10)
        
        # Attempt registration (but don't fail if it doesn't work)
        try:
            registration_success = register_with_tool_registry(agent_name)
            if registration_success:
                logger.info(f"Registration successful for {agent_name}")
            else:
                logger.warning(f"Registration failed for {agent_name}, continuing anyway...")
        except Exception as e:
            logger.error(f"Registration error for {agent_name}: {e}, continuing anyway...")
    
    # Always start the main application regardless of registration status
    logger.info(f"Starting Gunicorn server for {agent_name}...")
    start_gunicorn(agent_name)

if __name__ == "__main__":
    main()
