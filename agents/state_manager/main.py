"""
State Manager Service for Project Apex.

Provides centralized state management, event-driven synchronization, and
conflict resolution for multi-agent coordination.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict

from flask import Flask, request, jsonify, Response
import requests
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

# ---------------------------------------------------------------------------
# Configuration & Types
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")

class StateEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ACCESSED = "accessed"

class ConflictResolution(str, Enum):
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE = "merge"
    REJECT = "reject"

@dataclass
class StateEntry:
    key: str
    value: Any
    version: int
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    ttl: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS

@dataclass
class StateEvent:
    event_id: str
    event_type: StateEventType
    key: str
    agent_id: str
    timestamp: datetime
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = None

@dataclass
class Subscription:
    subscription_id: str
    agent_id: str
    key_pattern: str
    webhook_url: str
    event_types: Set[StateEventType]
    created_at: datetime
    active: bool = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("state_manager")

# ---------------------------------------------------------------------------
# State Manager
# ---------------------------------------------------------------------------
class StateManager:
    """Manages shared state across agents with versioning and event notifications."""
    
    def __init__(self):
        self.state_store: Dict[str, StateEntry] = {}
        self.event_history: List[StateEvent] = []
        self.subscriptions: Dict[str, Subscription] = {}
        self.locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.cleanup_interval = 300  # 5 minutes
        self.max_event_history = 10000
        self._last_cleanup = time.time()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return str(uuid.uuid4())
    
    def _cleanup_expired(self):
        """Clean up expired state entries and old events."""
        current_time = time.time()
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        now = datetime.now()
        expired_keys = []
        
        # Find expired entries
        for key, entry in self.state_store.items():
            if entry.ttl and entry.ttl < now:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            self._delete_state_internal(key, "system_cleanup")
        
        # Trim event history
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history:]
        
        self._last_cleanup = current_time
        
        if expired_keys:
            LOGGER.info(f"Cleaned up {len(expired_keys)} expired state entries")
    
    def _emit_event(self, event: StateEvent):
        """Emit state change event and notify subscribers."""
        self.event_history.append(event)
        LOGGER.debug(f"Emitted event: {event.event_type} for key {event.key}")
        
        # Notify subscribers (async in production)
        self._notify_subscribers(event)
    
    def _notify_subscribers(self, event: StateEvent):
        """Notify relevant subscribers about state changes."""
        for subscription in self.subscriptions.values():
            if not subscription.active:
                continue
                
            # Check if event matches subscription
            if event.event_type not in subscription.event_types:
                continue
                
            # Simple pattern matching (could be enhanced with regex)
            if subscription.key_pattern != "*" and subscription.key_pattern != event.key:
                if not (subscription.key_pattern.endswith("*") and 
                       event.key.startswith(subscription.key_pattern[:-1])):
                    continue
            
            # Send notification (simplified for local development)
            try:
                self._send_webhook_notification(subscription, event)
            except Exception as e:
                LOGGER.warning(f"Failed to notify subscriber {subscription.agent_id}: {e}")
    
    def _send_webhook_notification(self, subscription: Subscription, event: StateEvent):
        """Send webhook notification to subscriber."""
        payload = {
            "event": asdict(event),
            "subscription_id": subscription.subscription_id
        }
        
        # Convert datetime objects to ISO strings
        payload["event"]["timestamp"] = event.timestamp.isoformat()
        
        if os.getenv("PUBSUB_EMULATOR_HOST"):
            # Local mode
            requests.post(subscription.webhook_url, json=payload, timeout=5)
        else:
            # Production mode with authentication
            auth_req = grequests.Request()
            id_token_credentials = id_token.fetch_id_token(auth_req, subscription.webhook_url)
            
            headers = {
                "Authorization": f"Bearer {id_token_credentials}",
                "Content-Type": "application/json",
                "User-Agent": "project-apex-state-manager/1.0",
            }
            
            requests.post(subscription.webhook_url, json=payload, headers=headers, timeout=5)
    
    def get_state(self, key: str, agent_id: str) -> Optional[StateEntry]:
        """Get state value by key."""
        self._cleanup_expired()
        
        with self.locks[key]:
            entry = self.state_store.get(key)
            if entry and (not entry.ttl or entry.ttl > datetime.now()):
                # Emit access event
                event = StateEvent(
                    event_id=self._generate_event_id(),
                    event_type=StateEventType.ACCESSED,
                    key=key,
                    agent_id=agent_id,
                    timestamp=datetime.now()
                )
                self._emit_event(event)
                return entry
            
            return None
    
    def set_state(self, key: str, value: Any, agent_id: str, 
                  ttl_seconds: Optional[int] = None,
                  conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
                  metadata: Optional[Dict[str, Any]] = None) -> StateEntry:
        """Set state value with conflict resolution."""
        self._cleanup_expired()
        
        now = datetime.now()
        ttl = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None
        
        with self.locks[key]:
            existing = self.state_store.get(key)
            
            if existing:
                # Handle conflict resolution
                if conflict_resolution == ConflictResolution.FIRST_WRITE_WINS:
                    LOGGER.info(f"Rejected update to {key} due to FIRST_WRITE_WINS policy")
                    return existing
                elif conflict_resolution == ConflictResolution.REJECT:
                    raise ValueError(f"State key {key} already exists and updates are rejected")
                elif conflict_resolution == ConflictResolution.MERGE:
                    # Simple merge strategy for dict values
                    if isinstance(existing.value, dict) and isinstance(value, dict):
                        merged_value = {**existing.value, **value}
                        value = merged_value
                
                # Create updated entry
                entry = StateEntry(
                    key=key,
                    value=value,
                    version=existing.version + 1,
                    created_at=existing.created_at,
                    updated_at=now,
                    created_by=existing.created_by,
                    updated_by=agent_id,
                    ttl=ttl,
                    metadata=metadata or {},
                    conflict_resolution=conflict_resolution
                )
                
                event_type = StateEventType.UPDATED
                old_value = existing.value
            else:
                # Create new entry
                entry = StateEntry(
                    key=key,
                    value=value,
                    version=1,
                    created_at=now,
                    updated_at=now,
                    created_by=agent_id,
                    updated_by=agent_id,
                    ttl=ttl,
                    metadata=metadata or {},
                    conflict_resolution=conflict_resolution
                )
                
                event_type = StateEventType.CREATED
                old_value = None
            
            self.state_store[key] = entry
            
            # Emit change event
            event = StateEvent(
                event_id=self._generate_event_id(),
                event_type=event_type,
                key=key,
                agent_id=agent_id,
                timestamp=now,
                old_value=old_value,
                new_value=value,
                metadata=metadata
            )
            self._emit_event(event)
            
            return entry
    
    def _delete_state_internal(self, key: str, agent_id: str) -> bool:
        """Internal delete method without lock acquisition."""
        entry = self.state_store.get(key)
        if not entry:
            return False
        
        del self.state_store[key]
        
        # Emit delete event
        event = StateEvent(
            event_id=self._generate_event_id(),
            event_type=StateEventType.DELETED,
            key=key,
            agent_id=agent_id,
            timestamp=datetime.now(),
            old_value=entry.value
        )
        self._emit_event(event)
        
        return True
    
    def delete_state(self, key: str, agent_id: str) -> bool:
        """Delete state entry."""
        with self.locks[key]:
            return self._delete_state_internal(key, agent_id)
    
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all state keys, optionally filtered by prefix."""
        self._cleanup_expired()
        
        keys = list(self.state_store.keys())
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        
        return sorted(keys)
    
    def subscribe(self, agent_id: str, key_pattern: str, webhook_url: str,
                  event_types: Set[StateEventType]) -> str:
        """Subscribe to state change events."""
        subscription_id = str(uuid.uuid4())
        
        subscription = Subscription(
            subscription_id=subscription_id,
            agent_id=agent_id,
            key_pattern=key_pattern,
            webhook_url=webhook_url,
            event_types=event_types,
            created_at=datetime.now()
        )
        
        self.subscriptions[subscription_id] = subscription
        LOGGER.info(f"Created subscription {subscription_id} for {agent_id}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from state change events."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            LOGGER.info(f"Removed subscription {subscription_id}")
            return True
        return False
    
    def get_events(self, since: Optional[datetime] = None, 
                   key_pattern: Optional[str] = None,
                   limit: int = 100) -> List[StateEvent]:
        """Get state change events with optional filtering."""
        events = self.event_history
        
        if since:
            events = [e for e in events if e.timestamp > since]
        
        if key_pattern and key_pattern != "*":
            if key_pattern.endswith("*"):
                prefix = key_pattern[:-1]
                events = [e for e in events if e.key.startswith(prefix)]
            else:
                events = [e for e in events if e.key == key_pattern]
        
        return events[-limit:] if limit else events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        now = datetime.now()
        
        return {
            "total_keys": len(self.state_store),
            "total_events": len(self.event_history),
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.active]),
            "expired_keys": len([e for e in self.state_store.values() 
                               if e.ttl and e.ttl < now]),
            "memory_usage_bytes": len(str(self.state_store).encode('utf-8')),
            "last_cleanup": datetime.fromtimestamp(self._last_cleanup).isoformat()
        }

# ---------------------------------------------------------------------------
# Global State Manager Instance
# ---------------------------------------------------------------------------
state_manager = StateManager()

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "state_manager"}), 200

@app.route("/state/<key>", methods=["GET"])
def get_state_endpoint(key: str) -> Response:
    """Get state value by key."""
    agent_id = request.headers.get("X-Agent-ID", "unknown")
    
    entry = state_manager.get_state(key, agent_id)
    if not entry:
        return jsonify({"error": "key not found"}), 404
    
    response = asdict(entry)
    # Convert datetime objects to ISO strings
    response["created_at"] = entry.created_at.isoformat()
    response["updated_at"] = entry.updated_at.isoformat()
    if entry.ttl:
        response["ttl"] = entry.ttl.isoformat()
    
    return jsonify(response), 200

@app.route("/state/<key>", methods=["PUT"])
def set_state_endpoint(key: str) -> Response:
    """Set state value."""
    try:
        payload = request.get_json()
        agent_id = request.headers.get("X-Agent-ID", "unknown")
        
        value = payload["value"]
        ttl_seconds = payload.get("ttl_seconds")
        conflict_resolution = ConflictResolution(payload.get("conflict_resolution", "last_write_wins"))
        metadata = payload.get("metadata", {})
        
        entry = state_manager.set_state(
            key=key,
            value=value,
            agent_id=agent_id,
            ttl_seconds=ttl_seconds,
            conflict_resolution=conflict_resolution,
            metadata=metadata
        )
        
        response = asdict(entry)
        response["created_at"] = entry.created_at.isoformat()
        response["updated_at"] = entry.updated_at.isoformat()
        if entry.ttl:
            response["ttl"] = entry.ttl.isoformat()
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/state/<key>", methods=["DELETE"])
def delete_state_endpoint(key: str) -> Response:
    """Delete state entry."""
    agent_id = request.headers.get("X-Agent-ID", "unknown")
    
    success = state_manager.delete_state(key, agent_id)
    if not success:
        return jsonify({"error": "key not found"}), 404
    
    return jsonify({"message": "deleted"}), 200

@app.route("/state", methods=["GET"])
def list_keys_endpoint() -> Response:
    """List all state keys."""
    prefix = request.args.get("prefix")
    keys = state_manager.list_keys(prefix)
    
    return jsonify({"keys": keys, "count": len(keys)}), 200

@app.route("/subscriptions", methods=["POST"])
def subscribe_endpoint() -> Response:
    """Subscribe to state change events."""
    try:
        payload = request.get_json()
        agent_id = request.headers.get("X-Agent-ID", "unknown")
        
        key_pattern = payload["key_pattern"]
        webhook_url = payload["webhook_url"]
        event_types = set(StateEventType(t) for t in payload.get("event_types", ["created", "updated", "deleted"]))
        
        subscription_id = state_manager.subscribe(agent_id, key_pattern, webhook_url, event_types)
        
        return jsonify({"subscription_id": subscription_id}), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/subscriptions/<subscription_id>", methods=["DELETE"])
def unsubscribe_endpoint(subscription_id: str) -> Response:
    """Unsubscribe from state change events."""
    success = state_manager.unsubscribe(subscription_id)
    if not success:
        return jsonify({"error": "subscription not found"}), 404
    
    return jsonify({"message": "unsubscribed"}), 200

@app.route("/events", methods=["GET"])
def get_events_endpoint() -> Response:
    """Get state change events."""
    since_str = request.args.get("since")
    key_pattern = request.args.get("key_pattern")
    limit = int(request.args.get("limit", 100))
    
    since = datetime.fromisoformat(since_str) if since_str else None
    events = state_manager.get_events(since, key_pattern, limit)
    
    # Convert events to JSON-serializable format
    events_data = []
    for event in events:
        event_dict = asdict(event)
        event_dict["timestamp"] = event.timestamp.isoformat()
        events_data.append(event_dict)
    
    return jsonify({"events": events_data, "count": len(events_data)}), 200

@app.route("/statistics", methods=["GET"])
def get_statistics_endpoint() -> Response:
    """Get state manager statistics."""
    stats = state_manager.get_statistics()
    return jsonify(stats), 200

if __name__ == "__main__":
    # Register with Tool Registry at startup
    try:
        from agents.common.tool_caller import register_agent_with_registry
        port = int(os.environ.get("PORT", 8080))
        base_url = os.getenv("STATE_MANAGER_URL", f"http://localhost:{port}")
        register_agent_with_registry("state_manager", base_url)
    except Exception as e:
        LOGGER.warning(f"Failed to register with Tool Registry: {e}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
