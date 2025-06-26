#!/usr/bin/env python3
"""
Simple test script to verify ADK Orchestrator is working properly
"""
import base64
import json
import requests

def test_orchestrator():
    """Test the ADK Orchestrator with a minimal payload"""
    
    # Create a simple test message
    message_data = json.dumps({
        "run_id": "test-run-123",
        "csv_gcs_path": "gs://test-bucket/test.csv",
        "pit_gcs_path": "gs://test-bucket/test_pit.json",
        "fuel_gcs_path": "gs://test-bucket/test_fuel.json"
    }).encode("utf-8")
    
    push_message = {
        "message": {
            "data": base64.b64encode(message_data).decode("utf-8"),
            "messageId": "test-message-1",
        },
        "subscription": "projects/local-dev/subscriptions/local-trigger"
    }
    
    print("Testing ADK Orchestrator connectivity...")
    
    try:
        # Test GET request (should return 405)
        resp = requests.get("http://localhost:8087", timeout=10)
        print(f"GET request status: {resp.status_code} (expected 405)")
        
        # Test POST request
        print("Sending POST request to ADK Orchestrator...")
        resp = requests.post("http://localhost:8087", json=push_message, timeout=30)
        print(f"POST request status: {resp.status_code}")
        print(f"Response headers: {dict(resp.headers)}")
        print(f"Response body: {resp.text}")
        
        if resp.status_code == 200:
            print("✅ ADK Orchestrator is working correctly!")
        else:
            print("❌ ADK Orchestrator returned an error")
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_orchestrator()
