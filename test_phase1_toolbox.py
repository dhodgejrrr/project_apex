#!/usr/bin/env python3
"""
Test script for Phase 1 toolbox implementation.

Tests the new toolbox endpoints for CoreAnalyzer and Visualizer.
"""
import json
import os
import sys
import requests
from typing import Dict, Any

# Set environment variables for local testing
os.environ["PUBSUB_EMULATOR_HOST"] = "localhost:8085"  # Indicate local mode
os.environ["CORE_ANALYZER_URL"] = "http://localhost:8080"
os.environ["VISUALIZER_URL"] = "http://localhost:8081"

def test_core_analyzer_capabilities():
    """Test CoreAnalyzer capability discovery."""
    print("🔍 Testing CoreAnalyzer capabilities...")
    
    try:
        response = requests.get("http://localhost:8080/tools/capabilities")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ CoreAnalyzer capabilities: {len(data['tools'])} tools available")
            for tool in data['tools']:
                print(f"   - {tool['name']}: {tool['description']}")
            return True
        else:
            print(f"❌ CoreAnalyzer capabilities failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ CoreAnalyzer capabilities error: {e}")
        return False

def test_visualizer_capabilities():
    """Test Visualizer capability discovery."""
    print("🔍 Testing Visualizer capabilities...")
    
    try:
        response = requests.get("http://localhost:8081/tools/capabilities")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Visualizer capabilities: {len(data['tools'])} tools available")
            for tool in data['tools']:
                print(f"   - {tool['name']}: {tool['description']}")
            return True
        else:
            print(f"❌ Visualizer capabilities failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Visualizer capabilities error: {e}")
        return False

def test_tool_caller():
    """Test the ToolCaller infrastructure."""
    print("🔍 Testing ToolCaller infrastructure...")
    
    try:
        # Import the tool caller
        sys.path.append('/Users/davidhodge/Documents/GitHub/project_apex')
        from agents.common.tool_caller import tool_caller
        
        # Test capability discovery
        capabilities = tool_caller.get_all_capabilities()
        print(f"✅ ToolCaller discovered {len(capabilities)} agents")
        
        for agent_name, caps in capabilities.items():
            if 'tools' in caps:
                print(f"   - {agent_name}: {len(caps['tools'])} tools")
            elif 'error' in caps:
                print(f"   - {agent_name}: ERROR - {caps['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ToolCaller test error: {e}")
        return False

def main():
    """Run all Phase 1 tests."""
    print("🚀 Phase 1 Toolbox Infrastructure Tests")
    print("=" * 50)
    
    results = []
    
    # Test individual services
    results.append(test_core_analyzer_capabilities())
    results.append(test_visualizer_capabilities())
    results.append(test_tool_caller())
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 Phase 1 toolbox infrastructure is ready!")
        print("\n📋 Next Steps:")
        print("   1. Deploy CoreAnalyzer with new toolbox endpoints")
        print("   2. Deploy Visualizer with individual plot endpoints")
        print("   3. Update other agents to use ToolCaller")
        print("   4. Begin Phase 2: Autonomous Agent Logic")
    else:
        print("⚠️  Some tests failed. Check logs and fix issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
