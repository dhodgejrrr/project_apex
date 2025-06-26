#!/usr/bin/env python3
"""
Test script for Phase 2 autonomous agent implementation.

Tests the new autonomous logic for InsightHunter and Publicist.
"""
import json
import os
import sys
import requests
from typing import Dict, Any

# Set environment variables for local testing
os.environ["PUBSUB_EMULATOR_HOST"] = "localhost:8085"
os.environ["USE_AI_ENHANCED"] = "true"
os.environ["CORE_ANALYZER_URL"] = "http://localhost:8080"
os.environ["VISUALIZER_URL"] = "http://localhost:8081"
os.environ["INSIGHT_HUNTER_URL"] = "http://localhost:8082"
os.environ["PUBLICIST_URL"] = "http://localhost:8083"

def test_autonomous_insight_hunter():
    """Test the autonomous InsightHunter workflow."""
    print("🧠 Testing Autonomous InsightHunter...")
    
    # Mock analysis data
    test_payload = {
        "analysis_path": "gs://test-bucket/test-run/analysis.json",
        "use_autonomous": True
    }
    
    try:
        response = requests.post("http://localhost:8082/", json=test_payload)
        if response.status_code == 200:
            data = response.json()
            method = data.get("generation_method")
            print(f"✅ InsightHunter autonomous mode: {method}")
            
            if method == "autonomous":
                print("   - Successfully used autonomous investigation workflow")
                return True
            else:
                print(f"   - Warning: Expected autonomous mode, got {method}")
                return False
        else:
            print(f"❌ InsightHunter test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ InsightHunter test error: {e}")
        return False

def test_autonomous_publicist():
    """Test the autonomous Publicist workflow with self-correction."""
    print("📱 Testing Autonomous Publicist...")
    
    # Mock briefing data for new autonomous workflow
    test_payload = {
        "briefing_path": "gs://test-bucket/test-run/briefing.json",
        "analysis_path": "gs://test-bucket/test-run/analysis.json",
        "use_autonomous": True
    }
    
    try:
        response = requests.post("http://localhost:8083/", json=test_payload)
        if response.status_code == 200:
            data = response.json()
            method = data.get("generation_method")
            attempts = data.get("attempts", 1)
            posts_count = data.get("posts_count", 0)
            
            print(f"✅ Publicist autonomous mode: {method}")
            print(f"   - Generated {posts_count} posts in {attempts} attempts")
            
            if method in ["autonomous_with_correction", "autonomous_with_correction_incomplete"]:
                print("   - Successfully used autonomous generation with self-correction")
                return True
            else:
                print(f"   - Warning: Expected autonomous mode, got {method}")
                return False
        else:
            print(f"❌ Publicist test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Publicist test error: {e}")
        return False

def test_tool_integration():
    """Test that agents can call each other's tools."""
    print("🔧 Testing Inter-Agent Tool Integration...")
    
    try:
        # Import and test the tool caller
        sys.path.append('/Users/davidhodge/Documents/GitHub/project_apex')
        from agents.common.tool_caller import tool_caller
        
        # Test CoreAnalyzer tool calling
        try:
            caps = tool_caller.discover_capabilities("core_analyzer")
            if caps.get("tools"):
                print(f"✅ CoreAnalyzer tools discovered: {len(caps['tools'])} available")
            else:
                print("⚠️  CoreAnalyzer tools not accessible")
                return False
        except Exception as e:
            print(f"❌ CoreAnalyzer tool discovery failed: {e}")
            return False
        
        # Test Visualizer tool calling
        try:
            caps = tool_caller.discover_capabilities("visualizer")
            if caps.get("tools"):
                print(f"✅ Visualizer tools discovered: {len(caps['tools'])} available")
            else:
                print("⚠️  Visualizer tools not accessible")
                return False
        except Exception as e:
            print(f"❌ Visualizer tool discovery failed: {e}")
            return False
        
        print("✅ Inter-agent tool integration working")
        return True
        
    except Exception as e:
        print(f"❌ Tool integration test error: {e}")
        return False

def test_backward_compatibility():
    """Test that legacy workflows still work."""
    print("🔄 Testing Backward Compatibility...")
    
    results = []
    
    # Test legacy InsightHunter
    try:
        legacy_payload = {
            "analysis_path": "gs://test-bucket/test-run/analysis.json",
            "use_autonomous": False
        }
        response = requests.post("http://localhost:8082/", json=legacy_payload)
        if response.status_code == 200:
            data = response.json()
            if data.get("generation_method") in ["traditional", None]:
                print("✅ InsightHunter legacy mode working")
                results.append(True)
            else:
                print("⚠️  InsightHunter legacy mode unexpected behavior")
                results.append(False)
        else:
            print("❌ InsightHunter legacy mode failed")
            results.append(False)
    except Exception as e:
        print(f"❌ InsightHunter legacy test error: {e}")
        results.append(False)
    
    # Test legacy Publicist
    try:
        legacy_payload = {
            "analysis_path": "gs://test-bucket/test-run/analysis.json",
            "insights_path": "gs://test-bucket/test-run/insights.json"
        }
        response = requests.post("http://localhost:8083/", json=legacy_payload)
        if response.status_code == 200:
            data = response.json()
            if data.get("generation_method") == "legacy":
                print("✅ Publicist legacy mode working")
                results.append(True)
            else:
                print("⚠️  Publicist legacy mode unexpected behavior")
                results.append(False)
        else:
            print("❌ Publicist legacy mode failed")
            results.append(False)
    except Exception as e:
        print(f"❌ Publicist legacy test error: {e}")
        results.append(False)
    
    return all(results)

def main():
    """Run all Phase 2 tests."""
    print("🚀 Phase 2 Autonomous Agent Logic Tests")
    print("=" * 50)
    
    results = []
    
    # Test autonomous features
    results.append(test_autonomous_insight_hunter())
    results.append(test_autonomous_publicist())
    results.append(test_tool_integration())
    results.append(test_backward_compatibility())
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 Phase 2 autonomous agent logic is ready!")
        print("\n📋 Key Features Implemented:")
        print("   ✅ Autonomous InsightHunter with LLM-driven investigation")
        print("   ✅ Autonomous Publicist with self-correction loop")
        print("   ✅ Inter-agent tool calling infrastructure")
        print("   ✅ Backward compatibility maintained")
        print("\n🎯 Next Steps:")
        print("   1. Deploy updated agents with autonomous logic")
        print("   2. Begin Phase 3: Advanced Inter-Agent Communication")
        print("   3. Implement tool discovery and state management")
    else:
        print("⚠️  Some tests failed. Check logs and fix issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
