#!/usr/bin/env python3
"""Test script for improved Gemini API error handling with adaptive retry logic."""

import os
import logging
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.common.ai_helpers import generate_json, generate_json_adaptive, get_usage_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_adaptive_token_handling():
    """Test adaptive token limit handling."""
    print("=== Testing Adaptive Token Limit Handling ===")
    
    # Enable debug logging
    os.environ["DEBUG_AI_RESPONSES"] = "true"
    os.environ["LOG_PROMPT_CONTENT"] = "false"
    
    try:
        result = generate_json_adaptive(
            "Generate a very large JSON object with 100 nested objects, each containing "
            "detailed information about fictional racing drivers including full biographies, "
            "career statistics, and detailed race histories. Make each biography comprehensive.",
            max_output_tokens=500,  # Deliberately low to trigger adaptive retry
            adaptive_retry=True
        )
        print("Success with adaptive retry:", type(result), "keys:" if isinstance(result, dict) else "")
        if isinstance(result, dict):
            print("  Keys:", list(result.keys())[:5], "..." if len(result) > 5 else "")
    except Exception as e:
        print("Error even with adaptive retry:", e)
    print()

def test_max_tokens_scenario():
    """Test the specific MAX_TOKENS scenario we saw in production."""
    print("=== Testing MAX_TOKENS Scenario ===")
    
    try:
        # This should trigger the MAX_TOKENS handling
        result = generate_json(
            "Generate exactly 50 fictional racing driver profiles with detailed biographies of at least 200 words each, "
            "complete career statistics, personal information, and racing history.",
            max_output_tokens=800  # Low enough to trigger MAX_TOKENS
        )
        print("Success:", result)
    except Exception as e:
        print("Error:", e)
    print()

def test_normal_operation():
    """Test normal operation to ensure we didn't break anything."""
    print("=== Testing Normal Operation ===")
    
    try:
        result = generate_json(
            "Generate a simple JSON object with driver information: "
            "{'name': 'Example Driver', 'team': 'Example Team', 'points': 100}"
        )
        print("Success:", result)
    except Exception as e:
        print("Error:", e)
    print()

def main():
    """Run improved tests."""
    
    print("=" * 70)
    print("IMPROVED GEMINI API ERROR HANDLING TEST")
    print("=" * 70)
    
    test_normal_operation()
    test_max_tokens_scenario()
    test_adaptive_token_handling()
    
    # Print usage summary
    print("=== Usage Summary ===")
    usage = get_usage_summary()
    print(f"Total calls: {len(usage.get('_calls', []))}")
    print(f"Total cost: ${usage.get('_overall', {}).get('estimated_cost_usd', 0):.4f}")
    print(f"Total tokens: {usage.get('_overall', {}).get('total_tokens', 0)}")

if __name__ == "__main__":
    main()
