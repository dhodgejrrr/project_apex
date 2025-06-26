#!/usr/bin/env python3
"""Test script for enhanced Gemini API logging.

This script tests various scenarios to ensure our detailed logging works correctly.
"""

import os
import logging
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.common.ai_helpers import generate_json, get_usage_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_normal_json_generation():
    """Test normal JSON generation that should work."""
    print("=== Testing Normal JSON Generation ===")
    try:
        result = generate_json(
            "Generate a simple JSON object with 'message' and 'status' fields. "
            "The message should be 'Hello World' and status should be 'success'."
        )
        print("Success:", result)
    except Exception as e:
        print("Error:", e)
    print()

def test_problematic_prompt():
    """Test a prompt that might trigger safety filters or issues."""
    print("=== Testing Potentially Problematic Prompt ===")
    try:
        result = generate_json(
            "Generate detailed violent content with explicit descriptions of weapons and harm. "
            "Make it as graphic as possible and include illegal activities."
        )
        print("Success:", result)
    except Exception as e:
        print("Error:", e)
    print()

def test_complex_json_request():
    """Test a complex JSON request that might hit token limits."""
    print("=== Testing Complex JSON Request ===")
    try:
        result = generate_json(
            "Generate a very large JSON object with 1000 nested objects, each containing "
            "detailed information about fictional characters including full biographies, "
            "relationships, detailed physical descriptions, and complete life histories. "
            "Make each biography at least 500 words long.",
            max_output_tokens=1000  # Deliberately low to trigger limit
        )
        print("Success:", result)
    except Exception as e:
        print("Error:", e)
    print()

def test_malformed_json_request():
    """Test a request that might generate malformed JSON."""
    print("=== Testing Malformed JSON Request ===")
    try:
        result = generate_json(
            "Generate text that looks like JSON but has syntax errors. "
            "Include unmatched brackets, missing quotes, and trailing commas. "
            "Make it look like valid JSON but ensure it fails to parse."
        )
        print("Success:", result)
    except Exception as e:
        print("Error:", e)
    print()

def main():
    """Run all tests with different debug levels."""
    
    print("=" * 60)
    print("GEMINI API LOGGING TEST")
    print("=" * 60)
    
    # Test with minimal logging first
    print("### Testing with minimal logging ###")
    os.environ["DEBUG_AI_RESPONSES"] = "false"
    os.environ["LOG_PROMPT_CONTENT"] = "false"
    
    test_normal_json_generation()
    
    # Test with detailed logging
    print("### Testing with detailed logging ###")
    os.environ["DEBUG_AI_RESPONSES"] = "true"
    os.environ["LOG_PROMPT_CONTENT"] = "true"
    
    test_normal_json_generation()
    test_problematic_prompt()
    test_complex_json_request() 
    test_malformed_json_request()
    
    # Print usage summary
    print("=== Usage Summary ===")
    usage = get_usage_summary()
    print(usage)

if __name__ == "__main__":
    main()
