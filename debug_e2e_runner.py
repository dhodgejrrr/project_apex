#!/usr/bin/env python3
"""
Enhanced E2E test runner with detailed Gemini API debugging.

This script runs the E2E test with various debugging levels to help
identify and resolve Gemini API issues.
"""

import os
import sys
import subprocess
import time
from typing import Dict, Any

def run_e2e_with_debug_level(debug_level: str, description: str) -> Dict[str, Any]:
    """Run E2E test with specified debug level."""
    print(f"\n{'='*60}")
    print(f"Running E2E Test: {description}")
    print(f"Debug Level: {debug_level}")
    print(f"{'='*60}")
    
    # Set environment variables based on debug level
    env = os.environ.copy()
    
    if debug_level == "minimal":
        env["DEBUG_AI_RESPONSES"] = "false"
        env["LOG_PROMPT_CONTENT"] = "false"
    elif debug_level == "standard":
        env["DEBUG_AI_RESPONSES"] = "true"
        env["LOG_PROMPT_CONTENT"] = "false"
    elif debug_level == "full":
        env["DEBUG_AI_RESPONSES"] = "true"
        env["LOG_PROMPT_CONTENT"] = "true"
    
    start_time = time.time()
    
    try:
        # Run the E2E test
        result = subprocess.run(
            ["python", "run_local_e2e.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        print(f"\nTest Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Return Code: {result.returncode}")
        
        if not success:
            print(f"\nSTDERR:\n{result.stderr}")
            print(f"\nSTDOUT:\n{result.stdout}")
        
        return {
            "debug_level": debug_level,
            "success": success,
            "duration": duration,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 30 minutes")
        return {
            "debug_level": debug_level,
            "success": False,
            "duration": 1800,
            "return_code": -1,
            "error": "Timeout"
        }
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return {
            "debug_level": debug_level,
            "success": False,
            "duration": 0,
            "return_code": -1,
            "error": str(e)
        }

def analyze_gemini_errors(test_results: Dict[str, Any]) -> None:
    """Analyze test results for Gemini-specific errors."""
    print(f"\n{'='*60}")
    print("GEMINI API ERROR ANALYSIS")
    print(f"{'='*60}")
    
    stdout = test_results.get("stdout", "")
    stderr = test_results.get("stderr", "")
    combined_output = stdout + stderr
    
    # Look for specific Gemini error patterns
    patterns = {
        "MAX_TOKENS": "Finish Reason: 2",
        "Safety Filter": "blocked by safety filters",
        "JSON Parse Error": "JSONDecodeError",
        "Connection Error": "ConnectionError",
        "Rate Limit": "rate limit",
        "API Error": "GEMINI API ERROR",
        "Truncated Response": "truncated",
        "Empty Response": "Empty response"
    }
    
    found_issues = {}
    for issue_type, pattern in patterns.items():
        count = combined_output.lower().count(pattern.lower())
        if count > 0:
            found_issues[issue_type] = count
    
    if found_issues:
        print("Found Gemini API Issues:")
        for issue_type, count in found_issues.items():
            print(f"  • {issue_type}: {count} occurrences")
    else:
        print("✅ No specific Gemini API issues detected")
    
    # Extract detailed error logs if available
    lines = combined_output.split('\n')
    error_sections = []
    in_error_section = False
    current_section = []
    
    for line in lines:
        if "=== DETAILED GEMINI API ERROR ANALYSIS ===" in line:
            in_error_section = True
            current_section = [line]
        elif "=== END DETAILED ANALYSIS ===" in line:
            if in_error_section:
                current_section.append(line)
                error_sections.append('\n'.join(current_section))
                current_section = []
                in_error_section = False
        elif in_error_section:
            current_section.append(line)
    
    if error_sections:
        print(f"\n📊 Found {len(error_sections)} detailed error analysis sections")
        for i, section in enumerate(error_sections[:3], 1):  # Show first 3
            print(f"\n--- Error Analysis {i} ---")
            print(section[:500] + "..." if len(section) > 500 else section)

def main():
    """Run E2E tests with different debug levels."""
    print("🚀 ENHANCED GEMINI API DEBUGGING E2E TEST")
    print("This will run multiple E2E tests with different logging levels")
    print("to help identify and resolve Gemini API issues.\n")
    
    # Test scenarios
    scenarios = [
        ("minimal", "Minimal logging (production-like)"),
        ("standard", "Standard debug logging (detailed errors only)"),
        ("full", "Full debug logging (includes prompts)")
    ]
    
    results = []
    
    for debug_level, description in scenarios:
        result = run_e2e_with_debug_level(debug_level, description)
        results.append(result)
        
        # If we found issues, analyze them
        if not result["success"]:
            analyze_gemini_errors(result)
        
        # Add a pause between tests to avoid overwhelming the API
        if debug_level != scenarios[-1][0]:  # Not the last scenario
            print("\n⏸️  Pausing 30 seconds between tests...")
            time.sleep(30)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        duration = result.get("duration", 0)
        print(f"{result['debug_level']:10} | {status:10} | {duration:6.1f}s")
    
    successful_tests = [r for r in results if r["success"]]
    
    if successful_tests:
        print(f"\n✅ {len(successful_tests)}/{len(results)} tests passed")
        avg_duration = sum(r["duration"] for r in successful_tests) / len(successful_tests)
        print(f"📊 Average success duration: {avg_duration:.1f} seconds")
    else:
        print(f"\n❌ All {len(results)} tests failed")
        print("🔍 Check the detailed error analysis above for debugging information")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print("1. Review the detailed error logs above")
        print("2. Check if MAX_TOKENS errors can be resolved with higher limits")
        print("3. Verify API credentials and quota limits")
        print("4. Consider implementing the adaptive retry logic in production")
    
    if successful_tests:
        print("5. Consider using the minimal logging level for production")
        print("6. Enable standard debug logging for troubleshooting")

if __name__ == "__main__":
    main()
