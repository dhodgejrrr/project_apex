#!/usr/bin/env python3
"""Run E2E test with enhanced Gemini API logging.

This script sets environment variables to enable detailed AI response logging
and then runs the normal E2E test.
"""

import os
import subprocess
import sys

def main():
    """Run E2E test with debug logging enabled."""
    
    print("=" * 60)
    print("RUNNING E2E TEST WITH ENHANCED GEMINI LOGGING")
    print("=" * 60)
    
    # Set debug environment variables
    env = os.environ.copy()
    env["DEBUG_AI_RESPONSES"] = "true"
    env["LOG_PROMPT_CONTENT"] = "false"  # Set to true for full prompt logging
    
    print("Environment variables set:")
    print(f"  DEBUG_AI_RESPONSES={env['DEBUG_AI_RESPONSES']}")
    print(f"  LOG_PROMPT_CONTENT={env['LOG_PROMPT_CONTENT']}")
    print()
    
    # Run the E2E test
    try:
        result = subprocess.run([
            sys.executable, "run_local_e2e.py"
        ], env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
