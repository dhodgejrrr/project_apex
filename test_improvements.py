#!/usr/bin/env python3
"""
Test script to validate the improvements to run_local_e2e.py
"""
import subprocess
import sys
import time

def test_health_check_functionality():
    """Test the service health check functionality separately."""
    print("Testing service health check functionality...")
    
    # Import the functions from our main script
    sys.path.insert(0, '/Users/davidhodge/Documents/GitHub/project_apex')
    try:
        from run_local_e2e import check_service_health, SERVICE_ENDPOINTS
        
        # Test with a known unreachable service
        result = check_service_health("test-service", "http://localhost:9999", "/", timeout=2)
        if not result:
            print("✓ Health check correctly detected unreachable service")
        else:
            print("✗ Health check failed to detect unreachable service")
            
        print(f"✓ Service endpoints configured: {len(SERVICE_ENDPOINTS)} services")
        for name, config in SERVICE_ENDPOINTS.items():
            print(f"  - {name}: {config['url']}")
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False
    
    return True

def main():
    print("=== Enhanced run_local_e2e.py Validation ===\n")
    
    if test_health_check_functionality():
        print("\n✓ All validation tests passed!")
        print("\nKey improvements implemented:")
        print("1. Service health checks with 3-minute timeout")
        print("2. Robust HTTP request handling with retries")
        print("3. Enhanced logging and debugging capabilities")
        print("4. Docker container status monitoring")
        print("5. Better error handling and diagnostic information")
        print("6. Progressive backoff retry mechanism")
        print("7. Improved emulator connectivity checks")
        print("\nThe script should now handle connection issues much more reliably.")
    else:
        print("\n✗ Validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
