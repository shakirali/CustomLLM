#!/usr/bin/env python3
"""
Simple test runner that checks if pytest is available and runs tests.
If pytest is not available, provides installation instructions.
"""

import sys
import subprocess
import os

def check_pytest():
    """Check if pytest is available."""
    try:
        import pytest
        return True
    except ImportError:
        return False

def main():
    """Run tests if pytest is available."""
    if not check_pytest():
        print("=" * 70)
        print("pytest is not installed.")
        print("=" * 70)
        print("\nTo install pytest, run:")
        print("  pip install pytest")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        print("\nAfter installing, run:")
        print("  pytest tests/ -v")
        print("=" * 70)
        return 1
    
    # Run pytest
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    cmd = [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"]
    
    print("Running unit tests...")
    print("=" * 70)
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())

