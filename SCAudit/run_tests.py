#!/usr/bin/env python3
"""
Run all SCAudit tests.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_tests():
    """Run all tests and return success status."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    test_dir = Path(__file__).parent / 'tests'
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)