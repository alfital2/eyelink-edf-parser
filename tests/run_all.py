#!/usr/bin/env python3
"""
Test Runner Script for ASD Analysis Project
Author: Tal Alfi
Date: May 2025

This script runs all the tests in the tests directory and provides a summary of the results.
Run this script from the project root directory to execute all tests.

Usage:
    python3 tests/run_all.py

Options:
    --verbose/-v     Show more detailed test output
    --quiet/-q       Show less output (only failures)
    --failfast/-f    Stop on first failure
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Set offscreen matplotlib backend if we're in CI/GitHub Actions environment
# This needs to be done before importing matplotlib or any module that uses it
if 'CI' in os.environ or 'GITHUB_ACTIONS' in os.environ:
    print("CI environment detected, using 'Agg' backend for matplotlib...")
    import matplotlib
    matplotlib.use('Agg')
    
    # Set QT_QPA_PLATFORM to 'offscreen' for Qt
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    print("Set QT_QPA_PLATFORM to 'offscreen'")
    
    # This can help with Qt issues in headless environments
    os.environ['QT_DEBUG_PLUGINS'] = '1'


def get_test_modules():
    """Find all test modules in the tests directory."""
    # Get the directory containing this script
    tests_dir = Path(__file__).parent
    
    # Find all Python files that start with "test_"
    test_files = tests_dir.glob("test_*.py")
    
    # Convert to module names (e.g., tests.test_parser)
    return [f"tests.{file.stem}" for file in test_files]


def run_all_tests(verbosity=1, failfast=False):
    """Run all tests and return a TestResult object."""
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test modules to the suite
    test_modules = get_test_modules()
    for module_name in test_modules:
        try:
            # Load the module's tests
            tests = loader.loadTestsFromName(module_name)
            suite.addTest(tests)
            print(f"Added tests from {module_name}")
        except Exception as e:
            print(f"Error loading tests from {module_name}: {e}")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    return runner.run(suite)


def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run all tests for ASD Analysis project.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show more detailed test output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show less output (only failures)")
    parser.add_argument("-f", "--failfast", action="store_true", help="Stop on first failure")
    return parser.parse_args()


if __name__ == "__main__":
    # Make sure we're running from the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Add the project root to the Python path
    sys.path.insert(0, str(project_root))
    
    # Parse command line arguments
    args = parse_args()
    
    # Determine verbosity level
    verbosity = 2 if args.verbose else 1
    if args.quiet:
        verbosity = 0
    
    # Print a header
    print("=" * 80)
    print(f"Running all tests for ASD Analysis project")
    print("=" * 80)
    
    # Record the start time
    start_time = time.time()
    
    # Run the tests
    result = run_all_tests(verbosity=verbosity, failfast=args.failfast)
    
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    
    # Print a summary
    print("\n" + "=" * 80)
    print(f"Test Summary:")
    print(f"  Ran {result.testsRun} tests in {elapsed_time:.3f} seconds")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 80)
    
    # Return the appropriate exit code
    sys.exit(len(result.failures) + len(result.errors))