#!/usr/bin/env python3
"""
Comprehensive Test Runner for Eye Movement Analysis Application
Author: Combined from existing test runners
Date: May 2025

This script runs all tests for the ASD analysis application, providing
a unified way to execute the complete test suite or specific test modules.

Usage:
    python tests/run_all.py                   # Run all tests
    python tests/run_all.py -p gui            # Run tests matching pattern "gui"
    python tests/run_all.py -f tests/test_gui.py  # Run a specific test file
    python tests/run_all.py -l                # List all available test files
    python tests/run_all.py -v 2              # Verbosity level 2 (more output)
    python tests/run_all.py --failfast        # Stop on first failure
"""

import os
import sys
import unittest
import argparse
import time
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set up colored output if available
try:
    from termcolor import colored
except ImportError:
    # Mock the colored function if termcolor is not available
    def colored(text, *args, **kwargs):
        return text

# Set environment variables for headless testing
def setup_headless_environment():
    """Set up environment variables for headless testing."""
    # Set offscreen matplotlib backend
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for all tests
    
    # Set QT_QPA_PLATFORM to 'offscreen' for Qt
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # For CI environments, enable Qt debug mode
    if 'CI' in os.environ or 'GITHUB_ACTIONS' in os.environ:
        os.environ['QT_DEBUG_PLUGINS'] = '1'
        print(colored("CI environment detected, using headless configuration", "yellow"))


def discover_and_run_tests(pattern=None, verbosity=2, failfast=False):
    """Discover and run tests matching the given pattern."""
    start_time = time.time()
    
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover tests
    if pattern:
        print(colored(f"Running tests matching pattern: {pattern}", "cyan"))
        suite = loader.discover(test_dir, pattern=f"*{pattern}*.py")
    else:
        print(colored("Running all tests", "cyan"))
        suite = loader.discover(test_dir)
    
    # Count tests
    test_count = suite.countTestCases()
    print(colored(f"Discovered {test_count} tests", "cyan"))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print(colored(f"TEST SUMMARY", "yellow", attrs=["bold"]))
    print(colored(f"Ran {test_count} tests in {execution_time:.2f} seconds", "yellow"))
    print(colored(f"Successes: {test_count - len(result.failures) - len(result.errors)}", "green"))
    
    if result.failures:
        print(colored(f"Failures: {len(result.failures)}", "red"))
    
    if result.errors:
        print(colored(f"Errors: {len(result.errors)}", "red"))
    
    if result.skipped:
        print(colored(f"Skipped: {len(result.skipped)}", "yellow"))
    
    print("="*70)
    
    # Return system exit code based on test results
    return 0 if result.wasSuccessful() else 1


def run_specific_test_file(file_path, verbosity=2, failfast=False):
    """Run tests from a specific file."""
    if not os.path.exists(file_path):
        print(colored(f"Error: Test file not found: {file_path}", "red"))
        return 1
    
    # Get the module name from the file path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_path = os.path.relpath(file_path, project_root)
    module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
    
    print(colored(f"Running tests from: {module_name}", "cyan"))
    
    # Import the module
    try:
        __import__(module_name)
        module = sys.modules[module_name]
    except ImportError as e:
        print(colored(f"Error importing module {module_name}: {e}", "red"))
        return 1
    
    # Create test loader and load tests from the module
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)
    
    # Count tests
    test_count = suite.countTestCases()
    print(colored(f"Found {test_count} tests in module", "cyan"))
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print(colored(f"TEST SUMMARY FOR {os.path.basename(file_path)}", "yellow", attrs=["bold"]))
    print(colored(f"Ran {test_count} tests in {execution_time:.2f} seconds", "yellow"))
    print(colored(f"Successes: {test_count - len(result.failures) - len(result.errors)}", "green"))
    
    if result.failures:
        print(colored(f"Failures: {len(result.failures)}", "red"))
    
    if result.errors:
        print(colored(f"Errors: {len(result.errors)}", "red"))
    
    if result.skipped:
        print(colored(f"Skipped: {len(result.skipped)}", "yellow"))
    
    print("="*70)
    
    # Return system exit code based on test results
    return 0 if result.wasSuccessful() else 1


def run_by_module_names(verbosity=2, failfast=False):
    """Run tests by discovering and importing module names."""
    # Get the directory containing this script
    tests_dir = Path(__file__).parent
    
    # Find all Python files that start with "test_"
    test_files = tests_dir.glob("test_*.py")
    
    # Convert to module names (e.g., tests.test_parser)
    test_modules = [f"tests.{file.stem}" for file in test_files]
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test modules to the suite
    for module_name in test_modules:
        try:
            # Load the module's tests
            tests = loader.loadTestsFromName(module_name)
            suite.addTest(tests)
            print(colored(f"Added tests from {module_name}", "cyan"))
        except Exception as e:
            print(colored(f"Error loading tests from {module_name}: {e}", "red"))
    
    # Count tests
    test_count = suite.countTestCases()
    print(colored(f"Total: {test_count} tests to run", "cyan"))
    
    # Run the tests
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print(colored(f"TEST SUMMARY", "yellow", attrs=["bold"]))
    print(colored(f"Ran {test_count} tests in {execution_time:.2f} seconds", "yellow"))
    print(colored(f"Successes: {test_count - len(result.failures) - len(result.errors)}", "green"))
    
    if result.failures:
        print(colored(f"Failures: {len(result.failures)}", "red"))
    
    if result.errors:
        print(colored(f"Errors: {len(result.errors)}", "red"))
    
    if result.skipped:
        print(colored(f"Skipped: {len(result.skipped)}", "yellow"))
    
    print("="*70)
    
    # Return system exit code based on test results
    return 0 if result.wasSuccessful() else 1


def list_all_test_files():
    """List all available test files."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = []
    
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    
    # Print list of test files
    print(colored("Available test files:", "cyan"))
    for i, file in enumerate(test_files, 1):
        rel_path = os.path.relpath(file, test_dir)
        print(f"{i}. {rel_path}")
    
    return test_files


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run tests for the ASD analysis application'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        help='Run tests matching the given pattern (e.g., "gui" for all GUI tests)'
    )
    
    parser.add_argument(
        '-f', '--file',
        help='Run tests from a specific file'
    )
    
    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List all available test files'
    )
    
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Test output verbosity (0-2, default: 2)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (equivalent to --verbosity=0)'
    )
    
    parser.add_argument(
        '--failfast',
        action='store_true',
        help='Stop on first failing test'
    )
    
    parser.add_argument(
        '--modules',
        action='store_true',
        help='Run tests by module names rather than discovery'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Force headless mode for GUI and visualization tests'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Make sure we're running from the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Add the project root to the Python path
    sys.path.insert(0, str(project_root))
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Adjust verbosity if quiet flag is set
    if args.quiet:
        args.verbosity = 0
    
    # Always set up headless environment for CI or if requested
    if 'CI' in os.environ or 'GITHUB_ACTIONS' in os.environ or args.headless:
        setup_headless_environment()
        print(colored("Using headless mode for GUI and visualization tests", "yellow"))
    
    # Handle list option
    if args.list:
        list_all_test_files()
        sys.exit(0)
    
    # Print a header
    print("=" * 80)
    print(colored(f"Running tests for ASD Analysis project", "cyan"))
    print("=" * 80)
    
    # Run tests based on options
    if args.file:
        exit_code = run_specific_test_file(args.file, args.verbosity, args.failfast)
    elif args.modules:
        exit_code = run_by_module_names(args.verbosity, args.failfast)
    else:
        exit_code = discover_and_run_tests(args.pattern, args.verbosity, args.failfast)
    
    # Exit with appropriate code
    sys.exit(exit_code)