#!/usr/bin/env python3
"""
Script to run tests for the EyeLinkASCParser.

This script executes the test suite and provides a summary of the results.
It assumes the sample ASC file (sample_test.asc) is already in the tests directory.
"""

import os
import unittest
import sys
import time
import importlib
from pathlib import Path


def run_tests():
    """Run the test suite and display results."""
    start_time = time.time()

    # Setup test environment
    test_file = check_test_environment()

    # Discover and run tests
    print("\n")
    print("=" * 70)
    print(f"RUNNING TESTS FOR EYELINK ASC PARSER")
    print("=" * 70)
    print(f"Sample ASC file: {Path(os.path.join(os.path.dirname(__file__), 'asc_files/sample_test.asc')).resolve()}")
    print(f"Test file: {Path(test_file).resolve()}")
    print("-" * 70)

    # Run tests with detailed output
    loader = unittest.TestLoader()
    tests = loader.discover(os.path.dirname(test_file), pattern=os.path.basename(test_file))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)

    # Calculate and display test duration
    duration = time.time() - start_time

    # Display summary
    print("\n")
    print("=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Time elapsed: {duration:.2f} seconds")
    print("-" * 70)

    # Print failures and errors in detail if any
    if result.failures or result.errors:
        print("\nDetails of failed tests:")
        for i, (test, traceback) in enumerate(result.failures + result.errors):
            print(f"\n--- Failure/Error {i + 1} ---")
            print(f"Test: {test}")
            print(f"Details:\n{traceback}")
            print("-" * 50)

    # Return appropriate exit code
    if result.wasSuccessful():
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


def check_test_environment():
    """Check if the test environment is properly set up."""
    # Verify the current directory is the tests directory
    tests_dir = os.path.dirname(__file__)

    # Check if the sample ASC file exists
    sample_asc_path = os.path.join(tests_dir, "asc_files/sample_test.asc")
    if not os.path.exists(sample_asc_path):
        print(f"ERROR: Sample ASC file not found at {sample_asc_path}")
        print("Please ensure the sample_test.asc file is in the tests directory.")
        sys.exit(1)

    # Check if the test file exists
    test_file = os.path.join(tests_dir, "test_parser.py")
    if not os.path.exists(test_file):
        print(f"Warning: Test file not found at expected location: {test_file}")
        # Look for alternate test file names
        potential_test_files = [f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py")]
        if potential_test_files:
            test_file = os.path.join(tests_dir, potential_test_files[0])
            print(f"Using alternate test file: {test_file}")
        else:
            print("No test files found. Make sure you have a test file in the tests directory.")
            sys.exit(1)

    # Check if the parser module is accessible
    # Add parent directory to path to import parser
    parent_dir = str(Path(tests_dir).parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Try to import the parser
    try:
        importlib.import_module("parser")
    except ImportError as e:
        print(f"ERROR: Unable to import parser module: {e}")
        print("Make sure parser.py is in the parent directory.")
        sys.exit(1)

    return test_file


def count_expected_events():
    """Count the expected events in the ASC file to help with testing."""
    tests_dir = os.path.dirname(__file__)
    sample_asc_path = os.path.join(tests_dir, "asc_files/sample_test.asc")

    event_counts = {
        'SFIX L': 0, 'EFIX L': 0,
        'SFIX R': 0, 'EFIX R': 0,
        'SSACC L': 0, 'ESACC L': 0,
        'SSACC R': 0, 'ESACC R': 0,
        'SBLINK L': 0, 'EBLINK L': 0,
        'SBLINK R': 0, 'EBLINK R': 0,
        'MSG': 0,
        'Play_Movie_Start FRAME': 0,
        'Samples': 0
    }

    try:
        with open(sample_asc_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Count event markers
                for event_type in event_counts.keys():
                    if event_type in line:
                        event_counts[event_type] += 1

                # Count sample lines (lines that start with a timestamp)
                if line and line[0].isdigit() and not any(
                        event in line for event in ['SFIX', 'EFIX', 'SSACC', 'ESACC', 'SBLINK', 'EBLINK']):
                    event_counts['Samples'] += 1

        print("\nExpected counts from the ASC file:")
        print("-" * 40)
        print(f"Samples: {event_counts['Samples']}")
        print(f"Left fixations: {min(event_counts['SFIX L'], event_counts['EFIX L'])}")
        print(f"Right fixations: {min(event_counts['SFIX R'], event_counts['EFIX R'])}")
        print(f"Left saccades: {min(event_counts['SSACC L'], event_counts['ESACC L'])}")
        print(f"Right saccades: {min(event_counts['SSACC R'], event_counts['ESACC R'])}")
        print(f"Left blinks: {min(event_counts['SBLINK L'], event_counts['EBLINK L'])}")
        print(f"Right blinks: {min(event_counts['SBLINK R'], event_counts['EBLINK R'])}")
        print(f"Messages: {event_counts['MSG']}")
        print(f"Frame markers: {event_counts['Play_Movie_Start FRAME']}")
        print("-" * 40)

        return event_counts
    except FileNotFoundError:
        print(f"ERROR: Could not find ASC file at {sample_asc_path}")
        return None
    except Exception as e:
        print(f"ERROR counting events: {str(e)}")
        return None


if __name__ == "__main__":
    # Print expected event counts first to help with test validation
    print("\nAnalyzing sample file for expected events...")
    count_expected_events()

    # Run the tests
    sys.exit(run_tests())
