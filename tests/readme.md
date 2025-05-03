# ASD Analysis Testing Framework

This directory contains the test suite and verification tools for the ASD Analysis project, which includes eye tracking analysis tools for autism research.

## Running Tests

### Run All Tests

The simplest way to run all tests is to use the provided `run_all.py` script:

```bash
# Run all tests with normal output
python3 tests/run_all.py

# Run with verbose output
python3 tests/run_all.py -v

# Run quietly (only show failures)
python3 tests/run_all.py -q

# Stop on first failure
python3 tests/run_all.py -f
```

### Run Individual Test Modules

You can also run individual test modules using unittest:

```bash
# Run a specific test file
python3 -m unittest tests.test_parser
python3 -m unittest tests.test_roi_manager

# Run a specific test case
python3 -m unittest tests.test_parser.TestEyeLinkASCParser
```

## Directory Structure

```
tests/
├── README.md                    # This file
├── run_all.py                   # Script to run all tests
├── conftest.py                  # Shared test fixtures and utilities
├── test_parser.py               # Tests for the EyeLink ASC parser
├── test_roi_manager.py          # Tests for the ROI manager
├── test_roi_integration.py      # Tests for ROI integration
├── test_animated_scanpath.py    # Tests for animated scanpath visualization
├── test_animated_roi_scanpath.py # Tests for animated ROI scanpath
├── test_eyelink_visualizer.py   # Tests for the EyeLink visualizer
├── test_movie_visualizer_integration.py # Tests for movie visualizer
├── asc_files/                   # Sample ASC files for testing
│   ├── sample_test.asc
│   └── smiley_test.asc
├── test_data/                   # Test data files
│   ├── test_roi.json
│   └── movie1/, movie2/        # Test movie folders
└── plots/                       # Output directory for visualization tests
    ├── gaze_positions.png
    └── roi_test_visualization.png
```

## Test Modules

The tests are organized by module:

1. **Parser Tests** (`test_parser.py`) - Tests for the EyeLink ASC parser
   - File reading and metadata extraction
   - Event detection (fixations, saccades, blinks)
   - Message and frame marker parsing
   - Feature extraction for machine learning

2. **ROI Tests** (`test_roi_manager.py`, `test_roi_integration.py`) - Tests for ROI functionality
   - ROI data loading from JSON
   - Point-in-polygon detection
   - ROI visualization
   - ROI integration with eye tracking data

3. **Visualization Tests** 
   - `test_animated_scanpath.py` - Tests for animated scan path visualization
   - `test_animated_roi_scanpath.py` - Tests for animated ROI scan path
   - `test_eyelink_visualizer.py` - Tests for the EyeLink visualizer
   - `test_movie_visualizer_integration.py` - Tests for movie visualizer integration

## Helper Files

- `conftest.py`: Contains shared test fixtures and utilities for all tests
  - Sample data generators
  - Test directory creation
  - Mock classes for external dependencies

- `run_all.py`: Script to run all tests at once with options for verbosity

## Test Data

Test data is stored in the following locations:

- `test_data/`: Contains test data files like ROI definitions
- `asc_files/`: Contains sample ASC files for testing the parser
- `plots/`: Output directory for visualization tests

## Requirements for Running Tests

To run all tests, you need the following dependencies:

- Python 3.7 or higher
- pandas
- numpy
- matplotlib (some tests will mock this dependency)
- PyQt5 (for GUI tests)

## Adding New Tests

To add a new test:

1. Create a new test file named `test_<module_name>.py` if needed
2. Import the module you want to test
3. Create a test class that inherits from `unittest.TestCase`
4. Write test methods that begin with `test_`
5. Use the fixtures from `conftest.py` if needed
6. Run the test to make sure it passes

Example:

```python
import unittest
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from your_module import YourClass

class TestYourClass(unittest.TestCase):
    def test_your_method(self):
        # Arrange
        instance = YourClass()
        
        # Act
        result = instance.your_method()
        
        # Assert
        self.assertEqual(result, expected_value)
```

## Troubleshooting

If tests are failing:

1. Run a specific test with verbose output:
   ```bash
   python3 -m unittest tests.test_parser.TestEyeLinkASCParser.test_specific_method -v
   ```

2. Check for dependency issues, especially with libraries like matplotlib or seaborn

3. If sample counts don't match expected counts, check how the parser is detecting events in the ASC file

4. For matplotlib/seaborn dependencies, the tests use mocking to avoid direct dependencies in test execution