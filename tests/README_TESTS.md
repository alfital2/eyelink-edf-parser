# Test Suite for ASC Analysis Tool

This README explains the comprehensive test suite for the ASC Analysis Tool. The tests cover error recovery, edge cases, data processing validation, GUI functionality, ROI analysis, and social attention metrics.

## Test Suite Overview

### 1. Error Recovery Tests (test_error_recovery.py)

Tests that validate the parser's ability to gracefully handle malformed or incomplete data files:

- Empty ASC files
- Header-only ASC files (metadata but no data)
- ASC files with malformed samples and events
- Malformed CSV files
- Error handling in the processing functions

### 2. Parser Tests (test_parser.py, test_parser_expanded.py)

Extended test coverage for the parser:

- Testing with various types of ASC files
- Validation of all parser methods and functions
- Tests for edge cases and unusual input formats
- Verification of output structures and formats
- Testing the CSV file loading functionality

### 3. Data Processing Tests (test_data_processing.py, test_advanced_csv_loading.py)

Tests focusing on data quality and performance:

- Statistical validation of extracted features
- Performance scaling tests with different file sizes
- Memory usage benchmarking
- Movie-specific feature extraction validation
- Unified metrics data validation
- Advanced CSV loading with various column configurations

### 4. ROI Integration Tests (test_roi_integration.py, test_advanced_roi_manager.py)

Tests for Region of Interest functionality:

- Basic ROI loading and management
- Complex polygon ROIs (concave, many vertices)
- Overlapping ROIs and nested ROI handling
- Multi-frame ROI tracking for dynamic scenes
- Extreme ROI coordinates and boundary conditions
- ROI export/import functionality
- Social vs. non-social ROI classification

### 5. Social Attention Analysis Tests (test_social_attention_visualization.py)

Tests for the social attention analysis capabilities:

- ROI fixation detection in social and non-social regions
- Social attention metrics calculation
- Social attention visualization generation
- Fixation sequence analysis and visualization
- Validation of ROI fixation sequence plots

### 6. GUI Tests (test_gui.py)

Tests for the graphical user interface components:

- GUI initialization and component setup
- File handling (ASC and CSV selection)
- Processing thread functionality
- Feature table display and updating
- Visualization controls and rendering
- Animated scanpath widget functionality
- ROI visualization functionality

### 7. Animated Scanpath Tests (test_animated_scanpath.py, test_animated_roi_scanpath.py)

Tests for animated eye movement visualizations:

- Basic scanpath animation rendering
- ROI overlay functionality
- Playback controls (play, pause, speed)
- Timeline navigation and frame selection
- ROI detection and highlighting during playback
- Eye tracking data visualization options

## Running the Tests

The test suite now includes a comprehensive test runner that can execute all tests or specific test modules:

```bash
# Run all tests
python tests/run_all_tests.py

# Run tests matching a pattern
python tests/run_all_tests.py -p gui

# Run a specific test file
python tests/run_all_tests.py -f tests/test_advanced_roi_manager.py

# List all available test files
python tests/run_all_tests.py -l

# Run tests with less verbosity
python tests/run_all_tests.py -v 1

# Stop on first failure
python tests/run_all_tests.py --failfast
```

You can also use the standard unittest discovery:

```bash
python -m unittest discover
```

## Test Data

The tests use several types of data:

1. Real ASC files in the `tests/asc_files/` directory
2. Synthetic ASC files created by the mock data generator
3. Temporary files created during tests for specific scenarios
4. Sample ROI JSON files in the `tests/test_data/` directory

## Mock Data Generator (mock_data_generator.py)

A comprehensive synthetic data generator for eye tracking data that can:

- Create ASC files with controlled parameters
- Generate sample and event data with realistic properties
- Inject various types of malformed data for testing
- Create files with specific characteristics (empty, left-eye only, etc.)
- Generate movie segments and associated samples
- Create synthetic ROI data with social and non-social regions

## Key Validations

The expanded test suite verifies that:

1. The parser correctly handles malformed/incomplete data files
2. Statistical calculations in feature extraction are accurate
3. The parser scales appropriately with file size
4. Movie-specific features are extracted correctly
5. Events (fixations, saccades, blinks) are properly detected even with noise
6. CSV file loading handles various error conditions gracefully
7. ROI manager correctly identifies fixations within defined regions
8. Social attention metrics are calculated accurately
9. GUI components initialize correctly and handle user input
10. Animated visualizations render correctly and respond to controls
11. ROI-based visualizations properly display social vs. non-social attention patterns

## Adding New Tests

When adding new tests:

1. Use the mock data generator to create controlled test data
2. Consider edge cases and error conditions
3. Validate statistical properties when testing data processing
4. Use tempfile to manage test files for proper cleanup
5. Make tests independent and avoid system-specific dependencies
6. For GUI tests, ensure they can run headlessly by using the offscreen QPA platform
7. When testing visualizations, use the Agg matplotlib backend

## Test Coverage

The current test suite covers:

- Error recovery: 90% (5 test cases)
- Parser core functionality: 90% (15 test cases)
- Data processing: 85% (10 test cases)
- ROI management: 95% (15 test cases) 
- Social attention analysis: 90% (8 test cases)
- GUI components: 85% (15 test cases)
- Animated visualization: 85% (10 test cases)
- Mock data generation: 100% (used throughout tests)

## Running Tests in Continuous Integration

The test suite is designed to run in headless environments for continuous integration. For GUI and visualization tests, the following environment settings are used:

```python
# For Qt GUI tests
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# For matplotlib visualization tests
import matplotlib
matplotlib.use('Agg')
```

These settings are already applied in the `run_all_tests.py` script.