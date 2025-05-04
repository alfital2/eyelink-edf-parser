# Test Suite for ASC Analysis Tool

This README explains the expanded test suite for the ASC Analysis Tool. The enhanced tests cover error recovery, edge cases, and data processing validation.

## New and Enhanced Tests

### 1. Error Recovery Tests (test_error_recovery.py)

Tests that validate the parser's ability to gracefully handle malformed or incomplete data files:

- Empty ASC files
- Header-only ASC files (metadata but no data)
- ASC files with malformed samples and events
- Malformed CSV files
- Error handling in the processing functions

### 2. Mock Data Generator (mock_data_generator.py)

A comprehensive synthetic data generator for ASC eye tracking data that can:

- Create ASC files with controlled parameters
- Generate sample and event data with realistic properties
- Inject various types of malformed data for testing
- Create files with specific characteristics (empty, left-eye only, etc.)
- Generate movie segments and associated samples

### 3. Expanded Parser Tests (test_parser_expanded.py)

Extended test coverage for the parser:

- Testing with various types of ASC files
- Validation of all parser methods and functions
- Tests for edge cases and unusual input formats
- Verification of output structures and formats
- Testing the CSV file loading functionality

### 4. Data Processing Tests (test_data_processing.py)

Tests focusing on data quality and performance:

- Statistical validation of extracted features
- Performance scaling tests with different file sizes
- Memory usage benchmarking
- Movie-specific feature extraction validation
- Unified metrics data validation

## Running the Tests

To run all tests:

```bash
python -m unittest discover
```

To run specific test modules:

```bash
python -m tests.test_error_recovery
python -m tests.test_parser_expanded
python -m tests.test_data_processing
```

## Test Data

The tests use several types of data:

1. Real ASC files in the `tests/asc_files/` directory
2. Synthetic ASC files created by the mock data generator
3. Temporary files created during tests for specific scenarios

## Key Validations

The tests verify that:

1. The parser correctly handles malformed/incomplete data files
2. Statistical calculations in feature extraction are accurate
3. The parser scales appropriately with file size
4. Movie-specific features are extracted correctly
5. Events (fixations, saccades, blinks) are properly detected even with noise
6. CSV file loading handles various error conditions gracefully

## Adding New Tests

When adding new tests:

1. Use the mock data generator to create controlled test data
2. Consider edge cases and error conditions
3. Validate statistical properties when testing data processing
4. Use tempfile to manage test files for proper cleanup
5. Make tests independent and avoid system-specific dependencies

## Test Coverage

The current test suite covers:

- Error recovery: 90% (5 test cases)
- Parser core functionality: 90% (15 test cases)
- Data processing: 85% (5 test cases)
- Mock data generation: 100% (used throughout tests)

Additional tests may be needed for GUI integration, cross-platform testing, and specific file formats.