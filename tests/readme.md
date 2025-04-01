# EyeLink ASC Parser Testing Framework

This directory contains the test suite and verification tools for the EyeLink ASC Parser used in autism research. The framework helps ensure that the parser correctly extracts eye movement data and features from ASC files.

## Directory Structure

```
tests/
├── README.md                   # This file
├── run_tests.py                # Script to run the test suite
├── test_parser.py              # Test suite for the parser
├── verify_parser_output.py     # Tool to verify parser output
├── sample_test.asc             # Sample ASC file for testing
└── verification_output/        # Created by verify_parser_output.py
    ├── parser_verification_report.txt
    └── plots/
        ├── gaze_positions.png
        ├── pupil_sizes.png
        └── event_durations.png
```

## Testing Components

The testing framework consists of three main components:

1. **Comprehensive test suite** (`test_parser.py`) - Contains detailed unit tests for each function in the parser
2. **Test runner** (`run_tests.py`) - Discovers and runs the tests with detailed reporting
3. **Verification tool** (`verify_parser_output.py`) - Generates detailed reports and visualizations

## Running the Tests

To run the full test suite:

```bash
python run_tests.py
```

This will:
1. Count the expected events in the sample ASC file
2. Run all tests in `test_parser.py`
3. Display a detailed summary of test results

## Verifying Parser Output

To generate a detailed report and visualizations of the parser's output:

```bash
python verify_parser_output.py
```

This will:
1. Process the sample ASC file
2. Generate a detailed report with statistics on samples, events, messages, etc.
3. Create visualizations of gaze positions, pupil sizes, and event durations

The report and visualizations will be saved in the `verification_output` directory.

## Test Coverage

The test suite (`test_parser.py`) comprehensively tests:

1. **File reading** - Ensures the correct number of lines are read
2. **Metadata extraction** - Validates specific metadata fields and values
3. **Message parsing** - Verifies message counts and content
4. **Calibration info** - Checks extraction of calibration quality data
5. **Sample parsing** - Validates sample counts and specific sample values
6. **Fixation parsing** - Verifies fixation counts and specific fixation properties
7. **Saccade parsing** - Tests saccade detection and properties
8. **Blink parsing** - Confirms blink detection and durations
9. **Frame marker parsing** - Checks video frame marker extraction
10. **DataFrame conversion** - Validates conversion to pandas DataFrames
11. **Unified metrics** - Tests creation of unified eye metrics
12. **Feature extraction** - Verifies extraction of machine learning features
13. **CSV output** - Tests saving data to CSV files
14. **Helper functions** - Tests the process_asc_file and process_multiple_files functions

## Adding New Tests

To add a new test:

1. Open `test_parser.py`
2. Add a new test method that starts with `test_`
3. Run the tests using `python run_tests.py`

Example:

```python
def test_my_new_feature(self):
    """Test my new parser feature."""
    result = self.parser.my_new_feature()
    self.assertIsNotNone(result)
    # Add specific assertions to validate your feature
```

## Troubleshooting

If tests are failing:

1. Run the verification tool to get detailed information about the parser output:
   ```bash
   python verify_parser_output.py
   ```

2. Check the test output for specific assertions that failed

3. If sample counts don't match expected counts, check how the parser is detecting events in the ASC file

4. For development, you can run a specific test using:
   ```bash
   python -m unittest tests.test_parser.TestEyeLinkASCParser.test_specific_method
   ```

## Extending the Framework

To add more verification capabilities:

1. Add new visualization functions to `verify_parser_output.py`
2. Enhance the test suite with more specific tests for your features
3. Consider adding performance benchmarks for large file processing
