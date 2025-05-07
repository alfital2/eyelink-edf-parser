# Eye Tracking Analysis Test Suite

This directory contains a comprehensive test suite for the ASD eye tracking analysis system, comprising 130 individual tests across multiple modules. These tests ensure the reliability and correctness of all components, from basic file parsing to complex social attention analysis.

## Test Suite Structure

### Basic Eye Tracking Data Processing (test_parser.py)

**TestParser** focuses on the fundamental functionality of the ASC file parser:

1. **test_parse_header** - Tests correct parsing of ASC file headers, using minimal ASC files with participant details and sampling rates.
2. **test_parse_samples** - Tests extraction of eye tracking samples from ASC files, verifying eye positions, pupil sizes, and timestamps.
3. **test_parse_events** - Tests parsing of eye tracking events (fixations, saccades, blinks) from ASC files.
4. **test_process_asc_file** - Tests the complete ASC file processing pipeline, validating DataFrame creation and statistics.
5. **test_feature_extraction** - Tests extraction of statistical features from eye tracking data, including means, standard deviations, and other metrics.

### Advanced CSV Loading (test_advanced_csv_loading.py)

**TestAdvancedCSVLoading** focuses on loading and processing complex CSV files:

1. **test_load_csv_with_missing_columns** - Tests handling of CSV files with missing columns, verifying system behavior when critical columns are absent.
2. **test_load_multiple_csv_files** - Tests functionality to merge multiple CSV files, validating the resulting DataFrame structure.
3. **test_load_csv_normalization** - Tests normalization of eye position data, ensuring coordinates are normalized to [0,1] range.
4. **test_load_csv_with_feature_extraction** - Tests feature extraction during CSV loading.
5. **test_load_csv_error_handling** - Tests system behavior when loading invalid CSV files.
6. **test_load_csv_with_empty_data** - Tests handling of empty CSV files.

### Animated Scanpath Visualization (test_animated_scanpath.py & test_animated_roi_scanpath.py)

**TestAnimatedScanpath** tests scanpath animation functionality:

1. **test_scanpath_animation_creation** - Tests creation of scanpath animations from eye tracking data.
2. **test_scanpath_with_heatmap** - Tests the combination of scanpath and heatmap visualizations.

**TestAnimatedROIScanpath** tests ROI-enhanced scanpath animations:

1. **test_roi_scanpath_creation** - Tests creation of scanpath animations with ROI overlays.
2. **test_roi_detection** - Tests ROI detection during scanpath playback.

### Advanced Data Processing (test_data_processing.py)

**TestDataProcessing** tests advanced data processing algorithms:

1. **test_fixation_detection** - Tests accurate detection of fixations in eye tracking data.
2. **test_saccade_analysis** - Tests saccade analysis including length, velocity, and direction calculation.
3. **test_pupil_dilation_analysis** - Tests analysis of pupil dilation patterns over time.
4. **test_combined_metrics** - Tests calculation of combined metrics from eye tracking data.
5. **test_data_filtering** - Tests noise filtering algorithms for eye tracking data.

### Extended Parser Testing (test_parser_expanded.py)

**TestExpandedParser** provides extended testing of the ASC parser:

1. **test_multipart_asc_files** - Tests parsing of multi-part ASC files containing sequential experiments.
2. **test_message_extraction** - Tests extraction of specific messages from ASC files.
3. **test_extended_sample_formats** - Tests support for various sample data formats.
4. **test_merge_binocular_data** - Tests merging of left and right eye data.
5. **test_complex_trials** - Tests parsing of complex trials with multiple experimental phases.

### Complex ROI Testing (test_complex_roi_cases.py)

**TestComplexROICases** tests edge cases in ROI handling:

1. **test_concave_polygon** - Tests support for concave polygons, using C-shaped ROIs.
2. **test_overlapping_rois** - Tests handling of overlapping ROIs, verifying priority when ROIs nest.
3. **test_many_vertices** - Tests performance with complex polygons with many vertices.
4. **test_boundary_crossing_roi** - Tests handling of ROIs that cross screen boundaries.
5. **test_simulated_eye_tracking** - Tests integration of eye tracking data with ROIs using synthetic data.

### Error Recovery (test_error_recovery.py)

**TestErrorRecovery** tests the system's resilience to invalid inputs:

1. **test_recover_from_corrupt_file** - Tests recovery from corrupt ASC files.
2. **test_recover_from_missing_data** - Tests recovery when data is incomplete.
3. **test_recover_from_extreme_values** - Tests handling of extreme/invalid values.

### Robust Parsing (test_robust_parsing.py)

**TestRobustParsing** tests the parser's ability to handle non-standard formats:

1. **test_mixed_recording_modes** - Tests parsing data with mixed recording modes (monocular/binocular).
2. **test_partial_data** - Tests parsing partially complete data.
3. **test_malformed_lines** - Tests parsing files with malformed lines.
4. **test_format_variations** - Tests handling of variations in the ASC format.

### GUI Testing (test_gui.py)

**TestGUIInitialization** & **TestGUIFileHandling** test the graphical interface:

1. **test_window_initialization** - Tests correct initialization of the main window.
2. **test_tab_initialization** - Tests creation of all required interface tabs.
3. **test_file_selection** - Tests file selection functionality for ASC/CSV files.
4. **test_processing_thread** - Tests background processing functionality.

### Movie Visualization Integration (test_movie_visualizer_integration.py)

**TestMovieVisualizerIntegration** tests integration with video content:

1. **test_video_frame_synchronization** - Tests synchronization between video frames and eye tracking data.
2. **test_gaze_overlay** - Tests overlay of gaze data on videos.
3. **test_roi_rendering_in_video** - Tests rendering of ROIs on video frames.

### Social Attention Analysis (test_social_attention_analysis.py & test_social_attention_visualization.py)

**TestSocialAttentionAnalysis** tests social attention analysis:

1. **test_social_vs_nonsocial_ratio** - Tests calculation of social vs. non-social attention ratio.
2. **test_face_fixation_analysis** - Tests analysis of fixations on faces.
3. **test_social_attention_transitions** - Tests analysis of transitions between social and non-social areas.
4. **test_first_fixation_latency** - Tests measurement of time to first fixation on social stimuli.

**TestSocialAttentionVisualization** tests visualization of social attention metrics:

1. **test_roi_fixation_detection** - Tests detection of fixations within social ROIs.
2. **test_social_attention_metrics** - Tests calculation of social attention metrics.
3. **test_social_attention_plot_generation** - Tests generation of social attention plots.

### ROI Management (test_roi_manager.py & test_advanced_roi_manager.py)

**TestROIManager** & **TestAdvancedROIManager** test the ROI management system:

1. **test_load_roi_file** - Tests loading of ROI definitions from JSON files.
2. **test_get_frame_rois** - Tests extraction of ROIs for specific frames.
3. **test_find_roi_at_point** - Tests detection of ROIs at specific gaze points.
4. **test_overlapping_rois** - Tests handling of overlapping ROIs.
5. **test_multi_frame_roi_tracking** - Tests tracking of ROIs across multiple frames.
6. **test_social_roi_classification** - Tests classification of ROIs as social/non-social.

### CSV Loading (test_csv_loader.py)

**TestCSVLoader** tests basic CSV loading functionality:

1. **test_load_basic_csv** - Tests loading basic CSV files.
2. **test_validate_csv_columns** - Tests validation of required columns in CSV files.
3. **test_combine_csv_files** - Tests combining multiple CSV files.
4. **test_csv_to_dataframe** - Tests conversion of CSV data to pandas DataFrames.

### ROI Integration (test_roi_integration.py)

**TestROIIntegration** tests the integration of ROIs with eye tracking data:

1. **test_load_sample_data** - Tests loading sample eye tracking and ROI data.
2. **test_create_integrated_visualization** - Tests creation of visualizations combining eye data and ROIs.
3. **test_eye_data_roi_matching** - Tests matching eye positions to ROIs.

## Running Tests

### Run All Tests

```bash
python tests/run_all.py
```

### Run Tests with Options

```bash
# Run with verbose output
python tests/run_all.py -v

# Stop on first failure
python tests/run_all.py -f

# Run specific test file
python -m unittest tests.test_parser
```

## Test Data Files

The tests use several types of test data:

1. **ASC Files** - Located in `tests/asc_files/`
   - Sample eye tracking recordings with known properties
   - Contains fixations, saccades, and blinks

2. **ROI Definition Files** - Located in `tests/test_data/`
   - JSON files defining regions of interest
   - Contains social and non-social regions
   - Includes complex polygon shapes for testing edge cases

3. **Benchmark Files** - Located in `tests/benchmark_files/`
   - Reference data for comparing algorithm outputs

4. **Generated Test Data** - Created on-the-fly during tests
   - Generated using mock_data_generator.py
   - Customized for specific test scenarios

## Requirements

To run the tests, you need:

- Python 3.7+
- pandas
- numpy
- matplotlib
- PyQt5 (for GUI tests)
- pytest (optional, for additional test functionality)

## Adding New Tests

Follow these guidelines when adding new tests:

1. Create test methods that start with `test_`
2. Document what each test verifies
3. Use clear assertions with descriptive messages
4. For complex tests, break them into setup, action, and assertion phases
5. Use temporary files/directories for test outputs
6. Clean up after tests complete

## Test Coverage

The current test suite provides comprehensive coverage of the key components in the system:

### Core Components
- Parser: 85%
- ROI Manager: 74%
- ROI Integration: 75%
- Animated ROI Scanpath: 66%
- Animated Scanpath: 59%

### Test Modules
- test_parser: 99%
- test_parser_expanded: 99%
- test_robust_parsing: 99%
- test_roi_manager: 99%
- test_complex_roi_cases: 99%
- test_advanced_csv_loading: 97%
- test_error_recovery: 97%
- test_social_attention_analysis: 96%
- test_social_attention_visualization: 91%
- test_animated_roi_scanpath: 91%
- test_roi_integration: 93%
- test_data_processing: 87%
- test_movie_visualizer_integration: 86%
- test_animated_scanpath: 81%
- mock_data_generator: 95%

### GUI Components
- GUI/gui.py: 26% (Testing GUI components remains a challenge)
- Documentation: 100%

### Visualization
- eyelink_visualizer.py: 5% (Needs improved test coverage)
- movie_visualizer_integration.py: 0% (Needs test implementation)

Overall code coverage: 64%

These coverage statistics were generated using pytest-cov. Areas with lower coverage, particularly GUI components and visualization modules, would benefit from additional test development.