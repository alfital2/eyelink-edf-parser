# Eye Tracking Analysis Test Suite

This directory contains a comprehensive test suite for the ASD eye tracking analysis system, comprising 130 individual tests across multiple modules. These tests ensure the reliability and correctness of all components, from basic file parsing to complex social attention analysis.

## Test Suite Structure

### Basic Eye Tracking Data Processing (test_parser.py)

**TestEyeLinkASCParser** focuses on the fundamental functionality of the ASC file parser:

1. **test_file_reading** - Tests that the file is read correctly with the right number of lines.
2. **test_metadata_extraction** - Tests extraction of metadata from the file header.
3. **test_message_parsing** - Tests that messages are parsed correctly with the right count and content.
4. **test_sample_parsing** - Tests parsing of eye movement samples.
5. **test_fixation_parsing** - Tests parsing of fixation events.
6. **test_saccade_parsing** - Tests parsing of saccade events.
7. **test_blink_parsing** - Tests parsing of blink events.
8. **test_frame_marker_parsing** - Tests parsing of video frame markers.
9. **test_dataframe_conversion** - Tests conversion of parsed data to pandas DataFrames.
10. **test_unified_metrics_df** - Tests creation of the unified eye metrics DataFrame.
11. **test_feature_extraction** - Tests extraction of aggregate features for machine learning.
12. **test_extract_features_by_movie** - Tests extraction of features per movie.
13. **test_movie_segment_parsing** - Tests parsing of movie segments and related information.
14. **test_process_asc_file_function** - Tests the process_asc_file helper function.

### Advanced CSV Loading (test_advanced_csv_loading.py)

**TestAdvancedCSVLoading** focuses on loading and processing complex CSV files:

1. **test_standard_csv_loading** - Tests loading a standard unified eye metrics CSV file.
2. **test_missing_columns** - Tests loading a CSV file with missing columns.
3. **test_multiple_csv_with_different_columns** - Tests loading multiple CSV files with different column sets.
4. **test_varying_row_count** - Tests loading multiple CSV files with varying row counts.
5. **test_additional_columns** - Tests loading a CSV file with additional columns.
6. **test_column_type_mismatches** - Tests loading CSV files with column type mismatches.
7. **test_feature_extraction** - Tests feature extraction from CSV data.

### Animated Scanpath Visualization (test_animated_scanpath.py & test_animated_roi_scanpath.py)

**TestAnimatedScanpath** tests scanpath animation functionality:

1. **test_create_animated_scanpath** - Tests creating an animated scanpath widget.
2. **test_multiple_movie_loading** - Tests loading multiple movies into the widget.
3. **test_error_handling_with_missing_columns** - Tests error handling when loading data with missing columns.
4. **test_empty_dataframe_handling** - Tests handling of empty dataframes.

**TestAnimatedROIScanpath** tests ROI-enhanced scanpath animations:

1. **test_roi_widget_initialization** - Tests initialization of the animated ROI scanpath widget.
2. **test_roi_toggle_functions** - Tests the toggle functions for ROI display options.
3. **test_roi_detection** - Tests ROI detection during scanpath playback.

### Advanced Data Processing (test_data_processing.py)

**TestDataProcessing** tests advanced data processing algorithms:

1. **test_performance_scaling** - Tests parser performance with different file sizes.
2. **test_statistical_validation** - Tests that statistical calculations in feature extraction are accurate.
3. **test_memory_usage** - Tests memory usage with large files.
4. **test_movie_specific_features** - Tests extracting features for specific movies.
5. **test_unified_metrics_validation** - Tests that unified metrics correctly combines data.

### Extended Parser Testing (test_parser_expanded.py)

**TestExpandedParser** provides extended testing of the ASC parser:

1. **test_parser_initialization** - Tests the parser initialization with different file types.
2. **test_read_file** - Tests reading different types of ASC files.
3. **test_parse_metadata** - Tests metadata extraction from different file types.
4. **test_parse_messages** - Tests message parsing from different file types.
5. **test_parse_samples** - Tests sample parsing from different file types.
6. **test_parse_events** - Tests event parsing from different file types.
7. **test_parse_file** - Tests the complete file parsing process.
8. **test_to_dataframes** - Tests conversion to pandas DataFrames.
9. **test_create_unified_metrics_df** - Tests creation of unified metrics DataFrame.
10. **test_save_to_csv** - Tests saving data to CSV files.
11. **test_extract_features** - Tests feature extraction from different file types.
12. **test_extract_features_per_movie** - Tests extracting features for each movie segment.
13. **test_feature_validation** - Tests that extracted features have reasonable values.
14. **test_process_asc_file** - Tests the process_asc_file function.
15. **test_load_csv_file** - Tests loading a CSV file with the load_csv_file function.

### Complex ROI Testing (test_complex_roi_cases.py)

**TestComplexROICases** tests edge cases in ROI handling:

1. **test_concave_polygon** - Tests support for concave polygons, using C-shaped ROIs.
2. **test_overlapping_rois** - Tests handling of overlapping ROIs, verifying priority when ROIs nest.
3. **test_many_vertices** - Tests performance with complex polygons with many vertices.
4. **test_boundary_crossing_roi** - Tests handling of ROIs that cross screen boundaries.
5. **test_simulated_eye_tracking** - Tests integration of eye tracking data with ROIs using synthetic data.
6. **test_complex_roi_combination** - Tests a combination of complex ROI scenarios.
7. **test_edge_case_coordinates** - Tests ROI detection with edge case coordinate values.
8. **test_roi_data_analytics** - Tests the ROI manager's ability to analyze ROI data across frames.

### Error Recovery (test_error_recovery.py)

**TestErrorRecovery** tests the system's resilience to invalid inputs:

1. **test_empty_asc_file** - Tests handling of an empty ASC file.
2. **test_header_only_asc_file** - Tests handling of an ASC file with only header info.
3. **test_process_asc_malformed_file** - Tests process_asc_file with a malformed ASC file.
4. **test_malformed_sample_asc_file** - Tests handling of an ASC file with malformed samples.
5. **test_malformed_csv_file** - Tests handling of a malformed CSV file.

### Robust Parsing (test_robust_parsing.py)

**TestRobustParsing** tests the parser's ability to handle non-standard formats:

1. **test_mixed_recording_modes** - Tests parsing data with mixed recording modes (monocular/binocular).
2. **test_malformed_data_lines** - Tests handling of files with malformed data lines.
3. **test_extreme_values** - Tests handling of files with extreme values.
4. **test_incomplete_header** - Tests handling of files with incomplete headers.
5. **test_missing_message_timestamps** - Tests handling of files with missing timestamps in messages.
6. **test_empty_sections** - Tests handling of files with empty sections.
7. **test_inconsistent_event_formats** - Tests handling of files with inconsistent event formats.

### GUI Testing (test_gui.py)

**TestGUIInitialization** tests the graphical interface initialization:

1. **test_window_initialization** - Tests that the main window initializes correctly.
2. **test_tab_initialization** - Tests that all required tabs are created.
3. **test_feature_tables_initialization** - Tests that feature tables are properly initialized.
4. **test_animated_scanpath_widget_initialization** - Tests that the animated scanpath widget initializes correctly.

**TestGUIFileHandling** tests file and directory handling in the GUI:

1. **test_asc_file_selection** - Tests ASC file selection.
2. **test_csv_file_selection** - Tests CSV file selection.
3. **test_output_directory_selection** - Tests output directory selection.
4. **test_process_button_enabled** - Tests that process button is enabled when both files and output dir are selected.

**TestProcessingThread** tests background processing:

1. **test_processing_thread_csv_single** - Tests processing thread with a single CSV file.

**TestAnimatedROIScanpathWidget** tests the ROI scanpath widget:

1. **test_widget_initialization** - Tests that the widget initializes correctly.
2. **test_data_loading** - Tests loading data into the widget.
3. **test_roi_file_loading** - Tests loading ROI data into the widget.
4. **test_display_options** - Tests the display options of the widget.
5. **test_playback_controls** - Tests the playback controls of the widget.

### Movie Visualization Integration (test_movie_visualizer_integration.py)

**TestMovieVisualizerIntegration** tests integration with video content:

1. **test_generate_movie_visualizations** - Tests generating visualizations for movie folders.
2. **test_generate_specific_plot_with_real_data** - Tests generating a specific plot from the movie data.
3. **test_generate_all_plots** - Tests generating all plots for a movie.
4. **test_error_handling** - Tests that the code handles errors gracefully.
5. **test_empty_dataframe_handling** - Tests handling of empty dataframes.
6. **test_unknown_plot_type** - Tests handling of unknown plot types.

### Social Attention Analysis (test_social_attention_analysis.py)

**TestSocialAttentionAnalysis** tests social attention analysis:

1. **test_roi_social_classification** - Tests classification of ROIs into social and non-social categories.
2. **test_fixation_roi_detection** - Tests that fixations are correctly associated with ROIs.
3. **test_social_attention_metrics** - Tests calculation of basic social attention metrics.
4. **test_roi_dwell_time_calculation** - Tests calculation of dwell time for different ROIs.
5. **test_first_fixation_latency** - Tests measurement of time to first fixation on social stimuli.

### Social Attention Visualization (test_social_attention_visualization.py)

**TestSocialAttentionVisualization** tests visualization of social attention metrics:

1. **test_roi_fixation_detection** - Tests that fixations in ROIs are correctly detected.
2. **test_social_attention_metrics** - Tests calculation of social attention metrics.
3. **test_social_attention_plot_generation** - Tests generation of social attention plots.

**TestROIFixationSequencePlot** tests sequence plot generation:

1. **test_roi_fixation_sequence_no_annotations** - Tests that the ROI fixation sequence plot doesn't contain pagination and footnote.

### ROI Management (test_roi_manager.py)

**TestROIManager** tests the basic ROI management system:

1. **test_roi_file_loading** - Tests that the ROI file is loaded correctly.
2. **test_get_frame_rois** - Tests the get_frame_rois method.
3. **test_find_roi_at_point** - Tests the find_roi_at_point method.
4. **test_get_nearest_frame** - Tests the get_nearest_frame method.
5. **test_point_in_polygon** - Tests the point_in_polygon method.
6. **test_is_gaze_in_roi** - Tests the is_gaze_in_roi method.
7. **test_get_unique_labels** - Tests the get_unique_labels method.
8. **test_find_roi_at_gaze** - Tests the find_roi_at_gaze method.
9. **test_empty_roi_file** - Tests loading an empty ROI file.
10. **test_invalid_roi_file** - Tests loading an invalid ROI file.

### Advanced ROI Management (test_advanced_roi_manager.py)

**TestAdvancedROIManager** tests advanced ROI management features:

1. **test_overlapping_rois** - Tests ROI manager behavior with overlapping ROIs.
2. **test_roi_export_import** - Tests exporting and reimporting ROI data.
3. **test_invalid_roi_handling** - Tests handling of invalid ROI data.
4. **test_multi_frame_roi_tracking** - Tests ROI manager with changing ROIs across multiple frames.
5. **test_social_roi_classification** - Tests social vs. non-social ROI classification.
6. **test_complex_polygon_rois** - Tests ROI manager with complex polygon shapes.
7. **test_extreme_roi_coordinates** - Tests ROI manager with extreme coordinate values.
8. **test_polygon_containment_algorithms** - Tests the algorithms used for polygon containment.

### CSV Loading (test_csv_loader.py)

**TestCSVLoader** tests CSV loading functionality:

1. **test_load_csv_file** - Tests loading a CSV file.
2. **test_extract_events_from_unified** - Tests extracting events from the unified DataFrame.
3. **test_extract_features_from_unified** - Tests extracting features from the unified DataFrame.
4. **test_movie_specific_features** - Tests extracting features for specific movies from the unified DataFrame.
5. **test_load_csv_with_movie_features** - Tests loading a CSV file with multiple movies and extracting per-movie features.

### ROI Integration (test_roi_integration.py)

**TestROIIntegration** tests the integration of ROIs with eye tracking data:

1. **test_load_sample_data** - Tests loading sample eye tracking and ROI data.
2. **test_create_integrated_visualization** - Tests creation of visualizations combining eye data and ROIs.
3. **test_eye_data_roi_matching** - Tests matching eye positions to ROIs.

### EyeLink Visualizer (test_eyelink_visualizer.py)

**TestEyeLinkVisualizer** tests the EyeLink visualization functionality:

1. **test_initialization** - Tests initialization of the visualizer.
2. **test_discover_movie_folders** - Tests discovering movie folders.
3. **test_load_movie_data** - Tests loading movie data.
4. **test_ensure_plots_directory** - Tests ensuring plots directory exists.
5. **test_plot_scanpath** - Tests plotting scanpath visualization.
6. **test_time_window** - Tests plotting with time window constraints.

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

3. **Generated Test Data** - Created on-the-fly during tests
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

## Test Coverage

Core Components:
- Parser: 85%
- ROI Manager: 74%
- ROI Integration: 75%
- Animated ROI Scanpath: 66%
- Animated Scanpath: 59%
- GUI/gui.py: 26% (Testing GUI components remains a challenge)
- eyelink_visualizer.py: 5% (Needs improved test coverage)
- movie_visualizer_integration.py: 0% (Needs test implementation)

Overall code coverage: 64%