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

### GUI Testing 

#### Basic GUI Structure and Initialization (test_gui.py)

**TestGUIInitialization** tests the graphical interface initialization:

1. **test_window_initialization** - Tests that the main window initializes correctly with proper window title, geometry, and initial state.
2. **test_tab_initialization** - Tests that all required tabs (Data Processing, Results & Visualization, Extracted Features, Documentation) are created.
3. **test_feature_tables_initialization** - Tests that feature tables are properly initialized with correct categories and components.
4. **test_animated_scanpath_widget_initialization** - Tests that the animated scanpath widget initializes correctly with proper defaults.

**TestGUIFileHandling** tests file and directory handling in the GUI:

1. **test_asc_file_selection** - Tests ASC file selection: clicking "Load Source File(s)" → selecting ASC file → verifying UI updates.
2. **test_csv_file_selection** - Tests CSV file selection: clicking "Load Source File(s)" → selecting CSV file → verifying UI updates.
3. **test_output_directory_selection** - Tests output directory selection: clicking "Select Output Directory" → choosing directory → verifying UI updates.
4. **test_process_button_enabled** - Tests that process button is enabled when both files and output dir are selected, disabled otherwise.

**TestProcessingThread** tests background processing:

1. **test_processing_thread_csv_single** - Tests processing thread with a single CSV file: initialization → data loading → feature extraction → signal emission.

**TestAnimatedROIScanpathWidget** tests the ROI scanpath widget:

1. **test_widget_initialization** - Tests that the widget initializes correctly with proper default states.
2. **test_data_loading** - Tests loading data into the widget: loading eye tracking data → updating UI → enabling controls.
3. **test_roi_file_loading** - Tests loading ROI data: selecting ROI file → loading ROI definitions → updating display.
4. **test_display_options** - Tests the display options: toggling show/hide options → verifying display state changes.
5. **test_playback_controls** - Tests the playback controls: play/pause → reset → timeline slider → checking state changes.

#### GUI Data Loading and Processing (test_gui_data_loading.py)

**TestGUIDataLoading** tests data loading and processing flows:

1. **test_file_selection_updates_gui** - Tests file selection flow: selecting CSV file → verifying file path storage → checking UI label updates.
2. **test_output_dir_selection_updates_gui** - Tests output directory selection: choosing directory → verifying path storage → checking UI updates.
3. **test_process_button_enabled_when_ready** - Tests process button state: selecting file → choosing output directory → verifying button becomes enabled.
4. **test_process_data_starts_thread** - Tests process button action: clicking process → verifying thread starts → checking UI updates during processing.
5. **test_processing_thread_signal_connections** - Tests signal connections: emitting signals from thread → verifying proper GUI updates → checking data flow.
6. **test_update_progress_changes_progress_bar** - Tests progress updates: sending progress signals → verifying progress bar updates.
7. **test_update_status_updates_status_label_and_log** - Tests status updates: sending status messages → checking status label → verifying log updates.
8. **test_processing_error_shows_message_box** - Tests error handling: triggering processing error → verifying error dialog appears.

#### File Operations (test_file_operations.py)

**TestFileOperations** tests file operations in the GUI:

1. **test_asc_file_selection** - Tests ASC file selection: choosing ASC file → verifying file path storage → checking UI updates.
2. **test_csv_file_selection** - Tests CSV file selection: choosing CSV file → verifying file path storage → checking file type detection.
3. **test_multiple_file_selection** - Tests selecting multiple files: choosing multiple files → verifying UI updates → checking file type detection.
4. **test_output_directory_selection** - Tests output directory selection: choosing directory → verifying storage → checking label updates.
5. **test_roi_file_selection** - Tests ROI file selection: selecting ROI file → checking path storage → verifying UI updates → enabling social buttons.
6. **test_save_features** - Tests saving features to CSV: generating features → selecting save location → writing file → verifying file contents.
7. **test_save_features_cancel** - Tests canceling save dialog: opening save dialog → canceling → verifying no file is written.
8. **test_save_features_error_handling** - Tests error handling during save: triggering write error → verifying error dialog appears.
9. **test_save_features_no_data** - Tests save attempt with no data: trying to save with no features → verifying warning appears.
10. **test_save_features_success_message** - Tests success feedback: saving features successfully → verifying success dialog appears.
11. **test_open_report** - Tests report opening: generating report → clicking open report → verifying browser launches with correct URL.
12. **test_open_report_not_found** - Tests missing report handling: setting invalid report path → clicking open → verifying warning appears.

#### Signal and UI Interaction (test_signal_connections.py)

**TestSignalConnections** tests signal-slot connections and UI interactions:

1. **test_process_button_connection** - Tests process button click: clicking process button → verifying processing starts.
2. **test_select_file_button_connection** - Tests file button click: clicking file selection button → verifying file dialog opens.
3. **test_select_output_button_connection** - Tests output button click: clicking output selection button → verifying directory dialog opens.
4. **test_movie_combo_connection** - Tests movie selection dropdown: changing selected movie → verifying visualization updates.
5. **test_viz_type_combo_connection** - Tests visualization type dropdown: changing visualization type → verifying display updates.
6. **test_feature_movie_combo_connection** - Tests feature movie dropdown: changing selected movie → verifying feature display updates.
7. **test_processing_thread_signal_connections** - Tests thread signal connections: emitting thread signals → verifying UI responses.
8. **test_gui_process_data_creates_thread** - Tests thread creation: clicking process → verifying thread creation with correct parameters.

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

### Negative Test Cases (test_negative_cases.py)

**TestNegativeCases** tests error handling and edge cases:

1. **test_parser_with_empty_file** - Tests that the parser handles empty files gracefully.
2. **test_parser_with_malformed_file** - Tests that the parser handles malformed ASC files.
3. **test_missing_required_columns** - Tests handling data with missing required columns.
4. **test_process_asc_file_with_invalid_path** - Tests process_asc_file with an invalid file path.
5. **test_load_csv_with_missing_columns** - Tests that CSV loading handles missing columns appropriately.
6. **test_roi_manager_with_invalid_json** - Tests ROI manager with invalid JSON data.
7. **test_invalid_fixation_data** - Tests handling invalid fixation data.
8. **test_convert_numpy_types** - Tests proper conversion of NumPy types to standard Python types.

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

**TestMovieEyeTrackingVisualizer** tests the EyeLink visualization functionality:

1. **test_validate_plot_data** - Tests the validation of data for plotting.
2. **test_create_plot_filename** - Tests creating plot filenames with various parameters.
3. **test_discover_movie_folders** - Tests discovering movie folders with unified eye metrics.
4. **test_load_movie_data** - Tests loading movie data from folders.
5. **test_ensure_plots_directory** - Tests creating and accessing the plots directory.
6. **test_save_plot** - Tests saving plots to the filesystem.
7. **test_plot_scanpath_with_valid_data** - Tests creating scanpath plots with valid data.
8. **test_plot_scanpath_with_time_window** - Tests creating scanpath plots with time window constraints.
9. **test_plot_scanpath_with_empty_data** - Tests handling empty data in scanpath plotting.
10. **test_plot_heatmap** - Tests creating heatmap visualizations of gaze density.
11. **test_plot_fixation_duration_distribution** - Tests plotting fixation duration distributions.
12. **test_plot_pupil_size_timeseries** - Tests plotting pupil size over time.
13. **test_generate_report** - Tests generating HTML reports with visualizations.
14. **test_negative_cases** - Tests error handling when visualizing missing or invalid data.

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

## GUI Test Best Practices

The GUI tests follow these best practices for reliable automated testing:

1. **Dialog Mocking**: All dialog boxes (QFileDialog, QMessageBox) are mocked to prevent dialogs appearing during automated test runs.

2. **Signal Connection Testing**: Signal-slot connections are tested by directly connecting and emitting signals, then verifying the correct slots are called.

3. **Direct Module-Level Mocking**: PyQt dialog classes are mocked at the module level instead of using decorator-based patching for more reliable testing.

4. **Component Testing**: Individual components like the feature table manager and animated scanpath widget are tested in isolation.

5. **Integration Testing**: Full integration tests verify the workflow from file selection through processing to visualization.

See `tests/gui_testing_notes.md` for detailed implementation guidance on dialog mocking.

## Test Coverage

Core Components:
- Parser: 85%
- ROI Manager: 76% 
- ROI Integration: 80%
- Animated ROI Scanpath: 66%
- Animated Scanpath: 59%
- GUI/gui.py: 51%
- GUI/visualization/eyelink_visualizer.py: 47% (Improved from 5%)
- GUI/visualization/plot_generator.py: 42% (New tests added)
- movie_visualizer_integration.py: 0% (Still needs implementation)

### Recent Improvements

We've made significant improvements to the test coverage for visualization components:

1. **Added Negative Test Cases**: Created dedicated test file (`test_negative_cases.py`) to test error handling throughout the system, including tests for NumPy type conversion which fixed a segmentation fault issue.

2. **Enhanced Visualization Testing**: Added tests for both `eyelink_visualizer.py` (47% coverage) and `plot_generator.py` (42% coverage) with tests for key visualization functions.

3. **Improved Error Handling**: Added tests to verify proper error handling for edge cases, invalid inputs, and missing data.

Overall code coverage: 66%