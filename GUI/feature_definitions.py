# feature_definitions.py
"""
Constants module containing feature category definitions used in the eye movement analysis GUI.
This centralizes the feature structure to separate it from display logic.
"""

# Categories of eye movement metrics with their display configuration
# Format: (category_name, features_data, grid_row, grid_column)
FEATURE_CATEGORIES = [
    # Basic participant information
    ("Basic Information", ["participant_id"], 0, 0),  # row 0, col 0
    
    # Combined eye metrics tables with left/right columns
    ("Pupil Size", [
        {"name": "Mean Pupil Size", "left": "pupil_left_mean", "right": "pupil_right_mean"},
        {"name": "Pupil Size Std", "left": "pupil_left_std", "right": "pupil_right_std"},
        {"name": "Min Pupil Size", "left": "pupil_left_min", "right": "pupil_right_min"},
        {"name": "Max Pupil Size", "left": "pupil_left_max", "right": "pupil_right_max"}
    ], 0, 1),  # row 0, col 1
    
    ("Gaze Position", [
        {"name": "X Standard Deviation", "left": "gaze_left_x_std", "right": "gaze_right_x_std"},
        {"name": "Y Standard Deviation", "left": "gaze_left_y_std", "right": "gaze_right_y_std"},
        {"name": "Gaze Dispersion", "left": "gaze_left_dispersion", "right": "gaze_right_dispersion"}
    ], 0, 2),  # row 0, col 2
    
    ("Fixation Metrics", [
        {"name": "Fixation Count", "left": "fixation_left_count", "right": "fixation_right_count"},
        {"name": "Mean Duration (ms)", "left": "fixation_left_duration_mean", "right": "fixation_right_duration_mean"},
        {"name": "Duration Std (ms)", "left": "fixation_left_duration_std", "right": "fixation_right_duration_std"},
        {"name": "Fixation Rate", "left": "fixation_left_rate", "right": "fixation_right_rate"}
    ], 1, 0),  # row 1, col 0
    
    ("Saccade Metrics", [
        {"name": "Saccade Count", "left": "saccade_left_count", "right": "saccade_right_count"},
        {"name": "Mean Amplitude (°)", "left": "saccade_left_amplitude_mean", "right": "saccade_right_amplitude_mean"},
        {"name": "Amplitude Std (°)", "left": "saccade_left_amplitude_std", "right": "saccade_right_amplitude_std"},
        {"name": "Mean Duration (ms)", "left": "saccade_left_duration_mean", "right": "saccade_right_duration_mean"}
    ], 1, 1),  # row 1, col 1
    
    ("Blink Metrics", [
        {"name": "Blink Count", "left": "blink_left_count", "right": "blink_right_count"},
        {"name": "Mean Duration (ms)", "left": "blink_left_duration_mean", "right": "blink_right_duration_mean"},
        {"name": "Blink Rate", "left": "blink_left_rate", "right": "blink_right_rate"}
    ], 1, 2),  # row 1, col 2
    
    ("Head Movement", [
        {"name": "Mean", "key": "head_movement_mean"},
        {"name": "Standard Deviation", "key": "head_movement_std"},
        {"name": "Maximum", "key": "head_movement_max"},
        {"name": "Frequency", "key": "head_movement_frequency"}
    ], 2, 0)  # row 2, col 0
]