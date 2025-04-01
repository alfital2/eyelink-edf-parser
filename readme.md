# Eye Movement Analysis for Autism Classification

A comprehensive toolkit for parsing, analyzing, and visualizing eye-tracking data to support autism research.

## Overview

This project provides a set of tools for processing eye movement data recorded using SR Research EyeLink systems. The toolkit is specifically designed to extract features and visualizations relevant to autism spectrum disorder (ASD) research, focusing on the characteristic differences in visual attention patterns between individuals with ASD and neurotypical controls.

## Features

- Parses raw EyeLink ASC files into structured data
- Extracts eye movement metrics, fixations, saccades, and blinks
- Calculates head movement using corneal reflection data
- Generates comprehensive visualizations of eye-tracking patterns
- Extracts features for machine learning models


## Usage

### Basic Usage

```bash
python main.py --input path/to/file.asc --output results --visualize
```

### Command Line Options

```
--input, -i       Path to ASC file or directory containing ASC files
--output, -o      Output directory for parsed data and visualizations (default: 'output')
--visualize       Generate visualizations
--screen_width    Screen width in pixels (default: 1280)
--screen_height   Screen height in pixels (default: 1024)
--no_features     Skip feature extraction
```

### Processing Multiple Files

```bash
python main.py --input path/to/directory/ --output results --visualize
```

## Data Files Generated

The parser generates several CSV files containing various aspects of the eye-tracking data:

### 1. Unified Eye Metrics CSV (`file_name_unified_eye_metrics.csv`)

This is the primary data file containing all eye-tracking metrics in a single integrated format:

- `timestamp`: Time of the sample in milliseconds
- `x_left`, `y_left`: Left eye position coordinates
- `pupil_left`: Left eye pupil size
- `x_right`, `y_right`: Right eye position coordinates
- `pupil_right`: Right eye pupil size
- `input`: Input signal value
- `cr_left`, `cr_right`: Corneal reflection positions
- `head_movement_left_x`, `head_movement_right_x`: Head movement metrics
- `head_movement_magnitude`: Overall head movement magnitude
- `inter_pupil_distance`: Distance between pupils (depth/vergence)
- `gaze_velocity_left`, `gaze_velocity_right`: Gaze velocity in pixels/second
- `is_fixation_left/right`: Boolean flag indicating a fixation
- `is_saccade_left/right`: Boolean flag indicating a saccade
- `is_blink_left/right`: Boolean flag indicating a blink

### 2. Features CSV (`file_name_features.csv`)

Contains aggregated features relevant for machine learning classification:

- `participant_id`: Identifier for the participant
- **Pupil Metrics**: `pupil_left/right_mean`, `pupil_left/right_std`, `pupil_left/right_min`, `pupil_left/right_max`
- **Gaze Variability**: `gaze_left/right_x_std`, `gaze_left/right_y_std`, `gaze_left/right_dispersion`
- **Head Movement**: `head_movement_mean`, `head_movement_std`, `head_movement_max`, `head_movement_frequency`
- **Fixation Metrics**: `fixation_left/right_count`, `fixation_left/right_duration_mean`, `fixation_left/right_duration_std`, `fixation_left/right_rate`
- **Saccade Metrics**: `saccade_left/right_count`, `saccade_left/right_amplitude_mean`, `saccade_left/right_amplitude_std`, `saccade_left/right_duration_mean`, `saccade_left/right_peak_velocity_mean`
- **Blink Metrics**: `blink_left/right_count`, `blink_left/right_duration_mean`, `blink_left/right_rate`

### 3. Combined Files

When processing multiple participants, the system generates:

- `all_participants_unified_metrics.csv`: Combined eye metrics from all participants
- `combined_features.csv`: Features from all participants for ML analysis
- `feature_summary.csv`: Statistical summary of features

## Visualizations

The toolkit generates a variety of visualizations to help researchers understand eye movement patterns:

### 1. Scanpath Plot

 Shows the trajectory of eye movements over time, with markers for fixations.

**Interpretation**: 
- Typical development: More systematic, predictable scanning patterns
- ASD: Often shows more variable and idiosyncratic scanning, with less focus on socially relevant areas

### 2. Heatmap


 Color-coded visualization of gaze density, with warmer colors indicating areas that received more visual attention.

**Interpretation**:
- Typical development: Hotspots usually centered on socially relevant features (faces, eyes)
- ASD: May show more diffuse attention or focus on non-social elements

### 3. Fixation Duration Distribution


 Histogram showing the distribution of fixation durations.

**Interpretation**:
- Typical development: Moderately distributed fixation durations
- ASD: May show either unusually brief fixations (suggesting difficulty maintaining attention) or unusually long fixations (suggesting perseveration)

### 4. Saccade Amplitude Distribution


 Histogram showing the distribution of saccade amplitudes (distance between fixations).

**Interpretation**:
- Typical development: Wide range of saccade amplitudes, with peaks corresponding to typical reading or scene viewing
- ASD: May show unusual patterns with either very small saccades (local focus) or very large saccades (jumping between distant points)

### 5. Pupil Size Timeseries


 Graph showing pupil size changes over time.

**Interpretation**:
- Reflects cognitive load and emotional arousal
- ASD: May show atypical pupillary responses to social stimuli

### 6. Head Movement


 Visualization of head movement magnitude over time.

**Interpretation**:
- Typical development: Moderate head movements, often correlated with attention shifts
- ASD: May show increased head movements or different patterns of movement

### 7. Velocity Profile


 Graph showing eye movement velocity over time, with markers for saccades.

**Interpretation**:
- Shows the dynamics of eye movements
- ASD: May show different velocity profiles, potentially reflecting different neural control of eye movements

### 8. Fixation Density Comparison



 Compares the spatial distribution of fixations between left and right eyes.

**Interpretation**:
- Typical development: High correlation between eyes
- ASD: May show more asymmetry between eyes

### 9. Fixation-Saccade Distribution


 Spatial visualization showing fixations and the saccades connecting them.

**Interpretation**:
- Shows the complete spatial pattern of visual exploration
- ASD: May show unusual patterns of exploration compared to typical development

## Research Applications

This toolkit is designed to support several key areas of autism research:

1. **Classification**: Identifying potential biomarkers for ASD through machine learning on eye movement features
2. **Severity Assessment**: Quantifying the relationship between eye movement patterns and autism symptom severity
3. **Subtype Identification**: Uncovering potential subtypes within the autism spectrum based on different gaze patterns
4. **Early Detection**: Supporting research into early indicators of ASD through eye tracking

## Technical Notes

- The parser is optimized for EyeLink 1000+ data but should work with other EyeLink systems
- Head movement is calculated using the corneal reflection method, which requires that the EyeLink system was configured to track CR
- Feature extraction is designed to capture metrics relevant to autism research but can be extended for other applications


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SR Research for the EyeLink system documentation
- Professor Ohad Ben-Shahar for project guidance
- Department of Computer Science, Ben-Gurion University of the Negev
