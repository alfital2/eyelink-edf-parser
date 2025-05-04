# Eye Movement Analysis for Autism Classification

A comprehensive toolkit for analyzing eye-tracking data to support autism research, with interactive animations, deep learning integration, and advanced visualization capabilities.

## Overview

This project provides specialized tools for processing and analyzing eye movement data recorded using SR Research EyeLink systems. The toolkit is specifically designed for autism spectrum disorder (ASD) research, focusing on characterizing visual attention patterns during movie viewing and extracting features that may serve as potential biomarkers for ASD classification.

## Key Features

- **Advanced Eye Movement Analysis**: Extract detailed metrics from raw EyeLink ASC files including fixations, saccades, blinks, and pupil size
- **Movie-Specific Processing**: Analyze eye-tracking data separately for each movie stimulus
- **Interactive Animated Scanpaths**: View eye movements with playback controls, adjustable speed, and timeline navigation
- **Deep Learning Integration**: Ready for machine learning approaches to classify ASD from eye movement patterns
- **Rich Visualization Library**: Generate multiple visualization types revealing different aspects of visual attention
- **Social Attention Analysis**: Special focus on social vs. non-social attention patterns, a key area of difference in ASD
- **ROI Support**: Analyze attention to specific regions of interest within visual stimuli
- **GUI Interface**: User-friendly graphical interface with interactive visualization options
- **HTML Report Generation**: Create comprehensive, dynamically structured reports for sharing results
- **Animation Export**: Export eye movement animations as MP4 or GIF files

## Installation

```bash
git clone https://github.com/alfital2/eye-movement-analysis.git
cd eye-movement-analysis
pip install -r requirements.txt
```

## Usage Guide

### Command Line Interface

Process a single ASC file with visualizations:

```bash
python main.py --input path/to/file.asc --output results --visualize
```

Process multiple files:

```bash
python main.py --input path/to/directory/ --output results --visualize
```

Generate HTML report:

```bash
python main.py --input path/to/directory/ --output results --visualize --report
```

### Graphical User Interface

Launch the GUI for interactive analysis:

```bash
python GUI/gui.py
```

The GUI provides intuitive access to:
- File selection and batch processing
- Visualization browsing by movie and type
- Feature extraction and exploration
- Comprehensive documentation

### Command Line Options

```
--input, -i         Path to ASC/CSV file or directory containing ASC/CSV files
--output, -o        Output directory for parsed data and visualizations (default: 'output')
--use_csv           Process CSV files instead of ASC files (faster loading of pre-processed data)
--visualize         Generate visualizations for each movie
--report            Generate comprehensive HTML visualization report
--screen_width      Screen width in pixels (default: 1280)
--screen_height     Screen height in pixels (default: 1024)
--no_features       Skip feature extraction
--unified_only      Only save unified eye metrics CSV files
```

## Directory Structure

The toolkit generates a structured output directory:

```
results/
├── timestamp/                      # Timestamped results folder
│   ├── data/                       # Data output
│   │   ├── Movie1Name/             # Movie-specific folder
│   │   │   ├── plots/              # Visualizations for Movie1
│   │   │   └── *_unified_eye_metrics.csv  # Eye metrics for Movie1
│   │   ├── Movie2Name/             # Movie-specific folder
│   │   │   ├── plots/              # Visualizations for Movie2
│   │   │   └── *_unified_eye_metrics.csv  # Eye metrics for Movie2
│   │   └── general/                # General data across all movies
│   │       ├── *_fixations_left.csv
│   │       ├── *_fixations_right.csv
│   │       ├── *_saccades_left.csv
│   │       └── ... (other CSV files)
│   ├── plots/                      # General visualization folder
│   │   └── report/                 # HTML report folder (if --report used)
│   │       ├── visualization_report.html
│   │       └── ... (report resources)
│   └── features/                   # Extracted features for ML/DL
│       ├── *_features.csv          # Individual features
│       └── combined_features.csv   # Combined features (for multiple files)
```

## Visualizations

The toolkit generates multiple visualization types for each movie:

### 1. Eye Movement Visualizations

- **Scanpaths**: Display the trajectory of eye movements with fixation markers
- **Interactive Animated Scanpaths**: Dynamic visualization with playback controls, timeline navigation, and adjustable display settings
- **Heatmaps**: Show gaze density across the screen for each eye
- **Fixation-Saccade Distribution**: Visualize fixations and connecting saccades

### 2. Temporal Visualizations

- **Pupil Size Timeseries**: Track pupil size changes throughout the movie
- **Pupil Size and Eye Events**: Show relationship between pupil size and different eye events
- **Frame-synchronized Playback**: Visualize eye movements in relation to movie frames

### 3. Statistical Distributions

- **Fixation Duration Distribution**: Histogram of fixation durations for both eyes
- **Saccade Amplitude Distribution**: Histogram of saccade amplitudes in visual degrees
- **Statistical Summary**: Automatically calculated metrics including means, medians, and ranges

### 4. Social Attention Analysis

- **Social vs. Non-social Attention**: Pie chart showing allocation of attention
- **Social Attention Timeline**: Shows changes in social attention over the course of the movie
- **Region of Interest (ROI) Analysis**: Quantify attention to specific social regions like faces and eyes

## Key Features for Autism Research

### Feature Extraction for Machine Learning

The toolkit extracts over 30 features specifically relevant to autism research:

- **Pupil Dynamics**: Size variations that may reflect differences in autonomic nervous system function
- **Gaze Patterns**: Variability and dispersion metrics that quantify scanning behavior
- **Fixation Characteristics**: Duration, count, and rate metrics that may indicate attentional differences
- **Saccade Properties**: Amplitude and velocity metrics that reflect visual search strategies
- **Head Movement**: Metrics that capture potential restlessness or compensation strategies

### Social Attention Analysis

The toolkit includes specialized analysis of attention to social versus non-social elements:

- Quantifies attention to faces, body parts, and other social regions
- Tracks changes in social attention over time
- Provides visualization of social attention patterns
- Supports research on social attention as a potential biomarker for ASD

## Working with Deep Learning

The features extracted by this toolkit are designed to be used with deep learning approaches:

1. **Extract Features**: Process ASC files to generate standardized features
2. **Combine Data**: Use the combined features CSV with your ML/DL framework
3. **Classification**: Train models to distinguish ASD from control groups
4. **Severity Prediction**: Use regression models to predict symptom severity

## Testing Framework

A comprehensive testing framework ensures reliable processing:

- **Automated Tests**: Verify correct parsing and feature extraction
- **Verification Tools**: Generate reports on processing accuracy
- **Visual Verification**: Create plots of eye movement data for validation

## GUI Features

The graphical user interface provides:

- **Intuitive Workflow**: Simple file selection and batch processing
- **Interactive Visualization**: Browse all plots by movie and type
- **Animated Scanpath Tab**: Dedicated interface for interactive eye movement animation
- **Animation Controls**: Play/pause, speed adjustment, timeline scrubbing, and custom display options
- **Feature Exploration**: Examine extracted features with detailed explanations and tooltips
- **Export Capabilities**: Save animations as MP4 or GIF files
- **Comprehensive Documentation**: Access detailed information about features and visualizations with research context

## Research Applications

This toolkit supports several key areas of autism research:

1. **Visual Attention in Movie Viewing**: Compare scanning patterns between ASD and control groups during socially relevant movie stimuli
2. **Biomarker Identification**: Extract consistent eye movement features that correlate with ASD diagnosis
3. **Severity Assessment**: Quantify relationships between eye movement patterns and symptom severity
4. **Longitudinal Analysis**: Track changes in visual attention over time or across interventions
5. **Deep Learning Integration**: Use extracted features for classification or regression models
6. **Social Attention Dynamics**: Analyze how attention to social stimuli changes throughout movie viewing
7. **ROI-Based Analysis**: Quantify attention to specific regions of interest in social scenes
8. **Multi-Movie Consistency**: Examine consistency of gaze patterns across different movie stimuli

## Advanced Usage Examples

### Extract Features Only

```bash
python main.py --input path/to/directory/ --output results --no_visualize
```

### Process Files and Generate Report

```bash
python main.py --input path/to/directory/ --output results --visualize --report
```

### Load Pre-processed CSV Files (Faster)

First, process your ASC files with the unified_only option to generate CSV files:

```bash
python main.py --input path/to/directory/ --output results --unified_only
```

Later, you can load these CSV files directly (much faster than re-processing ASC files):

```bash
python main.py --input path/to/results/data/ --output new_results --use_csv --visualize
```

### Using the Animated Scanpath Feature

1. Process your data through the GUI or command line
2. In the GUI, navigate to the "Animated Scanpath" tab
3. Select a movie from the dropdown menu
4. Use playback controls to view eye movements in real time
5. Adjust playback speed (0.25x to 4x)
6. Modify trail length and display options
7. Use the timeline slider to navigate to specific points
8. Click "Save Animation" to export as MP4 or GIF

## Technical Notes

- The parser is optimized for EyeLink 1000+ data but works with other EyeLink systems
- For accurate head movement calculation, the EyeLink must be configured to track corneal reflection
- Memory usage scales with file size; very large ASC files may require more RAM
- When visualizing frame data, each movie is processed separately to maintain context

## Troubleshooting

- **NaN Values in Visualizations**: Some plots may fail with NaN values; this typically indicates missing data in the ASC file
- **Missing Frames**: If frame information is absent, certain plots may not display frame markers
- **Social Attention Analysis**: Without manually defined AOIs, this plot uses simulated regions and should be interpreted accordingly

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SR Research for the EyeLink system documentation
- Professor Ohad Ben-Shahar for project guidance
- Department of Computer Science, Ben-Gurion University of the Negev