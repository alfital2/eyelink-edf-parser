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
- **Custom Aspect Ratio Support**: Analyze eye-tracking data with different screen resolutions and aspect ratios

## Installation

### Prerequisites

- Python 3.7 or newer
- pip (Python package installer)
- Git (for cloning the repository)

### Platform-Specific Setup

#### Windows

1. Open Command Prompt or PowerShell:
   ```
   git clone https://github.com/alfital2/eye-movement-analysis.git
   cd eye-movement-analysis
   pip install -r requirements.txt
   ```

2For PyQt5 issues on Windows:
   ```
   python -m pip install --upgrade pip
   python -m pip install PyQt5
   ```

#### macOS / Linux

1. Clone and install:
   ```
   git clone https://github.com/alfital2/eye-movement-analysis.git
   cd eye-movement-analysis
   pip3 install -r requirements.txt
   ```

2. For M1/M2 Macs (Apple Silicon), you may need:
   ```
   pip3 install matplotlib --no-binary matplotlib
   ```

### Virtual Environment (Recommended for All Platforms)

Using a virtual environment is recommended for all platforms to avoid package conflicts:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

### Updating Requirements

If you add new dependencies to the project, update the requirements.txt file:

```bash
pip3 freeze > requirements.txt
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

### Graphical User Interface (Recommanded)

Launch the GUI for interactive analysis:

```bash
# Default GUI launch
python main.py
```

For testing purposes:
```bash
python main.py --gui --test_mode --source_file path/to/test.asc --destination_folder path/to/output
```

The GUI provides intuitive access to:
- File selection and batch processing
- Visualization browsing by movie and type
- Feature extraction and exploration
- Comprehensive documentation
- Screen resolution/aspect ratio selection

### Command Line Options

```
# General options
--gui                Start the graphical user interface
--test_mode          Run GUI in test mode with predefined files
--source_file        Path to source file (.asc or .csv) for GUI test mode
--destination_folder Output folder for GUI test mode

# Processing options
--input, -i          Path to ASC/CSV file or directory containing ASC/CSV files
--output, -o         Output directory for parsed data and visualizations (default: 'output')
--use_csv            Process CSV files instead of ASC files (faster loading of pre-processed data)
--visualize          Generate visualizations for each movie
--report             Generate comprehensive HTML visualization report
--screen_width       Screen width in pixels (default: 1280)
--screen_height      Screen height in pixels (default: 1024)
--no_features        Skip feature extraction
--unified_only       Only save unified eye metrics CSV files
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

The toolkit generates multiple visualization types for each movie (if exist in the ASC file):

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

The toolkit extracts features relevant to  research:

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

## Screen Resolution and Aspect Ratio Support

The toolkit now provides comprehensive support for different screen resolutions:

- **Configurable Aspect Ratios**: Choose from common screen resolutions (1280x1024, 1920x1080, etc.)
- **Persistent Settings**: Selected aspect ratio is saved between sessions
- **Dynamic Visualization Updates**: Visualizations automatically adjust to the selected screen dimensions
- **Real-time Updates**: Change aspect ratio at any time and see immediate updates to visualizations

This feature allows researchers to analyze data collected on different screen setups, ensuring accurate visualization across various experimental configurations.

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
- **Isolated Test Environment**: Proper test isolation prevents interference between tests

To run the tests:

```bash
# Windows
python -m unittest discover tests

# macOS/Linux
python3 -m unittest discover tests

# Run specific test file
python -m unittest tests/test_parser.py
```

## GUI Features

The graphical user interface provides:

- **Intuitive Workflow**: Simple file selection and batch processing
- **Interactive Visualization**: Browse all plots by movie and type
- **Animated Scanpath Tab**: Dedicated interface for interactive eye movement animation
- **Animation Controls**: Play/pause, speed adjustment, timeline scrubbing, and custom display options
- **Feature Exploration**: Examine extracted features with detailed explanations and tooltips
- **Export Capabilities**: Save animations as MP4 or GIF files
- **Comprehensive Documentation**: Access detailed information about features and visualizations with research context
- **Screen Resolution Selection**: Easily switch between different aspect ratios for analysis

## Advanced Usage Examples

### Extract Features Only

```bash
python main.py --input path/to/directory/ --output results --no_features
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

#### Using CSV Files With The GUI

For an even more convenient workflow with pre-processed data:

1. Launch the GUI: `python main.py`
2. Select "CSV Files" from the dropdown in the Data Processing tab
3. Click "Select File(s)" and choose your unified_eye_metrics*.csv files
4. Set an output directory and processing options
5. Click "Process Data" to quickly load and visualize the pre-processed data

This approach significantly speeds up the workflow when you've already processed ASC files
and want to revisit visualizations or perform additional analysis.

### Using the Animated Scanpath Feature

1. Process your data through the GUI or command line
2. In the GUI, navigate to the Results & Visualization tab
3. Select a movie from the dropdown menu
4. Select "Animated Scanpath" from the Visualization Type dropdown
5. Use playback controls to view eye movements in real time
6. Adjust playback speed (0.25x to 4x)
7. Modify trail length and display options
8. Use the timeline slider to navigate to specific points
9. Save the animation if desired

### Setting Custom Screen Dimensions

Specify screen dimensions for properly scaled visualizations:

```bash
python main.py --input path/to/file.asc --output results --visualize --screen_width 1920 --screen_height 1080
```

Or use the GUI's aspect ratio selector for common resolutions:

1. Launch the GUI: `python main.py`
2. In the Data Processing tab, select your preferred aspect ratio
3. Your choice is saved for future sessions
4. All visualizations dynamically adjust to the selected dimensions

## Technical Notes

- The parser is optimized for EyeLink 1000+ data but works with other EyeLink systems
- For accurate head movement calculation, the EyeLink must be configured to track corneal reflection with head marker
- Memory usage scales with file size; very large ASC files may require more RAM
- When visualizing frame data, each movie is processed separately to maintain context
- Screen resolution settings are persistent across sessions through a settings manager
- The application uses PyQt5 for the graphical interface

## Troubleshooting

### General Issues

- **NaN Values in Visualizations**: Some plots may fail with NaN values; this typically indicates missing data in the ASC file
- **Missing Frames**: If frame information is absent, certain plots may not display frame markers
- **Social Attention Analysis**: Without manually defined AOIs, this plot uses simulated regions and should be interpreted accordingly
- **Aspect Ratio Issues**: If visualizations appear distorted, check that the correct screen resolution is selected in the GUI settings
- **Memory Errors**: For large ASC files, try:
  ```bash
  # Increase Python's memory limit (adjust value as needed)
  export PYTHONMEM=4G  # For Unix/macOS
  set PYTHONMEM=4G     # For Windows (Command Prompt)
  ```
- **Processing Speed**: For faster processing of multiple files, use the `--unified_only` flag first, then process the resulting CSV files with `--use_csv`

## Contributing

Contributions to this project are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add some feature'`
4. Push to your branch: `git push origin feature-name`
5. Submit a pull request

When adding new dependencies, remember to update the requirements.txt file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SR Research for the EyeLink system documentation
- Professor Ohad Ben-Shahar for project guidance
- Department of Computer Science, Ben-Gurion University of the Negev