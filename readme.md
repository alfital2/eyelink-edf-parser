# Eye Movement Analysis for Autism Classification

A comprehensive toolkit for parsing, analyzing, and visualizing eye-tracking data to support autism research with movie-specific analysis capabilities.

## Overview

This project provides a specialized set of tools for processing eye movement data recorded using SR Research EyeLink systems. The toolkit is specifically designed for autism spectrum disorder (ASD) research, focusing on characterizing visual attention patterns during movie viewing. It enables researchers to compare gaze behavior between individuals with ASD and neurotypical controls across multiple movie stimuli.

## Key Features

- **Movie-Specific Analysis**: Processes and visualizes eye-tracking data separately for each movie stimulus
- **Hierarchical Organization**: Maintains clear directory structure with movie-specific visualizations
- **Comprehensive Data Extraction**: Parses raw EyeLink ASC files into structured data formats
- **Rich Visualization Library**: Generates multiple visualization types revealing different aspects of visual attention
- **ML Feature Extraction**: Computes standardized features for machine learning classification

## Installation

```bash
git clone https://github.com/username/eye-movement-analysis.git
cd eye-movement-analysis
pip install -r requirements.txt
```

## Usage Guide

### Basic Usage

Process a single ASC file with visualizations:

```bash
python main.py --input path/to/file.asc --output results --visualize
```

### Processing Multiple Files

Process an entire directory of ASC files:

```bash
python main.py --input path/to/directory/ --output results --visualize
```

### Generate HTML Report

Process files and create a comprehensive HTML report:

```bash
python main.py --input path/to/directory/ --output results --visualize --report
```

### Command Line Options

```
--input, -i         Path to ASC file or directory containing ASC files
--output, -o        Output directory for parsed data and visualizations (default: 'output')
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
│   └── features/                   # Extracted features for ML
│       ├── *_features.csv          # Individual features
│       └── combined_features.csv   # Combined features (for multiple files)
```

## Visualizations

The toolkit generates multiple visualization types for each movie:

### 1. Eye Movement Visualizations

- **Heatmaps**: Show gaze density across the screen for each eye
- **Scanpaths**: Display the trajectory of eye movements with fixation markers
- **Fixation-Saccade Distribution**: Visualize fixations and connecting saccades

### 2. Temporal Visualizations

- **Pupil Size Timeseries**: Track pupil size changes throughout the movie
- **Pupil Size and Eye Events**: Show relationship between pupil size and different eye events

### 3. Statistical Distributions

- **Fixation Duration Distribution**: Histogram of fixation durations for both eyes
- **Saccade Amplitude Distribution**: Histogram of saccade amplitudes in visual degrees

### 4. Social Attention Analysis

- **Social vs. Non-social Attention**: Pie chart showing allocation of attention (note: requires manual AOI data)
- **Social Attention Timeline**: Shows changes in social attention over the course of the movie

## Working with Social Attention Analysis

The Social Attention Analysis visualization requires manually defined Areas of Interest (AOIs):

1. **Default Behavior**: Without AOI data, the visualizer creates simulated AOIs for demonstration purposes
2. **For Research Use**: You should provide real AOI data in the following format:

```python
# Format for ROI data: Dict mapping frame numbers to lists of ROIs
roi_data = {
    1: [(x1, y1, width1, height1, 'face'), (x2, y2, width2, height2, 'eyes')],
    2: [(x1, y1, width1, height1, 'face'), (x2, y2, width2, height2, 'eyes')],
    # ... more frames
}

# Then pass to the visualization function:
visualizer.plot_social_attention_analysis(data, plots_dir, prefix, roi_data=roi_data)
```

## Data Files Generated

### 1. Unified Eye Metrics CSV

Primary data file containing integrated eye-tracking metrics:
- Eye positions, pupil sizes, velocity measures
- Event flags (fixations, saccades, blinks)
- Frame information

### 2. Features CSV

Aggregated features relevant for machine learning:
- Pupil metrics (mean, std, min, max)
- Gaze variability measures
- Fixation, saccade, and blink statistics
- Head movement metrics

## Research Applications

This toolkit supports several key areas of autism research:

1. **Visual Attention in Movie Viewing**: Compare scanning patterns between ASD and control groups during socially relevant movie stimuli
2. **Biomarker Identification**: Extract consistent eye movement features that correlate with ASD diagnosis
3. **Severity Assessment**: Quantify relationships between eye movement patterns and symptom severity
4. **Longitudinal Analysis**: Track changes in visual attention over time or across interventions
5. **Machine Learning Integration**: Use extracted features for classification or regression models

## Advanced Usage Examples

### Extract Features Only

```bash
python main.py --input path/to/directory/ --output results --no_visualize
```

### Process Files and Generate Report

```bash
python main.py --input path/to/directory/ --output results --visualize --report
```

## Technical Notes

- The parser is optimized for EyeLink 1000+ data but works with other EyeLink systems
- For accurate head movement calculation, the EyeLink must be configured to track corneal reflection
- Memory usage scales with file size; very large ASC files may require more RAM
- When visualizing frame data, each movie is processed separately to maintain context

## Troubleshooting

- **NaN Values in Visualizations**: Some plots may fail with NaN values; this typically indicates missing data in the ASC file
- **Missing Frames**: If frame information is absent, certain plots may not display frame markers
- **Social Attention Analysis**: Without manually defined AOIs, this plot uses simulated regions and should be interpreted accordingly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SR Research for the EyeLink system documentation
- Professor Ohad Ben-Shahar for project guidance
- Department of Computer Science, Ben-Gurion University of the Negev