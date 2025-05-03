"""
Shared test fixtures and utility functions for ASD Analysis tests
Author: Tal Alfi
Date: May 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Common test data generation function
def create_sample_eye_data(num_samples=10, include_normalized=True):
    """
    Create sample eye tracking data for tests.
    
    Args:
        num_samples: Number of samples to generate
        include_normalized: Whether to include normalized coordinates
        
    Returns:
        DataFrame with sample eye tracking data
    """
    # Base timestamps and frame numbers
    timestamps = list(range(1000, 1000 + num_samples * 10, 10))
    frame_numbers = [i // 2 + 1 for i in range(num_samples)]
    
    # Eye positions
    x_left = [256 + i * 10 for i in range(num_samples)]
    y_left = [200 + i * 5 for i in range(num_samples)]
    x_right = [830 + i * 5 for i in range(num_samples)]
    y_right = [600 + i * 5 for i in range(num_samples)]
    
    # Pupil sizes
    pupil_left = [1200 + i * 10 for i in range(num_samples)]
    pupil_right = [1100 + i * 10 for i in range(num_samples)]
    
    # Fixation flags (alternating)
    is_fixation_left = [i % 2 == 0 for i in range(num_samples)]
    is_fixation_right = [i % 2 == 0 for i in range(num_samples)]
    
    # Create data frame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'frame_number': frame_numbers,
        'x_left': x_left,
        'y_left': y_left,
        'x_right': x_right,
        'y_right': y_right,
        'pupil_left': pupil_left,
        'pupil_right': pupil_right,
        'is_fixation_left': is_fixation_left,
        'is_fixation_right': is_fixation_right,
    })
    
    # Add normalized coordinates if requested
    if include_normalized:
        screen_width, screen_height = 1280, 1024
        data['x_left_norm'] = data['x_left'] / screen_width
        data['y_left_norm'] = data['y_left'] / screen_height
        data['x_right_norm'] = data['x_right'] / screen_width
        data['y_right_norm'] = data['y_right'] / screen_height
    
    return data


# Common test directory setup function
def create_test_directories():
    """
    Create a temporary directory structure for tests.
    
    Returns:
        A tuple of (temp_dir, base_dir, movie1_folder, movie2_folder, movie1_plots_dir, movie2_plots_dir)
    """
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    base_dir = temp_dir.name
    
    # Create movie folders
    movie1_folder = os.path.join(base_dir, "test_movie1")
    movie2_folder = os.path.join(base_dir, "test_movie2")
    os.makedirs(movie1_folder, exist_ok=True)
    os.makedirs(movie2_folder, exist_ok=True)
    
    # Create plots directories
    movie1_plots_dir = os.path.join(movie1_folder, "plots")
    movie2_plots_dir = os.path.join(movie2_folder, "plots")
    os.makedirs(movie1_plots_dir, exist_ok=True)
    os.makedirs(movie2_plots_dir, exist_ok=True)
    
    return (temp_dir, base_dir, movie1_folder, movie2_folder, movie1_plots_dir, movie2_plots_dir)


# Mock classes for external dependencies

# Mock MovieEyeTrackingVisualizer for testing without seaborn/matplotlib
class MockMovieEyeTrackingVisualizer:
    """Mock implementation of MovieEyeTrackingVisualizer for testing."""
    
    def __init__(self, base_dir, screen_size=(1280, 1024), dpi=150):
        self.base_dir = base_dir
        self.screen_width, self.screen_height = screen_size
        self.dpi = dpi
        self.colors = {
            'left_eye': '#1f77b4',  # Blue
            'right_eye': '#ff7f0e',  # Orange
            'fixation': '#2ca02c',  # Green
            'saccade': '#d62728',  # Red
            'blink': '#9467bd',  # Purple
            'head_movement': '#8c564b',  # Brown
            'asd': '#d62728',  # Red for ASD group
            'control': '#1f77b4'  # Blue for control group
        }
        self.default_figsize = (12, 8)
    
    def discover_movie_folders(self):
        """Find movie folders in base_dir."""
        return discover_movie_folders_func(self.base_dir)
    
    def load_movie_data(self, movie_folder):
        """Load eye tracking data for a movie."""
        return load_movie_data_func(movie_folder)
    
    def ensure_plots_directory(self, movie_folder):
        """Ensure the plots directory exists for a movie."""
        return ensure_plots_directory_func(movie_folder)
    
    def plot_scanpath(self, *args, **kwargs):
        """Mock implementation for plot_scanpath."""
        plot_scanpath_func(*args, **kwargs)
    
    def plot_heatmap(self, *args, **kwargs):
        """Mock implementation for plot_heatmap."""
        plot_heatmap_func(*args, **kwargs)
    
    def plot_fixation_duration_distribution(self, *args, **kwargs):
        """Mock implementation for plot_fixation_duration_distribution."""
        plot_fixation_duration_distribution_func(*args, **kwargs)
    
    def plot_saccade_amplitude_distribution(self, *args, **kwargs):
        """Mock implementation for plot_saccade_amplitude_distribution."""
        plot_saccade_amplitude_distribution_func(*args, **kwargs)
    
    def plot_pupil_size_timeseries(self, *args, **kwargs):
        """Mock implementation for plot_pupil_size_timeseries."""
        plot_pupil_size_timeseries_func(*args, **kwargs)
    
    def plot_social_attention_analysis(self, *args, **kwargs):
        """Mock implementation for plot_social_attention_analysis."""
        plot_social_attention_analysis_func(*args, **kwargs)


# Standalone functions for the mock class
def discover_movie_folders_func(base_dir):
    """Find movie folders in base_dir."""
    # Return movie folders that end with _movie or have movie in name
    movie_folders = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if "movie" in dir_name.lower():
                movie_folders.append(os.path.join(root, dir_name))
        break  # Only look at top level
    return movie_folders

def load_movie_data_func(movie_folder):
    """Load eye tracking data for a movie."""
    movie_name = os.path.basename(movie_folder)
    # Create sample data
    data = pd.DataFrame({
        'timestamp': range(10),
        'x_left': range(10),
        'y_left': range(10),
    })
    return movie_name, data

def ensure_plots_directory_func(movie_folder):
    """Ensure the plots directory exists for a movie."""
    plots_dir = os.path.join(movie_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_scanpath_func(*args, **kwargs):
    """Mock implementation for plot_scanpath."""
    print("Generating scanpath visualization...")

def plot_heatmap_func(*args, **kwargs):
    """Mock implementation for plot_heatmap."""
    eye = kwargs.get('eye', 'left')
    print(f"Generating {eye} eye heatmap...")

def plot_fixation_duration_distribution_func(*args, **kwargs):
    """Mock implementation for plot_fixation_duration_distribution."""
    print("Generating fixation duration distribution...")

def plot_saccade_amplitude_distribution_func(*args, **kwargs):
    """Mock implementation for plot_saccade_amplitude_distribution."""
    print("Generating saccade amplitude distribution...")

def plot_pupil_size_timeseries_func(*args, **kwargs):
    """Mock implementation for plot_pupil_size_timeseries."""
    print("Generating pupil size timeseries...")

def plot_social_attention_analysis_func(*args, **kwargs):
    """Mock implementation for plot_social_attention_analysis."""
    print("Generating social attention analysis...")


# Create a README.md file for the tests directory
if not os.path.exists(os.path.join(Path(__file__).parent, 'README.md')):
    with open(os.path.join(Path(__file__).parent, 'README.md'), 'w') as f:
        f.write("""# ASD Analysis Tests

This directory contains tests for the ASD Analysis project.

## Running Tests

You can run all tests with:

```bash
python3 tests/run_all.py
```

Or run individual test modules:

```bash
python3 -m unittest tests.test_parser
python3 -m unittest tests.test_roi_manager
```

## Test Organization

- `test_parser.py`: Tests for the EyeLink ASC parser
- `test_roi_manager.py`: Tests for the ROI manager
- `test_roi_integration.py`: Tests for ROI integration functionality
- `test_animated_scanpath.py`: Tests for animated scanpath visualization
- `test_animated_roi_scanpath.py`: Tests for animated ROI scanpath visualization
- `test_eyelink_visualizer.py`: Tests for the EyeLink visualizer
- `test_movie_visualizer_integration.py`: Tests for movie visualizer integration

## Test Data

Test data is stored in the following locations:

- `test_data/`: Contains test data files like ROI definitions
- `asc_files/`: Contains sample ASC files for testing the parser

## Helper Files

- `conftest.py`: Contains shared test fixtures and utilities
- `run_all.py`: Script to run all tests at once
""")