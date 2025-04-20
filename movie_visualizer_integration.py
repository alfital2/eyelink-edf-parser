"""
Movie Visualization Integration Module
Author: Tal Alfi
Date: April 2025
"""

import os
from typing import List, Tuple, Dict, Optional
import pandas as pd

from eyelink_visualizer import MovieEyeTrackingVisualizer


def generate_movie_visualizations(data_dir: str, 
                                 screen_size: Tuple[int, int] = (1280, 1024),
                                 specific_movies: Optional[List[str]] = None,
                                 participant_id: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Generate visualizations for all movie folders in the data directory.
    
    Args:
        data_dir: Directory containing movie data folders
        screen_size: Screen dimensions in pixels (width, height)
        specific_movies: List of specific movie folder names to process (if None, process all)
        participant_id: Optional participant ID to use as prefix in filenames
        
    Returns:
        Dictionary mapping movie names to lists of generated plot paths
    """
    # Initialize the visualizer
    visualizer = MovieEyeTrackingVisualizer(base_dir=data_dir, screen_size=screen_size)
    
    # Discover all movie folders
    movie_folders = visualizer.discover_movie_folders()
    print(f"Found {len(movie_folders)} movie folders in {data_dir}")
    
    # Filter to specific movies if requested
    if specific_movies:
        movie_folders = [folder for folder in movie_folders 
                        if os.path.basename(folder) in specific_movies]
        print(f"Filtered to {len(movie_folders)} specified movie folders")
    
    generated_plots = {}
    
    # Process each movie folder
    for movie_folder in movie_folders:
        # Load the movie data
        movie_name, data = visualizer.load_movie_data(movie_folder)
        
        if data.empty:
            print(f"No data found for movie: {movie_name}, skipping")
            continue
            
        print(f"Processing movie: {movie_name} with {len(data)} data points")
        
        # Ensure plots directory exists
        plots_dir = visualizer.ensure_plots_directory(movie_folder)
        
        # Create prefix from participant ID if provided
        prefix = f"{participant_id}_" if participant_id else ""
        
        # Generate all available visualizations
        plots = generate_all_plots(visualizer, data, plots_dir, prefix)
        
        # Store generated plots
        generated_plots[movie_name] = plots
        
    return generated_plots


def generate_all_plots(visualizer: MovieEyeTrackingVisualizer,
                       data: pd.DataFrame,
                       plots_dir: str,
                       prefix: str = "") -> List[str]:
    """
    Generate all available visualizations for a movie.

    Args:
        visualizer: MovieEyeTrackingVisualizer instance
        data: DataFrame with unified eye metrics
        plots_dir: Directory to save plots
        prefix: Prefix for plot filenames

    Returns:
        List of paths to generated plots
    """
    generated_plots = []

    # Track which plots we've generated
    plots_generated = {}

    # 1. Generate scanpath visualization
    try:
        print("Generating scanpath visualization...")
        visualizer.plot_scanpath(data, plots_dir, prefix)
        plot_path = os.path.join(plots_dir, f"{prefix}scanpath.png")
        if os.path.exists(plot_path):
            generated_plots.append(plot_path)
            plots_generated['scanpath'] = True
    except Exception as e:
        print(f"Error generating scanpath visualization: {e}")

    # 2. Generate heatmaps for both eyes
    for eye in ['left', 'right']:
        try:
            print(f"Generating {eye} eye heatmap...")
            visualizer.plot_heatmap(data, plots_dir, prefix, eye=eye)
            plot_path = os.path.join(plots_dir, f"{prefix}heatmap_{eye}.png")
            if os.path.exists(plot_path):
                generated_plots.append(plot_path)
                plots_generated[f'heatmap_{eye}'] = True
        except Exception as e:
            print(f"Error generating {eye} eye heatmap: {e}")

    # 3. Generate fixation duration distribution
    try:
        print("Generating fixation duration distribution...")
        visualizer.plot_fixation_duration_distribution(data, plots_dir, prefix)
        plot_path = os.path.join(plots_dir, f"{prefix}fixation_duration_distribution.png")
        if os.path.exists(plot_path):
            generated_plots.append(plot_path)
            plots_generated['fixation_duration_distribution'] = True
    except Exception as e:
        print(f"Error generating fixation duration distribution: {e}")

    # 4. Generate saccade amplitude distribution
    try:
        print("Generating saccade amplitude distribution...")
        visualizer.plot_saccade_amplitude_distribution(data, plots_dir, prefix)
        plot_path = os.path.join(plots_dir, f"{prefix}saccade_amplitude_distribution.png")
        if os.path.exists(plot_path):
            generated_plots.append(plot_path)
            plots_generated['saccade_amplitude_distribution'] = True
    except Exception as e:
        print(f"Error generating saccade amplitude distribution: {e}")

    # 5. Generate pupil size timeseries
    try:
        print("Generating pupil size timeseries...")
        visualizer.plot_pupil_size_timeseries(data, plots_dir, prefix)
        plot_path = os.path.join(plots_dir, f"{prefix}pupil_size_timeseries.png")
        events_path = os.path.join(plots_dir, f"{prefix}pupil_size_events.png")

        if os.path.exists(plot_path):
            generated_plots.append(plot_path)
            plots_generated['pupil_size_timeseries'] = True

        if os.path.exists(events_path):
            generated_plots.append(events_path)
            plots_generated['pupil_size_events'] = True
    except Exception as e:
        print(f"Error generating pupil size visualizations: {e}")

    # 6. Generate social attention analysis
    try:
        print("Generating social attention analysis...")
        visualizer.plot_social_attention_analysis(data, plots_dir, prefix)
        plot_path = os.path.join(plots_dir, f"{prefix}social_attention_analysis.png")
        timeline_path = os.path.join(plots_dir, f"{prefix}social_attention_timeline.png")

        if os.path.exists(plot_path):
            generated_plots.append(plot_path)
            plots_generated['social_attention_analysis'] = True

        if os.path.exists(timeline_path):
            generated_plots.append(timeline_path)
            plots_generated['social_attention_timeline'] = True
    except Exception as e:
        print(f"Error generating social attention analysis: {e}")

    # Print summary of generated plots
    print(f"Generated {len(generated_plots)} plots")
    print(f"Plots generated: {', '.join(plots_generated.keys())}")

    return generated_plots


# When adding new visualization methods, update this function too
def generate_specific_plot(visualizer: MovieEyeTrackingVisualizer,
                           data: pd.DataFrame,
                           plots_dir: str,
                           plot_type: str,
                           prefix: str = "",
                           **kwargs) -> Optional[str]:
    """
    Generate a specific type of visualization.

    Args:
        visualizer: MovieEyeTrackingVisualizer instance
        data: DataFrame with unified eye metrics
        plots_dir: Directory to save plots
        plot_type: Type of plot to generate (e.g., 'scanpath', 'heatmap_left')
        prefix: Prefix for plot filenames
        **kwargs: Additional arguments for the specific plot function

    Returns:
        Path to generated plot or None if failed
    """
    if data.empty:
        print(f"Cannot generate {plot_type} plot: Empty dataframe")
        return None

    try:
        if plot_type == 'scanpath':
            visualizer.plot_scanpath(data, plots_dir, prefix, **kwargs)
            return os.path.join(plots_dir, f"{prefix}scanpath.png")

        elif plot_type.startswith('heatmap_'):
            eye = plot_type.split('_')[1]
            visualizer.plot_heatmap(data, plots_dir, prefix, eye=eye, **kwargs)
            return os.path.join(plots_dir, f"{prefix}heatmap_{eye}.png")

        elif plot_type == 'fixation_duration_distribution':
            visualizer.plot_fixation_duration_distribution(data, plots_dir, prefix)
            return os.path.join(plots_dir, f"{prefix}fixation_duration_distribution.png")

        elif plot_type == 'saccade_amplitude_distribution':
            visualizer.plot_saccade_amplitude_distribution(data, plots_dir, prefix)
            return os.path.join(plots_dir, f"{prefix}saccade_amplitude_distribution.png")

        elif plot_type == 'pupil_size_timeseries':
            visualizer.plot_pupil_size_timeseries(data, plots_dir, prefix, **kwargs)
            return os.path.join(plots_dir, f"{prefix}pupil_size_timeseries.png")

        elif plot_type == 'social_attention_analysis':
            visualizer.plot_social_attention_analysis(data, plots_dir, prefix, **kwargs)
            return os.path.join(plots_dir, f"{prefix}social_attention_analysis.png")

        else:
            print(f"Unknown plot type: {plot_type}")
            return None

    except Exception as e:
        print(f"Error generating {plot_type} plot: {e}")
        return None