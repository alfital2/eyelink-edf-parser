"""
Visualization Helper Module

Provides utility functions for working with visualization paths, names, and directories.
"""

import os
import pandas as pd


class VisualizationHelper:
    """
    Helper class for visualization-related operations including path management
    and display name generation.
    """
    
    # Mapping of filename patterns to friendly display names
    ROI_NAME_MAPPING = {
        "roi_fixation_sequence": "ROI Fixation Sequence",
        "roi_transition_matrix": "ROI Transition Matrix",
        "roi_first_fixation_latency": "ROI First Fixation Latency",
        "roi_fixation_duration_distribution": "ROI Fixation Duration Distribution",
        "roi_temporal_heatmap": "ROI Temporal Heatmap",
        "social_attention_roi_time": "ROI Attention Time"
    }
    
    @staticmethod
    def get_display_name_from_path(path):
        """Convert a file path to a user-friendly display name."""
        # Extract the base filename
        basename = os.path.basename(path)
        
        # Skip hidden files
        if basename.startswith('.'):
            return None
            
        # Check for ROI-specific names first
        for pattern, display_name in VisualizationHelper.ROI_NAME_MAPPING.items():
            if pattern in basename.lower():
                return display_name
                
        # General processing for other files
        # Remove participant prefix if present
        parts = basename.split('_')
        if len(parts) > 1:
            # If there's a prefix, join everything after the first underscore
            cleaned_name = '_'.join(parts[1:])
        else:
            cleaned_name = basename

        # Remove extension for display name
        display_name = os.path.splitext(cleaned_name)[0]

        # Convert to friendly display name
        return display_name.replace('_', ' ').title()
        
    @staticmethod
    def get_movie_data(movie_name, output_dir, visualization_results):
        """
        Find and load data for a specific movie.
        
        Args:
            movie_name: Name of the movie to find data for
            output_dir: Base output directory
            visualization_results: Dictionary of visualization results
            
        Returns:
            Dictionary with data, data_path, and data_dir if found, None otherwise
        """
        data_dir = None
        data_path = None

        # First check output directory data folder
        if output_dir:
            data_dir = os.path.join(output_dir, 'data')
            if os.path.exists(data_dir):
                # Try to find exact match for the movie
                data_path = VisualizationHelper._find_data_file(data_dir, movie_name)
                
        # If not found, try to extract from visualization paths
        if not data_path and movie_name in visualization_results:
            for paths in visualization_results[movie_name].values():
                if paths:  # If there's at least one path
                    # Get directory containing the visualization
                    viz_dir = os.path.dirname(paths[0])
                    # The movie directory is usually the parent of the plots directory
                    potential_data_dir = os.path.dirname(viz_dir)
                    if os.path.exists(potential_data_dir):
                        data_dir = potential_data_dir
                        data_path = VisualizationHelper._find_data_file(data_dir, movie_name)
                        if data_path:
                            break

        # Load the data if found
        if data_path and os.path.exists(data_path):
            data = pd.read_csv(data_path)
            
            # Check if necessary columns are present
            required_cols = ['timestamp', 'x_left', 'y_left', 'x_right', 'y_right']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols or data.empty:
                return None
            else:
                return {"data": data, "data_path": data_path, "data_dir": data_dir}
        
        return None
        
    @staticmethod
    def _find_data_file(dir_path, movie_name=None):
        """Helper to find a data file in the given directory."""
        # First look for exact match with movie name
        if movie_name:
            for file in os.listdir(dir_path):
                if 'unified_eye_metrics' in file and movie_name in file and file.endswith('.csv'):
                    return os.path.join(dir_path, file)
        
        # Fall back to any unified_eye_metrics file
        for file in os.listdir(dir_path):
            if 'unified_eye_metrics' in file and file.endswith('.csv'):
                return os.path.join(dir_path, file)
                
        return None
        
    @staticmethod
    def get_plots_directory(movie_name, output_dir, movie_data=None):
        """Get the appropriate plots directory for the given movie."""
        # First try using movie data
        if movie_data and "data_dir" in movie_data:
            plots_dir = os.path.join(movie_data["data_dir"], 'plots')
            if os.path.exists(plots_dir):
                return plots_dir
                
        # Then try output directory structure
        if output_dir and os.path.exists(output_dir):
            # Try data/plots subdirectory
            data_dir = os.path.join(output_dir, 'data')
            if os.path.exists(data_dir):
                plots_dir = os.path.join(data_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                return plots_dir
            
            # Fall back to main plots directory
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            return plots_dir
            
        # Last resort fallback
        default_plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
        os.makedirs(default_plots_dir, exist_ok=True)
        return default_plots_dir