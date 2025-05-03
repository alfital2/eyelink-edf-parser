"""
Movie-specific Eye Tracking Visualizer for Autism Research
Author: Tal Alfi
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import glob

from matplotlib import animation


class MovieEyeTrackingVisualizer:
    """
    A class for generating visualizations from movie-specific eye tracking data.

    This visualizer creates various plots to help understand eye movement patterns
    during specific movie stimuli, which is particularly useful for autism research
    where differences in visual attention to social stimuli may be apparent.
    """

    def __init__(self,
                 base_dir: str,
                 screen_size: Tuple[int, int] = (1280, 1024),
                 dpi: int = 150):
        """
        Initialize the visualizer with the base directory structure.

        Args:
            base_dir: Base directory containing movie folders
            screen_size: Screen dimensions in pixels (width, height)
            dpi: Resolution for saved plots
        """
        self.base_dir = base_dir
        self.screen_width, self.screen_height = screen_size
        self.dpi = dpi

        # Default plot style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.default_figsize = (12, 8)

        # Color scheme
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

    def discover_movie_folders(self):
        """
        Discover all movie folders in the base directory.

        Returns:
            List of paths to movie folders
        """
        # Look for folders that contain unified_eye_metrics CSV files
        movie_folders = []

        # Pattern: look for folders with unified_eye_metrics in their files
        for root, _, files in os.walk(self.base_dir):
            if any("unified_eye_metrics" in f for f in files if f.endswith('.csv')):
                movie_folders.append(root)

        return movie_folders

    def load_movie_data(self, movie_folder: str) -> Tuple[str, pd.DataFrame]:
        """
        Load the unified eye metrics data for a specific movie.

        Args:
            movie_folder: Path to the movie folder

        Returns:
            Tuple of (movie_name, unified_metrics_df)
        """
        # Find the unified eye metrics CSV file
        csv_files = glob.glob(os.path.join(movie_folder, '*unified_eye_metrics*.csv'))

        if not csv_files:
            print(f"No unified eye metrics CSV found in {movie_folder}")
            return os.path.basename(movie_folder), pd.DataFrame()

        # Load the first matching CSV file
        data = pd.read_csv(csv_files[0])

        # Extract movie name from folder
        movie_name = os.path.basename(movie_folder)

        return movie_name, data

    def ensure_plots_directory(self, movie_folder: str) -> str:
        """
        Ensure the plots directory exists for the given movie folder.

        Args:
            movie_folder: Path to the movie folder

        Returns:
            Path to the plots directory
        """
        plots_dir = os.path.join(movie_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        return plots_dir

    def save_plot(self, plots_dir: str, plot_name: str, fig=None, tight=True):
        """
        Save plot to the movie's plots directory.

        Args:
            plots_dir: Path to the plots directory
            plot_name: Name of the plot file (without extension)
            fig: Matplotlib figure object
            tight: Whether to use tight layout
        """
        if fig is None:
            fig = plt.gcf()

        # Create full path
        full_path = os.path.join(plots_dir, f"{plot_name}.png")

        if tight:
            plt.tight_layout()

        fig.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved plot to {full_path}")
        plt.close(fig)
        
    def _validate_plot_data(self, data: pd.DataFrame, required_columns=None, error_message=None):
        """Common data validation for plotting functions
        
        Args:
            data: DataFrame with eye tracking metrics
            required_columns: List of required columns
            error_message: Custom error message if data is invalid
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            print(error_message or "Empty dataframe, cannot generate plot.")
            return False
            
        if required_columns:
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                print(f"Missing required columns: {', '.join(missing_cols)}")
                return False
                
        return True
        
    def _create_plot_filename(self, prefix, base_name, **kwargs):
        """Create standardized filename for plots with consistent naming
        
        Args:
            prefix: Prefix for the filename
            base_name: Base name of the plot type
            **kwargs: Additional parts to add to the filename
            
        Returns:
            Formatted filename string
        """
        filename = f"{prefix}{base_name}"
        
        # Add any additional parts in a consistent order
        for key, value in sorted(kwargs.items()):
            if value is not None:
                if isinstance(value, tuple) and len(value) == 2:
                    filename += f"_{value[0]}_{value[1]}"
                else:
                    filename += f"_{key}_{value}"
                    
        return filename

    def plot_scanpath(self,
                      data: pd.DataFrame,
                      plots_dir: str,
                      prefix: str = '',
                      time_window: Optional[Tuple[int, int]] = None,
                      max_points: int = 5000,
                      alpha: float = 0.7):
        """
        Plot the scanpath (eye movement trajectory) for both eyes during a movie.

        This visualization shows how viewers scan the scene over time, which can reveal
        differences in visual attention patterns between ASD and control groups.

        Args:
            data: DataFrame with unified eye metrics
            plots_dir: Path to save the plot
            prefix: Prefix for the plot filename
            time_window: Optional tuple with (start_time, end_time) to plot a specific segment
            max_points: Maximum number of points to plot (for performance)
            alpha: Transparency of the points
        """
        # Validate input data
        required_columns = ['x_left', 'y_left', 'x_right', 'y_right']
        if not self._validate_plot_data(data, required_columns, "Empty dataframe, cannot plot scanpath."):
            return

        # Apply time window if specified
        if time_window:
            start_time, end_time = time_window
            df = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
        else:
            df = data

        # Subsample if necessary
        if len(df) > max_points:
            df = df.iloc[::len(df) // max_points]

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Plot left eye scanpath
        if 'x_left' in df.columns and 'y_left' in df.columns:
            ax.plot(df['x_left'], df['y_left'], 'o-', color=self.colors['left_eye'],
                    markersize=2, linewidth=0.5, alpha=alpha, label='Left Eye')

            # Mark fixations if available
            if 'is_fixation_left' in df.columns:
                fixations = df[df['is_fixation_left']]
                ax.scatter(fixations['x_left'], fixations['y_left'],
                           s=30, color=self.colors['fixation'], alpha=0.8, label='Left Fixations')

        # Plot right eye scanpath
        if 'x_right' in df.columns and 'y_right' in df.columns:
            ax.plot(df['x_right'], df['y_right'], 'o-', color=self.colors['right_eye'],
                    markersize=2, linewidth=0.5, alpha=alpha, label='Right Eye')

            # Mark fixations if available
            if 'is_fixation_right' in df.columns:
                fixations = df[df['is_fixation_right']]
                ax.scatter(fixations['x_right'], fixations['y_right'],
                           s=30, color=self.colors['fixation'], alpha=0.8, label='Right Fixations')

        if 'frame_number' in df.columns:
            unique_frames = df['frame_number'].unique()
            if len(unique_frames) > 1:
                valid_frames = [f for f in unique_frames if not pd.isna(f)]
                if valid_frames:
                    min_frame = int(min(valid_frames))
                    max_frame = int(max(valid_frames))

                    # Now use these values in the annotation:
                    ax.annotate(f"Frames: {min_frame} - {max_frame}",
                                xy=(0.02, 0.02), xycoords='axes fraction',
                                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Set axis limits to screen dimensions
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)  # Invert y-axis to match screen coordinates

        # Add annotations
        if time_window:
            title = f'Eye Movement Scanpath ({time_window[0] / 1000:.1f}s - {time_window[1] / 1000:.1f}s)'
        else:
            title = 'Eye Movement Scanpath (Full Movie)'

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        ax.legend(loc='best')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Create filename and save the plot
        filename = self._create_plot_filename(prefix, 'scanpath', time_window=time_window)
        self.save_plot(plots_dir, filename, fig)

    def generate_animated_scanpath(self,
                                   data: pd.DataFrame,
                                   plots_dir: str,
                                   prefix: str = '',
                                   max_points: int = 5000,
                                   fps: int = 30,
                                   trail_length: int = 100,
                                   save_path: str = None):
        """
        WARNING: This method appears to be unused in the current codebase.
        Consider using animated_scanpath.py for animated visualizations.
        """
        """
        Generate an animated version of the scanpath visualization.

        Args:
            data: DataFrame with unified eye metrics
            plots_dir: Path to save the animation
            prefix: Prefix for the filename
            max_points: Maximum number of points to include (for performance)
            fps: Frames per second for the animation
            trail_length: Length of the trailing path behind the current point
            save_path: Optional path to save the animation as a video file

        Returns:
            Path to the generated animation file or None if not saved
        """
        if data.empty:
            print("Empty dataframe, cannot generate animated scanpath.")
            return None

        # Subsample if necessary for performance
        if len(data) > max_points:
            step = len(data) // max_points
            df = data.iloc[::step].copy()
        else:
            df = data.copy()

        # Convert timestamp to seconds from start for better visualization
        df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Set axis limits
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)  # Invert y-axis

        # Add title and labels
        ax.set_title('Animated Eye Movement Scanpath', fontsize=16)
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Create line objects for animation
        left_line, = ax.plot([], [], 'o-', color=self.colors['left_eye'],
                             markersize=2, linewidth=0.5, alpha=0.7,
                             label='Left Eye')

        right_line, = ax.plot([], [], 'o-', color=self.colors['right_eye'],
                              markersize=2, linewidth=0.5, alpha=0.7,
                              label='Right Eye')

        left_point, = ax.plot([], [], 'o', color=self.colors['left_eye'],
                              markersize=8, alpha=1.0)

        right_point, = ax.plot([], [], 'o', color=self.colors['right_eye'],
                               markersize=8, alpha=1.0)

        # Add legend
        ax.legend(loc='upper right')

        # Text objects for time and frame display
        time_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        frame_text = ax.text(0.02, 0.08, '', transform=ax.transAxes, fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Animation initialization function
        def init():
            left_line.set_data([], [])
            right_line.set_data([], [])
            left_point.set_data([], [])
            right_point.set_data([], [])
            time_text.set_text('')
            frame_text.set_text('')
            return left_line, right_line, left_point, right_point, time_text, frame_text

        # Animation update function
        def update(frame_idx):
            # Calculate trail start index
            start_idx = max(0, frame_idx - trail_length)

            # Get slice of data for trail
            trail_data = df.iloc[start_idx:frame_idx + 1]

            # Update left eye trail and current position
            if 'x_left' in df.columns and 'y_left' in df.columns:
                x_left = trail_data['x_left'].values
                y_left = trail_data['y_left'].values

                # Handle NaN values
                mask_left = ~(np.isnan(x_left) | np.isnan(y_left))
                if any(mask_left):
                    left_line.set_data(x_left[mask_left], y_left[mask_left])

                    # Update current position point
                    current_x_left = df.iloc[frame_idx]['x_left']
                    current_y_left = df.iloc[frame_idx]['y_left']

                    if not (np.isnan(current_x_left) or np.isnan(current_y_left)):
                        left_point.set_data([current_x_left], [current_y_left])
                    else:
                        left_point.set_data([], [])
                else:
                    left_line.set_data([], [])
                    left_point.set_data([], [])

            # Update right eye trail and current position
            if 'x_right' in df.columns and 'y_right' in df.columns:
                x_right = trail_data['x_right'].values
                y_right = trail_data['y_right'].values

                # Handle NaN values
                mask_right = ~(np.isnan(x_right) | np.isnan(y_right))
                if any(mask_right):
                    right_line.set_data(x_right[mask_right], y_right[mask_right])

                    # Update current position point
                    current_x_right = df.iloc[frame_idx]['x_right']
                    current_y_right = df.iloc[frame_idx]['y_right']

                    if not (np.isnan(current_x_right) or np.isnan(current_y_right)):
                        right_point.set_data([current_x_right], [current_y_right])
                    else:
                        right_point.set_data([], [])
                else:
                    right_line.set_data([], [])
                    right_point.set_data([], [])

            # Update time text
            time_sec = df.iloc[frame_idx]['time_sec']
            total_duration = df['time_sec'].iloc[-1]
            time_text.set_text(f'Time: {time_sec:.2f}s / {total_duration:.2f}s')

            # Update frame text if frame information exists
            if 'frame_number' in df.columns:
                frame_num = df.iloc[frame_idx]['frame_number']
                if not pd.isna(frame_num):
                    frame_text.set_text(f'Frame: {int(frame_num)}')
                else:
                    frame_text.set_text('')
            else:
                frame_text.set_text('')

            return left_line, right_line, left_point, right_point, time_text, frame_text

        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=range(len(df)),
            init_func=init, blit=True, interval=1000 / fps
        )

        # Save animation if path is provided
        if save_path:
            # Determine file format based on extension
            if save_path.lower().endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
                ani.save(save_path, writer=writer)
            elif save_path.lower().endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=fps)

            print(f"Animated scanpath saved to {save_path}")
            return save_path

        # If not saving, just show the plot interactively
        plt.tight_layout()
        plt.show()

        return None

    def plot_heatmap(self,
                     data: pd.DataFrame,
                     plots_dir: str,
                     prefix: str = '',
                     bin_size: int = 20,
                     smoothing: int = 3,
                     eye: str = 'left',
                     only_fixations: bool = True,
                     frame_range: Optional[Tuple[int, int]] = None):
        """
        Plot a heatmap of eye gaze distribution during movie viewing.

        This visualization shows where viewers focused their attention, with warmer colors
        indicating areas that received more visual attention. Differences in heatmap patterns
        between ASD and control groups can reveal distinct visual preferences.

        Args:
            data: DataFrame with unified eye metrics
            plots_dir: Path to save the plot
            prefix: Prefix for the plot filename
            bin_size: Size of bins for the heatmap in pixels
            smoothing: Gaussian smoothing factor
            eye: Which eye to plot ('left' or 'right')
            only_fixations: Whether to use only fixation data or all samples
            frame_range: Optional tuple with (start_frame, end_frame) to plot a specific segment
        """
        # Validate input data
        x_col, y_col = f'x_{eye}', f'y_{eye}'
        required_columns = [x_col, y_col]
        if not self._validate_plot_data(data, required_columns, f"Empty dataframe or missing eye position data for {eye} eye."):
            return

        # Filter by frame range if specified
        if frame_range and 'frame_number' in data.columns:
            start_frame, end_frame = frame_range
            df = data[(data['frame_number'] >= start_frame) & (data['frame_number'] <= end_frame)]
        else:
            df = data

        if only_fixations:
            fixation_col = f'is_fixation_{eye}'
            if fixation_col in df.columns:
                df = df[df[fixation_col]]
            else:
                print("Fixation data not found, using all data points.")

        # Create 2D histogram
        x_bins = np.arange(0, self.screen_width + bin_size, bin_size)
        y_bins = np.arange(0, self.screen_height + bin_size, bin_size)

        hist, x_edges, y_edges = np.histogram2d(
            df[x_col].dropna(),
            df[y_col].dropna(),
            bins=[x_bins, y_bins]
        )

        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        hist = gaussian_filter(hist, sigma=smoothing)

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Plot heatmap
        import matplotlib.colors as mcolors
        mesh = ax.pcolormesh(
            x_edges, y_edges, hist.T,
            cmap='hot', alpha=0.8,
            norm=mcolors.PowerNorm(gamma=0.5)  # Adjust gamma for better visualization
        )

        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label('Gaze Density')

        # Set axis limits
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)  # Invert y-axis

        # Add annotations
        title = f'Gaze Heatmap ({eye.capitalize()} Eye)'
        if frame_range:
            title += f' - Frames {frame_range[0]}-{frame_range[1]}'

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)

        # Add frame range annotation
        if 'frame_number' in df.columns:
            actual_frames = df['frame_number'].dropna()
            if not actual_frames.empty:
                frame_info = f"Frames: {int(actual_frames.min())} - {int(actual_frames.max())}"
                ax.annotate(frame_info, xy=(0.02, 0.02), xycoords='axes fraction',
                            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Create filename and save the plot
        filename = self._create_plot_filename(prefix, f"heatmap_{eye}", frame_range=frame_range)
        self.save_plot(plots_dir, filename, fig)

    def plot_fixation_duration_distribution(self,
                                            data: pd.DataFrame,
                                            plots_dir: str,
                                            prefix: str = ''):
        """
        Plot the distribution of fixation durations during movie viewing.

        This visualization shows how long viewers fixate on specific areas, which can highlight
        differences in attentional focus between ASD and control groups. ASD individuals may
        show atypical fixation duration patterns.

        Args:
            data: DataFrame with unified eye metrics
            plots_dir: Path to save the plot
            prefix: Prefix for the plot filename
        """
        # Validate input data
        required_columns = ['timestamp']
        if not self._validate_plot_data(data, required_columns, "Empty dataframe, cannot plot fixation duration distribution."):
            return

        # Extract fixation durations by finding start and end times of fixation events
        fixation_durations = {'left': [], 'right': []}

        for eye in ['left', 'right']:
            fixation_col = f'is_fixation_{eye}'
            if fixation_col in data.columns:
                # Find transitions from False to True (fixation start)
                starts = data.index[~data[fixation_col].shift(1, fill_value=False) & data[fixation_col]].tolist()
                # Find transitions from True to False (fixation end)
                ends = data.index[data[fixation_col] & ~data[fixation_col].shift(-1, fill_value=False)].tolist()

                if len(starts) > 0 and len(ends) > 0:
                    # Make sure starts and ends match
                    if len(ends) < len(starts):
                        # Last fixation extends beyond data, use last data point
                        ends.append(data.index[-1])
                    elif len(starts) < len(ends):
                        # First fixation started before data, use first data point
                        starts.insert(0, data.index[0])

                    # Limit to minimum number of pairs
                    n_pairs = min(len(starts), len(ends))

                    # Calculate durations in milliseconds
                    for i in range(n_pairs):
                        start_time = data.iloc[starts[i]]['timestamp'] if starts[i] < len(data) else None
                        end_time = data.iloc[ends[i]]['timestamp'] if ends[i] < len(data) else None

                        if start_time is not None and end_time is not None:
                            duration = end_time - start_time
                            if duration > 0:  # Ensure positive duration
                                fixation_durations[eye].append(duration)

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Plot histograms for both eyes
        has_data = False

        for eye, durations in fixation_durations.items():
            if durations:
                has_data = True
                color = self.colors[f'{eye}_eye']
                sns.histplot(durations, ax=ax, color=color, alpha=0.7,
                             label=f'{eye.capitalize()} Eye', kde=True, bins=20)

        if not has_data:
            print("No fixation duration data available.")
            plt.close(fig)
            return

        # Add annotations
        ax.set_title('Fixation Duration Distribution', fontsize=16)
        ax.set_xlabel('Fixation Duration (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='best')

        # Add summary statistics
        stats_text = ""
        for eye, durations in fixation_durations.items():
            if durations:
                stats_text += f"{eye.capitalize()} Eye: "
                stats_text += f"Mean={np.mean(durations):.1f}ms, "
                stats_text += f"Median={np.median(durations):.1f}ms, "
                stats_text += f"Min={np.min(durations):.1f}ms, "
                stats_text += f"Max={np.max(durations):.1f}ms\n"

        if stats_text:
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Create filename and save the plot
        filename = self._create_plot_filename(prefix, "fixation_duration_distribution")
        self.save_plot(plots_dir, filename, fig)

    def plot_saccade_amplitude_distribution(self,
                                            data: pd.DataFrame,
                                            plots_dir: str,
                                            prefix: str = ''):
        """
        Plot the distribution of saccade amplitudes during movie viewing.

        Saccade amplitude represents the distance between fixations and provides insight
        into how viewers scan scenes. ASD individuals may show different scanning patterns
        with either more locally focused (small saccades) or more erratic (large saccades)
        eye movements compared to neurotypical controls.

        Args:
            data: DataFrame with unified eye metrics
            plots_dir: Path to save the plot
            prefix: Prefix for the plot filename
        """
        # Validate input data
        if not self._validate_plot_data(data, None, "Empty dataframe, cannot plot saccade amplitude distribution."):
            return

        # Calculate saccade amplitudes from the data
        saccade_amplitudes = {'left': [], 'right': []}

        for eye in ['left', 'right']:
            saccade_col = f'is_saccade_{eye}'
            x_col, y_col = f'x_{eye}', f'y_{eye}'
            
            # Check if required columns exist
            if (saccade_col not in data.columns or 
                x_col not in data.columns or 
                y_col not in data.columns):
                continue
                
            # Extract saccades
            saccade_segments = []
            current_segment = []

            # Identify continuous segments of saccades
            for i, is_saccade in enumerate(data[saccade_col]):
                if is_saccade:
                    current_segment.append(i)
                elif current_segment:
                    saccade_segments.append(current_segment)
                    current_segment = []

            # Don't forget the last segment if it exists
            if current_segment:
                saccade_segments.append(current_segment)

            # Calculate amplitude for each saccade
            for segment in saccade_segments:
                if len(segment) >= 2:  # Need at least start and end points
                    # Get start and end positions
                    start_x, start_y = data.iloc[segment[0]][x_col], data.iloc[segment[0]][y_col]
                    end_x, end_y = data.iloc[segment[-1]][x_col], data.iloc[segment[-1]][y_col]

                    # Calculate Euclidean distance (amplitude in pixels)
                    if not (np.isnan(start_x) or np.isnan(start_y) or np.isnan(end_x) or np.isnan(end_y)):
                        amplitude = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                        saccade_amplitudes[eye].append(amplitude)

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Plot histograms for both eyes
        has_data = False

        for eye, amplitudes in saccade_amplitudes.items():
            if amplitudes:
                has_data = True
                color = self.colors[f'{eye}_eye']

                # Convert to degrees if needed (approximation: 1 degree ≈ 35 pixels at 60cm viewing distance)
                # This is an approximation and should be calibrated based on your setup
                pixel_to_degree = 1 / 35  # Convert pixels to visual degrees
                amplitudes_degrees = [amp * pixel_to_degree for amp in amplitudes]

                sns.histplot(amplitudes_degrees, ax=ax, color=color, alpha=0.7,
                             label=f'{eye.capitalize()} Eye', kde=True, bins=20)

        if not has_data:
            print("No saccade amplitude data available.")
            plt.close(fig)
            return

        # Add annotations
        ax.set_title('Saccade Amplitude Distribution', fontsize=16)
        ax.set_xlabel('Saccade Amplitude (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='best')

        # Add summary statistics
        stats_text = ""
        for eye, amplitudes in saccade_amplitudes.items():
            if amplitudes:
                # Convert to degrees as above
                pixel_to_degree = 1 / 35
                amplitudes_degrees = [amp * pixel_to_degree for amp in amplitudes]

                stats_text += f"{eye.capitalize()} Eye: "
                stats_text += f"Mean={np.mean(amplitudes_degrees):.2f}°, "
                stats_text += f"Median={np.median(amplitudes_degrees):.2f}°, "
                stats_text += f"Min={np.min(amplitudes_degrees):.2f}°, "
                stats_text += f"Max={np.max(amplitudes_degrees):.2f}°\n"

        if stats_text:
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Create filename and save the plot
        filename = self._create_plot_filename(prefix, "saccade_amplitude_distribution")
        self.save_plot(plots_dir, filename, fig)

    def plot_pupil_size_timeseries(self,
                                   data: pd.DataFrame,
                                   plots_dir: str,
                                   prefix: str = '',
                                   window_size: int = 50,
                                   frame_markers: bool = True):
        """
        Plot pupil size over time during movie viewing with optional frame markers.

        Pupil size reflects cognitive and emotional arousal and can reveal different
        autonomic responses to social and emotional stimuli, which often differ between
        ASD and neurotypical individuals.

        Args:
            data: DataFrame with unified eye metrics
            plots_dir: Directory to save the plot
            prefix: Prefix for the plot filename
            window_size: Size of the moving average window for smoothing
            frame_markers: Whether to display movie frame markers
        """
        # Validate input data
        required_columns = ['timestamp']
        if not self._validate_plot_data(data, required_columns, "Empty dataframe or missing timestamp data, cannot plot pupil size timeseries."):
            return

        # Check if pupil data exists
        has_left_pupil = 'pupil_left' in data.columns and not data['pupil_left'].isna().all()
        has_right_pupil = 'pupil_right' in data.columns and not data['pupil_right'].isna().all()

        if not has_left_pupil and not has_right_pupil:
            print("No pupil size data available.")
            return

        # Create figure with white background
        fig, ax = plt.subplots(figsize=self.default_figsize, facecolor='white')
        ax.set_facecolor('white')

        # Convert timestamp to seconds from start
        time_sec = (data['timestamp'] - data['timestamp'].iloc[0]) / 1000.0

        # Plot left eye pupil size if available
        if has_left_pupil:
            # Apply moving average to smooth the data
            pupil_left_smooth = data['pupil_left'].rolling(window=window_size, center=True, min_periods=1).mean()

            ax.plot(time_sec, pupil_left_smooth, color=self.colors['left_eye'],
                    linewidth=1.5, label='Left Eye', alpha=0.8)

            # Mark only blink onset points, not every sample during a blink
            if 'is_blink_left' in data.columns:
                # Find transitions from False to True (blink start)
                blink_starts = data.index[(~data['is_blink_left'].shift(1, fill_value=False)) &
                                          data['is_blink_left']].tolist()

                if blink_starts:
                    # Get the timestamps and values for only the blink start points
                    blink_start_times = [time_sec.iloc[i] for i in blink_starts if i < len(time_sec)]
                    blink_start_values = [pupil_left_smooth.iloc[i] for i in blink_starts if i < len(pupil_left_smooth)]

                    # Plot just one marker per blink
                    ax.scatter(blink_start_times, blink_start_values,
                               marker='x', color='red', alpha=0.8, s=50, label='Left Blinks')

        # Plot right eye pupil size if available
        if has_right_pupil:
            # Apply moving average to smooth the data
            pupil_right_smooth = data['pupil_right'].rolling(window=window_size, center=True, min_periods=1).mean()

            ax.plot(time_sec, pupil_right_smooth, color=self.colors['right_eye'],
                    linewidth=1.5, label='Right Eye', alpha=0.8)

            # Mark only blink onset points for right eye
            if 'is_blink_right' in data.columns:
                # Find transitions from False to True (blink start)
                blink_starts = data.index[(~data['is_blink_right'].shift(1, fill_value=False)) &
                                          data['is_blink_right']].tolist()

                if blink_starts:
                    # Get the timestamps and values for only the blink start points
                    blink_start_times = [time_sec.iloc[i] for i in blink_starts if i < len(time_sec)]
                    blink_start_values = [pupil_right_smooth.iloc[i] for i in blink_starts if
                                          i < len(pupil_right_smooth)]

                    # Plot just one marker per blink
                    ax.scatter(blink_start_times, blink_start_values,
                               marker='x', color='blue', alpha=0.8, s=50, label='Right Blinks')

        # Add annotations
        ax.set_title('Pupil Size Over Time', fontsize=16)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Pupil Size', fontsize=12)
        ax.legend(loc='upper right')

        # Add grid that doesn't overwhelm the data
        ax.grid(True, linestyle='--', alpha=0.3, color='lightgray')

        # Calculate movie duration
        duration_sec = time_sec.iloc[-1]

        # Add movie duration annotation
        duration_text = f"Movie Duration: {duration_sec:.2f} seconds"
        ax.annotate(duration_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Create filename and save the plot
        filename = self._create_plot_filename(prefix, "pupil_size_timeseries")
        plt.tight_layout()
        self.save_plot(plots_dir, filename, fig)

        # Create an additional plot showing pupil size correlation with event markers
        self._plot_pupil_size_events(data, plots_dir, prefix, window_size)


    def _plot_pupil_size_events(self,
                                data: pd.DataFrame,
                                plots_dir: str,
                                prefix: str = '',
                                window_size: int = 50):
        """
        Plot pupil size changes with highlighted fixations, saccades, and blinks.

        This visualization helps identify how pupil size changes during different
        eye movement events, which can reveal attentional patterns.

        Args:
            data: DataFrame with unified eye metrics
            plots_dir: Path to save the plot
            prefix: Prefix for the plot filename
            window_size: Size of the moving average window for smoothing
        """
        # Validate input data
        required_columns = ['timestamp', 'pupil_left']
        if not self._validate_plot_data(data, required_columns, "Missing required pupil data for event visualization."):
            return

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Convert timestamp to seconds
        time_sec = (data['timestamp'] - data['timestamp'].iloc[0]) / 1000.0

        # Smooth pupil size
        pupil_smooth = data['pupil_left'].rolling(window=window_size, center=True, min_periods=1).mean()

        # Plot pupil size
        ax.plot(time_sec, pupil_smooth, color=self.colors['left_eye'],
                linewidth=1.5, label='Pupil Size')

        # Create shaded regions for different eye events
        event_types = {
            'is_fixation_left': {'color': self.colors['fixation'], 'alpha': 0.2, 'label': 'Fixation'},
            'is_saccade_left': {'color': self.colors['saccade'], 'alpha': 0.2, 'label': 'Saccade'},
            'is_blink_left': {'color': self.colors['blink'], 'alpha': 0.2, 'label': 'Blink'}
        }

        for event_col, props in event_types.items():
            if event_col in data.columns:
                # Find continuous segments of the event
                event_segments = []
                current_segment = []

                for i, is_event in enumerate(data[event_col]):
                    if is_event:
                        current_segment.append(i)
                    elif current_segment:
                        event_segments.append(current_segment)
                        current_segment = []

                # Don't forget the last segment
                if current_segment:
                    event_segments.append(current_segment)

                # Shade each segment
                legend_added = False
                for segment in event_segments:
                    if not segment:
                        continue

                    start_idx, end_idx = segment[0], segment[-1]
                    if start_idx >= len(time_sec) or end_idx >= len(time_sec):
                        continue

                    # Add shaded region
                    if not legend_added:
                        ax.axvspan(time_sec[start_idx], time_sec[end_idx],
                                   color=props['color'], alpha=props['alpha'],
                                   label=props['label'])
                        legend_added = True
                    else:
                        ax.axvspan(time_sec[start_idx], time_sec[end_idx],
                                   color=props['color'], alpha=props['alpha'])

        # Add annotations
        ax.set_title('Pupil Size and Eye Events', fontsize=16)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Pupil Size', fontsize=12)
        ax.legend(loc='best')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Create filename and save the plot
        filename = self._create_plot_filename(prefix, "pupil_size_events")
        self.save_plot(plots_dir, filename, fig)

    def plot_social_attention_analysis(self,
                                       data: pd.DataFrame,
                                       plots_dir: str,
                                       prefix: str = '',
                                       roi_data: Optional[Dict[str, List[Tuple[int, int, int, int]]]] = None):
        """
        Plot analysis of attention to social vs. non-social regions in the movie.

        This visualization specifically addresses core areas of research in autism by analyzing
        how viewers allocate attention to social elements (e.g., faces, eyes) compared to
        non-social elements, which is a key area of difference in ASD.

        Args:
            data: DataFrame with unified eye metrics
            plots_dir: Path to save the plot
            prefix: Prefix for the plot filename
            roi_data: Optional dictionary mapping frame numbers to lists of ROIs (x, y, width, height)
                     Each ROI should be labeled as 'face', 'eyes', or other social element
        """
        # Validate input data
        required_columns = ['frame_number']
        if not self._validate_plot_data(data, required_columns, "Empty dataframe or missing frame data, cannot plot social attention analysis."):
            return

        # If we don't have specific ROI data, we'll simulate it with frame-based random ROIs
        # In a real implementation, you would use actual annotated ROI data for accurate analysis
        if roi_data is None:
            # Create simulated ROIs for demonstration purposes
            unique_frames = data['frame_number'].dropna().unique()
            roi_data = {}

            for frame in unique_frames:
                if np.isnan(frame):
                    continue

                # Create simulated ROIs (would be replaced with actual annotated data)
                # Format: x, y, width, height, label
                frame_rois = [
                    # Simulated face ROI
                    (self.screen_width // 4, self.screen_height // 4,
                     self.screen_width // 5, self.screen_height // 5, 'face'),
                    # Simulated eyes ROI
                    (self.screen_width // 4 + 20, self.screen_height // 4 + 20,
                     self.screen_width // 10, self.screen_height // 20, 'eyes'),
                    # Simulated object ROI
                    (self.screen_width // 2, self.screen_height // 2,
                     self.screen_width // 8, self.screen_height // 8, 'object')
                ]
                roi_data[int(frame)] = frame_rois

        # If we still don't have ROI data, we can't create this visualization
        if not roi_data:
            print("No ROI data available for social attention analysis.")
            return

        # Analyze fixations within ROIs
        social_attention = {'face': 0, 'eyes': 0, 'other_social': 0, 'non_social': 0}
        total_fixations = 0
        fixations_in_rois = []

        # Use either left or right eye data, preferring left if available
        eye = 'left' if 'x_left' in data.columns and 'y_left' in data.columns else 'right'
        x_col, y_col = f'x_{eye}', f'y_{eye}'
        fixation_col = f'is_fixation_{eye}'

        if x_col not in data.columns or y_col not in data.columns or fixation_col not in data.columns:
            print(f"Required eye tracking data (position or fixation) not available for {eye} eye.")
            return

        # Filter to fixation data points
        fixation_data = data[data[fixation_col]]

        # For each fixation, check if it's within any ROI
        for _, row in fixation_data.iterrows():
            if pd.isna(row[x_col]) or pd.isna(row[y_col]) or pd.isna(row['frame_number']):
                continue

            frame = int(row['frame_number'])
            x, y = row[x_col], row[y_col]
            total_fixations += 1

            # Check if this frame has ROI data
            if frame in roi_data:
                fixation_classified = False

                # Check each ROI in this frame
                for roi in roi_data[frame]:
                    roi_x, roi_y, roi_width, roi_height, roi_label = roi

                    # Check if fixation is within this ROI
                    if (roi_x <= x <= roi_x + roi_width and
                            roi_y <= y <= roi_y + roi_height):

                        # Categorize the fixation
                        if roi_label == 'face':
                            social_attention['face'] += 1
                            fixations_in_rois.append((frame, x, y, 'face'))
                            fixation_classified = True
                        elif roi_label == 'eyes':
                            social_attention['eyes'] += 1
                            fixations_in_rois.append((frame, x, y, 'eyes'))
                            fixation_classified = True
                        elif 'social' in roi_label:
                            social_attention['other_social'] += 1
                            fixations_in_rois.append((frame, x, y, 'other_social'))
                            fixation_classified = True
                        else:
                            social_attention['non_social'] += 1
                            fixations_in_rois.append((frame, x, y, 'non_social'))
                            fixation_classified = True

                        # We've categorized this fixation, no need to check other ROIs
                        break

                # If fixation wasn't in any ROI, count as non-social
                if not fixation_classified:
                    social_attention['non_social'] += 1
            else:
                # Frame has no ROI data, count as non-social
                social_attention['non_social'] += 1

        # Calculate percentages
        if total_fixations > 0:
            social_percentages = {
                label: (count / total_fixations) * 100
                for label, count in social_attention.items()
            }
        else:
            print("No fixations detected for social attention analysis.")
            return

        # Create figure with 1 row, 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.default_figsize[0], self.default_figsize[1] // 2))

        # Plot 1: Pie chart of social vs. non-social attention
        social_sum = social_percentages['face'] + social_percentages['eyes'] + social_percentages['other_social']
        non_social = social_percentages['non_social']

        pie_labels = ['Social', 'Non-social']
        pie_values = [social_sum, non_social]
        pie_colors = ['#4CAF50', '#F44336']  # Green for social, red for non-social

        ax1.pie(pie_values, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
                startangle=90, wedgeprops={'alpha': 0.8})
        ax1.set_title('Social vs. Non-social Attention', fontsize=14)

        # Plot 2: Detailed breakdown of social attention
        categories = ['Face', 'Eyes', 'Other Social', 'Non-social']
        values = [social_percentages['face'], social_percentages['eyes'],
                  social_percentages['other_social'], social_percentages['non_social']]

        colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#F44336']  # Gradients of green for social, red for non-social

        ax2.bar(categories, values, color=colors, alpha=0.8)
        ax2.set_title('Detailed Attention Distribution', fontsize=14)
        ax2.set_ylabel('Percentage of Fixations (%)', fontsize=12)
        ax2.set_ylim(0, max(values) * 1.2)  # Add some headroom

        # Add value labels on bars
        for i, v in enumerate(values):
            ax2.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Create filename and save the plot
        filename = self._create_plot_filename(prefix, "social_attention_analysis")
        self.save_plot(plots_dir, filename, fig)

        # Create an additional visualization showing the distribution of social fixations over time
        self._plot_social_attention_timeline(data, fixations_in_rois, plots_dir, prefix)

    def _plot_social_attention_timeline(self,
                                        data: pd.DataFrame,
                                        fixations_in_rois: List[Tuple[int, float, float, str]],
                                        plots_dir: str,
                                        prefix: str = ''):
        """
        Plot how social attention changes over the course of the movie.

        Args:
            data: DataFrame with unified eye metrics
            fixations_in_rois: List of tuples (frame, x, y, category) for fixations
            plots_dir: Path to save the plot
            prefix: Prefix for plot filenames
        """
        # Validate there is fixation data to plot
        if not fixations_in_rois:
            print("No fixation data in ROIs available for timeline visualization.")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Sort fixations by frame
        fixations_in_rois.sort(key=lambda x: x[0])

        # Get frame to timestamp mapping
        frame_time_map = {}
        if 'frame_number' in data.columns and 'timestamp' in data.columns:
            for _, row in data.iterrows():
                if not pd.isna(row['frame_number']):
                    frame = int(row['frame_number'])
                    if frame not in frame_time_map:
                        frame_time_map[frame] = row['timestamp']

        # Convert frames to seconds if possible
        has_time_data = bool(frame_time_map)
        if has_time_data:
            # Get start time
            start_time = min(frame_time_map.values())
            x_data = [(frame_time_map.get(frame, start_time) - start_time) / 1000.0
                      for frame, _, _, _ in fixations_in_rois]
            x_label = 'Time (seconds)'
        else:
            # Use frame numbers if timestamps not available
            x_data = [frame for frame, _, _, _ in fixations_in_rois]
            x_label = 'Frame Number'

        # Set up a moving window to calculate percentages over time
        window_size = min(30, len(fixations_in_rois) // 10) if len(fixations_in_rois) > 10 else 5
        window_social = []
        window_times = []

        for i in range(0, len(fixations_in_rois) - window_size, window_size // 2):  # 50% overlap between windows
            window = fixations_in_rois[i:i + window_size]

            # Calculate percentage of social fixations in this window
            social_count = sum(1 for _, _, _, category in window
                               if category in ['face', 'eyes', 'other_social'])
            social_percent = (social_count / len(window)) * 100

            # Use middle of window for x-axis position
            mid_idx = i + window_size // 2
            if mid_idx < len(x_data):
                window_times.append(x_data[mid_idx])
                window_social.append(social_percent)

        # Plot the social attention timeline
        ax.plot(window_times, window_social, 'o-', color='#4CAF50', linewidth=2,
                markersize=5, alpha=0.8)

        # Highlight the 50% line
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% Level')

        # Add annotations
        ax.set_title('Social Attention Over Time', fontsize=16)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Percentage of Social Fixations (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Categorize different fixation types
        categories = {
            'face': {'color': '#4CAF50', 'marker': 'o', 'label': 'Face'},
            'eyes': {'color': '#8BC34A', 'marker': 's', 'label': 'Eyes'},
            'other_social': {'color': '#CDDC39', 'marker': '^', 'label': 'Other Social'},
            'non_social': {'color': '#F44336', 'marker': 'x', 'label': 'Non-social'}
        }

        # Add small markers at the bottom to show individual fixations by category
        y_pos = {
            'face': 5,
            'eyes': 3,
            'other_social': 1,
            'non_social': -1
        }

        # Plot a small sample of individual fixations (to avoid overcrowding)
        max_fixations = 100
        stride = max(1, len(fixations_in_rois) // max_fixations)

        for i in range(0, len(fixations_in_rois), stride):
            frame, _, _, category = fixations_in_rois[i]
            if i < len(x_data):
                ax.scatter(x_data[i], y_pos.get(category, 0),
                           color=categories[category]['color'],
                           marker=categories[category]['marker'],
                           s=20, alpha=0.7)

        # Add legend for fixation types
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker=props['marker'], color=props['color'],
                                  linestyle='', markersize=8, label=props['label'])
                           for category, props in categories.items()]

        ax.legend(handles=legend_elements, loc='upper right')

        # Create filename and save the plot
        filename = self._create_plot_filename(prefix, "social_attention_timeline")
        self.save_plot(plots_dir, filename, fig)

    def process_all_movies(self, participant_id: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Process all movies in the base directory, generating all available visualizations.

        This main method handles the complete visualization pipeline for all discovered
        movie folders, organizing results by movie.

        Args:
            participant_id: Optional identifier to use as prefix for plot filenames

        Returns:
            Dictionary with movie names as keys and dictionaries of plot types and paths as values
        """
        # Discover all movie folders
        movie_folders = self.discover_movie_folders()
        print(f"Found {len(movie_folders)} movie folders in {self.base_dir}")

        if not movie_folders:
            print(
                "No movie folders found. Ensure your directory structure contains movies with unified_eye_metrics CSV files.")
            return {}

        results = {}

        # Process each movie folder
        for movie_folder in movie_folders:
            movie_name, data = self.load_movie_data(movie_folder)

            if data.empty:
                print(f"No data found for movie: {movie_name}, skipping")
                continue

            print(f"\nProcessing movie: {movie_name} with {len(data)} data points")

            # Ensure plots directory exists
            plots_dir = self.ensure_plots_directory(movie_folder)

            # Create prefix from participant ID if provided
            prefix = f"{participant_id}_" if participant_id else ""

            # Dictionary to store all plot information for this movie
            movie_plots = {
                'general': [],
                'gaze': [],
                'fixation': [],
                'saccade': [],
                'pupil': [],
                'social': []
            }

            # Generate all visualizations

            # 1. General visualizations (scanpath, heatmaps)
            print("Generating general visualizations...")
            try:
                self.plot_scanpath(data, plots_dir, prefix)
                scanpath_filename = self._create_plot_filename(prefix, "scanpath")
                scanpath_path = os.path.join(plots_dir, f"{scanpath_filename}.png")
                if os.path.exists(scanpath_path):
                    movie_plots['general'].append(scanpath_path)
            except Exception as e:
                print(f"Error generating scanpath: {e}")

            for eye in ['left', 'right']:
                try:
                    self.plot_heatmap(data, plots_dir, prefix, eye=eye)
                    heatmap_filename = self._create_plot_filename(prefix, f"heatmap_{eye}")
                    heatmap_path = os.path.join(plots_dir, f"{heatmap_filename}.png")
                    if os.path.exists(heatmap_path):
                        movie_plots['gaze'].append(heatmap_path)
                except Exception as e:
                    print(f"Error generating {eye} eye heatmap: {e}")

            # 2. Fixation visualizations
            print("Generating fixation visualizations...")
            try:
                self.plot_fixation_duration_distribution(data, plots_dir, prefix)
                fixation_filename = self._create_plot_filename(prefix, "fixation_duration_distribution")
                fixation_path = os.path.join(plots_dir, f"{fixation_filename}.png")
                if os.path.exists(fixation_path):
                    movie_plots['fixation'].append(fixation_path)
            except Exception as e:
                print(f"Error generating fixation distribution: {e}")

            # 3. Saccade visualizations
            print("Generating saccade visualizations...")
            try:
                self.plot_saccade_amplitude_distribution(data, plots_dir, prefix)
                saccade_filename = self._create_plot_filename(prefix, "saccade_amplitude_distribution")
                saccade_path = os.path.join(plots_dir, f"{saccade_filename}.png")
                if os.path.exists(saccade_path):
                    movie_plots['saccade'].append(saccade_path)
            except Exception as e:
                print(f"Error generating saccade distribution: {e}")

            # 4. Pupil visualizations
            print("Generating pupil visualizations...")
            try:
                self.plot_pupil_size_timeseries(data, plots_dir, prefix)
                pupil_filename = self._create_plot_filename(prefix, "pupil_size_timeseries")
                events_filename = self._create_plot_filename(prefix, "pupil_size_events")
                pupil_path = os.path.join(plots_dir, f"{pupil_filename}.png")
                events_path = os.path.join(plots_dir, f"{events_filename}.png")
                if os.path.exists(pupil_path):
                    movie_plots['pupil'].append(pupil_path)
                if os.path.exists(events_path):
                    movie_plots['pupil'].append(events_path)
            except Exception as e:
                print(f"Error generating pupil visualizations: {e}")

            # 5. Social attention visualizations
            print("Generating social attention visualizations...")
            try:
                self.plot_social_attention_analysis(data, plots_dir, prefix)
                social_filename = self._create_plot_filename(prefix, "social_attention_analysis")
                timeline_filename = self._create_plot_filename(prefix, "social_attention_timeline")
                social_path = os.path.join(plots_dir, f"{social_filename}.png")
                timeline_path = os.path.join(plots_dir, f"{timeline_filename}.png")
                if os.path.exists(social_path):
                    movie_plots['social'].append(social_path)
                if os.path.exists(timeline_path):
                    movie_plots['social'].append(timeline_path)
            except Exception as e:
                print(f"Error generating social attention visualizations: {e}")

            # Store results for this movie
            results[movie_name] = movie_plots

            # Calculate total plots for this movie
            total_plots = sum(len(plots) for plots in movie_plots.values())
            print(f"Generated {total_plots} visualizations for {movie_name}")

        # Print final summary
        total_movies = len(results)
        total_visualizations = sum(sum(len(plots) for plots in movie_plots.values())
                                   for movie_plots in results.values())

        print(f"\nVisualization complete! Generated {total_visualizations} plots across {total_movies} movies.")

        return results

    def generate_report(self, results: Dict[str, Dict[str, List[str]]], output_dir: str) -> str:
        """
        Generate an HTML report with fully dynamic section discovery for each movie.
        No hardcoded category names or structure - adapts to whatever data is present.

        Args:
            results: Dictionary with movie names as keys and dictionaries of plot types and paths as values
            output_dir: Directory to save the HTML report

        Returns:
            Path to the generated HTML report
        """
        # Create report directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create HTML report
        report_path = os.path.join(output_dir, "visualization_report.html")

        # Get relative paths for images
        def get_relative_path(path):
            return os.path.relpath(path, output_dir)

        # Count total visualizations
        total_visualizations = sum(sum(len(plots) for plots in movie_plots.values())
                                   for movie_plots in results.values())

        # Dynamically determine all unique category names across all results
        all_categories = set()
        for movie_plots in results.values():
            all_categories.update(movie_plots.keys())

        # Create a function to convert category names to display titles
        def get_category_display_name(category):
            """Convert a category key to a user-friendly display name"""
            # Replace underscores with spaces and capitalize each word
            return category.replace('_', ' ').title()

        with open(report_path, 'w') as f:
            # Write HTML header with improved styling and modal functionality
            f.write("""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Eye Tracking Visualization Report</title>
        <style>
            :root {
                --primary-color: #2c3e50;
                --secondary-color: #3498db;
                --accent-color: #e74c3c;
                --bg-color: #f9f9f9;
                --text-color: #333;
                --border-color: #ddd;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: var(--bg-color);
            }

            h1, h2, h3, h4 {
                color: var(--primary-color);
                margin-bottom: 0.5em;
            }

            h1 {
                font-size: 2.5em;
                text-align: center;
                margin-bottom: 0.7em;
                padding-bottom: 0.5em;
                border-bottom: 2px solid var(--secondary-color);
            }

            h2 {
                font-size: 1.8em;
                margin-top: 1em;
            }

            .summary {
                background-color: #e8f4fc;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }

            .movie-section {
                margin-bottom: 30px;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }

            .movie-header {
                padding: 15px 20px;
                background-color: #f1f8fe;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: background-color 0.3s;
            }

            .movie-header:hover {
                background-color: #d9edfb;
            }

            .movie-header h2 {
                margin: 0;
                font-size: 1.5em;
            }

            .movie-header .count-badge {
                background-color: var(--secondary-color);
                color: white;
                border-radius: 20px;
                padding: 5px 12px;
                font-size: 0.9em;
                font-weight: 600;
            }

            .movie-content {
                display: none;
                padding: 20px;
            }

            .active .movie-content {
                display: block;
            }

            .plot-summary {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }

            .plot-category {
                margin-bottom: 25px;
                border-bottom: 1px solid #eee;
                padding-bottom: 15px;
            }

            .plot-category h3 {
                color: var(--secondary-color);
                margin-bottom: 15px;
                font-size: 1.4em;
            }

            .plot-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                gap: 20px;
            }

            .plot-container {
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 15px;
                background-color: white;
                transition: transform 0.2s, box-shadow 0.2s;
            }

            .plot-container:hover {
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }

            .plot-container img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                border: 1px solid #eee;
                cursor: pointer; /* Indicate clickable */
                transition: opacity 0.2s;
            }

            .plot-container img:hover {
                opacity: 0.9;
            }

            .plot-title {
                font-weight: 600;
                margin-bottom: 10px;
                color: var(--primary-color);
            }

            .timestamp {
                color: #777;
                font-style: italic;
                text-align: center;
                margin-top: 20px;
            }

            .footer {
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid var(--border-color);
                color: #777;
            }

            /* Modal styles for image zoom */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                overflow: auto;
                background-color: rgba(0,0,0,0.9);
                opacity: 0;
                transition: opacity 0.3s;
            }

            .modal.show {
                display: flex;
                opacity: 1;
                align-items: center;
                justify-content: center;
            }

            .modal-content {
                max-width: 95%;
                max-height: 95vh;
                margin: auto;
                display: block;
                animation: zoom 0.3s;
            }

            @keyframes zoom {
                from {transform: scale(0.9)}
                to {transform: scale(1)}
            }

            .modal-caption {
                position: absolute;
                bottom: 20px;
                left: 0;
                right: 0;
                text-align: center;
                color: white;
                background-color: rgba(0,0,0,0.5);
                padding: 10px;
                font-size: 1.2em;
            }

            .close-modal {
                position: absolute;
                top: 15px;
                right: 35px;
                color: #f1f1f1;
                font-size: 40px;
                font-weight: bold;
                transition: 0.3s;
                cursor: pointer;
            }

            .close-modal:hover {
                color: #bbb;
            }
        </style>
    </head>
    <body>
        <h1>Eye Tracking Visualization Report</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total movies processed:</strong> """ + str(len(results)) + """</p>
            <p><strong>Total visualizations generated:</strong> """ + str(total_visualizations) + """</p>
            <p><strong>Generated on:</strong> """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>

        <!-- Image Modal -->
        <div id="imageModal" class="modal">
            <span class="close-modal" onclick="closeModal()">&times;</span>
            <img class="modal-content" id="modalImg">
            <div id="modalCaption" class="modal-caption"></div>
        </div>

        <script>
            function toggleMovieSection(movieId) {
                const section = document.getElementById(movieId);
                section.classList.toggle('active');
            }

            // Modal functionality for image zoom
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImg');
            const modalCaption = document.getElementById('modalCaption');

            // Open modal when image is clicked
            function showModal(imgSrc, caption) {
                modalImg.src = imgSrc;
                modalCaption.textContent = caption;
                modal.classList.add('show');

                // Prevent scrolling on body when modal is open
                document.body.style.overflow = 'hidden';
            }

            // Close modal
            function closeModal() {
                modal.classList.remove('show');
                setTimeout(() => {
                    modalImg.src = '';
                }, 300);

                // Re-enable scrolling
                document.body.style.overflow = 'auto';
            }

            // Close modal when clicking outside the image
            modal.addEventListener('click', function(event) {
                if (event.target === modal) {
                    closeModal();
                }
            });

            // Close modal with Escape key
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape') {
                    closeModal();
                }
            });
        </script>
    """)

            # Write movie sections, each as a collapsible panel
            for i, (movie_name, movie_plots) in enumerate(results.items()):
                # Create a unique ID for this movie section
                movie_id = f"movie-{i}"

                # Count plots by category and total plots
                category_counts = {category: len(plots) for category, plots in movie_plots.items() if plots}
                total_plots = sum(category_counts.values())

                # Start movie section
                f.write(f'<div class="movie-section" id="{movie_id}">\n')

                # Movie header (always visible, clickable)
                f.write('    <div class="movie-header" onclick="toggleMovieSection(\'' + movie_id + '\')">\n')
                f.write(f'        <h2>{movie_name}</h2>\n')
                f.write(f'        <span class="count-badge">{total_plots} visualizations</span>\n')
                f.write('    </div>\n')

                # Movie content (initially hidden) - ALL visualization content goes inside this div
                f.write(f'    <div class="movie-content">\n')

                # Write summary for this movie
                f.write('        <div class="plot-summary">\n')
                f.write('            <ul>\n')
                for category, count in category_counts.items():
                    if count > 0:
                        category_display = get_category_display_name(category)
                        f.write(f'                <li><strong>{category_display}:</strong> {count} plot(s)</li>\n')
                f.write('            </ul>\n')
                f.write('        </div>\n')

                # Write plot categories - dynamically based on what's in the data
                for category, plots in movie_plots.items():
                    if plots:
                        category_display = get_category_display_name(category)
                        f.write('        <div class="plot-category">\n')
                        f.write(f'            <h3>{category_display}</h3>\n')
                        f.write('            <div class="plot-grid">\n')

                        for plot_path in plots:
                            plot_name = os.path.basename(plot_path)
                            # Make a more human-readable plot name by removing prefixes and extensions
                            readable_name = os.path.splitext(plot_name)[0]  # Remove extension
                            if '_' in readable_name:  # Remove prefix before underscore if present
                                readable_name = readable_name.split('_', 1)[1]
                            # Convert underscores to spaces and capitalize words
                            readable_name = readable_name.replace('_', ' ').title()

                            rel_path = get_relative_path(plot_path)

                            f.write('                <div class="plot-container">\n')
                            f.write(f'                    <div class="plot-title">{readable_name}</div>\n')
                            # Add onclick event to show modal with full-size image
                            f.write(
                                f'                    <img src="{rel_path}" alt="{readable_name}" loading="lazy" onclick="showModal(\'{rel_path}\', \'{readable_name}\');">\n')
                            f.write('                </div>\n')

                        f.write('            </div>\n')
                        f.write('        </div>\n')

                # Close movie content div
                f.write('    </div>\n')
                # Close movie section div
                f.write('</div>\n')

            # Write HTML footer
            f.write("""    <div class="timestamp">
            Report generated on: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """
        </div>

        <div class="footer">
            <p>Powered by Tal Alfi's Eye Movement Analysis for Autism Classification</p>
        </div>
    </body>
    </html>""")

            print(f"Generated dynamically structured HTML report at {report_path}")
            return report_path