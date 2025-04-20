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
        for root, dirs, files in os.walk(self.base_dir):
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
        if data.empty:
            print("Empty dataframe, cannot plot scanpath.")
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

        # Save the plot
        filename = f"{prefix}scanpath"
        if time_window:
            filename += f"_{time_window[0]}_{time_window[1]}"

        self.save_plot(plots_dir, filename, fig)

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
        if data.empty:
            print("Empty dataframe, cannot plot heatmap.")
            return

        # Filter data based on eye and fixation settings
        x_col, y_col = f'x_{eye}', f'y_{eye}'
        if x_col not in data.columns or y_col not in data.columns:
            print(f"Eye position data for {eye} eye not found.")
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

        # Generate filename
        filename = f"{prefix}heatmap_{eye}"
        if frame_range:
            filename += f"_frames_{frame_range[0]}_{frame_range[1]}"

        # Save the plot
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
        if data.empty:
            print("Empty dataframe, cannot plot fixation duration distribution.")
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

        # Save the plot
        filename = f"{prefix}fixation_duration_distribution"
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
        if data.empty:
            print("Empty dataframe, cannot plot saccade amplitude distribution.")
            return

        # Calculate saccade amplitudes from the data
        saccade_amplitudes = {'left': [], 'right': []}

        for eye in ['left', 'right']:
            saccade_col = f'is_saccade_{eye}'
            if saccade_col in data.columns:
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
                x_col, y_col = f'x_{eye}', f'y_{eye}'
                if x_col in data.columns and y_col in data.columns:
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

        # Save the plot
        filename = f"{prefix}saccade_amplitude_distribution"
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
            plots_dir: Path to save the plot
            prefix: Prefix for the plot filename
            window_size: Size of the moving average window for smoothing
            frame_markers: Whether to display movie frame markers
        """
        if data.empty:
            print("Empty dataframe, cannot plot pupil size timeseries.")
            return

        # Check if pupil data exists
        has_left_pupil = 'pupil_left' in data.columns and not data['pupil_left'].isna().all()
        has_right_pupil = 'pupil_right' in data.columns and not data['pupil_right'].isna().all()

        if not has_left_pupil and not has_right_pupil:
            print("No pupil size data available.")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Convert timestamp to seconds for better readability
        # First ensure timestamp column exists and is numeric
        if 'timestamp' not in data.columns:
            print("No timestamp column found.")
            return

        # Convert timestamp to seconds from start
        time_sec = (data['timestamp'] - data['timestamp'].iloc[0]) / 1000.0

        # Plot left eye pupil size if available
        if has_left_pupil:
            # Apply moving average to smooth the data
            pupil_left_smooth = data['pupil_left'].rolling(window=window_size, center=True, min_periods=1).mean()

            ax.plot(time_sec, pupil_left_smooth, color=self.colors['left_eye'],
                    linewidth=1.5, label='Left Eye')

            # Optionally highlight blinks
            if 'is_blink_left' in data.columns:
                blink_times = time_sec[data['is_blink_left']]
                if len(blink_times) > 0:
                    ax.scatter(blink_times, pupil_left_smooth[data['is_blink_left']],
                               marker='x', color=self.colors['blink'], alpha=0.5, label='Left Blinks')

        # Plot right eye pupil size if available
        if has_right_pupil:
            # Apply moving average to smooth the data
            pupil_right_smooth = data['pupil_right'].rolling(window=window_size, center=True, min_periods=1).mean()

            ax.plot(time_sec, pupil_right_smooth, color=self.colors['right_eye'],
                    linewidth=1.5, label='Right Eye')

            # Optionally highlight blinks
            if 'is_blink_right' in data.columns:
                blink_times = time_sec[data['is_blink_right']]
                if len(blink_times) > 0:
                    ax.scatter(blink_times, pupil_right_smooth[data['is_blink_right']],
                               marker='x', color=self.colors['blink'], alpha=0.5, label='Right Blinks')

        # Add frame markers if requested and available
        if frame_markers and 'frame_number' in data.columns:
            # Find transitions in frame numbers
            frame_changes = data.index[data['frame_number'].diff() > 0].tolist()

            if frame_changes:
                # For each frame change, add a vertical line
                for idx in frame_changes:
                    if idx < len(time_sec):
                        frame_num = data.iloc[idx]['frame_number']
                        ax.axvline(x=time_sec[idx], color='green', linestyle='--', alpha=0.5)

                        # Only label some frames to avoid cluttering
                        if int(frame_num) % 5 == 0:  # Label every 5th frame
                            ax.text(time_sec[idx], ax.get_ylim()[1] * 0.9, f"Frame {int(frame_num)}",
                                    rotation=90, va='top', ha='right', fontsize=8, alpha=0.7)

        # Add annotations
        ax.set_title('Pupil Size Over Time', fontsize=16)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Pupil Size', fontsize=12)
        ax.legend(loc='best')

        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Calculate movie duration
        duration_sec = time_sec.iloc[-1]

        # Add movie duration annotation
        duration_text = f"Movie Duration: {duration_sec:.2f} seconds"
        ax.annotate(duration_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Save the plot
        filename = f"{prefix}pupil_size_timeseries"
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
        if data.empty:
            return

        # We'll focus on one eye (left) for this visualization to avoid clutter
        if 'pupil_left' not in data.columns or data['pupil_left'].isna().all():
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

        # Save the plot
        filename = f"{prefix}pupil_size_events"
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
        if data.empty:
            print("Empty dataframe, cannot plot social attention analysis.")
            return

        # If we don't have specific ROI data, we'll simulate it with frame-based random ROIs
        # In a real implementation, you would use actual annotated ROI data for accurate analysis
        if roi_data is None and 'frame_number' in data.columns:
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

        # Save the plot
        filename = f"{prefix}social_attention_analysis"
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
        if not fixations_in_rois:
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

        # Save the plot
        filename = f"{prefix}social_attention_timeline"
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
                scanpath_path = os.path.join(plots_dir, f"{prefix}scanpath.png")
                if os.path.exists(scanpath_path):
                    movie_plots['general'].append(scanpath_path)
            except Exception as e:
                print(f"Error generating scanpath: {e}")

            for eye in ['left', 'right']:
                try:
                    self.plot_heatmap(data, plots_dir, prefix, eye=eye)
                    heatmap_path = os.path.join(plots_dir, f"{prefix}heatmap_{eye}.png")
                    if os.path.exists(heatmap_path):
                        movie_plots['gaze'].append(heatmap_path)
                except Exception as e:
                    print(f"Error generating {eye} eye heatmap: {e}")

            # 2. Fixation visualizations
            print("Generating fixation visualizations...")
            try:
                self.plot_fixation_duration_distribution(data, plots_dir, prefix)
                fixation_path = os.path.join(plots_dir, f"{prefix}fixation_duration_distribution.png")
                if os.path.exists(fixation_path):
                    movie_plots['fixation'].append(fixation_path)
            except Exception as e:
                print(f"Error generating fixation distribution: {e}")

            # 3. Saccade visualizations
            print("Generating saccade visualizations...")
            try:
                self.plot_saccade_amplitude_distribution(data, plots_dir, prefix)
                saccade_path = os.path.join(plots_dir, f"{prefix}saccade_amplitude_distribution.png")
                if os.path.exists(saccade_path):
                    movie_plots['saccade'].append(saccade_path)
            except Exception as e:
                print(f"Error generating saccade distribution: {e}")

            # 4. Pupil visualizations
            print("Generating pupil visualizations...")
            try:
                self.plot_pupil_size_timeseries(data, plots_dir, prefix)
                pupil_path = os.path.join(plots_dir, f"{prefix}pupil_size_timeseries.png")
                events_path = os.path.join(plots_dir, f"{prefix}pupil_size_events.png")
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
                social_path = os.path.join(plots_dir, f"{prefix}social_attention_analysis.png")
                timeline_path = os.path.join(plots_dir, f"{prefix}social_attention_timeline.png")
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
        Generate an HTML report summarizing all visualizations.

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

        with open(report_path, 'w') as f:
            # Write HTML header
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eye Tracking Visualization Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .movie-section {
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        .plot-category {
            margin-bottom: 20px;
        }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
        }
        .plot-container {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #eee;
        }
        .plot-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .summary {
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>Eye Tracking Visualization Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total movies processed: """ + str(len(results)) + """</p>
        <p>Total visualizations generated: """ + str(sum(sum(len(plots) for plots in movie_plots.values())
                                                         for movie_plots in results.values())) + """</p>
        <p>Generated on: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
""")

            # Write movie sections
            for movie_name, movie_plots in results.items():
                f.write(f'<div class="movie-section">\n')
                f.write(f'    <h2>Movie: {movie_name}</h2>\n')

                # Count plots by category
                category_counts = {category: len(plots) for category, plots in movie_plots.items() if plots}

                # Write summary for this movie
                f.write('    <div class="plot-summary">\n')
                f.write(f'        <p>Total visualizations: {sum(category_counts.values())}</p>\n')
                f.write('        <ul>\n')
                for category, count in category_counts.items():
                    if count > 0:
                        f.write(f'            <li>{category.capitalize()}: {count}</li>\n')
                f.write('        </ul>\n')
                f.write('    </div>\n')

                # Write plot categories
                category_titles = {
                    'general': 'General Visualizations',
                    'gaze': 'Gaze Distribution Visualizations',
                    'fixation': 'Fixation Visualizations',
                    'saccade': 'Saccade Visualizations',
                    'pupil': 'Pupil Size Visualizations',
                    'social': 'Social Attention Visualizations'
                }

                for category, plots in movie_plots.items():
                    if plots:
                        f.write(f'    <div class="plot-category">\n')
                        f.write(f'        <h3>{category_titles.get(category, category.capitalize())}</h3>\n')
                        f.write(f'        <div class="plot-grid">\n')

                        for plot_path in plots:
                            plot_name = os.path.basename(plot_path)
                            rel_path = get_relative_path(plot_path)

                            f.write(f'            <div class="plot-container">\n')
                            f.write(f'                <div class="plot-title">{plot_name}</div>\n')
                            f.write(f'                <img src="{rel_path}" alt="{plot_name}">\n')
                            f.write(f'            </div>\n')

                        f.write(f'        </div>\n')
                        f.write(f'    </div>\n')

                f.write('</div>\n')

            # Write HTML footer
            f.write("""</body>
</html>""")

        print(f"Generated HTML report at {report_path}")
        return report_path
