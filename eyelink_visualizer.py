import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Tuple, Optional


class EyeLinkVisualizer:
    """
    A class for generating visualizations from parsed EyeLink data.

    This visualizer creates various plots to help understand eye movement patterns:
    - Scanpaths
    - Heatmaps
    - Fixation distributions
    - Pupil size dynamics
    - Head movement patterns
    - Time series of various metrics
    """

    def __init__(self, data_dict: Dict[str, pd.DataFrame], output_dir: str = 'plots',
                 screen_size: Tuple[int, int] = (1280, 1024)):
        """
        Initialize the visualizer with parsed EyeLink data.

        Args:
            data_dict: Dictionary with DataFrames containing parsed EyeLink data
            output_dir: Base directory to save the plots
            screen_size: Screen dimensions in pixels (width, height)
        """
        self.data = data_dict
        self.base_output_dir = output_dir
        self.screen_width, self.screen_height = screen_size

        # Set the file name for the output directory
        if 'participant_id' in self.data:
            self.file_name = self.data['participant_id']
        else:
            self.file_name = 'unknown_participant'

        # Create output directory
        self.output_dir = os.path.join(self.base_output_dir, self.file_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Default plot style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.default_figsize = (12, 8)
        self.dpi = 150

        # Color scheme
        self.colors = {
            'left_eye': '#1f77b4',  # Blue
            'right_eye': '#ff7f0e',  # Orange
            'fixation': '#2ca02c',  # Green
            'saccade': '#d62728',  # Red
            'blink': '#9467bd',  # Purple
            'head_movement': '#8c564b'  # Brown
        }

    def save_plot(self, plot_name: str, fig=None, tight=True):
        """Save plot to the output directory"""
        if fig is None:
            fig = plt.gcf()

        # Create full path
        full_path = os.path.join(self.output_dir, f"{plot_name}.png")

        if tight:
            plt.tight_layout()

        fig.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved plot to {full_path}")
        plt.close(fig)

    def plot_scanpath(self, time_window: Optional[Tuple[int, int]] = None,
                      max_points: int = 5000, alpha: float = 0.7):
        """
        Plot the scanpath (eye movement trajectory) for both eyes.

        Args:
            time_window: Optional tuple with (start_time, end_time) to plot a specific segment
            max_points: Maximum number of points to plot (for performance)
            alpha: Transparency of the points
        """
        if 'unified_eye_metrics' not in self.data:
            print("Unified eye metrics data not found. Cannot plot scanpath.")
            return

        df = self.data['unified_eye_metrics']

        # Apply time window if specified
        if time_window:
            start_time, end_time = time_window
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

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

        # Set axis limits to screen dimensions
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)  # Invert y-axis to match screen coordinates

        # Add annotations
        ax.set_title('Eye Movement Scanpath', fontsize=16)
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        ax.legend(loc='best')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save the plot
        self.save_plot('scanpath', fig)

    def plot_heatmap(self, bin_size: int = 20, smoothing: int = 3,
                     eye: str = 'left', only_fixations: bool = True):
        """
        Plot a heatmap of eye gaze distribution.

        Args:
            bin_size: Size of bins for the heatmap in pixels
            smoothing: Gaussian smoothing factor
            eye: Which eye to plot ('left' or 'right')
            only_fixations: Whether to use only fixation data or all samples
        """
        if 'unified_eye_metrics' not in self.data:
            print("Unified eye metrics data not found. Cannot plot heatmap.")
            return

        df = self.data['unified_eye_metrics']

        # Filter data based on eye and fixation settings
        x_col, y_col = f'x_{eye}', f'y_{eye}'
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Eye position data for {eye} eye not found.")
            return

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
        ax.set_title(f'Gaze Heatmap ({eye.capitalize()} Eye)', fontsize=16)
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)

        # Save the plot
        self.save_plot(f'heatmap_{eye}', fig)

    def plot_fixation_duration_distribution(self):
        """Plot the distribution of fixation durations for both eyes."""
        has_fixation_data = False

        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Plot left eye fixation durations if available
        if 'fixations_left' in self.data and not self.data['fixations_left'].empty:
            df_left = self.data['fixations_left']
            if 'duration' in df_left.columns:
                sns.histplot(df_left['duration'], ax=ax, color=self.colors['left_eye'],
                             alpha=0.7, label='Left Eye', kde=True)
                has_fixation_data = True

        # Plot right eye fixation durations if available
        if 'fixations_right' in self.data and not self.data['fixations_right'].empty:
            df_right = self.data['fixations_right']
            if 'duration' in df_right.columns:
                sns.histplot(df_right['duration'], ax=ax, color=self.colors['right_eye'],
                             alpha=0.7, label='Right Eye', kde=True)
                has_fixation_data = True

        if not has_fixation_data:
            print("No fixation data found.")
            plt.close(fig)
            return

        # Add annotations
        ax.set_title('Fixation Duration Distribution', fontsize=16)
        ax.set_xlabel('Fixation Duration (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='best')

        # Save the plot
        self.save_plot('fixation_duration_distribution', fig)

    def plot_saccade_amplitude_distribution(self):
        """Plot the distribution of saccade amplitudes for both eyes."""
        has_saccade_data = False

        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Plot left eye saccade amplitudes if available
        if 'saccades_left' in self.data and not self.data['saccades_left'].empty:
            df_left = self.data['saccades_left']
            if 'amplitude' in df_left.columns:
                sns.histplot(df_left['amplitude'], ax=ax, color=self.colors['left_eye'],
                             alpha=0.7, label='Left Eye', kde=True)
                has_saccade_data = True

        # Plot right eye saccade amplitudes if available
        if 'saccades_right' in self.data and not self.data['saccades_right'].empty:
            df_right = self.data['saccades_right']
            if 'amplitude' in df_right.columns:
                sns.histplot(df_right['amplitude'], ax=ax, color=self.colors['right_eye'],
                             alpha=0.7, label='Right Eye', kde=True)
                has_saccade_data = True

        if not has_saccade_data:
            print("No saccade data found.")
            plt.close(fig)
            return

        # Add annotations
        ax.set_title('Saccade Amplitude Distribution', fontsize=16)
        ax.set_xlabel('Saccade Amplitude (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='best')

        # Save the plot
        self.save_plot('saccade_amplitude_distribution', fig)

    def plot_pupil_size_timeseries(self, window_size: int = 100):
        """
        Plot pupil size over time with a moving average smoothing.

        Args:
            window_size: Size of the moving average window
        """
        if 'unified_eye_metrics' not in self.data:
            print("Unified eye metrics data not found. Cannot plot pupil size time series.")
            return

        df = self.data['unified_eye_metrics']

        # Check if pupil data exists
        if 'pupil_left' not in df.columns and 'pupil_right' not in df.columns:
            print("No pupil size data found.")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Plot left eye pupil size if available
        if 'pupil_left' in df.columns:
            # Apply moving average
            pupil_left_smooth = df['pupil_left'].rolling(window=window_size, center=True, min_periods=1).mean()

            # Convert timestamp to seconds for better readability
            time_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0

            ax.plot(time_sec, pupil_left_smooth, color=self.colors['left_eye'],
                    linewidth=1.5, label='Left Eye')

        # Plot right eye pupil size if available
        if 'pupil_right' in df.columns:
            # Apply moving average
            pupil_right_smooth = df['pupil_right'].rolling(window=window_size, center=True, min_periods=1).mean()

            # Convert timestamp to seconds for better readability
            time_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0

            ax.plot(time_sec, pupil_right_smooth, color=self.colors['right_eye'],
                    linewidth=1.5, label='Right Eye')

        # Add blink indicators if available
        if 'is_blink_left' in df.columns:
            blink_times = time_sec[df['is_blink_left']]
            if len(blink_times) > 0:
                ymin, ymax = ax.get_ylim()
                ax.vlines(blink_times, ymin, ymax, color=self.colors['blink'],
                          alpha=0.3, label='Blinks')

        # Add annotations
        ax.set_title('Pupil Size Over Time', fontsize=16)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Pupil Size', fontsize=12)
        ax.legend(loc='best')

        # Save the plot
        self.save_plot('pupil_size_timeseries', fig)

    def plot_head_movement(self, window_size: int = 100):
        """
        Plot head movement magnitude over time.

        Args:
            window_size: Size of the moving average window
        """
        if 'unified_eye_metrics' not in self.data:
            print("Unified eye metrics data not found. Cannot plot head movement.")
            return

        df = self.data['unified_eye_metrics']

        # Check if head movement data exists
        if 'head_movement_magnitude' not in df.columns:
            # Try to calculate it if we have the necessary columns
            if 'x_left' in df.columns and 'cr_left' in df.columns:
                df['head_movement_left_x'] = df['x_left'] - df['cr_left']
                df['head_movement_right_x'] = df['x_right'] - df['cr_right']

                df['head_movement_magnitude'] = np.sqrt(
                    df['head_movement_left_x'] ** 2 + df['head_movement_right_x'] ** 2
                )
            else:
                print("No head movement data available.")
                return

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Apply moving average
        head_move_smooth = df['head_movement_magnitude'].rolling(window=window_size, center=True, min_periods=1).mean()

        # Convert timestamp to seconds for better readability
        time_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0

        # Plot head movement
        ax.plot(time_sec, head_move_smooth, color=self.colors['head_movement'],
                linewidth=1.5, label='Head Movement')

        # Add annotations
        ax.set_title('Head Movement Magnitude Over Time', fontsize=16)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Head Movement Magnitude (pixels)', fontsize=12)
        ax.legend(loc='best')

        # Save the plot
        self.save_plot('head_movement', fig)

    def plot_velocity_profile(self, window_size: int = 50, percentile: float = 99.5):
        """
        Plot the velocity profile of eye movements.

        Args:
            window_size: Size of the moving average window
            percentile: Upper percentile for y-axis limit to handle outliers
        """
        if 'unified_eye_metrics' not in self.data:
            print("Unified eye metrics data not found. Cannot plot velocity profile.")
            return

        df = self.data['unified_eye_metrics']

        # Check if velocity data exists or calculate it
        for eye in ['left', 'right']:
            velocity_col = f'gaze_velocity_{eye}'

            if velocity_col not in df.columns:
                # Calculate velocity if we have position data
                x_col, y_col = f'x_{eye}', f'y_{eye}'
                if x_col in df.columns and y_col in df.columns:
                    # Calculate position difference
                    x_diff = df[x_col].diff()
                    y_diff = df[y_col].diff()

                    # Calculate time difference in seconds
                    time_diff = df['timestamp'].diff() / 1000.0

                    # Calculate Euclidean distance
                    distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

                    # Calculate velocity (pixels/second)
                    df[velocity_col] = distance / time_diff

                    # Replace infinite values with NaN
                    df[velocity_col].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check if we have velocity data after calculations
        if 'gaze_velocity_left' not in df.columns and 'gaze_velocity_right' not in df.columns:
            print("Could not calculate velocity data.")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Convert timestamp to seconds for better readability
        time_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0

        # Plot velocity profiles for both eyes if available
        for eye, color in [('left', self.colors['left_eye']), ('right', self.colors['right_eye'])]:
            velocity_col = f'gaze_velocity_{eye}'

            if velocity_col in df.columns:
                # Apply moving average to smooth the data
                velocity_smooth = df[velocity_col].rolling(window=window_size, center=True, min_periods=1).mean()

                ax.plot(time_sec, velocity_smooth, color=color, linewidth=1.5, label=f'{eye.capitalize()} Eye')

                # Mark saccades if available
                saccade_col = f'is_saccade_{eye}'
                if saccade_col in df.columns:
                    saccade_times = time_sec[df[saccade_col]]
                    if len(saccade_times) > 0:
                        ymin, ymax = ax.get_ylim()
                        ax.vlines(saccade_times, 0, velocity_smooth[df[saccade_col]],
                                  color=self.colors['saccade'], alpha=0.3)

        # Set y-axis limit to handle extreme outliers
        if 'gaze_velocity_left' in df.columns:
            upper_limit = np.nanpercentile(df['gaze_velocity_left'], percentile)
            ax.set_ylim(0, upper_limit)
        elif 'gaze_velocity_right' in df.columns:
            upper_limit = np.nanpercentile(df['gaze_velocity_right'], percentile)
            ax.set_ylim(0, upper_limit)

        # Add saccade threshold line (typical threshold is around 30-50 deg/s)
        # Note: This is approximate and depends on calibration
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Typical Saccade Threshold')

        # Add annotations
        ax.set_title('Eye Movement Velocity Profile', fontsize=16)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Velocity (pixels/second)', fontsize=12)
        ax.legend(loc='best')

        # Save the plot
        self.save_plot('velocity_profile', fig)

    def plot_all(self):
        """Generate all available plots."""
        print(f"Generating plots for {self.file_name}...")

        # Check if we have eye tracking data
        if 'unified_eye_metrics' not in self.data:
            print("No unified eye metrics data found. Cannot generate plots.")
            return

        if self.data['unified_eye_metrics'].empty:
            print("Unified eye metrics dataframe is empty. Cannot generate plots.")
            return

        # Basic visualizations
        try:
            self.plot_scanpath()
        except Exception as e:
            print(f"Error generating scanpath plot: {e}")

        try:
            self.plot_heatmap(eye='left')
        except Exception as e:
            print(f"Error generating left eye heatmap: {e}")

        try:
            self.plot_heatmap(eye='right')
        except Exception as e:
            print(f"Error generating right eye heatmap: {e}")

        try:
            self.plot_fixation_duration_distribution()
        except Exception as e:
            print(f"Error generating fixation duration distribution: {e}")

        try:
            self.plot_saccade_amplitude_distribution()
        except Exception as e:
            print(f"Error generating saccade amplitude distribution: {e}")

        # Time series visualizations
        try:
            self.plot_pupil_size_timeseries()
        except Exception as e:
            print(f"Error generating pupil size timeseries: {e}")

        try:
            self.plot_head_movement()
        except Exception as e:
            print(f"Error generating head movement plot: {e}")

        try:
            self.plot_velocity_profile()
        except Exception as e:
            print(f"Error generating velocity profile: {e}")

        # Advanced visualizations
        try:
            self.plot_fixation_density_comparison()
        except Exception as e:
            print(f"Error generating fixation density comparison: {e}")

        try:
            self.plot_fixation_saccade_distribution()
        except Exception as e:
            print(f"Error generating fixation saccade distribution: {e}")

        print(f"All available plots saved to {self.output_dir}")

    # Add this debugging method to the EyeLinkVisualizer class
    def debug_data(self, method_name):
        """
        Print detailed debug information about the available data.

        Args:
            method_name: Name of the method requesting debug info
        """
        print(f"\n=== DEBUG INFO FOR {method_name} ===")

        # Print what keys are available in self.data
        print(f"Available keys in data dictionary: {list(self.data.keys())}")

        # Check fixation data
        if 'fixations_left' in self.data:
            print(f"fixations_left exists and has {len(self.data['fixations_left'])} rows")
            if not self.data['fixations_left'].empty:
                print(f"  Columns: {list(self.data['fixations_left'].columns)}")
                print(
                    f"  First row: {self.data['fixations_left'].iloc[0].to_dict() if len(self.data['fixations_left']) > 0 else 'Empty'}")
            else:
                print("  fixations_left is empty")
        else:
            print("fixations_left is not in data dictionary")

        if 'fixations_right' in self.data:
            print(f"fixations_right exists and has {len(self.data['fixations_right'])} rows")
            if not self.data['fixations_right'].empty:
                print(f"  Columns: {list(self.data['fixations_right'].columns)}")
                print(
                    f"  First row: {self.data['fixations_right'].iloc[0].to_dict() if len(self.data['fixations_right']) > 0 else 'Empty'}")
            else:
                print("  fixations_right is empty")
        else:
            print("fixations_right is not in data dictionary")

        # Check saccade data
        if 'saccades_left' in self.data:
            print(f"saccades_left exists and has {len(self.data['saccades_left'])} rows")
            if not self.data['saccades_left'].empty:
                print(f"  Columns: {list(self.data['saccades_left'].columns)}")
                print(
                    f"  First row: {self.data['saccades_left'].iloc[0].to_dict() if len(self.data['saccades_left']) > 0 else 'Empty'}")
            else:
                print("  saccades_left is empty")
        else:
            print("saccades_left is not in data dictionary")

        if 'saccades_right' in self.data:
            print(f"saccades_right exists and has {len(self.data['saccades_right'])} rows")
            if not self.data['saccades_right'].empty:
                print(f"  Columns: {list(self.data['saccades_right'].columns)}")
                print(
                    f"  First row: {self.data['saccades_right'].iloc[0].to_dict() if len(self.data['saccades_right']) > 0 else 'Empty'}")
            else:
                print("  saccades_right is empty")
        else:
            print("saccades_right is not in data dictionary")

        # Check unified metrics
        if 'unified_eye_metrics' in self.data:
            print(f"unified_eye_metrics exists and has {len(self.data['unified_eye_metrics'])} rows")
            if not self.data['unified_eye_metrics'].empty:
                print(f"  Columns: {list(self.data['unified_eye_metrics'].columns)}")
            else:
                print("  unified_eye_metrics is empty")
        else:
            print("unified_eye_metrics is not in data dictionary")

        print("=== END DEBUG INFO ===\n")


    def plot_fixation_density_comparison(self):
        """Plot a comparison of fixation densities between left and right eyes."""
        print("\nAttempting to plot fixation density comparison...")

        # Print debug info
        self.debug_data("plot_fixation_density_comparison")

        # Extract and validate fixation data
        has_left_data = False
        has_right_data = False

        try:
            # Check left eye data
            if 'fixations_left' in self.data:
                df_left = self.data['fixations_left']
                print(f"Left eye fixation data: {len(df_left)} rows, Empty: {df_left.empty}")
                if not df_left.empty:
                    has_left_data = 'x' in df_left.columns and 'y' in df_left.columns
                    print(f"Left eye has position data: {has_left_data}")

            # Check right eye data
            if 'fixations_right' in self.data:
                df_right = self.data['fixations_right']
                print(f"Right eye fixation data: {len(df_right)} rows, Empty: {df_right.empty}")
                if not df_right.empty:
                    has_right_data = 'x' in df_right.columns and 'y' in df_right.columns
                    print(f"Right eye has position data: {has_right_data}")

            if not has_left_data and not has_right_data:
                print("Position data missing in fixation records for both eyes. Cannot generate plot.")
                return

            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize)
            print("Created figure")

            # Plot fixation points for left eye if available
            if has_left_data:
                print("Plotting left eye fixations...")
                df_left = self.data['fixations_left']

                # Safety check for empty values
                valid_rows = df_left.dropna(subset=['x', 'y'])
                print(f"Left eye valid rows: {len(valid_rows)} after dropping NaN values")

                if len(valid_rows) > 0:
                    # Determine size parameter
                    if 'duration' in valid_rows.columns:
                        size = valid_rows['duration'] / 20
                        size = size.fillna(30)  # Fill any NaN values with default size
                    else:
                        size = 30

                    print(f"Plotting {len(valid_rows)} left eye points")
                    ax.scatter(valid_rows['x'], valid_rows['y'],
                               color=self.colors['left_eye'], alpha=0.7,
                               label='Left Eye Fixations', s=size)

                    # Plot density contours if enough points
                    if len(valid_rows) >= 5:
                        try:
                            print("Attempting left eye KDE plot...")
                            sns.kdeplot(x=valid_rows['x'], y=valid_rows['y'],
                                        ax=ax, color=self.colors['left_eye'],
                                        alpha=0.5, fill=True, levels=5, thresh=0.05)
                        except Exception as e:
                            print(f"Could not plot left eye KDE contours: {e}")
                    else:
                        print(f"Not enough left eye points for KDE (need at least 5, got {len(valid_rows)})")
                else:
                    print("No valid left eye fixation points after filtering NaN values")

            # Plot fixation points for right eye if available
            if has_right_data:
                print("Plotting right eye fixations...")
                df_right = self.data['fixations_right']

                # Safety check for empty values
                valid_rows = df_right.dropna(subset=['x', 'y'])
                print(f"Right eye valid rows: {len(valid_rows)} after dropping NaN values")

                if len(valid_rows) > 0:
                    # Determine size parameter
                    if 'duration' in valid_rows.columns:
                        size = valid_rows['duration'] / 20
                        size = size.fillna(30)  # Fill any NaN values with default size
                    else:
                        size = 30

                    print(f"Plotting {len(valid_rows)} right eye points")
                    ax.scatter(valid_rows['x'], valid_rows['y'],
                               color=self.colors['right_eye'], alpha=0.7,
                               label='Right Eye Fixations', s=size)

                    # Plot density contours if enough points
                    if len(valid_rows) >= 5:
                        try:
                            print("Attempting right eye KDE plot...")
                            sns.kdeplot(x=valid_rows['x'], y=valid_rows['y'],
                                        ax=ax, color=self.colors['right_eye'],
                                        alpha=0.5, fill=True, levels=5, thresh=0.05)
                        except Exception as e:
                            print(f"Could not plot right eye KDE contours: {e}")
                    else:
                        print(f"Not enough right eye points for KDE (need at least 5, got {len(valid_rows)})")
                else:
                    print("No valid right eye fixation points after filtering NaN values")

            # Set axis limits
            ax.set_xlim(0, self.screen_width)
            ax.set_ylim(self.screen_height, 0)  # Invert y-axis

            # Add annotations
            ax.set_title('Fixation Density Comparison', fontsize=16)
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)

            # Add legend only if we have labeled data
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='best')

            # Save the plot
            print("Saving fixation density comparison plot...")
            self.save_plot('fixation_density_comparison', fig)
            print("Fixation density comparison plot saved successfully")

        except Exception as e:
            import traceback
            print(f"Error in plot_fixation_density_comparison: {e}")
            traceback.print_exc()

    # Modified plot_fixation_saccade_distribution with detailed debugging
    def plot_fixation_saccade_distribution(self):
        """Plot the spatial distribution and relation between fixations and saccades."""
        print("\nAttempting to plot fixation-saccade distribution...")

        # Print debug info
        self.debug_data("plot_fixation_saccade_distribution")

        try:
            # Check if we have fixation and saccade data
            has_fixation_data = False
            has_saccade_data = False

            # Check for fixation data
            if 'fixations_left' in self.data and not self.data['fixations_left'].empty:
                df_fix_left = self.data['fixations_left']
                has_fixation_data = 'x' in df_fix_left.columns and 'y' in df_fix_left.columns
                print(f"Left eye fixation data available: {has_fixation_data}")

            if not has_fixation_data and 'fixations_right' in self.data and not self.data['fixations_right'].empty:
                df_fix_right = self.data['fixations_right']
                has_fixation_data = 'x' in df_fix_right.columns and 'y' in df_fix_right.columns
                print(f"Right eye fixation data available: {has_fixation_data}")

            # Check for saccade data
            if 'saccades_left' in self.data and not self.data['saccades_left'].empty:
                df_sacc_left = self.data['saccades_left']
                saccade_cols = ['start_x', 'start_y', 'end_x', 'end_y']
                missing_cols = [col for col in saccade_cols if col not in df_sacc_left.columns]
                has_saccade_data = len(missing_cols) == 0
                print(f"Left eye saccade data available: {has_saccade_data}")
                if not has_saccade_data:
                    print(f"Missing columns in left eye saccade data: {missing_cols}")

            if not has_saccade_data and 'saccades_right' in self.data and not self.data['saccades_right'].empty:
                df_sacc_right = self.data['saccades_right']
                saccade_cols = ['start_x', 'start_y', 'end_x', 'end_y']
                missing_cols = [col for col in saccade_cols if col not in df_sacc_right.columns]
                has_saccade_data = len(missing_cols) == 0
                print(f"Right eye saccade data available: {has_saccade_data}")
                if not has_saccade_data:
                    print(f"Missing columns in right eye saccade data: {missing_cols}")

            # Need at least one type of data to proceed
            if not has_fixation_data and not has_saccade_data:
                print("Insufficient fixation or saccade data for spatial distribution plot.")
                return

            # Create figure
            fig, ax = plt.subplots(figsize=self.default_figsize)
            print("Created figure")

            # Plot fixation data if available
            has_left_fix = False
            has_right_fix = False

            if 'fixations_left' in self.data and not self.data['fixations_left'].empty:
                df_fix = self.data['fixations_left']
                if 'x' in df_fix.columns and 'y' in df_fix.columns:
                    # Filter out NaN values
                    valid_rows = df_fix.dropna(subset=['x', 'y'])
                    print(f"Valid left eye fixation rows: {len(valid_rows)}")

                    if len(valid_rows) > 0:
                        has_left_fix = True
                        # Plot fixation points with size proportional to duration
                        if 'duration' in valid_rows.columns:
                            size = valid_rows['duration'] / 20
                            size = size.fillna(30)  # Default size for NaN values
                        else:
                            size = 30

                        print(f"Plotting {len(valid_rows)} left eye fixation points")
                        ax.scatter(valid_rows['x'], valid_rows['y'], s=size,
                                   color=self.colors['left_eye'], alpha=0.7, label='Left Eye Fixations')

            if not has_left_fix and 'fixations_right' in self.data and not self.data['fixations_right'].empty:
                df_fix = self.data['fixations_right']
                if 'x' in df_fix.columns and 'y' in df_fix.columns:
                    # Filter out NaN values
                    valid_rows = df_fix.dropna(subset=['x', 'y'])
                    print(f"Valid right eye fixation rows: {len(valid_rows)}")

                    if len(valid_rows) > 0:
                        has_right_fix = True
                        # Plot fixation points with size proportional to duration
                        if 'duration' in valid_rows.columns:
                            size = valid_rows['duration'] / 20
                            size = size.fillna(30)  # Default size for NaN values
                        else:
                            size = 30

                        print(f"Plotting {len(valid_rows)} right eye fixation points")
                        ax.scatter(valid_rows['x'], valid_rows['y'], s=size,
                                   color=self.colors['right_eye'], alpha=0.7, label='Right Eye Fixations')

            # Plot saccade data if available
            has_saccades = False

            if 'saccades_left' in self.data and not self.data['saccades_left'].empty:
                df_sacc = self.data['saccades_left']
                if all(col in df_sacc.columns for col in ['start_x', 'start_y', 'end_x', 'end_y']):
                    # Filter out NaN values
                    valid_rows = df_sacc.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
                    print(f"Valid left eye saccade rows: {len(valid_rows)}")

                    if len(valid_rows) > 0:
                        has_saccades = True
                        print(f"Plotting {len(valid_rows)} left eye saccade trajectories")
                        # Plot saccade trajectories
                        for _, sacc in valid_rows.iterrows():
                            ax.plot([sacc['start_x'], sacc['end_x']], [sacc['start_y'], sacc['end_y']],
                                    color=self.colors['saccade'], linewidth=0.5, alpha=0.3)

            if not has_saccades and 'saccades_right' in self.data and not self.data['saccades_right'].empty:
                df_sacc = self.data['saccades_right']
                if all(col in df_sacc.columns for col in ['start_x', 'start_y', 'end_x', 'end_y']):
                    # Filter out NaN values
                    valid_rows = df_sacc.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
                    print(f"Valid right eye saccade rows: {len(valid_rows)}")

                    if len(valid_rows) > 0:
                        has_saccades = True
                        print(f"Plotting {len(valid_rows)} right eye saccade trajectories")
                        # Plot saccade trajectories
                        for _, sacc in valid_rows.iterrows():
                            ax.plot([sacc['start_x'], sacc['end_x']], [sacc['start_y'], sacc['end_y']],
                                    color=self.colors['saccade'], linewidth=0.5, alpha=0.3)

            # At this point, if we still don't have any data to plot, return
            if not has_left_fix and not has_right_fix and not has_saccades:
                print("No valid fixation or saccade data to plot after filtering NaN values.")
                return

            # Set axis limits
            ax.set_xlim(0, self.screen_width)
            ax.set_ylim(self.screen_height, 0)  # Invert y-axis

            # Add annotations
            ax.set_title('Fixation-Saccade Spatial Distribution', fontsize=16)
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)

            # Add custom legend for saccades if we have saccade data
            if has_saccades:
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color=self.colors['saccade'], lw=1)]
                custom_labels = ['Saccades']

                # Add the custom legend alongside the auto-generated one
                handles, labels = ax.get_legend_handles_labels()
                if handles:  # Only if we have other legend entries
                    print("Adding legend with fixations and saccades")
                    first_legend = ax.legend(loc='upper right')
                    ax.add_artist(first_legend)
                    ax.legend(custom_lines, custom_labels, loc='upper left')
                else:
                    print("Adding legend with saccades only")
                    ax.legend(custom_lines, custom_labels, loc='best')
            else:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    print("Adding legend with fixations only")
                    ax.legend(loc='best')

            # Save the plot
            print("Saving fixation-saccade distribution plot...")
            self.save_plot('fixation_saccade_distribution', fig)
            print("Fixation-saccade distribution plot saved successfully")

        except Exception as e:
            import traceback
            print(f"Error in plot_fixation_saccade_distribution: {e}")
            traceback.print_exc()

def generate_visualizations(data_dict: Dict[str, pd.DataFrame],
                            output_dir: str = 'plots',
                            screen_size: Tuple[int, int] = (1280, 1024)):
    """
    Generate all visualizations for the given parsed EyeLink data.

    Args:
        data_dict: Dictionary with DataFrames containing parsed EyeLink data
        output_dir: Base directory to save the plots
        screen_size: Screen dimensions in pixels (width, height)

    Returns:
        Path to the directory containing the generated plots
    """
    visualizer = EyeLinkVisualizer(data_dict, output_dir, screen_size)
    visualizer.plot_all()
    return visualizer.output_dir


def generate_multiple_visualizations(parsed_data_list: List[Dict[str, pd.DataFrame]],
                                     output_dir: str = 'plots',
                                     screen_size: Tuple[int, int] = (1280, 1024)):
    """
    Generate visualizations for multiple parsed EyeLink data files.

    Args:
        parsed_data_list: List of dictionaries with parsed data
        output_dir: Base directory to save the plots
        screen_size: Screen dimensions in pixels

    Returns:
        List of paths to directories containing generated plots
    """
    output_dirs = []

    for data_dict in parsed_data_list:
        try:
            output_dir_i = generate_visualizations(data_dict, output_dir, screen_size)
            output_dirs.append(output_dir_i)
        except Exception as e:
            print(f"Error generating visualizations for {data_dict.get('participant_id', 'unknown')}: {e}")

    return output_dirs

