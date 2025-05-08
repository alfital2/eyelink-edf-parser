"""
Plot Generator Module

Provides functionality for generating advanced visualization plots for eye tracking data,
particularly focused on ROI (Region of Interest) analysis and social attention metrics.
"""

# Standard library imports
import os
import time
import json
import traceback
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd

# Data visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# PyQt5 imports
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QProgressBar, QMessageBox

# Local imports
from .eyelink_visualizer import MovieEyeTrackingVisualizer


class PlotGenerator:
    """
    Class for generating advanced visualization plots for eye tracking data with ROI analysis.
    """

    def __init__(self, screen_width, screen_height, visualization_results, movie_visualizations):
        """
        Initialize the PlotGenerator.
        
        Args:
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
            visualization_results: Dictionary to store visualization results
            movie_visualizations: Dictionary to store movie visualizations
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.visualization_results = visualization_results
        self.movie_visualizations = movie_visualizations
        self.report_path = None
        self.output_dir = None
        self.plots_dir = None  # This will be set by the GUI

    def create_advanced_roi_plots(self, movie, roi_durations, fixation_data, plots_dir,
                                  frame_keys, frame_range_map, polygon_check_cache, status_label, progress_bar,
                                  update_progress_func=None):
        """
        Generate advanced ROI-based social attention plots
        
        Args:
            movie: Name of the movie being analyzed
            roi_durations: Dictionary with ROI labels as keys and fixation counts as values
            fixation_data: DataFrame with fixation data
            plots_dir: Directory to save the plots
            frame_keys: Dictionary mapping frame numbers to ROI data
            frame_range_map: Dictionary for fast frame lookup
            polygon_check_cache: Cache for polygon checks 
            status_label: Label to update with status
            progress_bar: Progress bar to update
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator
        from collections import defaultdict
        
        # Initialize list to track created plots
        created_plots = []

        if not roi_durations:
            print("WARNING: No ROI hits found, cannot generate advanced plots")
            return

        print(f"DEBUG: Found {len(roi_durations)} ROIs with the following durations:")
        for roi, duration in sorted(roi_durations.items(), key=lambda x: x[1], reverse=True):
            print(f"DEBUG:   - {roi}: {duration} fixations")

        # Sort the roi_durations by value (descending) for consistent ordering in plots
        sorted_rois = sorted(roi_durations.items(), key=lambda x: x[1], reverse=True)
        roi_labels = [item[0] for item in sorted_rois]

        # Create a sequential fixation history to track ROI sequence over time
        # We'll use this for multiple plots
        status_label.setText("Analyzing fixation sequence...")
        progress_bar.setValue(97)
        QApplication.processEvents()

        # Initialize data structures for all plots
        roi_sequence = []  # For Fixation Sequence
        first_fixation_times = {roi: None for roi in roi_labels}  # For First Fixation Latency
        roi_revisits = {roi: 0 for roi in roi_labels}  # For Revisitation
        seen_rois = set()  # Track which ROIs have been seen

        # For duration distributions (new plot)
        fixation_durations = {roi: [] for roi in roi_labels}
        last_fixation_data = {'roi': None, 'start_time': None}

        # For Temporal Heatmap (new plot)
        # Create time bins (100ms bins across the entire stimulus)
        time_bin_size = 0.1  # 100ms bins
        max_time = fixation_data['timestamp'].max()
        min_time = fixation_data['timestamp'].min()
        total_duration = (max_time - min_time) / 1000.0  # in seconds
        num_bins = int(total_duration / time_bin_size) + 1

        # Create a 2D array: rows = ROIs, columns = time bins
        temporal_heatmap = {roi: np.zeros(num_bins) for roi in roi_labels}

        # For Transition Matrix
        transition_matrix = defaultdict(lambda: defaultdict(int))
        last_roi = None

        # Re-process fixations to collect data for all plots in a single pass
        fixation_count = len(fixation_data)
        fixation_data = fixation_data.sort_values(by='timestamp')  # Ensure time ordering

        # First timestamp for relative timing
        start_timestamp = fixation_data['timestamp'].min()

        # Process each fixation
        for idx, row in fixation_data.iterrows():
            if pd.isna(row['frame_number']):
                continue

            frame_num = int(row['frame_number'])

            # Find the nearest frame 
            nearest_frame = None
            for (start, end), frame in frame_range_map.items():
                if start <= frame_num < end:
                    nearest_frame = frame
                    break

            if nearest_frame is None:
                try:
                    nearest_frame = min(frame_keys.keys(), key=lambda x: abs(x - frame_num))
                except:
                    continue

            frame_distance = abs(nearest_frame - frame_num)
            if frame_distance > 1000:
                continue

            # Get the ROIs for this frame
            rois_in_frame = frame_keys[nearest_frame]

            # Get normalized coordinates
            if row['x_left'] > 1.0 or row['y_left'] > 1.0:
                x_norm = row['x_left'] / self.screen_width
                y_norm = row['y_left'] / self.screen_height
            else:
                x_norm = row['x_left']
                y_norm = row['y_left']

            # Find which ROI the fixation is in, if any
            current_roi = None
            for roi in rois_in_frame:
                if 'label' not in roi or 'coordinates' not in roi:
                    continue

                label = roi['label']
                coords = roi['coordinates']

                if label not in roi_labels:
                    continue  # Skip ROIs that didn't make it into the main plot

                # Use cached polygon checks
                cache_key = (tuple((coord['x'], coord['y']) for coord in coords), x_norm, y_norm)
                if cache_key in polygon_check_cache:
                    is_inside = polygon_check_cache[cache_key]
                else:
                    is_inside = self._point_in_polygon(x_norm, y_norm, coords)
                    polygon_check_cache[cache_key] = is_inside

                if is_inside:
                    current_roi = label
                    break

            if current_roi:
                # Add to sequence
                timestamp_sec = (row['timestamp'] - start_timestamp) / 1000.0  # Convert to seconds
                roi_sequence.append((timestamp_sec, current_roi))

                # Track fixation durations for the new ROI Fixation Duration Distribution plot
                if last_fixation_data['roi'] == current_roi and last_fixation_data['start_time'] is not None:
                    # Same ROI as last fixation - continue the duration
                    pass
                else:
                    # Different ROI or first fixation on this ROI
                    # If we were tracking a previous ROI, save its duration
                    if last_fixation_data['roi'] is not None and last_fixation_data['start_time'] is not None:
                        duration = timestamp_sec - last_fixation_data['start_time']
                        if duration > 0 and duration < 10:  # Filter out unreasonable durations
                            fixation_durations[last_fixation_data['roi']].append(duration)
                            if len(fixation_durations[last_fixation_data['roi']]) % 20 == 0:
                                print(
                                    f"DEBUG: Recorded fixation duration of {duration:.2f}s on {last_fixation_data['roi']}")

                    # Start tracking the new ROI
                    last_fixation_data['roi'] = current_roi
                    last_fixation_data['start_time'] = timestamp_sec

                # Debug - occasional sample of ROI hits
                if len(roi_sequence) % 100 == 0:
                    print(
                        f"DEBUG: Found ROI hit on '{current_roi}' at time {timestamp_sec:.2f}s (fixation #{len(roi_sequence)})")

                # First fixation time
                if first_fixation_times[current_roi] is None:
                    first_fixation_times[current_roi] = timestamp_sec
                    seen_rois.add(current_roi)
                    print(f"DEBUG: First fixation on '{current_roi}' at {timestamp_sec:.2f}s")
                elif current_roi in seen_rois:
                    # This ROI has been seen before, count as revisit
                    roi_revisits[current_roi] += 1

                # Update temporal heatmap
                # Calculate which time bin this fixation belongs to
                time_bin = int((timestamp_sec * 1000) / (time_bin_size * 1000))
                if time_bin < num_bins:
                    # Increment the count for this ROI at this time bin
                    temporal_heatmap[current_roi][time_bin] += 1

                # Transition matrix
                if last_roi is not None and last_roi != current_roi:
                    transition_matrix[last_roi][current_roi] += 1

                last_roi = current_roi
            else:
                # No ROI in focus - if we were tracking a fixation on an ROI, save its duration and reset
                timestamp_sec = (row['timestamp'] - start_timestamp) / 1000.0  # Convert to seconds
                if last_fixation_data['roi'] is not None and last_fixation_data['start_time'] is not None:
                    duration = timestamp_sec - last_fixation_data['start_time']
                    if duration > 0 and duration < 10:  # Filter out unreasonable durations
                        fixation_durations[last_fixation_data['roi']].append(duration)
                    last_fixation_data['roi'] = None
                    last_fixation_data['start_time'] = None

        # 1. ROI Social vs Non-Social Attention Balance Plot - SKIPPED
        # Variables needed for compatibility with later code
        social_rois = ['Face', 'Hand', 'Eyes', 'Mouth', 'Person', 'Body']
        nonsocial_rois = ['Background', 'Object', 'Bed', 'Couch', 'Torso', 'Floor', 'Wall', 'Toy']

        # Move directly to next plot in sequence
        if update_progress_func:
            update_progress_func(2, "Moving to next plot...")
        else:
            status_label.setText("Moving to next plot...")
            QApplication.processEvents()

        # Create mapping for all ROIs in the data
        roi_categories = {}
        for roi in roi_labels:
            # Try to categorize based on exact matches first
            if roi in social_rois:
                roi_categories[roi] = 'Social'
            elif roi in nonsocial_rois:
                roi_categories[roi] = 'Non-Social'
            else:
                # If no exact match, try partial matching
                if any(social_term in roi for social_term in ['face', 'hand', 'eye', 'mouth', 'person', 'body']):
                    roi_categories[roi] = 'Social'
                else:
                    roi_categories[roi] = 'Non-Social'

        print(f"DEBUG: ROI Categories: {roi_categories}")

        # Compute time spent looking at each ROI category
        social_time = 0
        nonsocial_time = 0
        other_time = 0

        for roi, duration in roi_durations.items():
            if roi in roi_categories:
                if roi_categories[roi] == 'Social':
                    social_time += duration
                elif roi_categories[roi] == 'Non-Social':
                    nonsocial_time += duration
            else:
                other_time += duration

        # Calculate total looking time and percentages
        total_time = social_time + nonsocial_time + other_time
        social_pct = (social_time / total_time * 100) if total_time > 0 else 0
        nonsocial_pct = (nonsocial_time / total_time * 100) if total_time > 0 else 0
        other_pct = (other_time / total_time * 100) if total_time > 0 else 0

        print(f"DEBUG: Social time: {social_time:.2f}s ({social_pct:.1f}%)")
        print(f"DEBUG: Non-social time: {nonsocial_time:.2f}s ({nonsocial_pct:.1f}%)")

        # Skip creating the social balance plot as requested
        # Define the path variables needed for compatibility with later code
        balance_filename = f"roi_social_balance_{movie.replace(' ', '_')}.png"
        balance_path = os.path.join(plots_dir, balance_filename)

        # Create the figure and axes to avoid NameError with ax2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Skip all pie chart visualization code
        # But keep the axes for the next plot
        # Sort ROIs by duration
        sorted_rois = sorted([(roi, time) for roi, time in roi_durations.items()],
                             key=lambda x: x[1], reverse=True)

        # Extract data for plotting
        roi_names = [item[0] for item in sorted_rois]
        roi_times = [item[1] for item in sorted_rois]

        # Define colors based on ROI category
        bar_colors = [
            '#ff9999' if roi in roi_categories and roi_categories[roi] == 'Social' else
            '#66b3ff' if roi in roi_categories and roi_categories[roi] == 'Non-Social' else
            '#c2c2f0'
            for roi in roi_names
        ]

        # Create the bar chart
        bars = ax2.barh(roi_names, roi_times, color=bar_colors)

        # Add social/non-social labels to the bars
        for i, (roi, time) in enumerate(zip(roi_names, roi_times)):
            category = roi_categories.get(roi, 'Other')
            ax2.text(time + 0.1, i, f"{category} ({time:.1f}s)",
                     va='center', fontsize=8, alpha=0.7)

        # Add title and labels
        # Skip creating and saving the social balance plot
        print(f"DEBUG: Skipping social balance plot generation (disabled)")

        # Skip adding to visualization results
        print(f"DEBUG: Skipping adding ROI Social Balance plot to visualization options")

        # 2. ROI Transition Matrix Plot
        if update_progress_func:
            update_progress_func(3, "Generating ROI Transition Matrix plot...")
        else:
            status_label.setText("Generating ROI Transition Matrix plot...")
            QApplication.processEvents()

        print(
            f"DEBUG: Creating ROI Transition Matrix with {sum(sum(v.values()) for v in transition_matrix.values())} transitions")
        print(f"DEBUG: Transition matrix has {len(transition_matrix)} source ROIs")

        if transition_matrix:
            # Create a dense representation for the heatmap
            transition_array = np.zeros((len(roi_labels), len(roi_labels)))

            # Fill the transition array
            for i, from_roi in enumerate(roi_labels):
                for j, to_roi in enumerate(roi_labels):
                    transition_array[i, j] = transition_matrix[from_roi][to_roi]

            # Create the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            # Use log scale if values range is large
            try:
                if np.max(transition_array) > 0:
                    if np.max(transition_array) / (np.min(transition_array[transition_array > 0]) or 1) > 10:
                        # Use log normalization for widely varying values
                        from matplotlib.colors import LogNorm
                        norm = LogNorm(vmin=max(1, np.min(transition_array[transition_array > 0])),
                                       vmax=max(2, np.max(transition_array)))
                        sns.heatmap(transition_array, cmap="YlOrRd", ax=ax,
                                    xticklabels=roi_labels, yticklabels=roi_labels,
                                    norm=norm, annot=True, fmt=".0f", linewidths=0.5)
                    else:
                        # Use regular normalization for more uniform values
                        sns.heatmap(transition_array, cmap="YlOrRd", ax=ax,
                                    xticklabels=roi_labels, yticklabels=roi_labels,
                                    annot=True, fmt=".0f", linewidths=0.5)
                else:
                    # Fallback for empty matrix
                    sns.heatmap(transition_array, cmap="YlOrRd", ax=ax,
                                xticklabels=roi_labels, yticklabels=roi_labels,
                                annot=True, fmt=".0f", linewidths=0.5)
            except Exception as e:
                print(f"ERROR in transition matrix heatmap: {e}")
                # Use a simpler approach if seaborn fails
                ax.imshow(transition_array, cmap="YlOrRd")
                ax.set_xticks(np.arange(len(roi_labels)))
                ax.set_yticks(np.arange(len(roi_labels)))
                ax.set_xticklabels(roi_labels)
                ax.set_yticklabels(roi_labels)

            # Add title and labels
            ax.set_title(f'ROI Transition Matrix for {movie}')
            ax.set_xlabel('To ROI')
            ax.set_ylabel('From ROI')

            # Adjust labels for readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Tight layout
            plt.tight_layout()

            # Save the plot
            matrix_filename = f"roi_transition_matrix_{movie.replace(' ', '_')}.png"
            matrix_path = os.path.join(plots_dir, matrix_filename)
            plt.savefig(matrix_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Add to visualization results
            self.visualization_results[movie]['social'].append(matrix_path)
            self.movie_visualizations[movie]["ROI Transition Matrix"] = matrix_path
            
            # Add to list of created plots
            if 'created_plots' in locals():
                created_plots.append(matrix_path)

        # 3. ROI First Fixation Latency Plot
        if update_progress_func:
            update_progress_func(4, "Generating ROI First Fixation Latency plot...")
        else:
            status_label.setText("Generating ROI First Fixation Latency plot...")
            QApplication.processEvents()

        if first_fixation_times:
            # Filter out ROIs that were never fixated
            valid_first_times = {roi: time for roi, time in first_fixation_times.items() if time is not None}

            if valid_first_times:
                # Sort by first fixation time
                sorted_latencies = sorted(valid_first_times.items(), key=lambda x: x[1])
                latency_rois = [item[0] for item in sorted_latencies]
                latency_times = [item[1] for item in sorted_latencies]

                # Create the bar chart
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot horizontal bars for better readability with many ROIs
                bars = ax.barh(latency_rois, latency_times, color='skyblue')

                # Add value labels on the bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                            f'{width:.2f}s', va='center')

                # Add title and labels
                ax.set_title(f'Time to First Fixation on Each ROI in {movie}')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('ROI')

                # Add gridlines
                ax.grid(True, linestyle='--', alpha=0.6, axis='x')

                # Tight layout
                plt.tight_layout()

                # Save the plot
                latency_filename = f"roi_first_fixation_latency_{movie.replace(' ', '_')}.png"
                latency_path = os.path.join(plots_dir, latency_filename)
                plt.savefig(latency_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Add to visualization results
                self.visualization_results[movie]['social'].append(latency_path)
                self.movie_visualizations[movie]["ROI First Fixation Latency"] = latency_path
                
                # Add to list of created plots
                if 'created_plots' in locals():
                    created_plots.append(latency_path)

        # ROI Dwell Time Comparison - Removed as it's now merged with ROI Attention Time plot
        # Skip this step in the overall progress
        # update_overall_progress(5, "Generating ROI Dwell Time Comparison plot...")

        # We don't need to modify total_plots here, that's handled in the calling function
        # This line was causing an UnboundLocalError
        # total_plots -= 1

        # ROI Revisitation Plot (Removed as per user request)
        # We still track roi_revisits data for potential future use, but don't generate the plot

        # Collect some statistics for debugging only
        if roi_revisits:
            # Filter to ROIs that were fixated at least once
            valid_revisits = {roi: count for roi, count in roi_revisits.items()
                              if first_fixation_times.get(roi) is not None}

            # Log some statistics but don't create the plot
            if valid_revisits:
                print(f"DEBUG: Revisit counts collected but plot generation skipped")
                for roi, count in sorted(valid_revisits.items(), key=lambda x: x[1], reverse=True):
                    print(f"DEBUG:   - {roi}: {count} revisits")

        # 6. ROI Fixation Duration Distribution plot (NEW)
        if update_progress_func:
            update_progress_func(6, "Generating ROI Fixation Duration Distribution plot...")
        else:
            status_label.setText("Generating ROI Fixation Duration Distribution plot...")
            QApplication.processEvents()

        print(
            f"DEBUG: Creating ROI Fixation Duration Distribution with data for {sum(1 for durations in fixation_durations.values() if durations)} ROIs")

        # Filter out ROIs with no duration data
        valid_durations = {roi: durs for roi, durs in fixation_durations.items() if durs}

        if valid_durations:
            # Print statistics about each ROI's durations
            for roi, durations in valid_durations.items():
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    print(f"DEBUG: {roi} has {len(durations)} fixations with average duration {avg_duration:.2f}s")

            # Create a violin plot or boxplot for each ROI's fixation durations
            try:
                # Create the figure
                fig, ax = plt.subplots(figsize=(12, 8))

                # Prepare data for the violin plot
                data_to_plot = []
                labels = []

                # Sort ROIs by median duration for better visualization
                sorted_rois = sorted(valid_durations.items(),
                                     key=lambda x: np.median(x[1]) if x[1] else 0,
                                     reverse=True)

                for roi, durations in sorted_rois:
                    if durations:  # Skip empty durations
                        data_to_plot.append(durations)
                        labels.append(roi)

                if data_to_plot:
                    # Select plot type based on number of data points
                    if all(len(d) >= 5 for d in data_to_plot):
                        # Use violin plot for sufficient data
                        sns.violinplot(data=data_to_plot, ax=ax, inner="box",
                                       palette="pastel", cut=0)

                        # Add individual points for more detail
                        sns.stripplot(data=data_to_plot, ax=ax, size=3, color=".3",
                                      alpha=0.4, jitter=True)
                    else:
                        # Use boxplot for less data
                        sns.boxplot(data=data_to_plot, ax=ax,
                                    palette="pastel", whis=1.5)

                        # Add individual points 
                        sns.stripplot(data=data_to_plot, ax=ax, size=4, color=".3",
                                      alpha=0.6, jitter=True)

                    # Set x-tick labels
                    ax.set_xticklabels(labels)

                    # Add title and labels
                    ax.set_title(f'Distribution of Fixation Durations by ROI in {movie}')
                    ax.set_xlabel('ROI')
                    ax.set_ylabel('Fixation Duration (seconds)')

                    # Rotate labels for better readability
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                    # Add a legend for quartiles
                    from matplotlib.patches import Patch
                    from matplotlib.lines import Line2D

                    # Create custom legend elements
                    legend_elements = [
                        Line2D([0], [0], color='k', lw=2, label='Median'),
                        Patch(facecolor='b', alpha=0.3, label='Durations Distribution')
                    ]

                    ax.legend(handles=legend_elements, loc='upper right')

                    # Add gridlines
                    ax.grid(True, linestyle='--', alpha=0.6, axis='y')

                    # Tight layout
                    plt.tight_layout()

                    # Save the plot
                    duration_filename = f"roi_fixation_duration_distribution_{movie.replace(' ', '_')}.png"
                    duration_path = os.path.join(plots_dir, duration_filename)

                    # Make sure the directory exists
                    os.makedirs(os.path.dirname(duration_path), exist_ok=True)
                    print(f"DEBUG: Saving ROI Fixation Duration Distribution plot to: {duration_path}")

                    plt.savefig(duration_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)

                    # Verify file was created
                    if os.path.exists(duration_path):
                        print(f"DEBUG: Successfully saved duration distribution plot to {duration_path}")
                    else:
                        print(f"ERROR: Failed to save duration distribution plot to {duration_path}")

                    # Add to visualization results
                    self.visualization_results[movie]['social'].append(duration_path)
                    print(f"DEBUG: Added ROI Fixation Duration Distribution plot to visualization_results")

                    # Make sure the movie is in the dictionary
                    if movie not in self.movie_visualizations:
                        self.movie_visualizations[movie] = {}

                    # Add to movie_visualizations with display name
                    self.movie_visualizations[movie]["ROI Fixation Duration Distribution"] = duration_path
                    print(f"DEBUG: Added ROI Fixation Duration Distribution plot to movie_visualizations")
                    
                    # Add to list of created plots
                    if 'created_plots' in locals():
                        created_plots.append(duration_path)
            except Exception as e:
                print(f"ERROR creating ROI Fixation Duration Distribution plot: {e}")
                import traceback
                traceback.print_exc()

        # 7. ROI Temporal Heatmap (NEW)
        if update_progress_func:
            update_progress_func(7, "Generating ROI Temporal Heatmap...")
            progress_bar.setValue(0)  # Reset for new task
        else:
            status_label.setText("Generating ROI Temporal Heatmap...")
            progress_bar.setValue(0)  # Reset for new task
            QApplication.processEvents()
            
        # Track the number of plots generated in this method
        print(f"DEBUG: Total plots created in advanced ROI plots: {len(created_plots)}")

        print(f"DEBUG: Creating ROI Temporal Heatmap with {num_bins} time bins")

        # Check if we have data for the temporal heatmap
        if temporal_heatmap:
            try:
                # Filter out empty ROIs (those with no fixations)
                active_rois = [roi for roi in roi_labels
                               if np.sum(temporal_heatmap[roi]) > 0]

                if active_rois:
                    # Create the figure
                    fig, ax = plt.subplots(figsize=(15, 8))

                    # Prepare data for the heatmap
                    heatmap_data = np.array([temporal_heatmap[roi] for roi in active_rois])

                    # Apply smoothing to the heatmap data for better visualization
                    from scipy.ndimage import gaussian_filter1d
                    smoothed_data = np.copy(heatmap_data)
                    for i in range(len(smoothed_data)):
                        # Apply moderate smoothing
                        smoothed_data[i] = gaussian_filter1d(smoothed_data[i], sigma=2)

                    # Create the heatmap
                    im = ax.imshow(smoothed_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')

                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Fixation Intensity')

                    # Set y-tick labels (ROI names)
                    ax.set_yticks(np.arange(len(active_rois)))
                    ax.set_yticklabels(active_rois)

                    # Set x-tick labels (time in seconds)
                    # Place ticks every 1 second
                    seconds_per_tick = 1.0  # 1 second between ticks
                    ticks_per_bin = int(seconds_per_tick / time_bin_size)
                    tick_positions = np.arange(0, num_bins, ticks_per_bin)
                    tick_labels = [f"{t * time_bin_size:.1f}" for t in tick_positions]

                    # Only show a subset of ticks if there are too many
                    if len(tick_positions) > 20:
                        tick_positions = tick_positions[::len(tick_positions) // 20]
                        tick_labels = [f"{t * time_bin_size:.1f}" for t in tick_positions]

                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels)

                    # Add title and labels
                    ax.set_title(f'ROI Attention Over Time in {movie}')
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('ROI')

                    # Add gridlines
                    ax.grid(False)

                    # Add time markers for significant events
                    # Calculate mean fixation time per ROI for annotation
                    for i, roi in enumerate(active_rois):
                        if first_fixation_times[roi] is not None:
                            # Mark the first fixation time
                            first_time_bin = int((first_fixation_times[roi] * 1000) / (time_bin_size * 1000))
                            if first_time_bin < num_bins:
                                ax.plot([first_time_bin, first_time_bin], [i - 0.4, i + 0.4], 'g-', linewidth=1)
                                ax.text(first_time_bin, i + 0.5, 'First', color='green',
                                        ha='center', va='bottom', fontsize=8)

                    # Tight layout
                    plt.tight_layout()

                    # Save the plot
                    temporal_filename = f"roi_temporal_heatmap_{movie.replace(' ', '_')}.png"
                    temporal_path = os.path.join(plots_dir, temporal_filename)

                    # Make sure the directory exists
                    os.makedirs(os.path.dirname(temporal_path), exist_ok=True)
                    print(f"DEBUG: Saving ROI Temporal Heatmap to: {temporal_path}")

                    plt.savefig(temporal_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)

                    # Verify file was created
                    if os.path.exists(temporal_path):
                        print(f"DEBUG: Successfully saved temporal heatmap to {temporal_path}")
                    else:
                        print(f"ERROR: Failed to save temporal heatmap to {temporal_path}")

                    # Add to visualization results
                    self.visualization_results[movie]['social'].append(temporal_path)
                    print(f"DEBUG: Added ROI Temporal Heatmap to visualization_results")

                    # Make sure the movie is in the dictionary
                    if movie not in self.movie_visualizations:
                        self.movie_visualizations[movie] = {}

                    # Add to movie_visualizations with display name
                    self.movie_visualizations[movie]["ROI Temporal Heatmap"] = temporal_path
                    print(f"DEBUG: Added ROI Temporal Heatmap to movie_visualizations")
                    
                    # Add to list of created plots
                    if 'created_plots' in locals():
                        created_plots.append(temporal_path)
            except Exception as e:
                print(f"ERROR creating ROI Temporal Heatmap: {e}")
                import traceback
                traceback.print_exc()
                
        # Return the list of created plots
        return created_plots

    def generate_social_attention_plots(self):
        """Generate social attention plots based on loaded ROI file"""
        if not hasattr(self, 'roi_file_path') or not self.roi_file_path:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                "ROI File Required",
                "Please load an ROI file first."
            )
            return

        # Get the currently selected movie
        if not hasattr(self, 'movie_combo') or self.movie_combo.count() == 0 or self.movie_combo.currentIndex() < 0:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                "No Movie Selected",
                "Please select a movie for analysis."
            )
            return

        movie = self.movie_combo.currentText()

        # Get the movie data
        movie_data = self._get_movie_data(movie)
        if movie_data is None or movie_data.get("data") is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                "No Data Available",
                f"Could not find eye tracking data for {movie}."
            )
            return

        # Load the ROI file
        import json
        try:
            with open(self.roi_file_path, 'r') as f:
                raw_roi_data = json.load(f)

            # Check for the new format with "annotations" key
            if "annotations" in raw_roi_data:
                print(f"DEBUG: Found 'annotations' key in ROI file, using new format")
                roi_data = raw_roi_data["annotations"]
            else:
                print(f"DEBUG: Using legacy ROI format")
                roi_data = raw_roi_data

            print(f"DEBUG: Processed ROI data contains {len(roi_data)} frame entries")
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                None,
                "Error Loading ROI File",
                f"Failed to load ROI data: {str(e)}"
            )
            return

        # Create a progress dialog with a progress bar
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
        progress_dialog = QDialog()
        progress_dialog.setWindowTitle("Generating Social Attention Plots")
        progress_dialog.setFixedSize(400, 100)

        dialog_layout = QVBoxLayout(progress_dialog)

        # Status label
        status_label = QLabel("Preparing data...")
        dialog_layout.addWidget(status_label)

        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        dialog_layout.addWidget(progress_bar)

        # Define the total number of plot generation steps
        # 1. ROI Attention Time (basic)
        # 2. ROI Social vs Non-Social Attention Balance (disabled)
        # 3. ROI Transition Matrix
        # 4. ROI First Fixation Latency
        # 5. ROI Dwell Time Comparison (merged with Attention Time)
        # 6. ROI Fixation Duration Distribution
        # 7. ROI Temporal Heatmap
        total_plots = 7
        current_plot = 0

        # Helper function to update overall progress
        def update_overall_progress(plot_num, status_text):
            nonlocal current_plot
            current_plot = plot_num
            overall_progress = int((plot_num / total_plots) * 100)
            progress_bar.setValue(overall_progress)
            status_label.setText(f"Generating plots for {movie}... ({plot_num}/{total_plots}): {status_text}")
            QApplication.processEvents()

            # Add a global progress indicator variable to be used in each plot
            global current_plot_progress
            current_plot_progress = f"{plot_num}/{total_plots}"

        # Show the dialog (non-modal)
        progress_dialog.show()
        QApplication.processEvents()

        try:
            # Create a plot showing time spent on each ROI
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            data = movie_data["data"]

            print(f"DEBUG: Starting social attention plot generation")
            print(f"DEBUG: ROI data keys: {list(roi_data.keys())}")
            print(f"DEBUG: Data shape: {data.shape}, columns: {data.columns}")

            # Get frame numbers from both data and ROI file
            if 'frame_number' not in data.columns:
                raise ValueError("Eye tracking data does not contain frame numbers")

            print(f"DEBUG: Frame numbers in data: {data['frame_number'].min()} to {data['frame_number'].max()}")
            status_label.setText("Processing ROI data...")
            progress_bar.setValue(10)
            QApplication.processEvents()

            # Count time spent on each ROI
            roi_durations = {}

            # First, convert frame keys to integers if they're strings - OPTIMIZATION: Do this once and cache
            frame_keys = {}
            for key in roi_data.keys():
                try:
                    frame_keys[int(key)] = roi_data[key]
                    # Only print a sample for debugging
                    if len(frame_keys) <= 3:
                        roi_sample = roi_data[key]
                        if roi_sample:
                            roi_labels = [roi['label'] for roi in roi_sample if 'label' in roi]
                            print(f"DEBUG: Frame {key} has {len(roi_sample)} ROIs: {roi_labels}")
                except ValueError:
                    print(f"DEBUG: Skipping non-integer key: {key}")
                    continue

            if not frame_keys:
                print(f"DEBUG: No valid frame keys found in ROI data")
                raise ValueError("No valid frame keys found in ROI data")

            status_label.setText("Analyzing fixations...")
            progress_bar.setValue(20)
            QApplication.processEvents()

            # OPTIMIZATION: Pre-process ROI data for faster lookup
            # Create a frame range map to quickly find the nearest frame
            frame_numbers = sorted(frame_keys.keys())
            frame_range_map = {}

            if frame_numbers:
                # Print some statistics about the frame distribution
                print(f"DEBUG: Frame key range: {min(frame_numbers)} to {max(frame_numbers)}")
                if len(frame_numbers) > 1:
                    # Calculate average interval between frames
                    intervals = [frame_numbers[i + 1] - frame_numbers[i] for i in range(len(frame_numbers) - 1)]
                    if intervals:
                        avg_interval = sum(intervals) / len(intervals)
                        print(f"DEBUG: Average interval between frames: {avg_interval:.2f}")

                # Create a map of frame ranges for faster nearest frame lookups
                for i, frame in enumerate(frame_numbers):
                    if i == 0:
                        # For the first frame, use it for anything less than the midpoint to the next frame
                        next_frame = frame_numbers[i + 1] if i + 1 < len(frame_numbers) else frame + 1000
                        frame_range_map[(0, (frame + next_frame) // 2)] = frame
                    elif i == len(frame_numbers) - 1:
                        # For the last frame, use it for anything greater than the midpoint from the previous frame
                        prev_frame = frame_numbers[i - 1]
                        frame_range_map[((prev_frame + frame) // 2, float('inf'))] = frame
                    else:
                        # For middle frames, use the midpoints between adjacent frames
                        prev_frame = frame_numbers[i - 1]
                        next_frame = frame_numbers[i + 1]
                        frame_range_map[((prev_frame + frame) // 2, (frame + next_frame) // 2)] = frame

            # Get fixation data - OPTIMIZATION: Filter for valid frames
            fixation_data = data[data['is_fixation_left'] | data['is_fixation_right']]
            fixation_data = fixation_data.dropna(subset=['frame_number', 'x_left', 'y_left'])
            fixation_count = len(fixation_data)
            print(f"DEBUG: Found {fixation_count} valid fixation data points")

            # Process fixations in batches for better progress reporting
            processed_count = 0
            hit_count = 0
            batch_size = max(1, fixation_count // 50)  # ~50 progress updates

            # OPTIMIZATION: Cache polygon checks to avoid redundant calculations
            polygon_check_cache = {}

            # Process each fixation
            for idx, row in fixation_data.iterrows():
                frame_num = int(row['frame_number'])
                processed_count += 1

                # Update progress every batch
                if processed_count % batch_size == 0:
                    progress = 20 + int(75 * processed_count / fixation_count)
                    progress_bar.setValue(progress)
                    status_label.setText(f"Processing fixations: {processed_count}/{fixation_count}")
                    QApplication.processEvents()

                # Find the nearest frame in ROI data - OPTIMIZATION: Use frame range map for faster lookup
                nearest_frame = None

                # Try the frame range map first
                for (start, end), frame in frame_range_map.items():
                    if start <= frame_num < end:
                        nearest_frame = frame
                        break

                # If no match in the range map, fall back to the slower nearest neighbor approach
                if nearest_frame is None:
                    try:
                        nearest_frame = min(frame_keys.keys(), key=lambda x: abs(x - frame_num))
                    except Exception as e:
                        print(f"DEBUG: Error finding nearest frame: {e}")
                        continue

                frame_distance = abs(nearest_frame - frame_num)

                # Skip if the frame distance is too large
                if frame_distance > 1000:  # Use a threshold based on your data
                    continue

                # Get the ROIs for this frame
                rois_in_frame = frame_keys[nearest_frame]

                # Get normalized coordinates
                if row['x_left'] > 1.0 or row['y_left'] > 1.0:
                    x_norm = row['x_left'] / self.screen_width
                    y_norm = row['y_left'] / self.screen_height
                else:
                    x_norm = row['x_left']
                    y_norm = row['y_left']

                # Check each ROI in this frame
                for roi in rois_in_frame:
                    if 'label' not in roi or 'coordinates' not in roi:
                        continue

                    label = roi['label']
                    coords = roi['coordinates']

                    # OPTIMIZATION: Use cached results for polygon checks
                    cache_key = (tuple((coord['x'], coord['y']) for coord in coords), x_norm, y_norm)
                    if cache_key in polygon_check_cache:
                        is_inside = polygon_check_cache[cache_key]
                    else:
                        # Check if point is inside polygon
                        is_inside = self._point_in_polygon(x_norm, y_norm, coords)
                        polygon_check_cache[cache_key] = is_inside

                    if is_inside:
                        # Add time spent to this ROI
                        if label not in roi_durations:
                            roi_durations[label] = 0
                        roi_durations[label] += 1  # Each fixation counts as one time unit
                        hit_count += 1
                        break  # Only count one ROI per fixation

            print(f"DEBUG: Processed {processed_count} fixations, found {hit_count} ROI hits")
            print(f"DEBUG: ROI durations: {roi_durations}")

            # Update progress
            update_overall_progress(1, "Creating ROI Attention Time plot...")
            progress_bar.setValue(95)

            # Create the plot - taller to accommodate the explanation text
            fig, ax = plt.subplots(figsize=(10, 7))

            # Initialize bars variable to avoid reference errors
            bars = None

            # Check if we found any ROI hits
            if not roi_durations:
                print(f"DEBUG: No ROI hits found! Creating empty plot with message.")
                # Display a message on the plot
                ax.text(0.5, 0.5, "No ROI fixations detected.\nCheck ROI file and eye tracking data alignment.",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_title(f'No ROI Hits Found in {movie}')
                # Set some reasonable axis limits for empty plot
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(-0.5, 0.5)
            else:
                # Sort ROIs by duration
                sorted_rois = sorted(roi_durations.items(), key=lambda x: x[1], reverse=True)
                labels = [item[0] for item in sorted_rois]
                durations = [item[1] for item in sorted_rois]

                # Calculate total fixation time for percentage
                total_duration = sum(durations)
                percentages = [(count / total_duration) * 100 for count in durations]

                # Plot bar chart
                bars = ax.bar(labels, durations, color='skyblue')

                # Add value labels with both count and percentage on top of bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    percentage = percentages[i]
                    # Position the raw count on top of the bar
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=10)
                    # Add percentage text in the middle of the bar
                    ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                            f'{percentage:.1f}%',
                            ha='center', va='center', fontsize=10,
                            color='black', fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, pad=2, boxstyle='round'))

                # Add title and labels with explanation
                ax.set_title(f'Time Spent on Each ROI in {movie}')
                ax.set_xlabel('ROI')
                ax.set_ylabel('Number of Fixations')

                # Add explanation text
                ax.text(0.5, -0.15,
                        "This chart shows the number of fixations on each ROI.\n" +
                        "Percentages indicate the proportion of total fixations on each ROI.",
                        ha='center', va='center', transform=ax.transAxes, fontsize=10,
                        style='italic', bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=0.5'))

                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')

                # Adjust layout to make room for the rotated labels
                plt.tight_layout()

            # Save the plot
            import os
            
            # Use the plots directory provided by the GUI
            plots_dir = self.plots_dir
            
            # If no plots directory was set, create a default one
            if not plots_dir:
                if self.output_dir and os.path.exists(self.output_dir):
                    plots_dir = os.path.join(self.output_dir, 'plots')
                else:
                    # Fallback to using the ROI file location
                    plots_dir = f"{os.path.dirname(self.roi_file_path)}/plots"
                
            os.makedirs(plots_dir, exist_ok=True)
            print(f"DEBUG: Using plots directory: {plots_dir}")
            
            # Initialize list to track created plot files
            created_plots = []

            attention_filename = f"roi_attention_time_{movie.replace(' ', '_')}.png"
            attention_path = os.path.join(plots_dir, attention_filename)
            print(f"DEBUG: Saving ROI Attention Time plot to: {attention_path}")
            plt.savefig(attention_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Add path to list of created plots
            created_plots.append(attention_path)

            # Add to visualization results
            if movie not in self.visualization_results:
                self.visualization_results[movie] = {'basic': [], 'social': []}

            self.visualization_results[movie]['basic'].append(attention_path)

            # Make sure the movie is in the dictionary
            if movie not in self.movie_visualizations:
                self.movie_visualizations[movie] = {}

            # Add to movie_visualizations with display name
            self.movie_visualizations[movie]["ROI Attention Time"] = attention_path

            # Call the advanced ROI plots function if we have ROI hits
            if roi_durations:
                # Now use the created directory for all plot outputs
                advanced_plots = self.create_advanced_roi_plots(
                    movie,
                    roi_durations,
                    fixation_data,
                    plots_dir,
                    frame_keys,
                    frame_range_map,
                    polygon_check_cache,
                    status_label,
                    progress_bar,
                    update_progress_func=update_overall_progress
                )
                
                # Add advanced plots to our created_plots list
                if advanced_plots:
                    created_plots.extend(advanced_plots)
                    print(f"DEBUG: Added {len(advanced_plots)} advanced plots to created_plots list")

            # Final update
            progress_bar.setValue(100)
            status_label.setText(f"Completed generating plots for {movie}")
            QApplication.processEvents()
            
            # Update HTML report if available
            report_updated = False
            if self.report_path:
                status_label.setText(f"Updating HTML report - adding new visualizations...")
                QApplication.processEvents()
                report_updated = self.update_html_report(movie)
            
            # Wait a bit before closing the dialog
            import time
            time.sleep(1)

            # Close the dialog
            progress_dialog.close()
            
            # Return results with information about report update and created plots
            result = {
                "success": True,
                "report_updated": report_updated,
                "movie": movie,
                "plots": created_plots
            }
            return result

        except Exception as e:
            import traceback
            print(f"ERROR during plot generation: {str(e)}")
            traceback.print_exc()

            # Close the dialog
            progress_dialog.close()

            # Show error message
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                None,
                "Error Generating Plots",
                f"An error occurred while generating plots: {str(e)}"
            )

            # Return failure with details
            return {
                "success": False,
                "error": str(e),
                "movie": movie,
                "plots": []
            }

    def _point_in_polygon(self, x, y, coordinates):
        """Check if a point is inside a polygon defined by coordinates using an optimized ray casting algorithm"""
        # Extract points from coordinates
        points = [(coord['x'], coord['y']) for coord in coordinates]

        # Need at least 3 points to form a polygon
        if len(points) < 3:
            return False

        # Ray casting algorithm
        inside = False
        j = len(points) - 1

        for i in range(len(points)):
            xi, yi = points[i]
            xj, yj = points[j]

            # Check if point is on an edge or vertex (exact match)
            if (yi == y and xi == x) or (yj == y and xj == x):
                return True

            # Check if the point is on a horizontal edge
            if (abs(yi - yj) < 1e-9) and (abs(yi - y) < 1e-9) and (min(xi, xj) <= x <= max(xi, xj)):
                return True

            # Ray casting - check if ray crosses this edge
            # Using a small epsilon for floating point comparison
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def update_html_report(self, movie):
        """
        Update the HTML report to include newly generated plots.
        
        Args:
            movie: The movie name for which plots were generated
            
        Returns:
            bool: True if the report was updated successfully, False otherwise
        """
        if not self.report_path or not os.path.exists(self.report_path):
            print("DEBUG: No report path available for update")
            return False
            
        try:
            # Get visualizer instance
            from .eyelink_visualizer import MovieEyeTrackingVisualizer
            
            # Determine the base directory for the visualizer
            movie_data = self._get_movie_data(movie)
            if movie_data and "data_path" in movie_data:
                base_dir = os.path.dirname(os.path.dirname(movie_data["data_path"]))
            else:
                # Fallback to output directory
                base_dir = self.output_dir
            
            print(f"DEBUG: Using base directory for report regeneration: {base_dir}")
            
            # First, scan the report directory to find existing images that might not be in visualization_results
            report_dir = os.path.dirname(self.report_path)
            
            # Try to find the plots directory - first use the one passed to the plot generator
            plots_dir = None
            if hasattr(self, 'plots_dir') and self.plots_dir and os.path.exists(self.plots_dir):
                plots_dir = self.plots_dir
                print(f"DEBUG: Using plots directory from plot generator: {plots_dir}")
            # Then try to find it relative to the report directory
            elif os.path.exists(os.path.join(os.path.dirname(report_dir), 'plots')):
                plots_dir = os.path.join(os.path.dirname(report_dir), 'plots')
            elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(report_dir)), 'plots')):
                plots_dir = os.path.join(os.path.dirname(os.path.dirname(report_dir)), 'plots')
            elif os.path.exists(os.path.join(report_dir, 'plots')):
                plots_dir = os.path.join(report_dir, 'plots')
                
            print(f"DEBUG: Looking for plots in: {plots_dir if plots_dir else 'No plots directory found'}")
            
            # Create a comprehensive visualization results dictionary including all plots
            all_visualizations = {}
            
            # Start with our existing visualization results
            # Make sure we at least have the current movie in the visualization results
            if movie not in self.visualization_results:
                self.visualization_results[movie] = {'social': [], 'basic': []}
                
            for m, categories in self.visualization_results.items():
                if m not in all_visualizations:
                    all_visualizations[m] = {}
                for category, plots in categories.items():
                    if category not in all_visualizations[m]:
                        all_visualizations[m][category] = []
                    all_visualizations[m][category].extend(plots)
            
            # If we found a plots directory, scan it for any additional plots
            if plots_dir and os.path.exists(plots_dir):
                for plot_file in os.listdir(plots_dir):
                    if plot_file.endswith('.png'):
                        # Try to determine which movie this plot belongs to
                        plot_path = os.path.join(plots_dir, plot_file)
                        
                        # Check if this plot is already in our visualization results
                        already_included = False
                        for m, categories in all_visualizations.items():
                            for category, plots in categories.items():
                                if plot_path in plots:
                                    already_included = True
                                    break
                            if already_included:
                                break
                        
                        # If this plot is not already included, add it to the appropriate movie
                        if not already_included:
                            movie_found = False
                            # Try to extract movie name from the plot filename
                            for m in all_visualizations.keys():
                                if m.lower().replace(' ', '_') in plot_file.lower():
                                    # This plot belongs to this movie
                                    if 'all' not in all_visualizations[m]:
                                        all_visualizations[m]['all'] = []
                                    all_visualizations[m]['all'].append(plot_path)
                                    print(f"DEBUG: Added previously untracked plot to report: {plot_file} (matched to {m})")
                                    movie_found = True
                                    break
                            
                            # If we couldn't find a movie match, add it to the current movie
                            if not movie_found:
                                if 'all' not in all_visualizations[movie]:
                                    all_visualizations[movie]['all'] = []
                                all_visualizations[movie]['all'].append(plot_path)
                                print(f"DEBUG: Added plot without specific movie match to current movie: {plot_file}")
            
            # Create the visualizer with appropriate screen size
            visualizer = MovieEyeTrackingVisualizer(base_dir=base_dir, 
                                                  screen_size=(self.screen_width, self.screen_height))
            
            # Generate a new report with all visualizations
            print(f"DEBUG: Regenerating report with comprehensive visualization set")
            visualizer.generate_report(all_visualizations, report_dir)
            print(f"DEBUG: Regenerated HTML report to include all plots")
            
            # Print out total number of plots included in the report
            total_plots = 0
            for m, categories in all_visualizations.items():
                for category, plots in categories.items():
                    total_plots += len(plots)
            print(f"DEBUG: Report contains {total_plots} total plots")
            
            return True
            
        except Exception as e:
            print(f"ERROR updating HTML report: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_movie_data(self, movie):
        """
        Retrieve movie data. 
        
        This method is a placeholder that will be overridden by the GUI's _get_movie_data
        method when generate_social_attention_plots is called.
        
        Args:
            movie: Name of the movie to get data for
            
        Returns:
            Dict containing movie data or None if not found
        """
        # This is a placeholder - the actual implementation will be provided by the GUI
        # through method injection in generate_social_attention_plots
        print("WARNING: _get_movie_data was called but not properly injected by GUI")
        return None