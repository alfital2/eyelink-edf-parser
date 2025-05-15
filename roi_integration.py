"""
ROI Integration Module for Eye Movement Analysis
Author: Tal Alfi
Date: April 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
                             QPushButton, QFileDialog, QMessageBox, QLabel)
from PyQt5.QtCore import Qt

# Import our modules
from roi_manager import ROIManager


def enhance_animated_scanpath_tab(animated_tab):
    """
    Enhance the existing animated scanpath tab with ROI visualization capabilities.
    Fixed version that properly creates all required UI elements and uses the correct attribute names.

    Args:
        animated_tab: The existing AnimatedScanpathTab instance
    """
    # Create ROI manager instance
    roi_manager = ROIManager()

    # Store reference on the scanpath widget
    animated_tab.scanpath_widget.roi_manager = roi_manager

    # Initialize ROI-related attributes
    animated_tab.scanpath_widget.show_rois = False
    animated_tab.scanpath_widget.show_roi_labels = True
    animated_tab.scanpath_widget.highlight_active_roi = True
    animated_tab.scanpath_widget.active_roi_id = None
    animated_tab.scanpath_widget.roi_dwell_times = {}
    animated_tab.scanpath_widget.current_roi_start_time = None

    # Find the display options layout
    options_layout = None
    for child in animated_tab.findChildren(QGroupBox):
        if "settings" in child.title().lower():
            # Likely the animation settings group
            for layout in child.findChildren(QVBoxLayout):
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget() and isinstance(item.widget(), QCheckBox):
                        options_layout = layout
                        break
                if options_layout:
                    break
            if options_layout:
                break

    # Add ROI selection button
    if options_layout:
        roi_button = QPushButton("Select ROI File")
        roi_button.clicked.connect(animated_tab.select_roi_file)
        options_layout.addWidget(roi_button)
        
        # Add status label for ROI file
        animated_tab.roi_status_label = QLabel("No ROI file loaded")
        options_layout.addWidget(animated_tab.roi_status_label)
        
        # Add ROI display controls
        show_rois_cb = QCheckBox("Show ROIs")
        show_rois_cb.setChecked(False)
        show_rois_cb.toggled.connect(animated_tab.toggle_roi_display)
        options_layout.addWidget(show_rois_cb)
        
        show_labels_cb = QCheckBox("Show ROI Labels")
        show_labels_cb.setChecked(True)
        show_labels_cb.toggled.connect(animated_tab.toggle_roi_labels)
        options_layout.addWidget(show_labels_cb)
        
        highlight_cb = QCheckBox("Highlight Active ROI")
        highlight_cb.setChecked(True)
        highlight_cb.toggled.connect(animated_tab.toggle_highlight_roi)
        options_layout.addWidget(highlight_cb)
        
        # Store references to the checkboxes
        animated_tab.show_rois_cb = show_rois_cb
        animated_tab.show_labels_cb = show_labels_cb
        animated_tab.highlight_cb = highlight_cb


def load_sample_data(eye_data_path, roi_file_path):
    """
    Load sample eye tracking and ROI data for visualization.
    
    Args:
        eye_data_path: Path to CSV file with eye tracking data
        roi_file_path: Path to JSON file with ROI data
        
    Returns:
        Tuple of (eye_data_df, roi_manager)
    """
    # Initialize ROI manager
    roi_manager = ROIManager()
    
    # Load eye tracking data
    try:
        eye_data = pd.read_csv(eye_data_path)
    except Exception as e:
        print(f"Error loading eye data: {str(e)}")
        return pd.DataFrame(), roi_manager
    
    # Load ROI data
    if roi_file_path and os.path.exists(roi_file_path):
        success = roi_manager.load_roi_file(roi_file_path)
        if success:
            # Extend ROI frames to match movie frame count if needed
            if 'frame_number' in eye_data.columns:
                try:
                    # Get the maximum frame number from the eye tracking data
                    max_frame = int(eye_data['frame_number'].max())
                    if max_frame > 0:
                        # Extend ROI frames if the movie has more frames
                        roi_manager.extend_roi_frames(max_frame)
                except Exception as e:
                    print(f"Warning: Failed to extend ROI frames: {str(e)}")
        else:
            print(f"Failed to load ROI data from {roi_file_path}")
    
    return eye_data, roi_manager


def create_integrated_visualization(eye_data, roi_manager, frame_number, save_path=None):
    """
    Create an integrated visualization showing eye positions and ROIs.
    
    Args:
        eye_data: DataFrame with eye tracking data
        roi_manager: ROIManager instance with loaded ROI data
        frame_number: Frame number to visualize
        save_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up normalized coordinates (0-1)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Invert y-axis to match screen coordinates
    
    # Ensure ROI frames match movie frames if not already done
    if 'frame_number' in eye_data.columns:
        try:
            # Get the maximum frame number from the eye tracking data
            max_frame = int(eye_data['frame_number'].max())
            if max_frame > 0 and len(roi_manager.frame_numbers) > 0 and max_frame > len(roi_manager.frame_numbers):
                # Extend ROI frames if the movie has more frames
                roi_manager.extend_roi_frames(max_frame)
        except Exception as e:
            print(f"Warning: Failed to extend ROI frames: {str(e)}")
    
    # Draw ROIs for this frame
    roi_manager.draw_rois_on_axis(ax, frame_number)
    
    # Filter eye data for this frame
    frame_data = eye_data[eye_data['frame_number'] == frame_number]
    
    # Plot left and right eye positions
    if 'x_left_norm' in frame_data.columns and 'y_left_norm' in frame_data.columns:
        ax.scatter(frame_data['x_left_norm'], frame_data['y_left_norm'], 
                  color='blue', label='Left Eye', s=50, alpha=0.7)
    elif 'x_left' in frame_data.columns and 'y_left' in frame_data.columns:
        # If normalized coordinates not available, use raw coordinates
        # Assuming screen dimensions of 1280x1024 for normalization
        ax.scatter(frame_data['x_left'] / 1280, frame_data['y_left'] / 1024, 
                  color='blue', label='Left Eye', s=50, alpha=0.7)
        
    if 'x_right_norm' in frame_data.columns and 'y_right_norm' in frame_data.columns:
        ax.scatter(frame_data['x_right_norm'], frame_data['y_right_norm'], 
                  color='red', label='Right Eye', s=50, alpha=0.7)
    elif 'x_right' in frame_data.columns and 'y_right' in frame_data.columns:
        # If normalized coordinates not available, use raw coordinates
        ax.scatter(frame_data['x_right'] / 1280, frame_data['y_right'] / 1024, 
                  color='red', label='Right Eye', s=50, alpha=0.7)
    
    # Add title and labels
    ax.set_title(f"Eye Positions and ROIs for Frame {frame_number}", fontsize=14)
    ax.set_xlabel("X Position (normalized)", fontsize=12)
    ax.set_ylabel("Y Position (normalized)", fontsize=12)
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    
    return fig


def is_social_roi(roi_label):
    """
    Determine if a ROI is social based on its label.
    
    Args:
        roi_label: The label of the ROI
        
    Returns:
        Boolean indicating if the ROI is considered social
    """
    social_labels = [
        "face", "eyes", "mouth", "nose", "ear", "ears", "hair", 
        "hand", "hands", "arm", "arms", "finger", "fingers",
        "torso", "body", "person", "people", "human", "head",
        "shoulder", "shoulders", "leg", "legs", "foot", "feet"
    ]
    
    if not roi_label:
        return False
        
    return any(social_label in roi_label.lower() for social_label in social_labels)

def analyze_roi_fixations(eye_data: pd.DataFrame, roi_manager: ROIManager, 
                          min_duration_ms: int = 100) -> Dict[str, Any]:
    """
    Analyze fixations within defined ROIs from eye tracking data.
    
    Args:
        eye_data: DataFrame with eye tracking samples 
        roi_manager: Instance of ROIManager with loaded ROI data
        min_duration_ms: Minimum duration (in ms) for a stable gaze to be considered a fixation
        
    Returns:
        Dictionary with fixation data including ROI information
    """
    # Initialize results
    fixations = []
    
    # Ensure we have necessary columns
    required_cols = ['timestamp', 'x_left', 'y_left', 'frame_number']
    for col in required_cols:
        if col not in eye_data.columns:
            print(f"Missing required column: {col} in eye_data columns: {eye_data.columns}")
            return {"error": f"Missing required column: {col}", "fixation_count": 0, "fixations": []}
            
    # Sort data by timestamp
    sorted_data = eye_data.sort_values('timestamp')
    
    # Prepare for fixation detection
    current_fixation = None
    fixation_samples = []
    fixation_count = 0
    
    # Debug information
    print(f"Analyzing eye data with {len(sorted_data)} samples")
    print(f"First few rows of data:")
    print(sorted_data.head())
    print(f"Data columns: {sorted_data.columns}")
    
    # Simple velocity-based fixation detection
    for i in range(1, len(sorted_data)):
        prev_row = sorted_data.iloc[i-1]
        curr_row = sorted_data.iloc[i]
        
        # Calculate displacement between consecutive samples
        dx = curr_row['x_left'] - prev_row['x_left']
        dy = curr_row['y_left'] - prev_row['y_left']
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Time difference between samples
        dt = curr_row['timestamp'] - prev_row['timestamp']
        
        # Simple threshold-based fixation detection (using normalized coordinates)
        # Use a slightly larger threshold to ensure we're detecting fixations
        if displacement < 0.05:  # Consider it part of a fixation if displacement is small (increased from 0.03)
            if current_fixation is None:
                # Start a new fixation
                current_fixation = {
                    'start_time': prev_row['timestamp'],
                    'end_time': curr_row['timestamp'],
                    'start_index': i-1,
                    'samples': [prev_row, curr_row],
                    'x_sum': prev_row['x_left'] + curr_row['x_left'],
                    'y_sum': prev_row['y_left'] + curr_row['y_left'],
                    'count': 2
                }
            else:
                # Continue current fixation
                current_fixation['end_time'] = curr_row['timestamp']
                current_fixation['samples'].append(curr_row)
                current_fixation['x_sum'] += curr_row['x_left']
                current_fixation['y_sum'] += curr_row['y_left']
                current_fixation['count'] += 1
        else:
            # End of fixation, save if it meets minimum duration
            if current_fixation is not None:
                current_fixation['duration'] = current_fixation['end_time'] - current_fixation['start_time']
                
                if current_fixation['duration'] >= min_duration_ms:
                    # Calculate centroid
                    x_mean = current_fixation['x_sum'] / current_fixation['count']
                    y_mean = current_fixation['y_sum'] / current_fixation['count']
                    
                    # Get middle frame for this fixation
                    frame_indices = [sample['frame_number'] for sample in current_fixation['samples'] 
                                    if 'frame_number' in sample and not pd.isna(sample['frame_number'])]
                    
                    if frame_indices:
                        frame_number = int(np.median(frame_indices))
                        
                        # Find which ROI this fixation belongs to
                        roi = roi_manager.find_roi_at_gaze(frame_number, x_mean, y_mean)
                        
                        # Create fixation record
                        roi_label = roi['label'] if roi and 'label' in roi else None
                        fixation_record = {
                            'start_time': current_fixation['start_time'],
                            'end_time': current_fixation['end_time'],
                            'duration': current_fixation['duration'],
                            'x': x_mean,
                            'y': y_mean,
                            'frame': frame_number,
                            'roi': roi_label,
                            'social': roi.get('social', is_social_roi(roi_label)) if roi else False
                        }
                        
                        fixations.append(fixation_record)
                        fixation_count += 1
                
                # Reset for next fixation
                current_fixation = None
    
    # Add one more fixation if we ended in the middle of one
    if current_fixation is not None:
        current_fixation['duration'] = current_fixation['end_time'] - current_fixation['start_time']
        
        if current_fixation['duration'] >= min_duration_ms:
            # Calculate centroid
            x_mean = current_fixation['x_sum'] / current_fixation['count']
            y_mean = current_fixation['y_sum'] / current_fixation['count']
            
            # Get middle frame for this fixation
            frame_indices = [sample['frame_number'] for sample in current_fixation['samples'] 
                            if 'frame_number' in sample and not pd.isna(sample['frame_number'])]
            
            if frame_indices:
                frame_number = int(np.median(frame_indices))
                
                # Find which ROI this fixation belongs to
                roi = roi_manager.find_roi_at_gaze(frame_number, x_mean, y_mean)
                
                # Create fixation record
                roi_label = roi['label'] if roi and 'label' in roi else None
                fixation_record = {
                    'start_time': current_fixation['start_time'],
                    'end_time': current_fixation['end_time'],
                    'duration': current_fixation['duration'],
                    'x': x_mean,
                    'y': y_mean,
                    'frame': frame_number,
                    'roi': roi_label,
                    'social': roi.get('social', is_social_roi(roi_label)) if roi else False
                }
                
                fixations.append(fixation_record)
                fixation_count += 1
    
    # Return fixation data
    return {
        "fixation_count": fixation_count,
        "fixations": fixations
    }


def compute_social_attention_metrics(fixation_data: Dict[str, Any], 
                                     eye_data: pd.DataFrame) -> Dict[str, float]:
    """
    Compute social attention metrics from fixation data.
    
    Args:
        fixation_data: Dictionary with fixation information
        eye_data: DataFrame with eye tracking samples
        
    Returns:
        Dictionary with social attention metrics
    """
    fixations = fixation_data.get('fixations', [])
    
    if not fixations:
        # Return all expected metrics with zero/default values
        return {
            "error": "No fixation data available",
            "social_attention_ratio": 0.0,
            "social_fixation_count": 0,
            "non_social_fixation_count": 0,
            "social_dwell_time": 0,
            "non_social_dwell_time": 0,
            "social_time_percent": 0.0,
            "social_first_fixation_latency": None,
            "time_to_first_social_fixation": None,
            "first_fixations_by_roi": {},
            "percent_fixations_social": 0.0
        }
    
    # Count fixations
    social_fixations = [f for f in fixations if f.get('social', False)]
    non_social_fixations = [f for f in fixations if not f.get('social', False)]
    
    social_count = len(social_fixations)
    non_social_count = len(non_social_fixations)
    total_count = social_count + non_social_count
    
    # Calculate ratio
    social_ratio = social_count / total_count if total_count > 0 else 0.0
    
    # Calculate dwell times - convert to regular Python float
    social_dwell_time = float(sum(float(f['duration']) for f in social_fixations))
    non_social_dwell_time = float(sum(float(f['duration']) for f in non_social_fixations))
    total_dwell_time = social_dwell_time + non_social_dwell_time
    
    # Calculate percentage of time spent on social vs. non-social
    social_time_percent = float((social_dwell_time / total_dwell_time * 100) if total_dwell_time > 0 else 0.0)
    
    # Find first fixation in each ROI category
    roi_first_fixations = {}
    for f in sorted(fixations, key=lambda x: x['start_time']):
        if f['roi'] and f['roi'] not in roi_first_fixations:
            # Convert to float to avoid NumPy types
            roi_first_fixations[f['roi']] = float(f['start_time'])
    
    # Calculate first fixation latency for social ROIs
    social_first_fixation = None
    for f in sorted(social_fixations, key=lambda x: x['start_time']):
        # Convert to float to avoid NumPy types
        social_first_fixation = float(f['start_time'])
        break
    
    # Calculate time to first social fixation (relative to start of recording)
    start_time = float(eye_data['timestamp'].min()) if 'timestamp' in eye_data.columns else 0
    time_to_first_social = float(social_first_fixation - start_time) if social_first_fixation is not None else None
    
    # Return metrics - convert all NumPy types to native Python types
    return {
        "social_attention_ratio": float(social_ratio),
        "social_fixation_count": int(social_count),
        "non_social_fixation_count": int(non_social_count),
        "social_dwell_time": float(social_dwell_time),
        "non_social_dwell_time": float(non_social_dwell_time),
        "social_time_percent": float(social_time_percent),
        "social_first_fixation_latency": float(social_first_fixation) if social_first_fixation is not None else None,
        "time_to_first_social_fixation": float(time_to_first_social) if time_to_first_social is not None else None,
        "first_fixations_by_roi": {k: float(v) for k, v in roi_first_fixations.items()},
        "percent_fixations_social": float(social_count / total_count * 100) if total_count > 0 else 0.0
    }


def plot_roi_fixation_sequence(fixations: List[Dict], eye_data: pd.DataFrame,
                              output_path: str = None, title: str = "ROI Fixation Sequence",
                              ax: plt.Axes = None) -> plt.Figure:
    """
    Generate a visualization showing the sequence of ROI fixations over time.
    
    Args:
        fixations: List of fixation dictionaries with ROI information
        eye_data: DataFrame with eye tracking samples
        output_path: Path to save the visualization (if None, just returns the figure)
        title: Title for the plot
        ax: Optional matplotlib Axes to draw on (if None, creates a new figure)
        
    Returns:
        matplotlib Figure object
    """
    # Get time range from eye data
    start_time = eye_data['timestamp'].min() if 'timestamp' in eye_data.columns else 0
    end_time = eye_data['timestamp'].max() if 'timestamp' in eye_data.columns else 1000 * 60  # 1 minute default
    
    # Convert to seconds for better display
    start_time_sec = start_time / 1000.0
    end_time_sec = end_time / 1000.0
    
    # Get all unique ROI labels
    roi_labels = sorted(set(f['roi'] for f in fixations if f.get('roi')))
    
    # Create a mapping of ROI labels to y-axis positions
    roi_y_pos = {roi: i for i, roi in enumerate(roi_labels)}
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Set up the plot
    ax.set_xlim(start_time_sec, end_time_sec)
    ax.set_ylim(-1, len(roi_labels))
    ax.set_yticks(list(range(len(roi_labels))))
    ax.set_yticklabels(roi_labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Region of Interest")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Choose colors based on social vs non-social
    colors = []
    for roi in roi_labels:
        # Find if this ROI is social based on our classification function
        is_social = is_social_roi(roi)
        
        # Also check in fixation data if available
        for f in fixations:
            if f.get('roi') == roi and f.get('social', False):
                is_social = True
                break
        
        # Use red for social ROIs, blue for non-social
        colors.append('salmon' if is_social else 'royalblue')
    
    # Plot fixations as vertical lines
    for f in fixations:
        if f.get('roi') in roi_y_pos:
            y_pos = roi_y_pos[f['roi']]
            start_sec = f['start_time'] / 1000.0
            end_sec = f['end_time'] / 1000.0
            duration_sec = (f['end_time'] - f['start_time']) / 1000.0
            
            # Get color based on social/non-social
            color = 'salmon' if f.get('social', False) else 'royalblue'
            
            # Plot vertical line for this fixation
            ax.plot([start_sec, start_sec], [y_pos - 0.4, y_pos + 0.4], 
                    color=color, linewidth=2, alpha=0.7)
            
            # For longer fixations, draw a horizontal line showing duration
            if duration_sec > 0.1:  # Only for fixations longer than 100ms
                ax.plot([start_sec, end_sec], [y_pos, y_pos], 
                        color=color, linewidth=4, alpha=0.5)
            
            # Add duration text for longer fixations (>500ms)
            if duration_sec > 0.5:
                # Display fixation duration in seconds
                ax.text(start_sec + duration_sec/2, y_pos + 0.2, 
                        f"{duration_sec:.1f}s", 
                        fontsize=8, ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Remove pagination indicator and footnote as requested
    # This keeps the plot cleaner without pagination indicators or explanatory notes
    
    # Improve layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
    
    return fig


def plot_social_attention_bar(metrics: Dict[str, Any], output_path: str = None,
                             title: str = "Social vs. Non-Social Attention") -> plt.Figure:
    """
    Generate a bar chart showing social vs. non-social attention metrics.
    
    Args:
        metrics: Dictionary with social attention metrics
        output_path: Path to save the visualization (if None, just returns the figure)
        title: Title for the plot
        
    Returns:
        matplotlib Figure object
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Prepare data for fixation counts
    social_count = metrics.get('social_fixation_count', 0)
    non_social_count = metrics.get('non_social_fixation_count', 0)
    
    # Prepare data for dwell times
    social_dwell = metrics.get('social_dwell_time', 0) / 1000.0  # Convert to seconds
    non_social_dwell = metrics.get('non_social_dwell_time', 0) / 1000.0
    
    # Plot fixation counts
    labels = ['Social', 'Non-Social']
    counts = [social_count, non_social_count]
    
    bars1 = ax1.bar(labels, counts, color=['salmon', 'royalblue'])
    ax1.set_title('Fixation Counts')
    ax1.set_ylabel('Number of Fixations')
    
    # Add count labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{counts[i]}', ha='center', va='bottom')
    
    # Plot dwell times
    dwell_times = [social_dwell, non_social_dwell]
    
    bars2 = ax2.bar(labels, dwell_times, color=['salmon', 'royalblue'])
    ax2.set_title('Dwell Times')
    ax2.set_ylabel('Time (seconds)')
    
    # Add time labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{dwell_times[i]:.1f}s', ha='center', va='bottom')
    
    # Add overall percentages
    social_percent = metrics.get('percent_fixations_social', 0)
    fig.suptitle(f"{title}\nSocial Attention: {social_percent:.1f}%", fontsize=14)
    
    # Improve layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
    
    return fig