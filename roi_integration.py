"""
ROI Integration Module for Eye Movement Analysis
Author: Tal Alfi
Date: April 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
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
                        # Found the display options section
                        options_layout = layout
                        break
                if options_layout:
                    break
            break

    # If we couldn't find the options layout, create one
    if not options_layout:
        print("Warning: Could not find existing display options layout, creating a new one")
        settings_group = animated_tab.findChild(QGroupBox)
        if settings_group:
            layout = settings_group.layout()
            if isinstance(layout, QHBoxLayout):
                # Create a new options section
                options_widget = QWidget()
                options_layout = QVBoxLayout(options_widget)
                layout.addWidget(options_widget)

    if options_layout:
        # Add ROI display checkbox
        animated_tab.show_roi_cb = QCheckBox("Show ROIs")
        animated_tab.show_roi_cb.setChecked(False)
        animated_tab.show_roi_cb.toggled.connect(lambda checked: toggle_roi_display(animated_tab, checked))
        options_layout.addWidget(animated_tab.show_roi_cb)

        # Add ROI labels checkbox
        animated_tab.show_roi_labels_cb = QCheckBox("Show ROI Labels")
        animated_tab.show_roi_labels_cb.setChecked(True)
        animated_tab.show_roi_labels_cb.toggled.connect(lambda checked: toggle_roi_labels(animated_tab, checked))
        options_layout.addWidget(animated_tab.show_roi_labels_cb)

        # Add highlight active ROI checkbox
        animated_tab.highlight_roi_cb = QCheckBox("Highlight Active ROI")
        animated_tab.highlight_roi_cb.setChecked(True)
        animated_tab.highlight_roi_cb.toggled.connect(lambda checked: toggle_roi_highlight(animated_tab, checked))
        options_layout.addWidget(animated_tab.highlight_roi_cb)

    # Create ROI file selection button and container
    roi_container = QWidget()
    roi_layout = QHBoxLayout(roi_container)

    roi_load_btn = QPushButton("Load ROI File")
    roi_load_btn.clicked.connect(lambda: select_roi_file(animated_tab))
    roi_layout.addWidget(roi_load_btn)

    # Add ROI status label
    animated_tab.roi_status_label = QLabel("No ROI file loaded")
    animated_tab.roi_status_label.setStyleSheet("text-align: left; padding: 5px;")
    roi_layout.addWidget(animated_tab.roi_status_label, 1)

    # Add roi_container to the parent layout
    parent_layout = animated_tab.layout()
    if parent_layout:
        # Add after settings group if found
        settings_group = animated_tab.findChild(QGroupBox)
        if settings_group:
            index = parent_layout.indexOf(settings_group)
            if index >= 0:
                parent_layout.insertWidget(index + 1, roi_container)
            else:
                parent_layout.addWidget(roi_container)
        else:
            parent_layout.addWidget(roi_container)

    # Add current ROI indicator
    animated_tab.current_roi_label = QLabel("Current ROI: None")
    if parent_layout:
        parent_layout.addWidget(animated_tab.current_roi_label)

    # Add ROI statistics label
    animated_tab.roi_stats_label = QLabel("ROI Dwell Times: Not available")
    if parent_layout:
        parent_layout.addWidget(animated_tab.roi_stats_label)

    # Patch the load_data method to accept ROI data
    original_load_data = animated_tab.load_data

    def enhanced_load_data(data, movie_name, screen_width=1280, screen_height=1024):
        """Enhanced load_data method that supports ROI visualization."""
        result = original_load_data(data, movie_name, screen_width, screen_height)

        if result and hasattr(animated_tab.scanpath_widget,
                              'roi_manager') and animated_tab.scanpath_widget.roi_manager.roi_data:
            # ROI data is already loaded, update the display
            animated_tab.scanpath_widget.redraw()
            return True

        return result

    # Replace the method
    animated_tab.load_data = enhanced_load_data

    # Patch the update_display method of the scanpath widget
    original_update_display = animated_tab.scanpath_widget.update_display

    def enhanced_update_display():
        """Enhanced update_display method that properly handles ROI visualization."""
        if not hasattr(animated_tab.scanpath_widget, 'data') or animated_tab.scanpath_widget.data is None:
            return

        # Clear previous ROI patches and text annotations
        for artist in animated_tab.scanpath_widget.ax.patches:
            artist.remove()

        # Clear text annotations (important to prevent labels from persisting)
        for text in animated_tab.scanpath_widget.ax.texts:
            text.remove()

        # Get current frame number for ROI detection
        current_frame_num = None
        if hasattr(animated_tab.scanpath_widget, 'current_frame'):
            frame_idx = animated_tab.scanpath_widget.current_frame
            if frame_idx < len(
                    animated_tab.scanpath_widget.data) and 'frame_number' in animated_tab.scanpath_widget.data.columns:
                frame_value = animated_tab.scanpath_widget.data.iloc[frame_idx]['frame_number']
                if not pd.isna(frame_value):
                    current_frame_num = int(frame_value)

        # Draw ROIs if enabled, frame number is available, and roi_manager has data
        if (animated_tab.scanpath_widget.show_rois and
                current_frame_num is not None and
                hasattr(animated_tab.scanpath_widget, 'roi_manager') and
                animated_tab.scanpath_widget.roi_manager is not None and
                animated_tab.scanpath_widget.roi_manager.roi_data):  # Ensure roi_data exists

            # Check for eye position to find active ROI
            active_roi = None
            if hasattr(animated_tab.scanpath_widget, 'data') and animated_tab.scanpath_widget.current_frame < len(
                    animated_tab.scanpath_widget.data):
                data_row = animated_tab.scanpath_widget.data.iloc[animated_tab.scanpath_widget.current_frame]

                # Try to detect active ROI with normalized coordinates
                # Check data for both eyes depending on display settings - using correct attributes
                eye_cols = []

                # Check if show_left_cb exists and is checked - FIXED ATTRIBUTE NAME
                if hasattr(animated_tab.scanpath_widget,
                           'show_left_cb') and animated_tab.scanpath_widget.show_left_cb.isChecked():
                    if 'x_left' in data_row and 'y_left' in data_row:
                        # Normalize coordinates if needed
                        if 'x_left_norm' in data_row and 'y_left_norm' in data_row:
                            eye_cols.append(('x_left_norm', 'y_left_norm'))
                        else:
                            x_norm = data_row['x_left'] / animated_tab.scanpath_widget.screen_width
                            y_norm = data_row['y_left'] / animated_tab.scanpath_widget.screen_height
                            eye_cols.append((x_norm, y_norm))

                # Check if show_right_cb exists and is checked - FIXED ATTRIBUTE NAME
                if hasattr(animated_tab.scanpath_widget,
                           'show_right_cb') and animated_tab.scanpath_widget.show_right_cb.isChecked():
                    if 'x_right' in data_row and 'y_right' in data_row:
                        # Normalize coordinates if needed
                        if 'x_right_norm' in data_row and 'y_right_norm' in data_row:
                            eye_cols.append(('x_right_norm', 'y_right_norm'))
                        else:
                            x_norm = data_row['x_right'] / animated_tab.scanpath_widget.screen_width
                            y_norm = data_row['y_right'] / animated_tab.scanpath_widget.screen_height
                            eye_cols.append((x_norm, y_norm))

                # Try each eye until we find a hit
                for x_col, y_col in eye_cols:
                    # Handle both attribute access and dictionary-like access
                    try:
                        if isinstance(x_col, str):
                            x_val = data_row[x_col] if x_col in data_row else None
                        else:
                            x_val = x_col

                        if isinstance(y_col, str):
                            y_val = data_row[y_col] if y_col in data_row else None
                        else:
                            y_val = y_col

                        if x_val is not None and y_val is not None and not pd.isna(x_val) and not pd.isna(y_val):
                            roi = animated_tab.scanpath_widget.roi_manager.find_roi_at_gaze(
                                current_frame_num, x_val, y_val)
                            if roi is not None:
                                active_roi = roi
                                break
                    except Exception as e:
                        print(f"Error checking eye position: {str(e)}")

            # Update active ROI tracking
            active_roi_id = None
            if active_roi and 'object_id' in active_roi:
                active_roi_id = active_roi['object_id']
                roi_label = active_roi.get('label', f"Unknown ({active_roi_id})")
                animated_tab.current_roi_label.setText(f"Current ROI: {roi_label}")
            else:
                animated_tab.current_roi_label.setText("Current ROI: None")

            animated_tab.scanpath_widget.active_roi_id = active_roi_id

            # Draw ROIs with potential highlighting
            try:
                animated_tab.scanpath_widget.roi_manager.draw_rois_on_axis(
                    animated_tab.scanpath_widget.ax,
                    current_frame_num,
                    show_labels=animated_tab.scanpath_widget.show_roi_labels,
                    highlighted_roi=active_roi_id if animated_tab.scanpath_widget.highlight_active_roi else None
                )
            except Exception as e:
                print(f"Error drawing ROIs: {str(e)}")

        # Call the original method to handle the rest of the display update
        try:
            original_update_display()
        except Exception as e:
            print(f"Error updating display: {str(e)}")

    # Replace the method with our enhanced version
    animated_tab.scanpath_widget.update_display = enhanced_update_display


def toggle_roi_display(animated_tab, checked):
    """Toggle ROI display on/off."""
    if hasattr(animated_tab.scanpath_widget, 'show_rois'):
        animated_tab.scanpath_widget.show_rois = checked
        animated_tab.scanpath_widget.redraw()


def toggle_roi_labels(animated_tab, checked):
    """Toggle ROI labels on/off."""
    if hasattr(animated_tab.scanpath_widget, 'show_roi_labels'):
        animated_tab.scanpath_widget.show_roi_labels = checked
        animated_tab.scanpath_widget.redraw()


def toggle_roi_highlight(animated_tab, checked):
    """Toggle ROI highlight on/off."""
    if hasattr(animated_tab.scanpath_widget, 'highlight_active_roi'):
        animated_tab.scanpath_widget.highlight_active_roi = checked
        animated_tab.scanpath_widget.redraw()


def select_roi_file(animated_tab):
    """Open file dialog to select an ROI JSON file."""
    file_path, _ = QFileDialog.getOpenFileName(
        animated_tab, "Select ROI File", "", "JSON Files (*.json)"
    )

    if file_path and os.path.exists(file_path):
        success = animated_tab.scanpath_widget.roi_manager.load_roi_file(file_path)
        if success:
            animated_tab.roi_status_label.setText(f"ROI File: {os.path.basename(file_path)}")
            # Enable the ROI display checkbox since we have data
            animated_tab.show_roi_cb.setEnabled(True)
            animated_tab.show_roi_cb.setChecked(True)  # Turn on ROI display
            animated_tab.scanpath_widget.show_rois = True
            # Redraw the visualization
            animated_tab.scanpath_widget.redraw()
        else:
            QMessageBox.warning(
                animated_tab,
                "ROI File Error",
                f"Failed to load ROI data from {file_path}. Please check file format."
            )
            animated_tab.roi_status_label.setText("Failed to load ROI file")


def integrate_roi_visualization(main_gui):
    """
    Integrate ROI visualization into the main GUI's existing Animated Scanpath tab.

    Args:
        main_gui: The main GUI instance
    """
    # Find the animated scanpath tab
    animated_tab = None

    # First try to get it directly if it's stored as an attribute
    if hasattr(main_gui, 'animated_viz_tab'):
        animated_tab = main_gui.animated_viz_tab
    else:
        # Otherwise search for it by name in the tab widget
        tab_widget = None
        for child in main_gui.centralWidget().findChildren(QTabWidget):
            tab_widget = child
            break

        if tab_widget:
            # Look for a tab with "Scanpath" or "Animated" in the title
            for i in range(tab_widget.count()):
                tab_title = tab_widget.tabText(i).lower()
                if "scanpath" in tab_title or "animated" in tab_title:
                    animated_tab = tab_widget.widget(i)
                    break

    if animated_tab:
        # Enhance the tab with ROI visualization
        enhance_animated_scanpath_tab(animated_tab)
        print("Successfully integrated ROI visualization into Animated Scanpath tab")
    else:
        print("Warning: Could not find Animated Scanpath tab to enhance")
        QMessageBox.warning(
            main_gui,
            "Integration Error",
            "Could not find Animated Scanpath tab to enhance with ROI visualization."
        )

# Helper functions for testing
def load_sample_data(eye_data_path, roi_data_path):
    """
    Load sample eye tracking data and ROI data for testing purposes.
    
    Args:
        eye_data_path: Path to eye tracking CSV file
        roi_data_path: Path to ROI JSON file
        
    Returns:
        Tuple of (eye_data, roi_manager)
    """
    import pandas as pd
    
    # Load eye tracking data
    try:
        eye_data = pd.read_csv(eye_data_path)
        print(f"Loaded eye tracking data with {len(eye_data)} samples")
    except Exception as e:
        print(f"Error loading eye tracking data: {str(e)}")
        eye_data = pd.DataFrame()
    
    # Load ROI data
    roi_manager = ROIManager()
    if os.path.exists(roi_data_path):
        success = roi_manager.load_roi_file(roi_data_path)
        if success:
            print(f"Loaded ROI data with {len(roi_manager.frame_numbers)} frames")
        else:
            print(f"Failed to load ROI data from {roi_data_path}")
    else:
        print(f"ROI data file not found: {roi_data_path}")
    
    return eye_data, roi_manager

def create_integrated_visualization(eye_data, roi_manager, frame_number=1, save_path=None):
    """
    Create a visualization that integrates eye tracking data with ROI overlays.
    
    Args:
        eye_data: DataFrame with eye tracking data
        roi_manager: ROIManager instance with loaded ROI data
        frame_number: Frame number to visualize
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set axis limits for normalized coordinates
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Invert y-axis to match screen coordinates
    
    # Find eye positions for this frame if available
    frame_data = eye_data[eye_data['frame_number'] == frame_number] if 'frame_number' in eye_data.columns else None
    
    if frame_data is not None and not frame_data.empty:
        # Plot eye positions
        for eye, color in [('left', 'blue'), ('right', 'orange')]:
            x_col, y_col = f'x_{eye}', f'y_{eye}'
            
            if x_col in frame_data.columns and y_col in frame_data.columns:
                # Normalize if needed
                if frame_data[x_col].max() > 1.0 or frame_data[y_col].max() > 1.0:
                    screen_width, screen_height = 1280, 1024  # Default screen dimensions
                    x_norm = frame_data[x_col] / screen_width
                    y_norm = frame_data[y_col] / screen_height
                else:
                    x_norm, y_norm = frame_data[x_col], frame_data[y_col]
                
                # Plot eye positions
                ax.scatter(x_norm, y_norm, c=color, label=f'{eye.title()} Eye', alpha=0.8, s=50)
    
    # Draw ROIs
    roi_manager.draw_rois_on_axis(ax, frame_number)
    
    # Add title and labels
    ax.set_title(f"Integrated Eye Tracking and ROI Visualization - Frame {frame_number}", fontsize=14)
    ax.set_xlabel("X Position (normalized)", fontsize=12)
    ax.set_ylabel("Y Position (normalized)", fontsize=12)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig

# Test the integration
if __name__ == "__main__":
    # Paths to sample data
    # eye_data_path = "eye_tracking_data.csv"  # Replace with actual path to eye data
    eye_data_path = ("results/20250430_121428/data/Children-play-finalXNewer/1017735502_unified_eye_metrics_Children"
                     "-play-finalXNewer.csv")
    roi_data_path = "roi_data.json"  # Current ROI file

    # Load data
    eye_data, roi_manager = load_sample_data(eye_data_path, roi_data_path)

    # Try to find a frame with data
    if 'frame_number' in eye_data.columns:
        frame_counts = eye_data['frame_number'].value_counts().sort_values(ascending=False)
        if len(frame_counts) > 0:
            # Get the frame with most data points
            best_frame = frame_counts.index[0]
            print(f"\nSelected frame {best_frame} with {frame_counts[best_frame]} data points for visualization")

            # Create integrated visualization with the frame that has the most data
            create_integrated_visualization(eye_data, roi_manager, frame_number=best_frame,
                                            save_path="integrated_visualization_best_frame.png")

    # Also try frame 1 for comparison
    create_integrated_visualization(eye_data, roi_manager, frame_number=1,
                                    save_path="integrated_visualization_frame1.png")
