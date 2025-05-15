"""
Animated ROI Scanpath Visualization
Author: Tal Alfi
Date: April 2025

This module extends the animated scanpath visualization with Region of Interest (ROI) 
capabilities, allowing users to visualize eye movements in relation to defined regions
in the stimulus.

The visualization can load data from either original ASC files or pre-processed CSV files,
providing flexibility and improved performance when working with previously analyzed data.
"""

import os
import numpy as np
import pandas as pd
# Configure matplotlib for thread safety
import matplotlib
# Only set the backend if it hasn't been set already (to avoid duplication with gui.py)
if matplotlib.get_backend() != 'Agg':
    matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QSlider, QLabel, QGroupBox, QSpinBox, QCheckBox,
                             QComboBox, QFileDialog, QSizePolicy, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, QSize

from GUI.visualization.roi_manager import ROIManager


class AnimatedROIScanpathWidget(QWidget):
    """Widget for displaying animated scanpath visualization with ROI overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.roi_manager = ROIManager()
        self.movie_name = None
        self.screen_width = 1280
        self.screen_height = 1024
        self.animation = None
        self.is_playing = False
        self.current_frame = 0
        self.playback_speed = 1.0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.current_roi = None  # Current ROI that gaze is in
        self.active_roi_id = None  # ID of ROI currently being gazed at
        
        # Store loaded movies data
        self.loaded_movies = {}  # Dictionary to store loaded movie data: {movie_name: data_dict}

        # Display options
        self.show_rois = True
        self.highlight_active_roi = True
        self.show_roi_labels = True
        # These attributes are now controlled by checkboxes directly
        self.trail_length = 100

        # For keeping track of real time
        self.last_update_time = None

        # ROI statistics
        self.roi_dwell_times = {}  # Label -> total time
        self.current_roi_start_time = None

        # Initialize UI
        self.init_ui()
        
    def sizeHint(self):
        """Suggested size for the widget."""
        return QSize(900, 700)  # Recommended default size
        
    def minimumSizeHint(self):
        """Minimum size for the widget."""
        return QSize(600, 450)  # Minimum usable size

    def init_ui(self):
        """Initialize the widget UI."""
        layout = QVBoxLayout(self)
        
        # Create a container for visualization that can manage minimum size
        viz_container = QWidget()
        viz_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)  # No margins to maximize space

        # Create matplotlib figure and canvas with adjusted figure size and tight layout
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.tight_layout(pad=2.0)  # Increase padding around the plot
        self.canvas = FigureCanvasQTAgg(self.fig)
        # Set size policy to make the canvas expand in both directions
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Use percentage of window height for minimum height
        self.canvas.setMinimumHeight(350)

        # Add canvas to visualization container
        viz_layout.addWidget(self.canvas)
        
        # Add visualization container to main layout
        layout.addWidget(viz_container, 1)  # Give it stretch factor to take available space
        
        # Create a dedicated container for playback controls
        playback_container = QWidget()
        playback_container.setMinimumHeight(50)  # Ensure minimum height for controls
        playback_container.setMaximumHeight(70)  # Limit maximum height
        playback_layout = QVBoxLayout(playback_container)
        playback_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create controls
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("▶ Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        # Reset button
        self.reset_button = QPushButton("⟲ Reset")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_animation)
        controls_layout.addWidget(self.reset_button)

        # Timeline slider
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self.slider_moved)
        controls_layout.addWidget(self.timeline_slider, 1)  # Give slider more space

        # Time display
        self.time_label = QLabel("0.0s / 0.0s")
        controls_layout.addWidget(self.time_label)

        # Add the controls to the playback container
        playback_layout.addLayout(controls_layout)
        
        # Add playback container to main layout
        layout.addWidget(playback_container)

        # Settings section with two groups in a vertical layout
        settings_container = QWidget()
        settings_main_layout = QVBoxLayout(settings_container)
        settings_main_layout.setSpacing(15)  # Increase spacing between groups
        settings_main_layout.setContentsMargins(10, 10, 10, 10)  # Add container margins
        
        # Create a more compact and efficient layout
        # Main container for all controls with horizontal arrangement
        controls_container = QWidget()
        controls_container.setMaximumHeight(130)  # Limit the height to maximize plot area
        main_controls_layout = QHBoxLayout(controls_container)
        main_controls_layout.setContentsMargins(5, 5, 5, 5)
        main_controls_layout.setSpacing(10)
        
        # === LEFT SECTION: Movie selection and playback settings ===
        left_section = QWidget()
        left_section_layout = QVBoxLayout(left_section)
        left_section_layout.setContentsMargins(0, 0, 0, 0)
        left_section_layout.setSpacing(5)
        
        # Movie selection row
        movie_selection_layout = QHBoxLayout()
        movie_selection_layout.addWidget(QLabel("Movie:"))
        self.movie_combo = QComboBox()
        self.movie_combo.setEnabled(False)
        self.movie_combo.currentIndexChanged.connect(self.movie_selected)
        self.movie_combo.setMinimumWidth(180)
        movie_selection_layout.addWidget(self.movie_combo, 1)
        left_section_layout.addLayout(movie_selection_layout)
        
        # Playback settings row
        playback_row = QHBoxLayout()
        
        # Speed controls
        playback_row.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.update_playback_speed)
        self.speed_combo.setFixedWidth(70)
        playback_row.addWidget(self.speed_combo)
        
        # Trail length controls
        playback_row.addWidget(QLabel("Trail:"))
        self.trail_spin = QSpinBox()
        self.trail_spin.setRange(10, 500)
        self.trail_spin.setValue(100)
        self.trail_spin.setSingleStep(10)
        self.trail_spin.valueChanged.connect(self.update_trail_length)
        self.trail_spin.setFixedWidth(70)
        playback_row.addWidget(self.trail_spin)
        
        left_section_layout.addLayout(playback_row)
        
        # === MIDDLE SECTION: Eye tracking controls ===
        middle_section = QWidget()
        middle_section_layout = QVBoxLayout(middle_section)
        middle_section_layout.setContentsMargins(0, 0, 0, 0)
        middle_section_layout.setSpacing(5)
        
        # Eye tracking controls in a grid
        eye_tracking_layout = QGridLayout()
        eye_tracking_layout.setSpacing(5)
        
        # Eye tracking checkboxes
        self.show_left_cb = QCheckBox("Show Left Eye")
        self.show_left_cb.setChecked(True)
        self.show_left_cb.toggled.connect(self.redraw)
        eye_tracking_layout.addWidget(self.show_left_cb, 0, 0)
        
        self.show_right_cb = QCheckBox("Show Right Eye")
        self.show_right_cb.setChecked(True)
        self.show_right_cb.toggled.connect(self.redraw)
        eye_tracking_layout.addWidget(self.show_right_cb, 0, 1)
        
        # ROI visibility options
        self.show_rois_cb = QCheckBox("Show ROIs")
        self.show_rois_cb.setChecked(True)
        self.show_rois_cb.toggled.connect(self.toggle_roi_display)
        eye_tracking_layout.addWidget(self.show_rois_cb, 1, 0)
        
        self.show_roi_labels_cb = QCheckBox("Show ROI Labels")
        self.show_roi_labels_cb.setChecked(True)
        self.show_roi_labels_cb.toggled.connect(self.toggle_roi_labels)
        eye_tracking_layout.addWidget(self.show_roi_labels_cb, 1, 1)
        
        # ROI highlighting
        self.highlight_active_roi_cb = QCheckBox("Highlight Active ROI")
        self.highlight_active_roi_cb.setChecked(True)
        self.highlight_active_roi_cb.toggled.connect(self.toggle_roi_highlight)
        eye_tracking_layout.addWidget(self.highlight_active_roi_cb, 2, 0, 1, 2)
        
        middle_section_layout.addLayout(eye_tracking_layout)
        
        # === RIGHT SECTION: ROI file controls ===
        right_section = QWidget()
        right_section_layout = QVBoxLayout(right_section)
        right_section_layout.setContentsMargins(0, 0, 0, 0)
        right_section_layout.setSpacing(5)
        
        # ROI file selection
        roi_file_layout = QHBoxLayout()
        roi_file_layout.setSpacing(5)
        
        roi_file_layout.addWidget(QLabel("ROI File:"))
        
        # ROI file selection button
        self.select_roi_btn = QPushButton("Select ROI File")
        self.select_roi_btn.setFixedWidth(110)
        self.select_roi_btn.clicked.connect(self.select_roi_file)
        roi_file_layout.addWidget(self.select_roi_btn)
        
        # File path display label (make it smaller)
        self.roi_file_label = QLabel("No ROI file selected")
        self.roi_file_label.setStyleSheet("padding: 2px; background-color: rgba(240, 240, 240, 50); border-radius: 3px;")
        roi_file_layout.addWidget(self.roi_file_label, 1)
        
        right_section_layout.addLayout(roi_file_layout)
        
        # Current ROI indicator
        self.current_roi_label = QLabel("Current ROI: None")
        right_section_layout.addWidget(self.current_roi_label)
        
        # ROI dwell time stats
        self.roi_stats_label = QLabel("ROI Dwell Times: Not available")
        right_section_layout.addWidget(self.roi_stats_label)
        
        # Add all sections to the main layout
        main_controls_layout.addWidget(left_section, 2)     # 2 parts width
        main_controls_layout.addWidget(middle_section, 2)   # 2 parts width
        main_controls_layout.addWidget(right_section, 3)    # 3 parts width (for file path)
        
        # Add the controls container to the main settings layout
        settings_main_layout.addWidget(controls_container)
        
        # Add settings container to main layout
        layout.addWidget(settings_container)

        # Remove status label entirely as it's not needed and takes up extra space

        # Initialize the plot
        self.init_plot()

        # Set initial playback speed
        self.update_playback_speed()

    def init_plot(self):
        """Initialize the plot with empty data."""
        # Clear the entire axis, including all text and patches
        self.ax.clear()
        
        # Make sure there are no lingering text elements
        while self.ax.texts:
            self.ax.texts[0].remove()
            
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(1, 0)  # Invert y-axis to match screen coordinates
        self.ax.set_title("Animated Scan Path with ROIs", fontsize=14)
        self.ax.set_xlabel("X Position (normalized)", fontsize=12)
        self.ax.set_ylabel("Y Position (normalized)", fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)

        # Create empty line objects for animation
        self.left_line, = self.ax.plot([], [], 'o-', color='blue',
                                       markersize=2, linewidth=0.5, alpha=0.7,
                                       label='Left Eye')

        self.right_line, = self.ax.plot([], [], 'o-', color='orange',
                                        markersize=2, linewidth=0.5, alpha=0.7,
                                        label='Right Eye')

        self.left_point, = self.ax.plot([], [], 'o', color='blue',
                                        markersize=8, alpha=1.0)

        self.right_point, = self.ax.plot([], [], 'o', color='orange',
                                         markersize=8, alpha=1.0)

        self.ax.legend(loc='upper right')
        
        # Apply tight layout to ensure plot fits properly
        self.fig.tight_layout(pad=2.0)
        
        self.canvas.draw()

    def toggle_roi_display(self, checked):
        """Toggle ROI display on/off."""
        self.show_rois = checked
        self.redraw()

    def toggle_roi_highlight(self, checked):
        """Toggle ROI highlight on/off."""
        self.highlight_active_roi = checked
        self.redraw()

    def toggle_roi_labels(self, checked):
        """Toggle ROI labels on/off."""
        self.show_roi_labels = checked
        self.redraw()
        
    def select_roi_file(self):
        """Handle ROI JSON file selection."""
        from PyQt5.QtWidgets import QFileDialog
        
        roi_file, _ = QFileDialog.getOpenFileName(
            self, "Select ROI File", "", "JSON Files (*.json)"
        )
        
        if roi_file and os.path.exists(roi_file):
            # Update label
            self.roi_file_label.setText(f"ROI File: {os.path.basename(roi_file)}")
            
            # Load the ROI file
            success = self.roi_manager.load_roi_file(roi_file)
            
            if success:
                # Enable the ROI display checkbox
                self.show_rois_cb.setChecked(True)
                self.show_rois = True
                
                # Extend ROI frames to match movie frame count if needed
                if self.data is not None and 'frame_number' in self.data.columns:
                    try:
                        # Get the maximum frame number from the movie data
                        movie_frame_count = int(self.data['frame_number'].max())
                        
                        # Extend ROI frames if movie has more frames
                        if movie_frame_count > 0:
                            self.roi_manager.extend_roi_frames(movie_frame_count)
                    except Exception as e:
                        print(f"Failed to extend ROI frames: {str(e)}")
                
                # Redraw with the new ROI data
                self.redraw()
                
                # ROI data loaded successfully - no status message needed
            else:
                # Silently handle error
                self.roi_file_label.setText("No ROI file selected")

    def movie_selected(self, index):
        """Handle movie selection and load the selected movie data."""
        if index < 0 or self.movie_combo.count() == 0:
            return
        
        movie_name = self.movie_combo.currentText()
        
        # Check if we have data for this movie
        if movie_name in self.loaded_movies:
            movie_data = self.loaded_movies[movie_name]
            
            # Get the data and ROI path
            eye_data = movie_data['data']
            roi_path = movie_data.get('roi_path', None)
            screen_width = movie_data.get('screen_width', 1280)
            screen_height = movie_data.get('screen_height', 1024)
            
            # Reset animation to initial state
            self.is_playing = False
            self.current_frame = 0
            self.last_update_time = None
            self.roi_dwell_times = {}
            self.current_roi_start_time = None
            self.active_roi_id = None
            
            # Store data and settings
            self.data = eye_data
            self.movie_name = movie_name
            self.screen_width = screen_width
            self.screen_height = screen_height
            
            # Load ROI data if available
            if roi_path and os.path.exists(roi_path):
                success = self.roi_manager.load_roi_file(roi_path)
                if success:
                    self.roi_file_label.setText(f"ROI File: {os.path.basename(roi_path)}")
                    self.show_rois = True
                    self.show_rois_cb.setChecked(True)
                    
                    # Extend ROI frames to match movie frame count if needed
                    if 'frame_number' in self.data.columns:
                        try:
                            # Get the maximum frame number from the movie data
                            movie_frame_count = int(self.data['frame_number'].max())
                            
                            # Extend ROI frames if movie has more frames
                            if movie_frame_count > 0:
                                self.roi_manager.extend_roi_frames(movie_frame_count)
                        except Exception as e:
                            print(f"Failed to extend ROI frames: {str(e)}")
                else:
                    self.roi_file_label.setText("No ROI file selected")
            
            # Update UI with loaded data
            # Calculate relative time in seconds for better display
            self.data['time_sec'] = (self.data['timestamp'] - self.data['timestamp'].iloc[0]) / 1000.0
            self.total_duration = self.data['time_sec'].iloc[-1]
            
            # Set up timeline slider
            self.timeline_slider.setMinimum(0)
            self.timeline_slider.setMaximum(len(self.data) - 1)
            self.timeline_slider.setValue(0)
            
            # Update time label
            self.time_label.setText(f"0.0s / {self.total_duration:.1f}s")
            
            # Enable controls
            self.play_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            
            # Try to normalize coordinates and ensure they exist
            try:
                # Check if we need to normalize
                if ('x_left_norm' not in self.data.columns or 
                    'y_left_norm' not in self.data.columns or
                    'x_right_norm' not in self.data.columns or
                    'y_right_norm' not in self.data.columns):
                    # Force normalization
                    self.data['x_left_norm'] = self.data['x_left'] / self.screen_width
                    self.data['y_left_norm'] = self.data['y_left'] / self.screen_height
                    self.data['x_right_norm'] = self.data['x_right'] / self.screen_width
                    self.data['y_right_norm'] = self.data['y_right'] / self.screen_height
                    print(f"Created normalization columns for movie: {movie_name}")
            except Exception as e:
                print(f"Error normalizing coordinates: {str(e)}")
                # Emergency fallback - create normalized columns from raw coordinates
                if 'x_left' in self.data.columns:
                    self.data['x_left_norm'] = self.data['x_left'] / self.screen_width
                    self.data['y_left_norm'] = self.data['y_left'] / self.screen_height
                    self.data['x_right_norm'] = self.data['x_right'] / self.screen_width
                    self.data['y_right_norm'] = self.data['y_right'] / self.screen_height
            
            # Initialize the plot with the loaded data
            self.redraw()
            
            # Data loaded successfully - no status message needed
            
            return True
        else:
            return False
    
    def _normalize_coordinates(self):
        """Normalize eye coordinates if they are in pixel values."""
        if self.data is None:
            return
        
        # First check if the normalization columns already exist
        if ('x_left_norm' in self.data.columns and 
            'y_left_norm' in self.data.columns and 
            'x_right_norm' in self.data.columns and 
            'y_right_norm' in self.data.columns):
            return
            
        # Add normalized columns if they don't exist
        try:    
            # Check if normalization is needed
            max_x = max(self.data['x_left'].max(), self.data['x_right'].max())
            max_y = max(self.data['y_left'].max(), self.data['y_right'].max())
            
            if max_x > 1.0 or max_y > 1.0:
                # Normalize to [0, 1] range
                self.data['x_left_norm'] = self.data['x_left'] / self.screen_width
                self.data['y_left_norm'] = self.data['y_left'] / self.screen_height
                self.data['x_right_norm'] = self.data['x_right'] / self.screen_width
                self.data['y_right_norm'] = self.data['y_right'] / self.screen_height
            else:
                # Already normalized
                self.data['x_left_norm'] = self.data['x_left']
                self.data['y_left_norm'] = self.data['y_left']
                self.data['x_right_norm'] = self.data['x_right']
                self.data['y_right_norm'] = self.data['y_right']
        except Exception as e:
            print(f"Error during normalization: {str(e)}")
            # If normalization fails, create backup columns to prevent errors
            self.data['x_left_norm'] = self.data['x_left']
            self.data['y_left_norm'] = self.data['y_left']
            self.data['x_right_norm'] = self.data['x_right']
            self.data['y_right_norm'] = self.data['y_right']
    
    def load_data(self, eye_data: pd.DataFrame, roi_data_path: str = None,
                  movie_name: str = "Unknown", screen_width: int = 1280,
                  screen_height: int = 1024) -> bool:
        """
        Load eye tracking data and ROI data for animation.

        Args:
            eye_data: DataFrame with unified eye metrics
            roi_data_path: Path to ROI JSON file
            movie_name: Name of the movie
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels

        Returns:
            True if loading was successful, False otherwise
        """
        if eye_data.empty:
            # Silently handle empty data
            return False

        # Check if required columns exist
        required_cols = ['timestamp', 'x_left', 'y_left', 'x_right', 'y_right', 'frame_number']
        missing_cols = [col for col in required_cols if col not in eye_data.columns]

        if missing_cols:
            # Silently handle missing columns
            return False

        # Store data in the loaded movies dictionary
        self.loaded_movies[movie_name] = {
            'data': eye_data.copy(),
            'roi_path': roi_data_path,
            'screen_width': screen_width,
            'screen_height': screen_height
        }
        
        # Update movie selection dropdown
        current_movie = self.movie_combo.currentText() if self.movie_combo.count() > 0 else None
        
        # Block signals to prevent triggering movie_selected while updating
        self.movie_combo.blockSignals(True)
        self.movie_combo.clear()
        self.movie_combo.addItems(sorted(self.loaded_movies.keys()))
        self.movie_combo.setEnabled(True)
        
        # Restore previous selection or select the new movie
        if current_movie and current_movie in self.loaded_movies:
            index = self.movie_combo.findText(current_movie)
            if index >= 0:
                self.movie_combo.setCurrentIndex(index)
        else:
            # Select the newly added movie
            index = self.movie_combo.findText(movie_name)
            if index >= 0:
                self.movie_combo.setCurrentIndex(index)
        
        # Unblock signals
        self.movie_combo.blockSignals(False)
        
        # Store data and settings for immediate use
        self.data = eye_data
        self.movie_name = movie_name
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Calculate relative time in seconds for better display
        self.data['time_sec'] = (self.data['timestamp'] - self.data['timestamp'].iloc[0]) / 1000.0
        self.total_duration = self.data['time_sec'].iloc[-1]

        # Load ROI data if provided
        if roi_data_path and os.path.exists(roi_data_path):
            success = self.roi_manager.load_roi_file(roi_data_path)
            if success:
                # Update the ROI file label
                self.roi_file_label.setText(f"ROI File: {os.path.basename(roi_data_path)}")
                
                # Get the movie frame count for ROI frame matching
                if 'frame_number' in self.data.columns:
                    try:
                        # Get the maximum frame number from the movie data
                        movie_frame_count = int(self.data['frame_number'].max())
                        
                        # Extend ROI frames if movie has more frames
                        if movie_frame_count > 0:
                            self.roi_manager.extend_roi_frames(movie_frame_count)
                    except Exception as e:
                        print(f"Failed to extend ROI frames: {str(e)}")
            else:
                # Failed to load ROI data - silently handle
                pass
        elif roi_data_path:
            # ROI data file not found - silently handle
            pass
        else:
            # No ROI file provided - that's okay
            pass

        # Normalize eye coordinates
        self._normalize_coordinates()

        # Reset animation state
        self.is_playing = False
        self.current_frame = 0
        self.last_update_time = None
        self.roi_dwell_times = {}
        self.current_roi_start_time = None
        self.active_roi_id = None

        # Set up timeline slider
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(len(self.data) - 1)
        self.timeline_slider.setValue(0)

        # Update time label
        self.time_label.setText(f"0.0s / {self.total_duration:.1f}s")

        # Enable controls
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.timeline_slider.setEnabled(True)

        # Initialize the plot with the loaded data
        self.redraw()

        # Data loaded successfully - no status message needed

        # Set trail length from spinner
        self.trail_length = self.trail_spin.value()

        return True

    def update_playback_speed(self):
        """Update the playback speed based on combo box selection."""
        speed_text = self.speed_combo.currentText()

        if speed_text == "0.25x":
            self.playback_speed = 0.25
        elif speed_text == "0.5x":
            self.playback_speed = 0.5
        elif speed_text == "1x":
            self.playback_speed = 1.0
        elif speed_text == "2x":
            self.playback_speed = 2.0
        elif speed_text == "4x":
            self.playback_speed = 4.0
        else:
            self.playback_speed = 1.0

    def update_trail_length(self):
        """Update the trail length setting and redraw."""
        self.trail_length = self.trail_spin.value()
        self.redraw()

    def toggle_play(self):
        """Toggle play/pause state."""
        if not self.is_playing:
            # Start playing
            self.is_playing = True
            self.play_button.setText("⏸ Pause")
            self.last_update_time = None  # Reset last update time
            self.timer.start(30)  # ~30fps
        else:
            # Pause
            self.is_playing = False
            self.play_button.setText("▶ Play")
            self.timer.stop()

    def update_animation(self):
        """Update the animation frame based on real time progression."""
        if not self.is_playing or self.data is None:
            return

        # Get the current time data
        current_time = self.data.iloc[self.current_frame]['time_sec']

        # Initialize last_update_time if first call
        if self.last_update_time is None:
            import datetime
            self.last_update_time = datetime.datetime.now()
            return

        # Calculate real elapsed time since last update (in seconds)
        import datetime
        now = datetime.datetime.now()
        real_elapsed = (now - self.last_update_time).total_seconds()
        self.last_update_time = now

        # Calculate how much video time should elapse based on playback speed
        video_time_increment = real_elapsed * self.playback_speed

        # Find the target video time
        target_time = current_time + video_time_increment

        # Find the frame closest to the target time
        time_column = self.data['time_sec'].values
        next_frame_idx = self.current_frame

        # Scan forward to find the frame closest to our target time
        while next_frame_idx < len(time_column) - 1:
            if time_column[next_frame_idx + 1] > target_time:
                break
            next_frame_idx += 1

        # Update the current frame
        self.current_frame = next_frame_idx

        # Check if we've reached the end
        if self.current_frame >= len(self.data) - 1:
            self.current_frame = len(self.data) - 1
            self.toggle_play()  # Pause at the end
            return

        # Update slider position (will trigger redraw through slider_moved)
        self.timeline_slider.setValue(self.current_frame)
        
        # Force an update of the ROI statistics display to ensure real-time updates
        self.update_roi_stats_display()

    def slider_moved(self, value):
        """Handle timeline slider movement."""
        if self.data is None:
            return

        self.current_frame = value
        self.update_display()

    def reset_animation(self):
        """Reset the animation to the beginning."""
        if self.data is None:
            return

        self.current_frame = 0
        self.timeline_slider.setValue(0)
        self.is_playing = False
        self.play_button.setText("▶ Play")
        self.timer.stop()
        self.roi_dwell_times = {}
        self.current_roi_start_time = None
        self.active_roi_id = None
        self.current_roi = None
        
        # Reset displays
        self.current_roi_label.setText("Current ROI: None")
        self.roi_stats_label.setText("ROI Dwell Times: Not available")
        self.update_roi_stats_display()

    def update_display(self):
        """Update the display based on current frame."""
        if self.data is None or self.current_frame >= len(self.data):
            return

        # Make sure normalization columns exist
        self._normalize_coordinates()
        
        # Check if normalization columns exist after attempted normalization
        if ('x_left_norm' not in self.data.columns or
            'y_left_norm' not in self.data.columns or
            'x_right_norm' not in self.data.columns or
            'y_right_norm' not in self.data.columns):
            print("Warning: Normalized coordinates are not available. Using raw coordinates.")
            # Create temporary normalization columns for this visualization
            self.data['x_left_norm'] = self.data['x_left'] / self.screen_width
            self.data['y_left_norm'] = self.data['y_left'] / self.screen_height
            self.data['x_right_norm'] = self.data['x_right'] / self.screen_width
            self.data['y_right_norm'] = self.data['y_right'] / self.screen_height

        # Get trail length
        trail_length = self.trail_length

        # Calculate trail start index
        start_idx = max(0, self.current_frame - trail_length)

        # Get slice of data for trail
        trail_data = self.data.iloc[start_idx:self.current_frame + 1]

        # Get current frame number for ROI detection
        current_frame_num = None
        if 'frame_number' in self.data.columns:
            current_frame_num = self.data.iloc[self.current_frame]['frame_number']
            if pd.isna(current_frame_num):
                current_frame_num = None
            else:
                current_frame_num = int(current_frame_num)

        # Clear ROI patches before redrawing
        for artist in self.ax.patches:
            artist.remove()
            
        # Clear text annotations (ROI labels)
        for text in self.ax.texts:
            text.remove()

        # Draw ROIs if enabled and frame number is available
        if self.show_rois and current_frame_num is not None:
            # Check for gaze position to detect active ROI
            active_roi = None
            if self.show_left_cb.isChecked():
                x_left = self.data.iloc[self.current_frame]['x_left_norm']
                y_left = self.data.iloc[self.current_frame]['y_left_norm']
                if not (pd.isna(x_left) or pd.isna(y_left)):
                    active_roi = self.roi_manager.find_roi_at_gaze(current_frame_num, x_left, y_left)
            elif self.show_right_cb.isChecked() and active_roi is None:
                # Only check right eye if left eye didn't find a hit
                x_right = self.data.iloc[self.current_frame]['x_right_norm']
                y_right = self.data.iloc[self.current_frame]['y_right_norm']
                if not (pd.isna(x_right) or pd.isna(y_right)):
                    active_roi = self.roi_manager.find_roi_at_gaze(current_frame_num, x_right, y_right)

            # Update current ROI and tracking for dwell times
            if active_roi is not None and 'object_id' in active_roi:
                active_roi_id = active_roi['object_id']
                if active_roi_id != self.active_roi_id:
                    # We've entered a new ROI
                    current_time = self.data.iloc[self.current_frame]['time_sec']

                    # End timing for previous ROI if applicable
                    if self.active_roi_id is not None and self.current_roi_start_time is not None:
                        roi_label = self.current_roi[
                            'label'] if self.current_roi and 'label' in self.current_roi else 'unknown'
                        dwell_time = current_time - self.current_roi_start_time
                        self.roi_dwell_times[roi_label] = self.roi_dwell_times.get(roi_label, 0) + dwell_time

                    # Start timing for new ROI
                    self.current_roi = active_roi
                    self.active_roi_id = active_roi_id
                    self.current_roi_start_time = current_time

                    # Update current ROI display in the right panel
                    if 'label' in active_roi:
                        self.current_roi_label.setText(f"Current ROI: {active_roi['label']}")
                    else:
                        self.current_roi_label.setText(f"Current ROI: Unknown (ID: {active_roi_id})")
            elif self.active_roi_id is not None:
                # We've exited an ROI
                current_time = self.data.iloc[self.current_frame]['time_sec']

                # End timing for previous ROI
                if self.current_roi_start_time is not None:
                    roi_label = self.current_roi[
                        'label'] if self.current_roi and 'label' in self.current_roi else 'unknown'
                    dwell_time = current_time - self.current_roi_start_time
                    self.roi_dwell_times[roi_label] = self.roi_dwell_times.get(roi_label, 0) + dwell_time

                # Reset current ROI tracking
                self.current_roi = None
                self.active_roi_id = None
                self.current_roi_start_time = None
                
                # Update current ROI display in the right panel
                self.current_roi_label.setText("Current ROI: None")

            # Draw ROIs with active ROI highlighting if enabled
            self.roi_manager.draw_rois_on_axis(
                self.ax,
                current_frame_num,
                show_labels=self.show_roi_labels,
                highlighted_roi=self.active_roi_id if self.highlight_active_roi else None
            )

        # Update left eye trail and current position
        if self.show_left_cb.isChecked():
            x_left = trail_data['x_left_norm'].values
            y_left = trail_data['y_left_norm'].values

            # Handle NaN values in the trail
            mask_left = ~(np.isnan(x_left) | np.isnan(y_left))
            if any(mask_left):
                self.left_line.set_data(x_left[mask_left], y_left[mask_left])

                # Update current position point
                current_x_left = self.data.iloc[self.current_frame]['x_left_norm']
                current_y_left = self.data.iloc[self.current_frame]['y_left_norm']

                if not (pd.isna(current_x_left) or pd.isna(current_y_left)):
                    self.left_point.set_data([current_x_left], [current_y_left])
                else:
                    self.left_point.set_data([], [])
            else:
                self.left_line.set_data([], [])
                self.left_point.set_data([], [])
        else:
            self.left_line.set_data([], [])
            self.left_point.set_data([], [])

        # Update right eye trail and current position
        if self.show_right_cb.isChecked():
            x_right = trail_data['x_right_norm'].values
            y_right = trail_data['y_right_norm'].values

            # Handle NaN values in the trail
            mask_right = ~(np.isnan(x_right) | np.isnan(y_right))
            if any(mask_right):
                self.right_line.set_data(x_right[mask_right], y_right[mask_right])

                # Update current position point
                current_x_right = self.data.iloc[self.current_frame]['x_right_norm']
                current_y_right = self.data.iloc[self.current_frame]['y_right_norm']

                if not (pd.isna(current_x_right) or pd.isna(current_y_right)):
                    self.right_point.set_data([current_x_right], [current_y_right])
                else:
                    self.right_point.set_data([], [])
            else:
                self.right_line.set_data([], [])
                self.right_point.set_data([], [])
        else:
            self.right_line.set_data([], [])
            self.right_point.set_data([], [])

        # Update time display
        current_time = self.data.iloc[self.current_frame]['time_sec']
        self.time_label.setText(f"{current_time:.1f}s / {self.total_duration:.1f}s")

        # Add frame number if available
        if current_frame_num is not None:
            self.ax.set_title(f"Animated Scan Path with ROIs - Frame {current_frame_num}", fontsize=14)
        else:
            self.ax.set_title("Animated Scan Path with ROIs", fontsize=14)

        # Adjust the figure size to match the canvas size
        if hasattr(self, 'canvas'):
            width, height = self.canvas.get_width_height()
            # Only adjust if we have reasonable dimensions
            if width > 100 and height > 100:
                self.fig.set_size_inches(width/100, height/100)
                
        # Apply tight layout to ensure plot fits properly
        self.fig.tight_layout(pad=2.0)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # Always update the ROI statistics display to ensure the UI shows the latest information
        self.update_roi_stats_display()
        
        # Redraw canvas
        self.canvas.draw()

    def update_roi_stats_display(self):
        """Update the ROI statistics display in the right panel."""
        # Update Current ROI label
        if self.current_roi is not None and 'label' in self.current_roi:
            self.current_roi_label.setText(f"Current ROI: {self.current_roi['label']}")
        else:
            self.current_roi_label.setText("Current ROI: None")
            
        # Update ROI Dwell Times label
        if not self.roi_dwell_times:
            self.roi_stats_label.setText("ROI Dwell Times: Not available")
            return

        # Format ROI dwell times for display
        stats_text = "ROI Dwell Times: "
        for label, time in sorted(self.roi_dwell_times.items(), key=lambda x: x[1], reverse=True):
            stats_text += f"{label}: {time:.2f}s, "

        # Remove trailing comma and space
        stats_text = stats_text.rstrip(", ")
        
        self.roi_stats_label.setText(stats_text)

    def redraw(self):
        """Completely redraw the plot."""
        # Clear everything before redrawing
        self.ax.clear()
        
        # Additionally clear any lingering text objects
        while self.ax.texts:
            self.ax.texts[0].remove()
            
        # Initialize the plot with fresh elements
        self.init_plot()
        
        # Update the display with current data
        if self.data is not None:
            self.update_display()
            
        # Apply tight layout to ensure plot fits properly
        self.fig.tight_layout(pad=2.0)
        
    def resizeEvent(self, event):
        """Handle widget resize events to adjust the figure."""
        super().resizeEvent(event)
        # Force a redraw to adjust to the new size
        if hasattr(self, 'data') and self.data is not None:
            self.update_display()


# Simple test for the widget
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    widget = AnimatedROIScanpathWidget()
    widget.show()

    # Load sample data if available
    eye_data_path = "results/20250430_121428/data/Children-play-finalXNewer/1017735502_unified_eye_metrics_Children-play-finalXNewer.csv"
    roi_data_path = "roi_data.json"

    if os.path.exists(eye_data_path) and os.path.exists(roi_data_path):
        eye_data = pd.read_csv(eye_data_path)
        widget.load_data(eye_data, roi_data_path, "Children Play")

    sys.exit(app.exec_())