"""
Animated Scan Path Visualization
Author: Tal Alfi
Date: April 2025

This module adds animation capabilities to the eye movement scan path visualization,
allowing users to see the temporal progression of eye movements during movie viewing.

The animation can load data from either original ASC files or pre-processed CSV files,
providing flexibility and improved performance when working with previously analyzed data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QSlider, QLabel, QGroupBox, QSpinBox, QCheckBox,
                             QComboBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer


class AnimatedScanpathWidget(QWidget):
    """Widget for displaying and controlling animated scan path visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.movie_name = None
        self.screen_width = 1280
        self.screen_height = 1024
        self.animation = None
        self.is_playing = False
        self.current_frame = 0
        self.playback_speed = 1.0  # Initialize playback_speed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        
        # Store loaded movies data
        self.loaded_movies = {}  # Dictionary to store loaded movie data: {movie_name: data_dict}

        # For keeping track of real time
        self.last_update_time = None

        # Initialize UI
        self.init_ui()



    def load_data(self, data, movie_name="Unknown", screen_width=1280, screen_height=1024):
        """
        Load eye tracking data for animation.

        Args:
            data: DataFrame with unified eye metrics
            movie_name: Name of the movie
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        if data.empty:
            self.status_label.setText("Error: Empty data")
            return False

        # Check for required columns
        required_cols = ['timestamp', 'x_left', 'y_left', 'x_right', 'y_right']
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            self.status_label.setText(f"Error: Missing columns: {', '.join(missing_cols)}")
            return False
            
        # Store data in the loaded movies dictionary
        self.loaded_movies[movie_name] = {
            'data': data.copy(),
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
        self.data = data
        self.movie_name = movie_name
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Reset animation state
        self.is_playing = False
        self.current_frame = 0
        self.last_update_time = None

        # Calculate relative time in seconds for better display
        self.data['time_sec'] = (self.data['timestamp'] - self.data['timestamp'].iloc[0]) / 1000.0
        self.total_duration = self.data['time_sec'].iloc[-1]

        # Set up timeline slider
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(len(data) - 1)
        self.timeline_slider.setValue(0)

        # Update time label
        self.time_label.setText(f"0.0s / {self.total_duration:.1f}s")

        # Enable controls
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.timeline_slider.setEnabled(True)
        self.export_button.setEnabled(True)

        # Initialize the plot with the loaded data
        self.redraw()

        # Update status
        self.status_label.setText(f"Loaded {len(data)} samples from {movie_name} "
                                  f"({self.total_duration:.1f} seconds)")

        return True

    def init_ui(self):
        """Initialize the widget UI."""
        layout = QVBoxLayout(self)

        # Create matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(500)

        # Add canvas to layout
        layout.addWidget(self.canvas)

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

        layout.addLayout(controls_layout)

        # Settings section with proper organization
        settings_container = QWidget()
        settings_main_layout = QVBoxLayout(settings_container)
        settings_main_layout.setSpacing(15)  # Increase spacing between groups
        settings_main_layout.setContentsMargins(10, 10, 10, 10)  # Add container margins
        
        # Movie selection section
        movie_selection_container = QWidget()
        movie_selection_layout = QHBoxLayout(movie_selection_container)
        movie_selection_layout.setContentsMargins(5, 5, 5, 5)
        movie_selection_layout.setSpacing(10)
        
        # Movie selection dropdown
        movie_selection_layout.addWidget(QLabel("Select Movie:"))
        self.movie_combo = QComboBox()
        self.movie_combo.setEnabled(False)
        self.movie_combo.currentIndexChanged.connect(self.movie_selected)
        self.movie_combo.setMinimumWidth(200)
        movie_selection_layout.addWidget(self.movie_combo, 1)
        
        # Add the movie selection section to the main layout
        settings_main_layout.addWidget(movie_selection_container)
        
        # Animation controls with eye tracking options
        animation_settings_group = QGroupBox("Animation Controls")
        animation_settings_group.setMinimumHeight(150)  # Ensure minimum height for better appearance
        animation_settings_layout = QHBoxLayout(animation_settings_group)
        animation_settings_layout.setContentsMargins(15, 25, 15, 15)  # Increase padding
        animation_settings_layout.setSpacing(25)  # Increase spacing between elements
        
        # Create a container for playback controls to align them nicely
        playback_section = QWidget()
        playback_section_layout = QVBoxLayout(playback_section)
        playback_section_layout.setContentsMargins(5, 0, 5, 0)
        playback_section_layout.setSpacing(15)  # Increase vertical spacing
        
        # Add a header for this section
        playback_header = QLabel("Playback Settings")
        playback_header.setAlignment(Qt.AlignCenter)
        playback_header.setStyleSheet("font-weight: bold;")
        playback_section_layout.addWidget(playback_header)
        
        # Create horizontal layout for playback controls
        playback_controls_layout = QHBoxLayout()
        playback_controls_layout.setSpacing(20)  # Space between speed and trail controls
        
        # Playback speed controls
        speed_layout = QVBoxLayout()
        speed_layout.setSpacing(8)  # Increase spacing
        speed_label = QLabel("Playback Speed:")
        speed_label.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(speed_label)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.update_playback_speed)
        self.speed_combo.setMinimumWidth(100)
        speed_layout.addWidget(self.speed_combo)
        playback_controls_layout.addLayout(speed_layout)
        
        # Trail length controls
        trail_layout = QVBoxLayout()
        trail_layout.setSpacing(8)  # Increase spacing
        trail_label = QLabel("Trail Length:")
        trail_label.setAlignment(Qt.AlignCenter)
        trail_layout.addWidget(trail_label)
        
        self.trail_spin = QSpinBox()
        self.trail_spin.setRange(10, 500)
        self.trail_spin.setValue(100)
        self.trail_spin.setSingleStep(10)
        self.trail_spin.valueChanged.connect(self.update_trail_length)
        self.trail_spin.setMinimumWidth(100)
        trail_layout.addWidget(self.trail_spin)
        playback_controls_layout.addLayout(trail_layout)
        
        # Add the horizontal playback controls to the playback section
        playback_section_layout.addLayout(playback_controls_layout)
        
        # Add the playback section to the main layout
        animation_settings_layout.addWidget(playback_section, 1)
        
        # Add a vertical separator line
        separator = QWidget()
        separator.setFixedWidth(1)
        separator.setStyleSheet("background-color: #d0d0d0;")  # Light gray line
        animation_settings_layout.addWidget(separator)
        
        # Create a container for eye tracking options to align them nicely
        eye_tracking_section = QWidget()
        eye_tracking_layout = QVBoxLayout(eye_tracking_section)
        eye_tracking_layout.setContentsMargins(5, 0, 5, 0)
        eye_tracking_layout.setSpacing(8)
        
        # Add a header for this section
        eye_tracking_header = QLabel("Eye Tracking Display")
        eye_tracking_header.setAlignment(Qt.AlignCenter)
        eye_tracking_header.setStyleSheet("font-weight: bold;")
        eye_tracking_layout.addWidget(eye_tracking_header)
        
        # Add checkboxes with better spacing
        checkbox_container = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_container)
        checkbox_layout.setSpacing(10)  # Increase spacing between checkboxes
        
        self.show_left_cb = QCheckBox("Show Left Eye")
        self.show_left_cb.setChecked(True)
        self.show_left_cb.toggled.connect(self.redraw)
        checkbox_layout.addWidget(self.show_left_cb)
        
        self.show_right_cb = QCheckBox("Show Right Eye")
        self.show_right_cb.setChecked(True)
        self.show_right_cb.toggled.connect(self.redraw)
        checkbox_layout.addWidget(self.show_right_cb)
        
        # Add some padding at the bottom for alignment
        checkbox_layout.addStretch(1)
        
        # Add the checkbox container to the eye tracking section
        eye_tracking_layout.addWidget(checkbox_container)
        
        # Add the eye tracking section to the main layout
        animation_settings_layout.addWidget(eye_tracking_section, 1)
        
        # Add a vertical separator line
        separator2 = QWidget()
        separator2.setFixedWidth(1)
        separator2.setStyleSheet("background-color: #d0d0d0;")  # Light gray line
        animation_settings_layout.addWidget(separator2)
        
        # Create a container for export options
        export_section = QWidget()
        export_layout = QVBoxLayout(export_section)
        export_layout.setContentsMargins(5, 0, 5, 0)
        export_layout.setSpacing(8)
        
        # Add a header for this section
        export_header = QLabel("Animation Export")
        export_header.setAlignment(Qt.AlignCenter)
        export_header.setStyleSheet("font-weight: bold;")
        export_layout.addWidget(export_header)
        
        # Add export button
        self.export_button = QPushButton("Save to File")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_animation)
        self.export_button.setMinimumWidth(100)
        export_layout.addWidget(self.export_button)
        
        # Add some padding at the bottom for alignment
        export_layout.addStretch(1)
        
        # Add the export section to the main layout
        animation_settings_layout.addWidget(export_section)
        
        # Add animation controls to main settings layout
        settings_main_layout.addWidget(animation_settings_group)
        
        # Add settings container to main layout
        layout.addWidget(settings_container)

        # Status label
        self.status_label = QLabel("Load eye tracking data to begin")
        layout.addWidget(self.status_label)

        # Initialize the plot
        self.init_plot()

        # Set initial playback speed
        self.update_playback_speed()

    def init_plot(self):
        """Initialize the plot with empty data."""
        self.ax.clear()
        self.ax.set_xlim(0, self.screen_width)
        self.ax.set_ylim(self.screen_height, 0)  # Invert y-axis to match screen coordinates
        self.ax.set_title("Animated Scan Path", fontsize=14)
        self.ax.set_xlabel("X Position (pixels)", fontsize=12)
        self.ax.set_ylabel("Y Position (pixels)", fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)

        # Create empty line objects for animation
        self.left_line, = self.ax.plot([], [], 'o-', color='#1f77b4',
                                       markersize=2, linewidth=0.5, alpha=0.7,
                                       label='Left Eye')

        self.right_line, = self.ax.plot([], [], 'o-', color='#ff7f0e',
                                        markersize=2, linewidth=0.5, alpha=0.7,
                                        label='Right Eye')

        self.left_point, = self.ax.plot([], [], 'o', color='#1f77b4',
                                        markersize=8, alpha=1.0)

        self.right_point, = self.ax.plot([], [], 'o', color='#ff7f0e',
                                         markersize=8, alpha=1.0)

        self.ax.legend(loc='upper right')
        self.canvas.draw()


    def export_animation(self):
        """Export the animation as a video file."""
        if self.data is None:
            return

        # Get save path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation", "", "MP4 Files (*.mp4);;GIF Files (*.gif)"
        )

        if not file_path:
            return

        # Show status
        self.status_label.setText("Exporting animation... Please wait, this may take a while.")
        self.export_button.setEnabled(False)
        self.update()  # Force UI update

        try:
            # Create a new figure for export (with better resolution)
            export_fig, export_ax = plt.subplots(figsize=(10, 8), dpi=100)
            export_ax.set_xlim(0, self.screen_width)
            export_ax.set_ylim(self.screen_height, 0)
            export_ax.set_title(f"Eye Movement Scan Path - {self.movie_name}", fontsize=14)
            export_ax.set_xlabel("X Position (pixels)", fontsize=12)
            export_ax.set_ylabel("Y Position (pixels)", fontsize=12)
            export_ax.grid(True, linestyle='--', alpha=0.7)

            # Create line objects for animation
            left_line, = export_ax.plot([], [], 'o-', color='#1f77b4',
                                        markersize=2, linewidth=0.5, alpha=0.7,
                                        label='Left Eye')

            right_line, = export_ax.plot([], [], 'o-', color='#ff7f0e',
                                         markersize=2, linewidth=0.5, alpha=0.7,
                                         label='Right Eye')

            left_point, = export_ax.plot([], [], 'o', color='#1f77b4',
                                         markersize=8, alpha=1.0)

            right_point, = export_ax.plot([], [], 'o', color='#ff7f0e',
                                          markersize=8, alpha=1.0)

            export_ax.legend(loc='upper right')

            # Get trail length
            trail_length = self.trail_spin.value()
            show_left = self.show_left_cb.isChecked()
            show_right = self.show_right_cb.isChecked()

            # Animation frame function
            def update_frame(frame_idx):
                # Calculate trail start index
                start_idx = max(0, frame_idx - trail_length)

                # Get slice of data for trail
                trail_data = self.data.iloc[start_idx:frame_idx + 1]

                # Update plot title with frame number or time
                if 'frame_number' in self.data.columns:
                    current_frame_num = self.data.iloc[frame_idx]['frame_number']
                    if not pd.isna(current_frame_num):
                        export_ax.set_title(f"Eye Movement Scan Path - Frame {int(current_frame_num)}", fontsize=14)

                # Update left eye trail and current position
                if show_left:
                    x_left = trail_data['x_left'].values
                    y_left = trail_data['y_left'].values

                    # Handle NaN values
                    mask_left = ~(np.isnan(x_left) | np.isnan(y_left))
                    if any(mask_left):
                        left_line.set_data(x_left[mask_left], y_left[mask_left])

                        # Update current position point
                        current_x_left = self.data.iloc[frame_idx]['x_left']
                        current_y_left = self.data.iloc[frame_idx]['y_left']

                        if not (np.isnan(current_x_left) or np.isnan(current_y_left)):
                            left_point.set_data([current_x_left], [current_y_left])
                        else:
                            left_point.set_data([], [])
                    else:
                        left_line.set_data([], [])
                        left_point.set_data([], [])
                else:
                    left_line.set_data([], [])
                    left_point.set_data([], [])

                # Update right eye trail and current position
                if show_right:
                    x_right = trail_data['x_right'].values
                    y_right = trail_data['y_right'].values

                    # Handle NaN values
                    mask_right = ~(np.isnan(x_right) | np.isnan(y_right))
                    if any(mask_right):
                        right_line.set_data(x_right[mask_right], y_right[mask_right])

                        # Update current position point
                        current_x_right = self.data.iloc[frame_idx]['x_right']
                        current_y_right = self.data.iloc[frame_idx]['y_right']

                        if not (np.isnan(current_x_right) or np.isnan(current_y_right)):
                            right_point.set_data([current_x_right], [current_y_right])
                        else:
                            right_point.set_data([], [])
                    else:
                        right_line.set_data([], [])
                        right_point.set_data([], [])
                else:
                    right_line.set_data([], [])
                    right_point.set_data([], [])

                return left_line, right_line, left_point, right_point

            # Create and save animation
            frames = len(self.data)
            step = max(1, frames // 300)  # Limit to max ~300 frames for reasonable file size

            ani = animation.FuncAnimation(
                export_fig, update_frame,
                frames=range(0, frames, step),
                blit=True, interval=30
            )

            # Determine file format based on extension
            if file_path.lower().endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=20, bitrate=1800)
                ani.save(file_path, writer=writer)
            else:  # GIF
                ani.save(file_path, writer='pillow', fps=15)

            plt.close(export_fig)

            self.status_label.setText(f"Animation exported to {file_path}")
        except Exception as e:
            self.status_label.setText(f"Error exporting animation: {str(e)}")

        self.export_button.setEnabled(True)


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

    def _normalize_coordinates(self):
        """Normalize eye coordinates if they are in pixel values."""
        if self.data is None:
            return
            
        # We don't need normalization in this widget as it uses pixel values directly
        # But we keep this method for consistency with ROI version
        pass
        
    def movie_selected(self, index):
        """Handle movie selection and load the selected movie data."""
        if index < 0 or self.movie_combo.count() == 0:
            return
        
        movie_name = self.movie_combo.currentText()
        
        # Check if we have data for this movie
        if movie_name in self.loaded_movies:
            movie_data = self.loaded_movies[movie_name]
            
            # Load the data for this movie
            self.status_label.setText(f"Loading data for movie: {movie_name}...")
            
            # Get the data
            data = movie_data['data']
            screen_width = movie_data.get('screen_width', 1280)
            screen_height = movie_data.get('screen_height', 1024)
            
            # Reset animation to initial state
            self.is_playing = False
            self.current_frame = 0
            self.last_update_time = None
            
            # Store data and settings
            self.data = data
            self.movie_name = movie_name
            self.screen_width = screen_width
            self.screen_height = screen_height
            
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
            self.export_button.setEnabled(True)
            
            # Initialize the plot with the loaded data
            self.redraw()
            
            # Update status
            self.status_label.setText(f"Loaded {len(self.data)} samples from {movie_name} "
                                    f"({self.total_duration:.1f} seconds)")
            
            return True
        else:
            self.status_label.setText(f"No data available for movie: {movie_name}")
            return False
    
    def update_trail_length(self):
        """Update the trail length setting and redraw."""
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
            from datetime import datetime
            self.last_update_time = datetime.now()
            return

        # Calculate real elapsed time since last update (in seconds)
        from datetime import datetime
        now = datetime.now()
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

        # Update slider position (will trigger redraw)
        self.timeline_slider.setValue(self.current_frame)

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

    def update_display(self):
        """Update the display based on current frame."""
        if self.data is None or self.current_frame is None:
            return

        # Get trail length
        trail_length = self.trail_spin.value()
        
        # Calculate trail start index
        start_idx = max(0, self.current_frame - trail_length)
        
        # Get slice of data for trail
        trail_data = self.data.iloc[start_idx:self.current_frame + 1]
        
        # Update left eye trail if enabled
        if self.show_left_cb.isChecked():
            x_left = trail_data['x_left'].values
            y_left = trail_data['y_left'].values
            
            # Handle NaN values
            mask_left = ~(np.isnan(x_left) | np.isnan(y_left))
            if any(mask_left):
                self.left_line.set_data(x_left[mask_left], y_left[mask_left])
                
                # Update current position point
                current_x_left = self.data.iloc[self.current_frame]['x_left']
                current_y_left = self.data.iloc[self.current_frame]['y_left']
                
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
            
        # Update right eye trail if enabled
        if self.show_right_cb.isChecked():
            x_right = trail_data['x_right'].values
            y_right = trail_data['y_right'].values
            
            # Handle NaN values
            mask_right = ~(np.isnan(x_right) | np.isnan(y_right))
            if any(mask_right):
                self.right_line.set_data(x_right[mask_right], y_right[mask_right])
                
                # Update current position point
                current_x_right = self.data.iloc[self.current_frame]['x_right']
                current_y_right = self.data.iloc[self.current_frame]['y_right']
                
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

        # Redraw canvas
        self.canvas.draw()


    def redraw(self):
        """Completely redraw the plot."""
        self.init_plot()
        if self.data is not None:
            self.ax.set_title(f"Animated Scan Path - {self.movie_name}", fontsize=14)
            self.update_display()



# This function is kept for backward compatibility with tests
def create_animated_scanpath(data, movie_name, screen_width=1280, screen_height=1024):
    """
    Create an animated scanpath visualization widget.
    
    This function is kept for backward compatibility with tests.
    In new code, prefer instantiating the AnimatedScanpathWidget class directly.
    
    Args:
        data: DataFrame with unified eye metrics
        movie_name: Name of the movie for display
        screen_width: Width of the screen in pixels
        screen_height: Height of the screen in pixels
        
    Returns:
        AnimatedScanpathWidget instance with data loaded
    """
    # Create a new widget
    widget = AnimatedScanpathWidget()
    
    # Load the data into the widget
    widget.load_data(
        data=data,
        movie_name=movie_name,
        screen_width=screen_width,
        screen_height=screen_height
    )
    
    return widget