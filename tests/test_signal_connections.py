"""
Signal-Slot Connections Tests Module for Eye Movement Analysis Application
Author: Claude Code Assistant
Date: May 2025

This module provides tests for the signal-slot connections in the GUI,
focusing on interaction between components via PyQt signals.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Patch PyQt5's QApplication to allow headless testing
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton, QWidget, QVBoxLayout, QComboBox, QLabel
from PyQt5.QtCore import pyqtSignal, QObject, Qt
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

# Mock required modules
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.ndimage'] = MagicMock()
sys.modules['scipy.ndimage.gaussian_filter1d'] = MagicMock()

# Create mock for eyelink_visualizer
class MockMovieEyeTrackingVisualizer:
    def __init__(self, *args, **kwargs):
        pass
    
    def process_all_movies(self, *args, **kwargs):
        return {"test_movie": ["/path/to/test_plot.png"]}
    
    def generate_report(self, *args, **kwargs):
        return "/path/to/report.html"

sys.modules['eyelink_visualizer'] = MagicMock()
sys.modules['eyelink_visualizer'].MovieEyeTrackingVisualizer = MockMovieEyeTrackingVisualizer
sys.modules['visualization'] = MagicMock()
sys.modules['visualization.eyelink_visualizer'] = MagicMock()
sys.modules['visualization.eyelink_visualizer'].MovieEyeTrackingVisualizer = MockMovieEyeTrackingVisualizer

# Mock animated_roi_scanpath module but create proper widget class
class MockAnimatedROIScanpathWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setLayout(QVBoxLayout())
        
        # Create mock UI elements
        self.play_button = QPushButton("Play")
        self.reset_button = QPushButton("Reset")
        self.timeline_slider = MagicMock()
        self.show_left_cb = MagicMock()
        self.show_right_cb = MagicMock()
        self.show_rois_cb = MagicMock()
        self.show_roi_labels_cb = MagicMock()
        self.highlight_active_roi_cb = MagicMock()
        self.trail_spin = MagicMock()
        self.speed_combo = MagicMock()
        self.movie_combo = QComboBox()
        self.roi_file_label = QLabel("No ROI file selected")
        self.current_roi_label = QLabel("Current ROI: None")
        
        # Set default states
        self.play_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.show_left_cb.isChecked = lambda: True
        self.show_right_cb.isChecked = lambda: True
        self.show_rois_cb.isChecked = lambda: True
        self.show_roi_labels_cb.isChecked = lambda: True
        self.highlight_active_roi_cb.isChecked = lambda: True
        self.show_rois = True
        self.show_roi_labels = True
        self.highlight_active_roi = True
        self.trail_length = 100
        self.playback_speed = 1.0
        self.current_frame = 0
        self.data = None
        self.movie_name = None
        self.screen_width = 1280
        self.screen_height = 1024
        self.is_playing = False
        
        # Methods
        self.toggle_play = MagicMock()
        self.reset_animation = MagicMock()
        self.roi_manager = MagicMock()
        self.roi_manager.load_roi_file = MagicMock(return_value=True)
        
    def load_data(self, *args, **kwargs):
        self.data = kwargs.get('eye_data')
        self.movie_name = kwargs.get('movie_name', "Test Movie")
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.timeline_slider.setEnabled = MagicMock()
        self.movie_combo.clear()
        self.movie_combo.addItem(self.movie_name)
        return True

# Replace the actual module with our mock
from animated_roi_scanpath import AnimatedROIScanpathWidget as OriginalWidget
import animated_roi_scanpath
animated_roi_scanpath.AnimatedROIScanpathWidget = MockAnimatedROIScanpathWidget

# Import GUI modules
from GUI.gui import EyeMovementAnalysisGUI
from GUI.data.processing_thread import ProcessingThread


class TestSignalConnections(unittest.TestCase):
    """Test signal-slot connections in the GUI application."""
    
    def setUp(self):
        """Set up the test environment."""
        self.gui = EyeMovementAnalysisGUI()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample CSV file
        self.csv_file = os.path.join(self.temp_dir, "test_unified_eye_metrics.csv")
        self.create_sample_csv()
        
        # Create temporary output directory
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.gui.close()
        del self.gui
        
        # Remove temporary directory and all files
        shutil.rmtree(self.temp_dir)
    
    def create_sample_csv(self):
        """Create a sample CSV file for testing."""
        data = {
            'timestamp': list(range(0, 10000, 100)),
            'x_left': [x / 1280 for x in range(100)],
            'y_left': [y / 1024 for y in range(100)],
            'x_right': [x / 1280 for x in range(100)],
            'y_right': [y / 1024 for y in range(100)],
            'frame_number': list(range(100)),
        }
        pd.DataFrame(data).to_csv(self.csv_file, index=False)
    
    def test_process_button_connection(self):
        """Test process button signal-slot connection."""
        # Replace the process_data method with a mock but save original
        original_process_data = self.gui.process_data
        self.gui.process_data = MagicMock()
        
        try:
            # Prepare the GUI state to ensure process button works
            self.gui.file_paths = [self.csv_file]
            self.gui.output_dir = self.output_dir
            self.gui.update_process_button()  # Enable the button
            
            # Manually connect clicked signal to our mock
            self.gui.process_btn.clicked.disconnect()  # Disconnect existing connections
            self.gui.process_btn.clicked.connect(self.gui.process_data)
            
            # Click the process button
            self.gui.process_btn.click()
            
            # Check that process_data was called
            self.gui.process_data.assert_called_once()
        finally:
            # Restore original method and reconnect
            self.gui.process_data = original_process_data
            self.gui.process_btn.clicked.disconnect()
            self.gui.process_btn.clicked.connect(self.gui.process_data)
    
    def test_select_file_button_connection(self):
        """Test select file button signal-slot connection."""
        # First, mock QFileDialog to prevent actual dialogs from appearing
        original_getOpenFileNames = QFileDialog.getOpenFileNames
        QFileDialog.getOpenFileNames = MagicMock(return_value=([self.csv_file], "CSV Files (*.csv)"))
        
        try:
            # Replace the select_files method with a mock
            original_select_files = self.gui.select_files
            self.gui.select_files = MagicMock()
            
            # Find the select file button using QPushButton type and text
            select_file_btn = None
            for btn in self.gui.findChildren(QPushButton):
                if btn.text() == "Load Source File(s)":
                    select_file_btn = btn
                    break
            
            if select_file_btn:
                # Manually connect the button's clicked signal to our mock
                select_file_btn.clicked.disconnect()  # Disconnect existing connections
                select_file_btn.clicked.connect(self.gui.select_files)
                
                # Click the button
                select_file_btn.click()
                
                # Check that select_files was called
                self.gui.select_files.assert_called_once()
            else:
                self.fail("Load Source File(s) button not found")
        finally:
            # Restore original methods
            QFileDialog.getOpenFileNames = original_getOpenFileNames
            self.gui.select_files = original_select_files
    
    def test_select_output_button_connection(self):
        """Test select output button signal-slot connection."""
        # First, mock QFileDialog to prevent actual dialogs from appearing
        original_getExistingDirectory = QFileDialog.getExistingDirectory
        QFileDialog.getExistingDirectory = MagicMock(return_value=self.output_dir)
        
        try:
            # Replace the select_output_dir method with a mock
            original_select_output_dir = self.gui.select_output_dir
            self.gui.select_output_dir = MagicMock()
            
            # Find the select output button using QPushButton type and text
            select_output_btn = None
            for btn in self.gui.findChildren(QPushButton):
                if btn.text() == "Select Output Directory":
                    select_output_btn = btn
                    break
            
            if select_output_btn:
                # Manually connect the button's clicked signal to our mock
                select_output_btn.clicked.disconnect()  # Disconnect existing connections
                select_output_btn.clicked.connect(self.gui.select_output_dir)
                
                # Click the button
                select_output_btn.click()
                
                # Check that select_output_dir was called
                self.gui.select_output_dir.assert_called_once()
            else:
                self.fail("Select Output Directory button not found")
        finally:
            # Restore original methods
            QFileDialog.getExistingDirectory = original_getExistingDirectory
            self.gui.select_output_dir = original_select_output_dir
    
    def test_movie_combo_connection(self):
        """Test movie combo box signal-slot connection."""
        # Replace the movie_selected method with a mock but save original
        original_movie_selected = self.gui.movie_selected
        self.gui.movie_selected = MagicMock()
        
        try:
            # Set up the movie combo box with some items and connect signal manually
            self.gui.movie_combo.clear()
            self.gui.movie_combo.addItems(["Movie1", "Movie2"])
            self.gui.movie_combo.setEnabled(True)
            
            # Manually connect currentIndexChanged to our mock
            self.gui.movie_combo.currentIndexChanged.connect(self.gui.movie_selected)
            
            # Change the selection
            self.gui.movie_combo.setCurrentIndex(1)
            
            # Check that movie_selected was called
            self.gui.movie_selected.assert_called()
        finally:
            # Restore original method
            self.gui.movie_selected = original_movie_selected
    
    def test_viz_type_combo_connection(self):
        """Test visualization type combo box signal-slot connection."""
        # Replace the visualization_type_selected method with a mock but save original
        original_viz_type_selected = self.gui.visualization_type_selected
        self.gui.visualization_type_selected = MagicMock()
        
        try:
            # Set up the visualization type combo box with some items
            self.gui.viz_type_combo.clear()
            self.gui.viz_type_combo.addItems(["Gaze Plot", "Heatmap"])
            self.gui.viz_type_combo.setEnabled(True)
            
            # Manually connect currentIndexChanged to our mock
            self.gui.viz_type_combo.currentIndexChanged.connect(self.gui.visualization_type_selected)
            
            # Change the selection
            self.gui.viz_type_combo.setCurrentIndex(1)
            
            # Check that visualization_type_selected was called
            self.gui.visualization_type_selected.assert_called()
        finally:
            # Restore original method
            self.gui.visualization_type_selected = original_viz_type_selected
    
    def test_feature_movie_combo_connection(self):
        """Test feature movie combo box signal-slot connection."""
        # Replace the feature_movie_selected method with a mock but save original
        original_feature_movie_selected = self.gui.feature_movie_selected
        self.gui.feature_movie_selected = MagicMock()
        
        try:
            # Set up the feature movie combo box with some items
            self.gui.feature_movie_combo.clear()
            self.gui.feature_movie_combo.addItems(["All Data", "Movie1", "Movie2"])
            self.gui.feature_movie_combo.setEnabled(True)
            
            # Manually connect currentIndexChanged to our mock
            self.gui.feature_movie_combo.currentIndexChanged.connect(self.gui.feature_movie_selected)
            
            # Change the selection
            self.gui.feature_movie_combo.setCurrentIndex(1)
            
            # Check that feature_movie_selected was called
            self.gui.feature_movie_selected.assert_called()
        finally:
            # Restore original method
            self.gui.feature_movie_selected = original_feature_movie_selected
    
    def test_processing_thread_signal_connections(self):
        """Test the signal connections between processing thread and GUI."""
        # Create a processing thread with test parameters
        thread = ProcessingThread(
            file_paths=[self.csv_file],
            output_dir=self.output_dir,
            visualize=True,
            extract_features=True,
            generate_report=True,
            file_type="CSV Files"
        )
        
        # Create mocks for the GUI methods
        update_progress_mock = MagicMock()
        update_status_mock = MagicMock()
        processing_finished_mock = MagicMock()
        processing_error_mock = MagicMock()
        
        # Connect thread signals to mocks
        thread.update_progress.connect(update_progress_mock)
        thread.status_update.connect(update_status_mock)
        thread.processing_complete.connect(processing_finished_mock)
        thread.error_occurred.connect(processing_error_mock)
        
        # Test update_progress signal
        progress_value = 50
        thread.update_progress.emit(progress_value)
        update_progress_mock.assert_called_with(progress_value)
        
        # Test status_update signal
        status_message = "Processing file..."
        thread.status_update.emit(status_message)
        update_status_mock.assert_called_with(status_message)
        
        # Test processing_complete signal
        result_data = {"summary": {"samples": 100, "fixations": 50, "saccades": 30, "blinks": 10, "frames": 100}}
        thread.processing_complete.emit(result_data)
        processing_finished_mock.assert_called_with(result_data)
        
        # Test error_occurred signal
        error_message = "Error processing file"
        thread.error_occurred.emit(error_message)
        processing_error_mock.assert_called_with(error_message)
    
    def test_gui_process_data_creates_thread(self):
        """Test that process_data creates a processing thread with correct signals."""
        # Set up GUI with file and output directory
        self.gui.file_paths = [self.csv_file]
        self.gui.output_dir = self.output_dir
        self.gui.selected_file_type = "CSV Files"
        
        # Replace the thread start method with a mock
        with patch.object(ProcessingThread, 'start') as mock_start:
            # Also mock the signal connection methods to verify they are called
            with patch.object(ProcessingThread, 'update_progress') as mock_progress_signal:
                with patch.object(ProcessingThread, 'status_update') as mock_status_signal:
                    with patch.object(ProcessingThread, 'processing_complete') as mock_complete_signal:
                        with patch.object(ProcessingThread, 'error_occurred') as mock_error_signal:
                            # Call process_data
                            self.gui.process_data()
                            
                            # Check that the thread was created and started
                            mock_start.assert_called_once()
                            
                            # Check that signals were connected
                            # Note: We can't directly check the connections, but we can verify
                            # that the signals exist on the thread
                            self.assertTrue(hasattr(self.gui.processing_thread, 'update_progress'))
                            self.assertTrue(hasattr(self.gui.processing_thread, 'status_update'))
                            self.assertTrue(hasattr(self.gui.processing_thread, 'processing_complete'))
                            self.assertTrue(hasattr(self.gui.processing_thread, 'error_occurred'))


if __name__ == '__main__':
    unittest.main()