"""
GUI Data Loading Tests Module for Eye Movement Analysis Application
Author: Claude Code Assistant
Date: May 2025

This module provides tests for data loading and connection mechanisms in the GUI,
focusing on proper data loading, file handling, and the interaction between
the GUI and the processing threads.
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
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

# Mock required modules
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.ndimage'] = MagicMock()
sys.modules['scipy.ndimage.gaussian_filter1d'] = MagicMock()

# Create a mock for eyelink_visualizer
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

# Import GUI modules
from GUI.gui import EyeMovementAnalysisGUI
from GUI.data.processing_thread import ProcessingThread


class TestGUIDataLoading(unittest.TestCase):
    """Test the data loading and connection mechanisms in the GUI."""
    
    def setUp(self):
        """Set up the test environment with mock data."""
        self.gui = EyeMovementAnalysisGUI()
        
        # Create temporary directory and test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample CSV file with eye tracking data
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
        """Create a sample CSV file with eye tracking data."""
        # Create realistic sample data with all required columns
        data = {
            'timestamp': np.arange(0, 10000, 100),
            'x_left': np.random.uniform(0, 1280, 100),
            'y_left': np.random.uniform(0, 1024, 100),
            'x_right': np.random.uniform(0, 1280, 100),
            'y_right': np.random.uniform(0, 1024, 100),
            'pupil_left': np.random.uniform(800, 1200, 100),
            'pupil_right': np.random.uniform(800, 1200, 100),
            'is_fixation_left': np.random.choice([True, False], 100),
            'is_fixation_right': np.random.choice([True, False], 100),
            'is_saccade_left': np.random.choice([True, False], 100),
            'is_saccade_right': np.random.choice([True, False], 100),
            'is_blink_left': np.random.choice([True, False], 100),
            'is_blink_right': np.random.choice([True, False], 100),
            'frame_number': np.arange(100)
        }
        pd.DataFrame(data).to_csv(self.csv_file, index=False)
    
    def test_file_selection_updates_gui(self):
        """Test that selecting files updates the GUI state."""
        # Directly mock QFileDialog.getOpenFileNames at the module level
        original_getOpenFileNames = QFileDialog.getOpenFileNames
        
        try:
            # Replace with our mock that returns predefined values
            QFileDialog.getOpenFileNames = MagicMock(return_value=([self.csv_file], "CSV Files (*.csv)"))
            
            # Call the select_files method
            self.gui.select_files()
            
            # Verify the GUI state was updated
            self.assertEqual(self.gui.file_paths, [self.csv_file])
            self.assertEqual(self.gui.selected_file_type, "CSV Files")
            self.assertEqual(self.gui.file_label.text(), f"Selected: {os.path.basename(self.csv_file)}")
            self.assertFalse(self.gui.process_btn.isEnabled())  # Process button should be disabled until output dir is set
        finally:
            # Restore the original method
            QFileDialog.getOpenFileNames = original_getOpenFileNames
    
    def test_output_dir_selection_updates_gui(self):
        """Test that selecting an output directory updates the GUI state."""
        # Directly mock QFileDialog.getExistingDirectory at the module level
        original_getExistingDirectory = QFileDialog.getExistingDirectory
        
        try:
            # Replace with our mock that returns predefined values
            QFileDialog.getExistingDirectory = MagicMock(return_value=self.output_dir)
            
            # Call the select_output_dir method
            self.gui.select_output_dir()
            
            # Verify the GUI state was updated
            self.assertEqual(self.gui.output_dir, self.output_dir)
            self.assertEqual(self.gui.output_label.text(), f"Output: {self.output_dir}")
            self.assertFalse(self.gui.process_btn.isEnabled())  # Process button should still be disabled until files are selected
        finally:
            # Restore the original method
            QFileDialog.getExistingDirectory = original_getExistingDirectory
    
    def test_process_button_enabled_when_ready(self):
        """Test that the process button is enabled when both files and output directory are selected."""
        # Directly mock both QFileDialog methods at the module level
        original_getOpenFileNames = QFileDialog.getOpenFileNames
        original_getExistingDirectory = QFileDialog.getExistingDirectory
        
        try:
            # Replace with our mocks that return predefined values
            QFileDialog.getOpenFileNames = MagicMock(return_value=([self.csv_file], "CSV Files (*.csv)"))
            QFileDialog.getExistingDirectory = MagicMock(return_value=self.output_dir)
            
            # First select files, then output directory
            self.gui.select_files()
            self.gui.select_output_dir()
            
            # Verify the process button is now enabled
            self.assertTrue(self.gui.process_btn.isEnabled())
        finally:
            # Restore the original methods
            QFileDialog.getOpenFileNames = original_getOpenFileNames
            QFileDialog.getExistingDirectory = original_getExistingDirectory
    
    @patch('GUI.data.processing_thread.ProcessingThread.start')
    def test_process_data_starts_thread(self, mock_thread_start):
        """Test that process_data method starts the processing thread."""
        # Setup GUI with file and output dir
        self.gui.file_paths = [self.csv_file]
        self.gui.output_dir = self.output_dir
        self.gui.selected_file_type = "CSV Files"
        
        # Call process_data method
        self.gui.process_data()
        
        # Verify the thread was started
        mock_thread_start.assert_called_once()
        
        # Verify GUI state changes
        self.assertFalse(self.gui.process_btn.isEnabled())  # Button should be disabled during processing
        self.assertEqual(self.gui.progress_bar.value(), 0)  # Progress bar should be reset
    
    def test_update_progress_changes_progress_bar(self):
        """Test that update_progress method updates the progress bar."""
        # Call update_progress with a test value
        test_value = 42
        self.gui.update_progress(test_value)
        
        # Verify progress bar was updated
        self.assertEqual(self.gui.progress_bar.value(), test_value)
    
    def test_update_status_updates_status_label_and_log(self):
        """Test that update_status method updates the status label and log."""
        # Call update_status with a test message
        test_message = "Test status message"
        self.gui.update_status(test_message)
        
        # Verify status label was updated
        self.assertEqual(self.gui.status_label.text(), test_message)
        
        # Verify status log contains the message
        self.assertTrue(test_message in self.gui.status_log.toPlainText())
    
    @patch('GUI.data.processing_thread.load_csv_file')
    def test_processing_thread_signal_connections(self, mock_load_csv):
        """Test that processing thread signals are properly connected to GUI slots."""
        # Mock the CSV loading function to return simple feature data
        mock_features = pd.DataFrame({
            'participant_id': ['test_participant'],
            'pupil_left_mean': [1028.7],
            'pupil_right_mean': [1032.4],
        })
        
        mock_load_csv.return_value = {
            'features': mock_features,
            'parser': MagicMock(),
            'summary': {'samples': 100, 'fixations': 50, 'saccades': 30, 'blinks': 10, 'frames': 100}
        }
        
        # Create processing thread
        thread = ProcessingThread(
            file_paths=[self.csv_file],
            output_dir=self.output_dir,
            visualize=False,
            extract_features=True,
            generate_report=False,
            file_type="CSV Files"
        )
        
        # Mock GUI slots
        self.gui.update_progress = MagicMock()
        self.gui.update_status = MagicMock()
        self.gui.processing_finished = MagicMock()
        self.gui.processing_error = MagicMock()
        
        # Connect signals and run thread
        thread.update_progress.connect(self.gui.update_progress)
        thread.status_update.connect(self.gui.update_status)
        thread.processing_complete.connect(self.gui.processing_finished)
        thread.error_occurred.connect(self.gui.processing_error)
        
        thread.run()  # Run synchronously for testing
        
        # Verify that slots were called
        self.gui.update_progress.assert_called()
        self.gui.update_status.assert_called()
        self.gui.processing_finished.assert_called_once()
        self.gui.processing_error.assert_not_called()
    
    def test_processing_error_shows_message_box(self):
        """Test that processing_error method shows an error message box."""
        # Save original method
        original_critical = QMessageBox.critical
        
        try:
            # Replace with mock
            QMessageBox.critical = MagicMock()
            
            # Call processing_error with a test error message
            test_error = "Test error message"
            self.gui.processing_error(test_error)
            
            # Verify that QMessageBox.critical was called with the error message
            QMessageBox.critical.assert_called_once()
            args = QMessageBox.critical.call_args[0]
            self.assertEqual(args[0], self.gui)  # First arg should be the parent widget
            self.assertEqual(args[1], "Processing Error")  # Second arg is the title
            self.assertEqual(args[2], test_error)  # Third arg is the message
        finally:
            # Restore original method
            QMessageBox.critical = original_critical


if __name__ == '__main__':
    unittest.main()