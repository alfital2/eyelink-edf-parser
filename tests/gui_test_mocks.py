"""
GUI Test Mocks Module for Eye Movement Analysis Application
Author: Claude Code Assistant
Date: May 2025

This module provides mock objects for GUI testing, including mock models,
controllers, and data objects that match the current architecture.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QWidget

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class MockEyeTrackingData(QObject):
    """
    Mock implementation of the eye tracking data model.
    
    This mock provides the same signal interface as the real EyeTrackingData class,
    but with simplified implementations for testing.
    """
    # Define signals
    data_loaded = pyqtSignal(object)
    data_processed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.file_paths = []
        self.file_type = None
        self.participant_id = None
        
        # Summary statistics
        self.summary = {
            'samples': 0,
            'fixations': 0,
            'saccades': 0,
            'blinks': 0,
            'frames': 0
        }
        
        # Output paths
        self.output_dir = None
    
    def load_from_files(self, file_paths, file_type):
        """Mock implementation of load_from_files."""
        self.file_paths = file_paths
        self.file_type = file_type
        
        # Set participant ID based on the first file
        if file_paths:
            base_name = os.path.basename(file_paths[0])
            self.participant_id = os.path.splitext(base_name)[0]
            
            # For CSV files, remove '_unified_eye_metrics' from the participant ID if present
            if self.file_type == "CSV Files" and self.participant_id.endswith('_unified_eye_metrics'):
                self.participant_id = self.participant_id.replace('_unified_eye_metrics', '')
        
        # Signal that files are ready
        self.data_loaded.emit(self)
        return True
    
    def set_processed_data(self, data, summary=None):
        """Mock implementation of set_processed_data."""
        self.processed_data = data
        
        if summary:
            self.summary.update(summary)
        
        # Emit signal with processed data
        self.data_processed.emit({
            'data': self.processed_data,
            'summary': self.summary
        })


class MockFeatureModel(QObject):
    """
    Mock implementation of the feature model.
    
    This mock provides the same signal interface as the real FeatureModel class,
    but with simplified implementations for testing.
    """
    # Define signals
    features_calculated = pyqtSignal(object)
    features_updated = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        
        # Data storage
        self.features = None
        self.movie_features = {}
    
    def calculate_features(self, eye_tracking_data):
        """Mock implementation of calculate_features."""
        # Create mock features data
        if eye_tracking_data and hasattr(eye_tracking_data, 'processed_data') and eye_tracking_data.processed_data is not None:
            # Create realistic mock features based on eye_tracking_data
            self.features = pd.DataFrame({
                'participant_id': [eye_tracking_data.participant_id or 'test_participant'],
                'pupil_left_mean': [1028.7],
                'pupil_right_mean': [1032.4],
                'gaze_left_x_std': [75.2],
                'gaze_right_x_std': [78.6],
                'fixation_left_count': [42],
                'fixation_right_count': [40]
            })
            
            # Create mock movie-specific features
            if hasattr(eye_tracking_data.processed_data, 'movie'):
                for movie in eye_tracking_data.processed_data['movie'].unique():
                    # Create slightly different values for each movie
                    movie_features = self.features.copy()
                    movie_features['fixation_left_count'] = [np.random.randint(30, 60)]
                    self.movie_features[movie] = movie_features
            
            # Signal that features were calculated
            self.features_calculated.emit(self.features)
            return self.features
        else:
            # If no data, return empty DataFrame
            self.features = pd.DataFrame()
            self.features_calculated.emit(self.features)
            return self.features
    
    def update_features(self, additional_features):
        """Mock implementation of update_features."""
        if self.features is not None and not self.features.empty:
            # Merge additional features with existing features
            for column, values in additional_features.items():
                self.features[column] = values[0] if isinstance(values, list) else values
            
            # Signal that features were updated
            self.features_updated.emit(self.features)
            return True
        return False


class MockROIModel(QObject):
    """
    Mock implementation of the ROI model.
    
    This mock provides the same signal interface as the real ROIModel class,
    but with simplified implementations for testing.
    """
    # Define signals
    roi_loaded = pyqtSignal(object)
    roi_analyzed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        # Data storage
        self.roi_data = None
        self.roi_file_path = None
        self.analysis_results = None
    
    def load_roi_data(self, file_path):
        """Mock implementation of load_roi_data."""
        self.roi_file_path = file_path
        
        # Create mock ROI data
        self.roi_data = {
            "0": {
                "objects": [
                    {
                        "object_id": "face1",
                        "label": "Face",
                        "vertices": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
                        "social": True
                    },
                    {
                        "object_id": "ball1",
                        "label": "Ball",
                        "vertices": [[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8]],
                        "social": False
                    }
                ]
            }
        }
        
        # Signal that ROI data was loaded
        self.roi_loaded.emit(self.roi_data)
        return True
    
    def analyze_fixations(self, fixations):
        """Mock implementation of analyze_fixations."""
        # Create mock analysis results
        self.analysis_results = {
            "social_attention_percentage": 65.2,
            "roi_dwell_times": {
                "Face": 3.5,
                "Ball": 1.2
            },
            "roi_fixation_counts": {
                "Face": 12,
                "Ball": 5
            },
            "roi_transition_matrix": {
                "Face": {"Face": 0.7, "Ball": 0.3},
                "Ball": {"Face": 0.6, "Ball": 0.4}
            }
        }
        
        # Signal that analysis is complete
        self.roi_analyzed.emit(self.analysis_results)
        return self.analysis_results


class MockProcessingThread(QObject):
    """
    Mock implementation of the processing thread.
    
    This mock provides the same signal interface as the real ProcessingThread class,
    but with simplified implementations for testing.
    """
    # Define signals
    update_progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    processing_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_paths, output_dir, visualize=True, extract_features=True, 
                 generate_report=False, file_type="ASC Files"):
        super().__init__()
        
        # Store parameters
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.visualize = visualize
        self.extract_features = extract_features
        self.generate_report = generate_report
        self.file_type = file_type
        self.should_fail = False  # For testing error handling
    
    def start(self):
        """Mock implementation that simulates a real thread."""
        # Use a timer to simulate asynchronous processing
        QTimer.singleShot(100, self.run)
    
    def run(self):
        """Mock implementation that emits signals in a realistic sequence."""
        try:
            if self.should_fail:
                raise RuntimeError("Simulated processing error")
            
            # Emit progress updates
            self.update_progress.emit(5)
            self.status_update.emit("Starting processing...")
            
            self.update_progress.emit(20)
            self.status_update.emit(f"Processing {len(self.file_paths)} file(s)...")
            
            self.update_progress.emit(50)
            self.status_update.emit("Extracting features...")
            
            self.update_progress.emit(75)
            self.status_update.emit("Generating visualizations...")
            
            self.update_progress.emit(90)
            self.status_update.emit("Finalizing results...")
            
            # Create mock results
            results = {
                'summary': {
                    'samples': 10000,
                    'fixations': 500,
                    'saccades': 300,
                    'blinks': 50,
                    'frames': 1000
                },
                'output_dir': self.output_dir
            }
            
            # Add features if requested
            if self.extract_features:
                # Create mock features data
                features = pd.DataFrame({
                    'participant_id': ['test_participant'],
                    'pupil_left_mean': [1028.7],
                    'pupil_right_mean': [1032.4],
                    'gaze_left_x_std': [75.2],
                    'gaze_right_x_std': [78.6],
                    'fixation_left_count': [42],
                    'fixation_right_count': [40],
                    'saccade_left_count': [43],
                    'saccade_right_count': [41],
                    'blink_left_count': [12],
                    'blink_right_count': [12]
                })
                
                results['features'] = features
                
                # Add movie-specific features
                movie_features = {
                    "All Data": features,
                    "Movie1": features.copy(),
                    "Movie2": features.copy()
                }
                
                # Make Movie2 features slightly different
                movie_features["Movie2"]["fixation_left_count"] = [60]
                
                results['movie_features'] = movie_features
            
            # Add visualizations if requested
            if self.visualize:
                visualizations = {
                    "Movie1": {
                        "basic": [
                            os.path.join(self.output_dir, "plots", "gaze_positions.png"),
                            os.path.join(self.output_dir, "plots", "fixation_density.png")
                        ]
                    },
                    "Movie2": {
                        "basic": [
                            os.path.join(self.output_dir, "plots", "gaze_positions.png"),
                            os.path.join(self.output_dir, "plots", "fixation_density.png")
                        ]
                    }
                }
                
                results['visualizations'] = visualizations
            
            # Add report path if requested
            if self.generate_report:
                results['report_path'] = os.path.join(self.output_dir, "report", "report.html")
            
            # Emit complete signal
            self.update_progress.emit(100)
            self.status_update.emit("Processing complete!")
            self.processing_complete.emit(results)
            
        except Exception as e:
            # Emit error signal
            self.status_update.emit(f"Error: {str(e)}")
            self.error_occurred.emit(f"Error: {str(e)}")


class MockFeatureTableManager:
    """
    Mock implementation of the feature table manager.
    
    This mock provides the same interface as the real FeatureTableManager class,
    but with simplified implementations for testing.
    """
    def __init__(self, parent=None, theme_manager=None, feature_explanations=None):
        # Data storage
        self.feature_tables = {}
        self.theme_manager = theme_manager
        self.feature_explanations = feature_explanations or {}
        self.parent = parent
    
    def create_feature_tables(self, parent_layout):
        """Mock implementation of create_feature_tables."""
        # Create mock tables for each category in FEATURE_CATEGORIES
        feature_categories = [
            ("Basic Information", ["participant_id"], 0, 0),
            ("Pupil Size", [
                {"name": "Mean Pupil Size", "left": "pupil_left_mean", "right": "pupil_right_mean"},
                {"name": "Pupil Size Std", "left": "pupil_left_std", "right": "pupil_right_std"}
            ], 0, 1),
            ("Fixation Metrics", [
                {"name": "Fixation Count", "left": "fixation_left_count", "right": "fixation_right_count"}
            ], 1, 0)
        ]
        
        for category_info in feature_categories:
            category_name = category_info[0]
            features = category_info[1]
            
            # Create mock table
            table = MagicMock()
            table.rowCount.return_value = 0
            
            # Store in feature_tables
            self.feature_tables[category_name] = {
                "table": table,
                "features": features,
                "is_combined": isinstance(features[0], dict) if features else False
            }
        
        return {category: [] for category in self.feature_tables.keys()}
    
    def update_feature_tables(self, features_df):
        """Mock implementation of update_feature_tables."""
        if features_df is None or features_df.empty:
            return
        
        # Update mock row counts
        for category_name, table_info in self.feature_tables.items():
            table = table_info["table"]
            
            # Set a row count based on the features
            if category_name == "Basic Information":
                table.rowCount.return_value = 1
            elif category_name == "Pupil Size":
                table.rowCount.return_value = 2
            elif category_name == "Fixation Metrics":
                table.rowCount.return_value = 1
            else:
                table.rowCount.return_value = 0


class MockAnimatedROIScanpathWidget(QWidget):
    """
    Mock implementation of the animated ROI scanpath widget.
    
    This mock provides the same interface as the real AnimatedROIScanpathWidget class,
    but with simplified implementations for testing.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.data = None
        self.roi_data = None
        self.movie_name = None
        self.screen_width = 1280
        self.screen_height = 1024
        
        # Mock UI elements
        self.play_button = MagicMock()
        self.reset_button = MagicMock()
        self.timeline_slider = MagicMock()
        self.current_roi_label = MagicMock()
    
    def load_data(self, eye_data, roi_data_path=None, movie_name="Unknown", screen_width=1280, screen_height=1024):
        """Mock implementation of load_data."""
        self.data = eye_data
        self.movie_name = movie_name
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        if roi_data_path and os.path.exists(roi_data_path):
            self.roi_data = roi_data_path
        
        # Enable UI elements
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.timeline_slider.setEnabled(True)
        
        return True


class MockPlotGenerator:
    """
    Mock implementation of the plot generator.
    
    This mock provides the same interface as the real PlotGenerator class,
    but with simplified implementations for testing.
    """
    def __init__(self, screen_width=1280, screen_height=1024, 
                visualization_results=None, movie_visualizations=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.visualization_results = visualization_results or {}
        self.movie_visualizations = movie_visualizations or {}
        self.roi_file_path = None
        self.movie_combo = None
        self._get_movie_data = None
        self.plots_dir = None
    
    def generate_social_attention_plots(self):
        """Mock implementation of generate_social_attention_plots."""
        # Check if ROI file is set
        if not self.roi_file_path:
            return {"success": False, "error": "No ROI file selected"}
        
        # Check if movie_combo is set
        if not self.movie_combo or self.movie_combo.count() == 0:
            return {"success": False, "error": "No movies available"}
        
        # Get current movie
        movie = self.movie_combo.currentText()
        
        # Create mock plot paths
        plot_paths = [
            os.path.join(self.plots_dir or "/tmp", f"social_attention_roi_time_{movie}.png"),
            os.path.join(self.plots_dir or "/tmp", f"social_vs_nonsocial_balance_{movie}.png")
        ]
        
        # Return success result
        return {
            "success": True,
            "movie": movie,
            "plots": plot_paths,
            "report_updated": True
        }


# Helper function to create a complete set of mock objects for GUI testing
def create_mock_objects():
    """Create a complete set of mock objects for GUI testing."""
    return {
        "eye_tracking_data": MockEyeTrackingData(),
        "feature_model": MockFeatureModel(),
        "roi_model": MockROIModel(),
        "processing_thread": MockProcessingThread(
            file_paths=["/path/to/test.csv"],
            output_dir="/path/to/output"
        ),
        "feature_table_manager": MockFeatureTableManager(),
        "animated_scanpath": MockAnimatedROIScanpathWidget(),
        "plot_generator": MockPlotGenerator()
    }


# Example usage
if __name__ == "__main__":
    # Create mock objects
    mocks = create_mock_objects()
    
    # Demonstrate how to use the mocks
    # Load data with the mock eye tracking data model
    mocks["eye_tracking_data"].load_from_files(["/path/to/test.csv"], "CSV Files")
    
    # Calculate features with the mock feature model
    mocks["feature_model"].calculate_features(mocks["eye_tracking_data"])
    
    # Load ROI data with the mock ROI model
    mocks["roi_model"].load_roi_data("/path/to/test_roi.json")
    
    # Run the mock processing thread
    mocks["processing_thread"].run()
    
    print("Mock objects created and demonstrated successfully.")