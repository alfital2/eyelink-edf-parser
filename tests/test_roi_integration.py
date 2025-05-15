import unittest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roi_integration import load_sample_data, create_integrated_visualization
from roi_manager import ROIManager

TEST_ROI_FILE = "test_data/test_roi.json"


class TestROIIntegration(unittest.TestCase):
    """
    Test suite for the ROI integration functionality.
    Tests validate the non-UI helper functions in roi_integration.py.
    """
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Create sample eye tracking data
        cls.eye_data = pd.DataFrame({
            'frame_number': [1, 1, 2, 2, 5, 5],
            'timestamp': [1000, 1016, 1033, 1050, 1066, 1083],
            'x_left': [256, 256, 320, 320, 512, 512],
            'y_left': [205, 205, 256, 256, 410, 410],
            'x_right': [831, 831, 896, 896, 640, 640],
            'y_right': [614, 614, 614, 614, 512, 512]
        })
        
        # Define screen dimensions for normalization
        cls.screen_width = 1280
        cls.screen_height = 1024
        
        # Normalize the coordinates
        for eye in ['left', 'right']:
            cls.eye_data[f'x_{eye}_norm'] = cls.eye_data[f'x_{eye}'] / cls.screen_width
            cls.eye_data[f'y_{eye}_norm'] = cls.eye_data[f'y_{eye}'] / cls.screen_height
        
        # Save the eye data to a temporary CSV file
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.eye_data_path = os.path.join(cls.temp_dir.name, "test_eye_data.csv")
        cls.eye_data.to_csv(cls.eye_data_path, index=False)
        
        # Get the full path to the test ROI file
        cls.roi_file_path = os.path.join(os.path.dirname(__file__), TEST_ROI_FILE)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory and files
        cls.temp_dir.cleanup()

    def test_load_sample_data(self):
        """Test loading sample eye tracking and ROI data."""
        # Load the sample data
        eye_data, roi_manager = load_sample_data(self.eye_data_path, self.roi_file_path)
        
        # Check that eye data was loaded correctly
        self.assertEqual(len(eye_data), 6, "Eye data should have 6 rows")
        self.assertTrue('frame_number' in eye_data.columns, "Eye data should have frame_number column")
        self.assertTrue('x_left_norm' in eye_data.columns, "Eye data should have normalized coordinates")
        
        # Check that ROI data was loaded correctly - frames should be extended based on eye_data
        self.assertEqual(len(roi_manager.frame_numbers), 5, "ROI manager should have 5 frames")
        # Original frames were 1, 2, 5 but they've been extended to match eye_data's max frame (5)
        self.assertTrue(all(frame in roi_manager.frame_numbers for frame in range(5)), 
                       "ROI manager should have frames 0 through 4 after extension")
        
        # Test with non-existent files
        bad_eye_path = "non_existent_eye_data.csv"
        bad_roi_path = "non_existent_roi_data.json"
        
        # Should return empty dataframe and ROI manager without data
        bad_eye_data, bad_roi_manager = load_sample_data(bad_eye_path, bad_roi_path)
        
        self.assertTrue(bad_eye_data.empty, "Should return empty dataframe for non-existent eye data file")
        self.assertEqual(len(bad_roi_manager.frame_numbers), 0, "Should have 0 frames for non-existent ROI file")

    def test_create_integrated_visualization(self):
        """Test creating an integrated visualization."""
        # Load the ROI data
        roi_manager = ROIManager()
        roi_manager.load_roi_file(self.roi_file_path)
        
        # Create a visualization - should not raise any exceptions
        try:
            # Use a temporary file to save the visualization
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Create visualization for frame 1
            fig = create_integrated_visualization(
                self.eye_data, 
                roi_manager, 
                frame_number=1, 
                save_path=temp_path
            )
            
            # Check that the figure was created
            self.assertIsNotNone(fig, "Figure should not be None")
            
            # Check that the file was created
            self.assertTrue(os.path.exists(temp_path), "Visualization file should exist")
            
            # Clean up
            plt.close(fig)
            os.unlink(temp_path)
        except Exception as e:
            self.fail(f"create_integrated_visualization raised exception: {str(e)}")

    def test_eye_data_roi_matching(self):
        """Test matching eye positions with ROIs."""
        # Load the ROI data
        roi_manager = ROIManager()
        roi_manager.load_roi_file(self.roi_file_path)
        
        # Test for each frame and eye position
        test_cases = [
            # Format: frame_number, x_norm, y_norm, expected_roi_label
            (1, 0.25, 0.25, "Face"),  # Inside Face ROI in frame 1
            (1, 0.65, 0.65, "Hand"),  # Inside Hand ROI in frame 1
            (1, 0.5, 0.5, None),      # Not in any ROI in frame 1
            (2, 0.3, 0.3, "Face"),    # Inside Face ROI in frame 2
            (5, 0.45, 0.45, "Torso")  # Inside Torso ROI in frame 5
        ]
        
        for frame, x, y, expected_label in test_cases:
            roi = roi_manager.find_roi_at_gaze(frame, x, y)
            
            if expected_label is None:
                self.assertIsNone(roi, f"Should not find ROI at ({x}, {y}) in frame {frame}")
            else:
                self.assertIsNotNone(roi, f"Should find ROI at ({x}, {y}) in frame {frame}")
                self.assertEqual(roi["label"], expected_label, f"ROI label mismatch at ({x}, {y}) in frame {frame}")
        
        # Test eye positions from actual data
        for _, row in self.eye_data.iterrows():
            frame = row['frame_number']
            x_left_norm = row['x_left_norm']
            y_left_norm = row['y_left_norm']
            
            # For this test data, we know certain positions should be in ROIs
            # Frame 1, left eye should be in Face ROI
            if frame == 1:
                # Left eye at (0.2, 0.2) should be inside/on edge of Face ROI
                roi = roi_manager.find_roi_at_gaze(frame, x_left_norm, y_left_norm)
                if x_left_norm == 0.2 and y_left_norm == 0.2:
                    self.assertIsNotNone(roi, f"Left eye should be in/on edge of ROI at frame {frame}")
                    self.assertEqual(roi["label"], "Face", "ROI should be Face")


if __name__ == '__main__':
    unittest.main()