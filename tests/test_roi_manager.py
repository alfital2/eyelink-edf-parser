import unittest
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path to import ROIManager
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roi_manager import ROIManager

TEST_ROI_FILE = "test_data/test_roi.json"


class TestROIManager(unittest.TestCase):
    """
    Test suite for the ROIManager class.
    Tests validate ROI loading, point detection, and visualization functionality.
    """
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Initialize the ROI manager and load the test data
        cls.roi_manager = ROIManager()
        
        # Get the full path to the test file
        cls.test_file_path = os.path.join(os.path.dirname(__file__), TEST_ROI_FILE)
        
        # Load the ROI data
        cls.load_success = cls.roi_manager.load_roi_file(cls.test_file_path)
        
        # Create test visualization
        if cls.load_success:
            cls.create_test_visualization()

    @classmethod
    def create_test_visualization(cls):
        """Generate a test visualization for visual verification."""
        tests_dir = os.path.dirname(__file__)
        output_dir = os.path.join(tests_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "roi_test_visualization.png")
        
        cls.roi_manager.create_test_visualization(
            frame_number=1,  # Use the first frame
            save_path=output_path
        )

    def test_roi_file_loading(self):
        """Test that the ROI file is loaded correctly."""
        self.assertTrue(self.load_success, "Failed to load the test ROI file")
        
        # Verify that we have the expected number of frames
        self.assertEqual(len(self.roi_manager.frame_numbers), 3, 
                         "Expected 3 frames in the test data")
        
        # Verify the frame numbers
        expected_frames = [1, 2, 5]
        self.assertEqual(self.roi_manager.frame_numbers, expected_frames,
                         f"Expected frames {expected_frames} but got {self.roi_manager.frame_numbers}")

    def test_get_unique_labels(self):
        """Test the get_unique_labels method."""
        expected_labels = ["Face", "Hand", "Torso"]
        actual_labels = self.roi_manager.get_unique_labels()
        
        self.assertEqual(sorted(actual_labels), sorted(expected_labels),
                         f"Expected labels {expected_labels} but got {actual_labels}")

    def test_get_frame_rois(self):
        """Test the get_frame_rois method."""
        # Check frame 1 (should have 2 ROIs)
        frame1_rois = self.roi_manager.get_frame_rois(1)
        self.assertEqual(len(frame1_rois), 2, "Expected 2 ROIs for frame 1")
        
        # Check frame 2 (should have 1 ROI)
        frame2_rois = self.roi_manager.get_frame_rois(2)
        self.assertEqual(len(frame2_rois), 1, "Expected 1 ROI for frame 2")
        
        # Check frame 5 (should have 1 ROI)
        frame5_rois = self.roi_manager.get_frame_rois(5)
        self.assertEqual(len(frame5_rois), 1, "Expected 1 ROI for frame 5")
        
        # Check non-existent frame (should return empty list)
        non_existent_rois = self.roi_manager.get_frame_rois(100)
        self.assertEqual(len(non_existent_rois), 0, "Expected 0 ROIs for non-existent frame")

    def test_point_in_polygon(self):
        """Test the point_in_polygon method."""
        # Simple square polygon from (0.2, 0.2) to (0.3, 0.3)
        square_polygon = [
            {"x": 0.2, "y": 0.2},
            {"x": 0.3, "y": 0.2},
            {"x": 0.3, "y": 0.3},
            {"x": 0.2, "y": 0.3}
        ]
        
        # Test point inside the polygon
        self.assertTrue(
            self.roi_manager.point_in_polygon(0.25, 0.25, square_polygon),
            "Point (0.25, 0.25) should be inside the square polygon"
        )
        
        # Test point outside the polygon
        self.assertFalse(
            self.roi_manager.point_in_polygon(0.15, 0.15, square_polygon),
            "Point (0.15, 0.15) should be outside the square polygon"
        )
        
        # Test point on the edge of the polygon
        self.assertTrue(
            self.roi_manager.point_in_polygon(0.2, 0.25, square_polygon),
            "Point (0.2, 0.25) should be on the edge of the square polygon"
        )
        
        # Test point on a vertex of the polygon
        self.assertTrue(
            self.roi_manager.point_in_polygon(0.2, 0.2, square_polygon),
            "Point (0.2, 0.2) should be on a vertex of the square polygon"
        )
        
        # Test with an empty polygon (should return False)
        empty_polygon = []
        self.assertFalse(
            self.roi_manager.point_in_polygon(0.5, 0.5, empty_polygon),
            "Point should not be in an empty polygon"
        )

    def test_is_gaze_in_roi(self):
        """Test the is_gaze_in_roi method."""
        # Get an ROI from the test data
        frame1_rois = self.roi_manager.get_frame_rois(1)
        face_roi = frame1_rois[0]  # Face ROI
        
        # Test point inside the ROI
        self.assertTrue(
            self.roi_manager.is_gaze_in_roi(0.25, 0.25, face_roi),
            "Point (0.25, 0.25) should be inside the Face ROI"
        )
        
        # Test point outside the ROI
        self.assertFalse(
            self.roi_manager.is_gaze_in_roi(0.15, 0.15, face_roi),
            "Point (0.15, 0.15) should be outside the Face ROI"
        )
        
        # Test with an ROI missing coordinates (should return False)
        bad_roi = {"label": "Bad ROI"}
        self.assertFalse(
            self.roi_manager.is_gaze_in_roi(0.5, 0.5, bad_roi),
            "is_gaze_in_roi should return False for an ROI missing coordinates"
        )

    def test_find_roi_at_point(self):
        """Test the find_roi_at_point method."""
        # Test point in Face ROI in frame 1
        face_roi = self.roi_manager.find_roi_at_point(1, 0.25, 0.25)
        self.assertIsNotNone(face_roi, "Should find an ROI at point (0.25, 0.25) in frame 1")
        self.assertEqual(face_roi["label"], "Face", "ROI at (0.25, 0.25) in frame 1 should be 'Face'")
        
        # Test point in Hand ROI in frame 1
        hand_roi = self.roi_manager.find_roi_at_point(1, 0.65, 0.65)
        self.assertIsNotNone(hand_roi, "Should find an ROI at point (0.65, 0.65) in frame 1")
        self.assertEqual(hand_roi["label"], "Hand", "ROI at (0.65, 0.65) in frame 1 should be 'Hand'")
        
        # Test point not in any ROI in frame 1
        no_roi = self.roi_manager.find_roi_at_point(1, 0.5, 0.5)
        self.assertIsNone(no_roi, "Should not find an ROI at point (0.5, 0.5) in frame 1")
        
        # Test point in Torso ROI in frame 5
        torso_roi = self.roi_manager.find_roi_at_point(5, 0.45, 0.45)
        self.assertIsNotNone(torso_roi, "Should find an ROI at point (0.45, 0.45) in frame 5")
        self.assertEqual(torso_roi["label"], "Torso", "ROI at (0.45, 0.45) in frame 5 should be 'Torso'")
        
        # Test point in non-existent frame
        no_frame_roi = self.roi_manager.find_roi_at_point(100, 0.5, 0.5)
        self.assertIsNone(no_frame_roi, "Should not find an ROI in non-existent frame")

    def test_get_nearest_frame(self):
        """Test the get_nearest_frame method."""
        # Test exact frame match
        self.assertEqual(
            self.roi_manager.get_nearest_frame(1), 1,
            "Nearest frame to 1 should be 1"
        )
        
        # Test frame before the first available
        self.assertEqual(
            self.roi_manager.get_nearest_frame(0), 1,
            "Nearest frame to 0 should be 1"
        )
        
        # Test frame between 1 and 2
        self.assertIn(
            self.roi_manager.get_nearest_frame(1.5), [1, 2],
            "Nearest frame to 1.5 should be either 1 or 2"
        )
        
        # Test frame between 2 and 5
        self.assertEqual(
            self.roi_manager.get_nearest_frame(3), 2,
            "Nearest frame to 3 should be 2"
        )
        
        # Test frame after the last available
        self.assertEqual(
            self.roi_manager.get_nearest_frame(10), 5,
            "Nearest frame to 10 should be 5"
        )

    def test_find_roi_at_gaze(self):
        """Test the find_roi_at_gaze method."""
        # Test with exact frame
        face_roi = self.roi_manager.find_roi_at_gaze(1, 0.25, 0.25)
        self.assertIsNotNone(face_roi, "Should find an ROI at point (0.25, 0.25) in frame 1")
        self.assertEqual(face_roi["label"], "Face", "ROI at (0.25, 0.25) in frame 1 should be 'Face'")
        
        # Test with nearest frame enabled (default)
        torso_roi = self.roi_manager.find_roi_at_gaze(4, 0.45, 0.45)
        self.assertIsNotNone(torso_roi, "Should find an ROI at point (0.45, 0.45) near frame 4")
        self.assertEqual(torso_roi["label"], "Torso", "ROI near frame 4 should be 'Torso'")
        
        # Test with nearest frame disabled
        no_roi = self.roi_manager.find_roi_at_gaze(4, 0.45, 0.45, use_nearest_frame=False)
        self.assertIsNone(no_roi, "Should not find an ROI when nearest frame is disabled")

    def test_invalid_roi_file(self):
        """Test loading an invalid ROI file."""
        # Create a temporary invalid JSON file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_file.write(b'{"invalid": "json"')  # Intentionally malformed JSON
            temp_path = temp_file.name

        try:
            # Test loading the invalid file
            test_manager = ROIManager()
            result = test_manager.load_roi_file(temp_path)
            self.assertFalse(result, "Loading invalid JSON should return False")
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    def test_empty_roi_file(self):
        """Test loading an empty ROI file."""
        # Create a temporary empty JSON file (valid but empty)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_file.write(b'{}')  # Empty JSON object
            temp_path = temp_file.name

        try:
            # Test loading the empty file
            test_manager = ROIManager()
            result = test_manager.load_roi_file(temp_path)
            self.assertFalse(result, "Loading empty JSON should return False (no frames)")
            self.assertEqual(len(test_manager.frame_numbers), 0, "Empty JSON should have 0 frames")
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()