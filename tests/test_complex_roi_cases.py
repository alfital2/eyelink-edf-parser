"""
Complex ROI Cases Tests
Author: Tal Alfi
Date: May 2025

This module tests complex edge cases for ROI handling, including:
- ROIs with complex shapes (concave polygons)
- ROIs with holes
- ROIs that overlap
- ROIs with large numbers of vertices
- ROIs that cross screen boundaries
"""

import unittest
import os
import sys
import tempfile
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import ROIManager
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roi_manager import ROIManager


class TestComplexROICases(unittest.TestCase):
    """
    Test suite for complex ROI cases.
    Tests edge cases and complex scenarios for ROI processing.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.roi_manager = ROIManager()
        self.temp_files = []  # Keep track of temporary files to clean up
    
    def tearDown(self):
        """Clean up after each test."""
        # Delete any temporary files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        self.temp_files = []

    def create_temp_roi_file(self, roi_data):
        """Helper to create a temporary ROI file with the provided data."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_file.write(json.dumps(roi_data).encode('utf-8'))
            temp_path = temp_file.name
            self.temp_files.append(temp_path)
            return temp_path

    def test_concave_polygon(self):
        """Test ROI detection with concave polygons."""
        # Create a concave polygon (C-shape)
        concave_polygon = {
            "1": [
                {
                    "label": "Concave",
                    "object_id": "concave1",
                    "coordinates": [
                        {"x": 0.1, "y": 0.1},  # Top-left
                        {"x": 0.3, "y": 0.1},  # Top-right
                        {"x": 0.3, "y": 0.2},  # Right indent start
                        {"x": 0.2, "y": 0.2},  # Indent top
                        {"x": 0.2, "y": 0.3},  # Indent bottom
                        {"x": 0.3, "y": 0.3},  # Right indent end
                        {"x": 0.3, "y": 0.4},  # Bottom-right
                        {"x": 0.1, "y": 0.4},  # Bottom-left
                    ]
                }
            ]
        }
        
        # Create and load the temp file
        temp_path = self.create_temp_roi_file(concave_polygon)
        self.roi_manager.load_roi_file(temp_path)
        
        # Points inside the concave polygon (on the outer edges)
        self.assertTrue(self.roi_manager.find_roi_at_point(1, 0.15, 0.15) is not None,
                        "Point in top section should be inside the concave polygon")
        self.assertTrue(self.roi_manager.find_roi_at_point(1, 0.15, 0.35) is not None,
                        "Point in bottom section should be inside the concave polygon")
        
        # Point in the "indent" should be outside
        self.assertTrue(self.roi_manager.find_roi_at_point(1, 0.25, 0.25) is None,
                       "Point in the concave indent should be outside the polygon")

    def test_overlapping_rois(self):
        """Test ROI detection with overlapping ROIs."""
        # Create overlapping ROIs
        overlapping_rois = {
            "1": [
                {
                    "label": "Background",
                    "object_id": "bg1",
                    "coordinates": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.5, "y": 0.1},
                        {"x": 0.5, "y": 0.5},
                        {"x": 0.1, "y": 0.5}
                    ]
                },
                {
                    "label": "Foreground",
                    "object_id": "fg1",
                    "coordinates": [
                        {"x": 0.3, "y": 0.3},
                        {"x": 0.7, "y": 0.3},
                        {"x": 0.7, "y": 0.7},
                        {"x": 0.3, "y": 0.7}
                    ]
                }
            ]
        }
        
        # Create and load the temp file
        temp_path = self.create_temp_roi_file(overlapping_rois)
        self.roi_manager.load_roi_file(temp_path)
        
        # Test point in the overlapping region - should return the first matching ROI (Background)
        roi = self.roi_manager.find_roi_at_point(1, 0.35, 0.35)
        self.assertIsNotNone(roi, "Should find an ROI in the overlapping region")
        self.assertEqual(roi["label"], "Background", 
                         "Should return the first ROI in the list for overlapping regions")
        
        # Test point only in Foreground
        roi = self.roi_manager.find_roi_at_point(1, 0.65, 0.65)
        self.assertIsNotNone(roi, "Should find an ROI at point only in Foreground")
        self.assertEqual(roi["label"], "Foreground", "Should identify Foreground ROI correctly")
        
        # Test point only in Background
        roi = self.roi_manager.find_roi_at_point(1, 0.15, 0.15)
        self.assertIsNotNone(roi, "Should find an ROI at point only in Background")
        self.assertEqual(roi["label"], "Background", "Should identify Background ROI correctly")

    def test_many_vertices(self):
        """Test ROI detection with a polygon having many vertices (performance test)."""
        # Create a "circle-like" polygon with many vertices
        num_vertices = 100
        center_x, center_y = 0.5, 0.5
        radius = 0.3
        
        # Generate vertices around a circle
        coordinates = []
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            coordinates.append({"x": x, "y": y})
        
        many_vertex_roi = {
            "1": [
                {
                    "label": "Circle",
                    "object_id": "circle1",
                    "coordinates": coordinates
                }
            ]
        }
        
        # Create and load the temp file
        temp_path = self.create_temp_roi_file(many_vertex_roi)
        self.roi_manager.load_roi_file(temp_path)
        
        # Test point inside the circle
        roi = self.roi_manager.find_roi_at_point(1, 0.5, 0.5)
        self.assertIsNotNone(roi, "Should find an ROI at the center of the circle")
        self.assertEqual(roi["label"], "Circle", "Should identify Circle ROI correctly")
        
        # Test point near the edge but inside
        roi = self.roi_manager.find_roi_at_point(1, 0.5 + 0.29, 0.5)
        self.assertIsNotNone(roi, "Should find an ROI near the edge of the circle")
        
        # Test point outside the circle
        roi = self.roi_manager.find_roi_at_point(1, 0.5 + 0.4, 0.5)
        self.assertIsNone(roi, "Should not find an ROI outside the circle")

    def test_boundary_crossing_roi(self):
        """Test ROI that would cross screen boundaries if unconstrained."""
        # Note: Normalized coordinates should be constrained to [0,1]
        # In this test, we check how the system handles coordinates slightly outside this range
        boundary_crossing_roi = {
            "1": [
                {
                    "label": "OffScreen",
                    "object_id": "offscreen1",
                    "coordinates": [
                        {"x": -0.1, "y": -0.1},  # Outside top-left
                        {"x": 1.1, "y": -0.1},   # Outside top-right
                        {"x": 1.1, "y": 1.1},    # Outside bottom-right
                        {"x": -0.1, "y": 1.1}    # Outside bottom-left
                    ]
                }
            ]
        }
        
        # Create and load the temp file
        temp_path = self.create_temp_roi_file(boundary_crossing_roi)
        self.roi_manager.load_roi_file(temp_path)
        
        # Test point at center of screen
        roi = self.roi_manager.find_roi_at_point(1, 0.5, 0.5)
        self.assertIsNotNone(roi, "Point at center should be inside the boundary-crossing ROI")
        
        # Test points at corners of screen
        roi = self.roi_manager.find_roi_at_point(1, 0.01, 0.01)
        self.assertIsNotNone(roi, "Point near top-left corner should be inside the ROI")
        
        roi = self.roi_manager.find_roi_at_point(1, 0.99, 0.99)
        self.assertIsNotNone(roi, "Point near bottom-right corner should be inside the ROI")

    def test_edge_case_coordinates(self):
        """Test ROI detection with edge case coordinate values."""
        edge_case_rois = {
            "1": [
                {
                    "label": "ZeroSize",
                    "object_id": "zero1",
                    "coordinates": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.1, "y": 0.1},  # Duplicate point
                        {"x": 0.1, "y": 0.1}   # Another duplicate
                    ]
                },
                {
                    "label": "NaN",
                    "object_id": "nan1",
                    "coordinates": [
                        {"x": float('nan'), "y": 0.5},
                        {"x": 0.5, "y": float('nan')},
                        {"x": 0.6, "y": 0.6}
                    ]
                }
            ]
        }
        
        # Create and load the temp file
        temp_path = self.create_temp_roi_file(edge_case_rois)
        self.roi_manager.load_roi_file(temp_path)
        
        # Test point at ZeroSize ROI
        roi = self.roi_manager.find_roi_at_point(1, 0.1, 0.1)
        # Some implementations might match this as a point ROI, others might not
        # We accept either behavior, just verify it doesn't crash
        self.assertTrue(roi is None or roi["label"] == "ZeroSize", 
                        "Zero-size ROI test should either match or not match, but not crash")
        
        # Test with a point near NaN coordinates
        # This shouldn't crash, though proper behavior depends on the implementation
        roi = self.roi_manager.find_roi_at_point(1, 0.6, 0.6)
        # Again, accept either behavior
        self.assertTrue(roi is None or roi["label"] == "NaN",
                        "NaN coordinate test should either match or not match, but not crash")

    def test_roi_data_analytics(self):
        """Test the ROI manager's ability to analyze ROI data across frames."""
        # Create test data with multiple frames and ROIs
        multi_frame_rois = {
            "1": [
                {
                    "label": "Face",
                    "object_id": "face1",
                    "coordinates": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.3, "y": 0.1},
                        {"x": 0.3, "y": 0.3},
                        {"x": 0.1, "y": 0.3}
                    ]
                },
                {
                    "label": "Hand",
                    "object_id": "hand1",
                    "coordinates": [
                        {"x": 0.6, "y": 0.6},
                        {"x": 0.8, "y": 0.6},
                        {"x": 0.8, "y": 0.8},
                        {"x": 0.6, "y": 0.8}
                    ]
                }
            ],
            "2": [
                {
                    "label": "Face",
                    "object_id": "face1",
                    "coordinates": [
                        {"x": 0.15, "y": 0.15},
                        {"x": 0.35, "y": 0.15},
                        {"x": 0.35, "y": 0.35},
                        {"x": 0.15, "y": 0.35}
                    ]
                }
            ],
            "3": [
                {
                    "label": "Object",
                    "object_id": "obj1",
                    "coordinates": [
                        {"x": 0.4, "y": 0.4},
                        {"x": 0.6, "y": 0.4},
                        {"x": 0.6, "y": 0.6},
                        {"x": 0.4, "y": 0.6}
                    ]
                }
            ]
        }
        
        # Create and load the temp file
        temp_path = self.create_temp_roi_file(multi_frame_rois)
        self.roi_manager.load_roi_file(temp_path)
        
        # Test unique labels - should be ["Face", "Hand", "Object"]
        unique_labels = self.roi_manager.get_unique_labels()
        self.assertEqual(sorted(unique_labels), sorted(["Face", "Hand", "Object"]),
                         "Should correctly identify all unique ROI labels")
        
        # Check the number of frames
        self.assertEqual(len(self.roi_manager.frame_numbers), 3,
                         "Should identify 3 distinct frames")
        
        # Check get_nearest_frame behavior
        self.assertEqual(self.roi_manager.get_nearest_frame(2.4), 2,
                         "Nearest frame to 2.4 should be 2")
        self.assertEqual(self.roi_manager.get_nearest_frame(2.6), 3,
                         "Nearest frame to 2.6 should be 3")

    def test_simulated_eye_tracking(self):
        """Test ROI detection with simulated eye tracking data."""
        # Create multi-frame ROI data
        # Face moves slightly across frames
        multi_frame_rois = {
            "1": [
                {
                    "label": "Face",
                    "object_id": "face1",
                    "coordinates": [
                        {"x": 0.4, "y": 0.4},
                        {"x": 0.6, "y": 0.4},
                        {"x": 0.6, "y": 0.6},
                        {"x": 0.4, "y": 0.6}
                    ]
                }
            ],
            "2": [
                {
                    "label": "Face",
                    "object_id": "face1",
                    "coordinates": [
                        {"x": 0.41, "y": 0.41},
                        {"x": 0.61, "y": 0.41},
                        {"x": 0.61, "y": 0.61},
                        {"x": 0.41, "y": 0.61}
                    ]
                }
            ],
            "3": [
                {
                    "label": "Face",
                    "object_id": "face1",
                    "coordinates": [
                        {"x": 0.42, "y": 0.42},
                        {"x": 0.62, "y": 0.42},
                        {"x": 0.62, "y": 0.62},
                        {"x": 0.42, "y": 0.62}
                    ]
                }
            ]
        }
        
        # Create simulated eye tracking data
        # Gaze following the center of the face
        eye_data = pd.DataFrame({
            'timestamp': [1000, 2000, 3000],
            'frame_number': [1, 2, 3],
            'x_left': [0.5, 0.51, 0.52],
            'y_left': [0.5, 0.51, 0.52],
            'is_fixation_left': [True, True, True]
        })
        
        # Create and load the temp file
        temp_path = self.create_temp_roi_file(multi_frame_rois)
        self.roi_manager.load_roi_file(temp_path)
        
        # Test each gaze point against the ROIs
        for _, row in eye_data.iterrows():
            frame = int(row['frame_number'])
            x, y = row['x_left'], row['y_left']
            
            # The gaze should be in the Face ROI for each frame
            roi = self.roi_manager.find_roi_at_point(frame, x, y)
            self.assertIsNotNone(roi, f"Should find an ROI at frame {frame}, point ({x}, {y})")
            self.assertEqual(roi["label"], "Face", f"ROI at frame {frame}, point ({x}, {y}) should be 'Face'")

    def test_complex_roi_combination(self):
        """Test a combination of complex ROI scenarios."""
        # Create a test file with various complex ROI combinations
        complex_roi_data = {
            "1": [
                # A concave C-shaped ROI
                {
                    "label": "Complex1",
                    "object_id": "complex1",
                    "coordinates": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.3, "y": 0.1},
                        {"x": 0.3, "y": 0.2},
                        {"x": 0.2, "y": 0.2},
                        {"x": 0.2, "y": 0.3},
                        {"x": 0.3, "y": 0.3},
                        {"x": 0.3, "y": 0.4},
                        {"x": 0.1, "y": 0.4}
                    ]
                },
                # A polygon with many vertices (approximating a circle)
                {
                    "label": "Complex2",
                    "object_id": "complex2",
                    "coordinates": [
                        {"x": 0.6 + 0.1 * np.cos(angle), "y": 0.6 + 0.1 * np.sin(angle)}
                        for angle in np.linspace(0, 2*np.pi, 20, endpoint=False)
                    ]
                }
            ],
            # Frame with small, partially overlapping ROIs
            "2": [
                {
                    "label": "Small1",
                    "object_id": "small1",
                    "coordinates": [
                        {"x": 0.45, "y": 0.45},
                        {"x": 0.55, "y": 0.45},
                        {"x": 0.55, "y": 0.55},
                        {"x": 0.45, "y": 0.55}
                    ]
                },
                {
                    "label": "Small2",
                    "object_id": "small2",
                    "coordinates": [
                        {"x": 0.5, "y": 0.5},
                        {"x": 0.6, "y": 0.5},
                        {"x": 0.6, "y": 0.6},
                        {"x": 0.5, "y": 0.6}
                    ]
                }
            ]
        }
        
        # Create and load the temp file
        temp_path = self.create_temp_roi_file(complex_roi_data)
        self.roi_manager.load_roi_file(temp_path)
        
        # Test points in the complex shapes
        self.assertIsNotNone(
            self.roi_manager.find_roi_at_point(1, 0.15, 0.15),
            "Point should be inside the C-shaped ROI"
        )
        
        self.assertIsNone(
            self.roi_manager.find_roi_at_point(1, 0.25, 0.25),
            "Point should be in the 'hole' of the C-shape"
        )
        
        self.assertIsNotNone(
            self.roi_manager.find_roi_at_point(1, 0.6, 0.6),
            "Point should be inside the circular ROI"
        )
        
        # Test overlapping point in frame 2 - should return the first ROI (Small1)
        roi = self.roi_manager.find_roi_at_point(2, 0.51, 0.51)
        self.assertIsNotNone(roi, "Should find an ROI at the overlapping point")
        self.assertEqual(roi["label"], "Small1", 
                         "Overlapping point should match the first ROI in the list")


if __name__ == '__main__':
    unittest.main()