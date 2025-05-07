"""
Advanced ROI Manager Testing Module
Author: Claude Code Assistant
Date: May 2025

This module provides comprehensive tests for the ROI manager functionality,
focusing on complex ROI scenarios, edge cases, and social attention classification.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Optional imports - handle gracefully if not available
try:
    from shapely.geometry import Polygon, Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: Shapely not available, some tests will be skipped")
    # Define placeholder classes for type checking
    class Polygon:
        pass
    class Point:
        pass

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the module to test
from roi_manager import ROIManager


class TestAdvancedROIManager(unittest.TestCase):
    """Test complex scenarios and edge cases for the ROI manager."""
    
    def setUp(self):
        """Set up test environment."""
        self.roi_manager = ROIManager()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create standard test data directory
        self.test_data_dir = os.path.join(self.temp_dir, 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
        plt.close('all')  # Close all matplotlib figures
    
    def test_complex_polygon_rois(self):
        """Test ROI manager with complex polygon shapes."""
        # Create a test ROI file with complex polygons (concave, many vertices)
        complex_roi_file = os.path.join(self.test_data_dir, 'complex_polygon_roi.json')
        
        # Define a concave polygon (C-shape)
        concave_polygon = [
            [0.1, 0.1], [0.4, 0.1], [0.4, 0.2],
            [0.2, 0.2], [0.2, 0.3], [0.4, 0.3],
            [0.4, 0.4], [0.1, 0.4]
        ]
        
        # Define a star-like polygon with many vertices
        star_points = 10
        star_polygon = []
        for i in range(star_points * 2):
            radius = 0.15 if i % 2 == 0 else 0.1
            angle = np.pi * i / star_points
            x = 0.7 + radius * np.cos(angle)
            y = 0.25 + radius * np.sin(angle)
            star_polygon.append([x, y])
        
        # Create ROI data with these complex polygons
        roi_data = {
            "frames": {
                "0": {
                    "objects": [
                        {
                            "object_id": "concave1",
                            "label": "Concave Shape",
                            "vertices": concave_polygon,
                            "social": False
                        },
                        {
                            "object_id": "star1",
                            "label": "Star Shape",
                            "vertices": star_polygon,
                            "social": True
                        }
                    ]
                }
            }
        }
        
        # Save to file
        with open(complex_roi_file, 'w') as f:
            json.dump(roi_data, f)
        
        # Load the ROI file
        self.assertTrue(self.roi_manager.load_roi_file(complex_roi_file),
                       "Failed to load complex polygon ROI file")
        
        # Test points inside and outside concave polygon
        # Point inside the C opening (should be outside the ROI)
        self.assertIsNone(self.roi_manager.find_roi_at_gaze(0, 0.3, 0.25),
                         "Point inside concave opening should not be in ROI")
        
        # Point inside the C shape
        self.assertIsNotNone(self.roi_manager.find_roi_at_gaze(0, 0.15, 0.15),
                            "Point inside concave shape should be in ROI")
        
        # Test points inside and outside star
        # Point at star center
        star_roi = self.roi_manager.find_roi_at_gaze(0, 0.7, 0.25)
        self.assertIsNotNone(star_roi, "Point at star center should be in ROI")
        self.assertEqual(star_roi['label'], "Star Shape", "Wrong ROI detected for star point")
        
        # Point outside star (between points)
        self.assertIsNone(self.roi_manager.find_roi_at_gaze(0, 0.7, 0.42),
                         "Point outside star should not be in ROI")
    
    def test_overlapping_rois(self):
        """Test ROI manager behavior with overlapping ROIs."""
        # Create a test ROI file with overlapping regions
        overlap_roi_file = os.path.join(self.test_data_dir, 'overlapping_roi.json')
        
        # Define overlapping rectangles and nested shapes
        roi_data = {
            "frames": {
                "0": {
                    "objects": [
                        {
                            "object_id": "bg",
                            "label": "Background",
                            "vertices": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
                            "social": False
                        },
                        {
                            "object_id": "face",
                            "label": "Face",
                            "vertices": [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]],
                            "social": True
                        },
                        {
                            "object_id": "eye",
                            "label": "Eye",
                            "vertices": [[0.25, 0.25], [0.35, 0.25], [0.35, 0.35], [0.25, 0.35]],
                            "social": True
                        }
                    ]
                }
            }
        }
        
        # Save to file
        with open(overlap_roi_file, 'w') as f:
            json.dump(roi_data, f)
        
        # Load the ROI file
        self.assertTrue(self.roi_manager.load_roi_file(overlap_roi_file),
                       "Failed to load overlapping ROI file")
        
        # Test points in different regions
        # The ROI manager should prioritize by area (smallest ROI), 
        # and then by order (first in list) for overlapping regions
        
        # Point in all three regions (background, face, eye)
        eye_point_roi = self.roi_manager.find_roi_at_gaze(0, 0.3, 0.3)
        self.assertIsNotNone(eye_point_roi, "Point in eye region should find a ROI")
        # This should be the Eye ROI because it's the smallest - the area-based selection should work
        self.assertEqual(eye_point_roi['label'], "Eye", 
                         "Point in eye should detect eye ROI (smallest one)")
        
        # Point in face but not eye
        face_point_roi = self.roi_manager.find_roi_at_gaze(0, 0.22, 0.22)
        self.assertIsNotNone(face_point_roi, "Point in face region should find a ROI")
        self.assertEqual(face_point_roi['label'], "Face", 
                         "Point in face should detect face ROI")
        
        # Point in background only
        bg_point_roi = self.roi_manager.find_roi_at_gaze(0, 0.12, 0.12)
        self.assertIsNotNone(bg_point_roi, "Point in background region should find a ROI")
        self.assertEqual(bg_point_roi['label'], "Background", 
                        "Point in background should detect background ROI")
    
    def test_multi_frame_roi_tracking(self):
        """Test ROI manager with changing ROIs across multiple frames."""
        # Create a test ROI file with ROIs that change positions across frames
        multi_frame_roi_file = os.path.join(self.test_data_dir, 'multi_frame_roi.json')
        
        # Create moving object across 5 frames
        roi_data = {"frames": {}}
        
        # Face moving diagonally across frames
        for frame in range(5):
            x_offset = 0.1 * frame
            y_offset = 0.1 * frame
            
            roi_data["frames"][str(frame)] = {
                "objects": [
                    {
                        "object_id": f"face_{frame}",
                        "label": "Face",
                        "vertices": [
                            [0.1 + x_offset, 0.1 + y_offset],
                            [0.3 + x_offset, 0.1 + y_offset],
                            [0.3 + x_offset, 0.3 + y_offset],
                            [0.1 + x_offset, 0.3 + y_offset]
                        ],
                        "social": True
                    },
                    {
                        "object_id": f"object_{frame}",
                        "label": "Object",
                        "vertices": [
                            [0.6, 0.1 + y_offset],
                            [0.8, 0.1 + y_offset],
                            [0.8, 0.3 + y_offset],
                            [0.6, 0.3 + y_offset]
                        ],
                        "social": False
                    }
                ]
            }
        
        # Save to file
        with open(multi_frame_roi_file, 'w') as f:
            json.dump(roi_data, f)
        
        # Load the ROI file
        self.assertTrue(self.roi_manager.load_roi_file(multi_frame_roi_file),
                       "Failed to load multi-frame ROI file")
        
        # Test ROI tracking across frames
        # Frame 0: Face at (0.1, 0.1) to (0.3, 0.3)
        frame0_face = self.roi_manager.find_roi_at_gaze(0, 0.2, 0.2)
        self.assertIsNotNone(frame0_face, "Face ROI not found at frame 0")
        self.assertEqual(frame0_face['label'], "Face")
        
        # Frame 2: Face moved to (0.3, 0.3) to (0.5, 0.5)
        frame2_face = self.roi_manager.find_roi_at_gaze(2, 0.4, 0.4)
        self.assertIsNotNone(frame2_face, "Face ROI not found at frame 2")
        self.assertEqual(frame2_face['label'], "Face")
        
        # The original position should no longer have a face in frame 2
        self.assertIsNone(self.roi_manager.find_roi_at_gaze(2, 0.2, 0.2),
                         "Original face position should be empty in frame 2")
        
        # Frame 4: Face moved to (0.5, 0.5) to (0.7, 0.7)
        frame4_face = self.roi_manager.find_roi_at_gaze(4, 0.6, 0.6)
        self.assertIsNotNone(frame4_face, "Face ROI not found at frame 4")
        self.assertEqual(frame4_face['label'], "Face")
        
        # Test non-existent frame (use frame 100 which is far away from any existing frames)
        self.assertIsNone(self.roi_manager.find_roi_at_gaze(100, 0.5, 0.5, use_nearest_frame=False),
                         "Non-existent frame should return None with use_nearest_frame=False")
    
    def test_extreme_roi_coordinates(self):
        """Test ROI manager with extreme coordinate values."""
        # Create a test ROI file with ROIs at extremes (edges of screen, tiny/large)
        extreme_roi_file = os.path.join(self.test_data_dir, 'extreme_roi.json')
        
        roi_data = {
            "frames": {
                "0": {
                    "objects": [
                        # ROI at top-left corner (edge case)
                        {
                            "object_id": "corner",
                            "label": "Corner",
                            "vertices": [[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]],
                            "social": False
                        },
                        # Very small ROI (1% of screen)
                        {
                            "object_id": "tiny",
                            "label": "Tiny",
                            "vertices": [[0.5, 0.5], [0.51, 0.5], [0.51, 0.51], [0.5, 0.51]],
                            "social": True
                        },
                        # Very large ROI (covers most of screen)
                        {
                            "object_id": "huge",
                            "label": "Huge",
                            "vertices": [[0.1, 0.6], [0.9, 0.6], [0.9, 0.95], [0.1, 0.95]],
                            "social": False
                        },
                        # ROI at bottom-right corner (edge case)
                        {
                            "object_id": "bottom_corner",
                            "label": "Bottom Corner",
                            "vertices": [[0.95, 0.95], [1.0, 0.95], [1.0, 1.0], [0.95, 1.0]],
                            "social": False
                        }
                    ]
                }
            }
        }
        
        # Save to file
        with open(extreme_roi_file, 'w') as f:
            json.dump(roi_data, f)
        
        # Load the ROI file
        self.assertTrue(self.roi_manager.load_roi_file(extreme_roi_file),
                       "Failed to load extreme ROI file")
        
        # Test corner ROI
        corner_roi = self.roi_manager.find_roi_at_gaze(0, 0.01, 0.01)
        self.assertIsNotNone(corner_roi, "Corner ROI not detected")
        self.assertEqual(corner_roi['label'], "Corner")
        
        # Test tiny ROI
        tiny_roi = self.roi_manager.find_roi_at_gaze(0, 0.505, 0.505)
        self.assertIsNotNone(tiny_roi, "Tiny ROI not detected")
        self.assertEqual(tiny_roi['label'], "Tiny")
        
        # Test huge ROI
        huge_roi = self.roi_manager.find_roi_at_gaze(0, 0.5, 0.8)
        self.assertIsNotNone(huge_roi, "Huge ROI not detected")
        self.assertEqual(huge_roi['label'], "Huge")
        
        # Test bottom-right corner ROI
        corner_roi = self.roi_manager.find_roi_at_gaze(0, 0.97, 0.97)
        self.assertIsNotNone(corner_roi, "Bottom corner ROI not detected")
        self.assertEqual(corner_roi['label'], "Bottom Corner")
        
        # Test exact boundary points
        # Point exactly on edge of tiny ROI
        edge_roi = self.roi_manager.find_roi_at_gaze(0, 0.5, 0.505)
        self.assertIsNotNone(edge_roi, "Edge point should be detected")
        self.assertEqual(edge_roi['label'], "Tiny")
    
    def test_roi_export_import(self):
        """Test exporting and reimporting ROI data."""
        # Create test ROI data
        roi_data = {
            "frames": {
                "0": {
                    "objects": [
                        {
                            "object_id": "test1",
                            "label": "Test 1",
                            "vertices": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
                            "social": True
                        },
                        {
                            "object_id": "test2",
                            "label": "Test 2",
                            "vertices": [[0.5, 0.5], [0.7, 0.5], [0.7, 0.7], [0.5, 0.7]],
                            "social": False
                        }
                    ]
                },
                "1": {
                    "objects": [
                        {
                            "object_id": "test1",
                            "label": "Test 1",
                            "vertices": [[0.15, 0.15], [0.35, 0.15], [0.35, 0.35], [0.15, 0.35]],
                            "social": True
                        }
                    ]
                }
            }
        }
        
        # Create test file
        test_roi_file = os.path.join(self.test_data_dir, 'test_roi.json')
        with open(test_roi_file, 'w') as f:
            json.dump(roi_data, f)
        
        # Load the ROI file
        self.assertTrue(self.roi_manager.load_roi_file(test_roi_file),
                       "Failed to load test ROI file")
        
        # Export to a new file
        export_roi_file = os.path.join(self.test_data_dir, 'exported_roi.json')
        self.roi_manager.export_roi_data(export_roi_file)
        
        # Create a new ROI manager and load the exported file
        new_roi_manager = ROIManager()
        self.assertTrue(new_roi_manager.load_roi_file(export_roi_file),
                       "Failed to load exported ROI file")
        
        # Verify data is preserved
        # Check frame 0, first object
        roi1 = self.roi_manager.find_roi_at_gaze(0, 0.2, 0.2)
        new_roi1 = new_roi_manager.find_roi_at_gaze(0, 0.2, 0.2)
        self.assertEqual(roi1['label'], new_roi1['label'], 
                        "ROI label not preserved in export/import")
        self.assertEqual(roi1['social'], new_roi1['social'],
                        "ROI social flag not preserved in export/import")
        
        # Check frame 1
        roi2 = self.roi_manager.find_roi_at_gaze(1, 0.25, 0.25)
        new_roi2 = new_roi_manager.find_roi_at_gaze(1, 0.25, 0.25)
        self.assertEqual(roi2['label'], new_roi2['label'],
                        "ROI label in frame 1 not preserved in export/import")
    
    def test_social_roi_classification(self):
        """Test social vs. non-social ROI classification."""
        # Create a test ROI file with explicit social flags
        social_roi_file = os.path.join(self.test_data_dir, 'social_roi.json')
        
        roi_data = {
            "frames": {
                "0": {
                    "objects": [
                        {
                            "object_id": "face",
                            "label": "Face",
                            "vertices": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
                            "social": True
                        },
                        {
                            "object_id": "eyes",
                            "label": "Eyes",
                            "vertices": [[0.15, 0.15], [0.25, 0.15], [0.25, 0.20], [0.15, 0.20]],
                            "social": True
                        },
                        {
                            "object_id": "hands",
                            "label": "Hands",
                            "vertices": [[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]],
                            "social": True
                        },
                        {
                            "object_id": "toy",
                            "label": "Toy",
                            "vertices": [[0.6, 0.6], [0.7, 0.6], [0.7, 0.7], [0.6, 0.7]],
                            "social": False
                        },
                        {
                            "object_id": "background",
                            "label": "Background",
                            "vertices": [[0.8, 0.8], [0.9, 0.8], [0.9, 0.9], [0.8, 0.9]],
                            "social": False
                        }
                    ]
                }
            }
        }
        
        # Save to file
        with open(social_roi_file, 'w') as f:
            json.dump(roi_data, f)
        
        # Load the ROI file
        self.assertTrue(self.roi_manager.load_roi_file(social_roi_file),
                       "Failed to load social ROI file")
        
        # Test social ROI detection
        # For the Face ROI, we want to ensure we get the Face ROI specifically
        # For this test, we'll directly search for the ROI with the expected object_id
        rois = self.roi_manager.get_frame_rois(0)
        face_roi = next((roi for roi in rois if roi['object_id'] == 'face'), None)
        self.assertIsNotNone(face_roi, "Face ROI not found in frame data")
        self.assertEqual(face_roi['label'], "Face")
        self.assertTrue(face_roi['social'], "Face ROI should be marked as social")
        
        # For the Eyes ROI, do the same thing
        eyes_roi = next((roi for roi in rois if roi['object_id'] == 'eyes'), None)
        self.assertIsNotNone(eyes_roi, "Eyes ROI not found in frame data")
        self.assertEqual(eyes_roi['label'], "Eyes")
        self.assertTrue(eyes_roi['social'], "Eyes ROI should be marked as social")
        
        # Verify that both points are within their respective ROIs
        self.assertTrue(self.roi_manager.is_gaze_in_roi(0.2, 0.2, face_roi), 
                     "Point (0.2, 0.2) should be within the Face ROI")
        self.assertTrue(self.roi_manager.is_gaze_in_roi(0.2, 0.17, eyes_roi), 
                     "Point (0.2, 0.17) should be within the Eyes ROI")
        
        hands_roi = self.roi_manager.find_roi_at_gaze(0, 0.45, 0.45)
        self.assertIsNotNone(hands_roi, "Hands ROI not detected")
        self.assertEqual(hands_roi['label'], "Hands")
        self.assertTrue(hands_roi['social'], "Hands ROI should be marked as social")
        
        # Test non-social ROI detection
        toy_roi = self.roi_manager.find_roi_at_gaze(0, 0.65, 0.65)
        self.assertIsNotNone(toy_roi, "Toy ROI not detected")
        self.assertEqual(toy_roi['label'], "Toy")
        self.assertFalse(toy_roi['social'], "Toy ROI should be marked as non-social")
        
        bg_roi = self.roi_manager.find_roi_at_gaze(0, 0.85, 0.85)
        self.assertIsNotNone(bg_roi, "Background ROI not detected")
        self.assertEqual(bg_roi['label'], "Background")
        self.assertFalse(bg_roi['social'], "Background ROI should be marked as non-social")
    
    def test_invalid_roi_handling(self):
        """Test handling of invalid ROI data."""
        # Create a test ROI file with invalid data
        invalid_roi_file = os.path.join(self.test_data_dir, 'invalid_roi.json')
        
        # Test case 1: ROI with < 3 vertices (not a polygon)
        roi_data_1 = {
            "frames": {
                "0": {
                    "objects": [
                        {
                            "object_id": "invalid1",
                            "label": "Invalid Polygon",
                            "vertices": [[0.1, 0.1], [0.3, 0.3]],  # Only 2 vertices
                            "social": True
                        }
                    ]
                }
            }
        }
        
        with open(invalid_roi_file, 'w') as f:
            json.dump(roi_data_1, f)
        
        # Load should succeed but skip invalid ROIs
        self.assertTrue(self.roi_manager.load_roi_file(invalid_roi_file),
                       "ROI manager should load file with invalid ROIs")
        
        # Check that the invalid ROI was skipped
        self.assertIsNone(self.roi_manager.find_roi_at_gaze(0, 0.2, 0.2),
                         "Invalid ROI should not be detected")
        
        # Test case 2: ROI with invalid coordinates (outside 0-1 range)
        roi_data_2 = {
            "frames": {
                "0": {
                    "objects": [
                        {
                            "object_id": "invalid2",
                            "label": "Out of Range",
                            "vertices": [[0.1, 0.1], [1.5, 0.1], [1.5, 0.3], [0.1, 0.3]],  # x > 1
                            "social": True
                        }
                    ]
                }
            }
        }
        
        with open(invalid_roi_file, 'w') as f:
            json.dump(roi_data_2, f)
        
        # Load should succeed but handle ROIs with out-of-range coordinates
        self.assertTrue(self.roi_manager.load_roi_file(invalid_roi_file),
                       "ROI manager should load file with out-of-range coordinates")
        
        # ROI should still be detected for valid portion
        self.assertIsNotNone(self.roi_manager.find_roi_at_gaze(0, 0.2, 0.2),
                            "Valid portion of out-of-range ROI should be detected")
        
        # Test case 3: Completely malformed JSON
        with open(invalid_roi_file, 'w') as f:
            f.write("{not valid json")
        
        # Load should fail
        self.assertFalse(self.roi_manager.load_roi_file(invalid_roi_file),
                        "ROI manager should reject malformed JSON")
    
    def test_polygon_containment_algorithms(self):
        """Test the algorithms used for polygon containment."""
        # Skip test if Shapely is not available
        if not SHAPELY_AVAILABLE:
            self.skipTest("Shapely library not available, skipping this test")
            
        # Create a simple test ROI file
        test_roi_file = os.path.join(self.test_data_dir, 'algorithm_test_roi.json')
        
        # Define a complex polygon with a hole
        # Outer square with inner square hole
        roi_data = {
            "frames": {
                "0": {
                    "objects": [
                        {
                            "object_id": "square",
                            "label": "Square",
                            "vertices": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
                            "social": False
                        }
                    ]
                }
            }
        }
        
        # Save to file
        with open(test_roi_file, 'w') as f:
            json.dump(roi_data, f)
        
        # Load the ROI file
        self.assertTrue(self.roi_manager.load_roi_file(test_roi_file),
                       "Failed to load algorithm test ROI file")
        
        # Test points
        # Inside the outer square
        self.assertIsNotNone(self.roi_manager.find_roi_at_gaze(0, 0.2, 0.2),
                            "Point inside square should be detected")
        
        # Outside the outer square
        self.assertIsNone(self.roi_manager.find_roi_at_gaze(0, 0.6, 0.6),
                         "Point outside square should not be detected")
        
        # On the edge of the square
        edge_roi = self.roi_manager.find_roi_at_gaze(0, 0.1, 0.3)
        self.assertIsNotNone(edge_roi, "Point on edge should be detected")
        self.assertEqual(edge_roi['label'], "Square")
        
        # At a vertex of the square
        vertex_roi = self.roi_manager.find_roi_at_gaze(0, 0.1, 0.1)
        self.assertIsNotNone(vertex_roi, "Point at vertex should be detected")
        self.assertEqual(vertex_roi['label'], "Square")
        
        # Test custom function by directly calling it with a custom polygon
        # Create a polygon with a hole
        outer_poly = Polygon([(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)])
        hole_poly = Polygon([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)])
        
        # Import the function directly for testing
        from roi_manager import ROIManager
        
        # Test point inside outer polygon but not in hole
        point = Point(0.15, 0.15)
        self.assertTrue(self.roi_manager._is_point_in_polygon(point, outer_poly),
                       "Point should be in outer polygon")
        
        # Create polygon with hole (using the constructor)
        poly_with_hole = Polygon([(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)], 
                                [[(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)]])
        
        # Test point in hole
        hole_point = Point(0.3, 0.3)
        self.assertFalse(self.roi_manager._is_point_in_polygon(hole_point, poly_with_hole),
                        "Point in hole should not be in polygon")


if __name__ == '__main__':
    unittest.main()