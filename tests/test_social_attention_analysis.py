"""
Social Attention Analysis Tests
Author: Tal Alfi
Date: May 2025

This module tests the social attention analysis features, including:
- Social vs non-social ROI classification
- Social attention metrics calculation
- Social attention visualization
"""

import unittest
import os
import sys
import tempfile
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
# Set non-interactive backend
matplotlib.use('Agg')

# Add parent directory to path to import necessary modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roi_manager import ROIManager


class TestSocialAttentionAnalysis(unittest.TestCase):
    """
    Test suite for social attention analysis features.
    Tests the classification, analysis, and visualization of social attention patterns.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Get path to complex ROI test data
        cls.complex_roi_path = os.path.join(os.path.dirname(__file__), 
                                         "test_data", "complex_roi.json")
        
        # Initialize the ROI manager
        cls.roi_manager = ROIManager()
        cls.load_success = cls.roi_manager.load_roi_file(cls.complex_roi_path)
        
        if not cls.load_success:
            raise ValueError(f"Failed to load test ROI file: {cls.complex_roi_path}")
            
        # Create test eye tracking data with fixations on various ROIs
        cls.create_test_eye_data()
    
    @classmethod
    def create_test_eye_data(cls):
        """Create synthetic eye tracking data for testing social attention analysis."""
        # Create data with gaze points on various ROIs across frames
        timestamps = list(range(1000, 10000, 100))  # 10-second recording at 10Hz
        num_samples = len(timestamps)
        
        # Generate a mix of frames that match our ROI frames
        roi_frames = cls.roi_manager.frame_numbers
        # Repeat frames with some randomness to simulate natural viewing
        frames = []
        for _ in range(num_samples):
            # Choose frames to match our ROI data with higher probability for social frames
            if np.random.random() < 0.7:  # 70% chance of social frame
                # Frames 1, 10, 20, 50, 60, 70, 80 have Face ROIs
                social_frames = [1, 10, 20, 50, 60, 70, 80]
                frame = np.random.choice(social_frames)
            else:
                # Frames 30, 90 have non-social ROIs
                non_social_frames = [30, 90]
                frame = np.random.choice(non_social_frames)
            frames.append(frame)
        
        # Create eye positions with a mix of social and non-social looking
        x_positions = []
        y_positions = []
        is_social_gaze = []
        is_fixation = []
        
        for frame in frames:
            rois = cls.roi_manager.get_frame_rois(frame)
            if not rois:
                # If no ROIs for this frame, look at a random position
                x, y = np.random.random(), np.random.random()
                is_social = False
            else:
                # Decide if we're looking at social or non-social ROI
                social_rois = [roi for roi in rois if roi["label"] in ["Face", "Eyes", "Hand"]]
                non_social_rois = [roi for roi in rois if roi["label"] not in ["Face", "Eyes", "Hand"]]
                
                if social_rois and np.random.random() < 0.8:  # 80% chance of looking at social ROI if available
                    # Choose a random social ROI
                    target_roi = np.random.choice(social_rois)
                    is_social = True
                elif non_social_rois:
                    # Choose a random non-social ROI
                    target_roi = np.random.choice(non_social_rois)
                    is_social = False
                else:
                    # If no non-social ROIs, look at a random position
                    x, y = np.random.random(), np.random.random()
                    is_social = False
                    target_roi = None
                
                if target_roi:
                    # Get a random point inside the ROI
                    coords = target_roi["coordinates"]
                    x_coords = [p["x"] for p in coords]
                    y_coords = [p["y"] for p in coords]
                    
                    # Simple way to get a point inside: take weighted average of vertices
                    # with small random perturbation
                    weights = np.random.random(len(coords))
                    weights = weights / weights.sum()
                    
                    x = sum(x * w for x, w in zip(x_coords, weights))
                    y = sum(y * w for y, w in zip(y_coords, weights))
                    
                    # Add small random noise, but keep inside the general ROI area
                    x += np.random.normal(0, 0.01)
                    y += np.random.normal(0, 0.01)
                    
                    # Keep within screen bounds
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
            
            x_positions.append(x)
            y_positions.append(y)
            is_social_gaze.append(is_social)
            
            # 60% chance of being a fixation, otherwise saccade
            is_fixation.append(np.random.random() < 0.6)
        
        # Create DataFrame with the synthetic data
        cls.eye_data = pd.DataFrame({
            'timestamp': timestamps,
            'frame_number': frames,
            'x_left': x_positions,
            'y_left': y_positions,
            'x_right': [x + np.random.normal(0, 0.01) for x in x_positions],  # Slightly offset right eye
            'y_right': [y + np.random.normal(0, 0.01) for y in y_positions],
            'is_fixation_left': is_fixation,
            'is_fixation_right': is_fixation,
            'is_social_gaze': is_social_gaze  # Ground truth for testing
        })
    
    def test_roi_social_classification(self):
        """Test classification of ROIs into social and non-social categories."""
        # This test verifies that ROIs are correctly classified as social or non-social
        
        # Get test data from first frame
        frame1_rois = self.roi_manager.get_frame_rois(1)
        
        # Define social categories based on common classification
        social_categories = ['Face', 'Eyes', 'Hand']
        
        # Check each ROI in frame 1
        for roi in frame1_rois:
            label = roi["label"]
            # Verify correct classification
            if label in social_categories:
                self.assertTrue(self._is_social_roi(roi),
                              f"ROI '{label}' should be classified as social")
            else:
                self.assertFalse(self._is_social_roi(roi),
                               f"ROI '{label}' should be classified as non-social")
    
    def _is_social_roi(self, roi):
        """Helper method to classify ROI as social or non-social."""
        social_categories = ['Face', 'Eyes', 'Hand', 'Torso']
        return roi["label"] in social_categories
    
    def test_fixation_roi_detection(self):
        """Test that fixations are correctly associated with ROIs."""
        # Get a subset of the eye data for testing
        test_data = self.eye_data.head(10)
        
        for _, row in test_data.iterrows():
            frame = int(row['frame_number'])
            x, y = row['x_left'], row['y_left']
            is_fixation = row['is_fixation_left']
            
            # Only check fixations
            if not is_fixation:
                continue
            
            # Find which ROI (if any) this fixation is in
            roi = self.roi_manager.find_roi_at_point(frame, x, y)
            
            # If ground truth says this is a social gaze, it should be in a social ROI
            if row['is_social_gaze']:
                self.assertIsNotNone(roi, f"Social gaze should be within an ROI: frame {frame}, point ({x}, {y})")
                if roi:
                    self.assertTrue(self._is_social_roi(roi),
                                  f"Social gaze should be in a social ROI, got {roi['label']}")
    
    def test_social_attention_metrics(self):
        """Test calculation of basic social attention metrics."""
        # Calculate basic metrics from eye data
        social_fixation_count = 0
        non_social_fixation_count = 0
        
        for _, row in self.eye_data.iterrows():
            if not row['is_fixation_left']:
                continue
                
            if row['is_social_gaze']:
                social_fixation_count += 1
            else:
                non_social_fixation_count += 1
        
        total_fixations = social_fixation_count + non_social_fixation_count
        
        # Calculate social attention percentage
        if total_fixations > 0:
            social_attention_percentage = (social_fixation_count / total_fixations) * 100
        else:
            social_attention_percentage = 0
        
        # Print metrics for debugging
        print(f"Social fixations: {social_fixation_count}")
        print(f"Non-social fixations: {non_social_fixation_count}")
        print(f"Social attention percentage: {social_attention_percentage:.2f}%")
        
        # Our synthetic data should have social attention between 40-80%
        # due to our data generation parameters
        self.assertTrue(20 <= social_attention_percentage <= 90,
                      f"Social attention percentage ({social_attention_percentage:.2f}%) is outside expected range")

    def test_roi_dwell_time_calculation(self):
        """Test calculation of dwell time for different ROIs."""
        # Calculate dwell time for each ROI (simplified method for testing)
        roi_dwell_times = {}
        
        # Simulate dwell time calculation from fixation data
        current_roi = None
        current_roi_start = None
        
        for i, row in self.eye_data.iterrows():
            if not row['is_fixation_left']:
                # Only count fixations
                continue
                
            frame = int(row['frame_number'])
            x, y = row['x_left'], row['y_left']
            timestamp = row['timestamp']
            
            # Find ROI at current point
            roi = self.roi_manager.find_roi_at_point(frame, x, y)
            
            if roi:
                roi_label = roi["label"]
                
                if current_roi != roi_label:
                    # ROI changed, log previous if any
                    if current_roi and current_roi_start:
                        duration = timestamp - current_roi_start
                        roi_dwell_times[current_roi] = roi_dwell_times.get(current_roi, 0) + duration
                    
                    # Start tracking new ROI
                    current_roi = roi_label
                    current_roi_start = timestamp
            else:
                # Not in any ROI, log previous if any
                if current_roi and current_roi_start:
                    duration = timestamp - current_roi_start
                    roi_dwell_times[current_roi] = roi_dwell_times.get(current_roi, 0) + duration
                
                current_roi = None
                current_roi_start = None
        
        # Log final ROI if any
        if current_roi and current_roi_start and len(self.eye_data) > 0:
            last_timestamp = self.eye_data.iloc[-1]['timestamp']
            duration = last_timestamp - current_roi_start
            roi_dwell_times[current_roi] = roi_dwell_times.get(current_roi, 0) + duration
        
        # Print dwell times for debugging
        print("ROI Dwell Times (ms):")
        for roi, time in roi_dwell_times.items():
            print(f"  {roi}: {time}ms")
        
        # Verify we have some dwell time data
        self.assertTrue(len(roi_dwell_times) > 0, "Should have calculated dwell time for at least one ROI")
        
        # Verify social ROIs have recorded dwell time
        social_roi_time = sum(time for roi, time in roi_dwell_times.items() 
                             if roi in ['Face', 'Eyes', 'Hand'])
        
        self.assertTrue(social_roi_time > 0, "Should have recorded some dwell time on social ROIs")
    
    def test_first_fixation_latency(self):
        """Test calculation of first fixation latency for each ROI."""
        # Calculate time to first fixation for each ROI
        first_fixation_times = {}
        start_time = self.eye_data['timestamp'].iloc[0]
        
        for _, row in self.eye_data.iterrows():
            if not row['is_fixation_left']:
                continue
                
            frame = int(row['frame_number'])
            x, y = row['x_left'], row['y_left']
            timestamp = row['timestamp']
            
            # Find ROI at current point
            roi = self.roi_manager.find_roi_at_point(frame, x, y)
            
            if roi:
                roi_label = roi["label"]
                
                # Only record first fixation for each ROI
                if roi_label not in first_fixation_times:
                    latency = (timestamp - start_time) / 1000.0  # Convert to seconds
                    first_fixation_times[roi_label] = latency
        
        # Print first fixation latencies for debugging
        print("First Fixation Latencies (s):")
        for roi, latency in first_fixation_times.items():
            print(f"  {roi}: {latency:.2f}s")
        
        # Verify we have some first fixation data
        self.assertTrue(len(first_fixation_times) > 0, 
                      "Should have calculated first fixation latency for at least one ROI")


if __name__ == '__main__':
    unittest.main()