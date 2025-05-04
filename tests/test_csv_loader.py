"""
Unit tests for the CSV loader functionality
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser import load_csv_file, extract_events_from_unified, extract_features_from_unified


class TestCSVLoader(unittest.TestCase):
    """Test cases for CSV loading functionality"""

    def setUp(self):
        """Create a test unified eye metrics DataFrame and save it to a CSV file"""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple unified eye metrics DataFrame
        data = []
        for i in range(100):
            timestamp = i * 10
            is_fix_left = i >= 10 and i < 20
            is_fix_right = i >= 15 and i < 25
            is_sacc_left = i >= 30 and i < 40
            is_sacc_right = i >= 35 and i < 45
            is_blink_left = i >= 50 and i < 60
            is_blink_right = i >= 55 and i < 65
            
            row = {
                'timestamp': timestamp,
                'movie_name': 'test_movie',
                'frame_number': i // 5,
                'x_left': 500 + np.sin(i/10) * 100,
                'y_left': 400 + np.cos(i/10) * 100,
                'pupil_left': 3000 + np.sin(i/5) * 200,
                'x_right': 500 + np.sin(i/10) * 100 + 10,
                'y_right': 400 + np.cos(i/10) * 100 + 10,
                'pupil_right': 3000 + np.sin(i/5) * 200 + 50,
                'is_fixation_left': is_fix_left,
                'is_fixation_right': is_fix_right,
                'is_saccade_left': is_sacc_left,
                'is_saccade_right': is_sacc_right,
                'is_blink_left': is_blink_left,
                'is_blink_right': is_blink_right,
                'gaze_velocity_left': 0.0 if is_fix_left else 200.0 if is_sacc_left else 50.0,
                'gaze_velocity_right': 0.0 if is_fix_right else 200.0 if is_sacc_right else 50.0,
                'head_movement_magnitude': np.abs(np.sin(i/20) * 10),
                'inter_pupil_distance': 60 + np.sin(i/15) * 5
            }
            data.append(row)
        
        self.test_df = pd.DataFrame(data)
        
        # Save to CSV
        self.csv_path = os.path.join(self.temp_dir, 'test_participant_unified_eye_metrics.csv')
        self.test_df.to_csv(self.csv_path, index=False)
        
        # Participant ID
        self.participant_id = 'test_participant'
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_load_csv_file(self):
        """Test loading a CSV file"""
        result = load_csv_file(self.csv_path, extract_features=True)
        
        # Check that we got the expected structure
        self.assertIn('summary', result)
        self.assertIn('dataframes', result)
        self.assertIn('features', result)
        
        # Check that unified_eye_metrics is loaded
        self.assertIn('unified_eye_metrics', result['dataframes'])
        
        # Check that events were extracted
        self.assertIn('fixations_left', result['dataframes'])
        self.assertIn('fixations_right', result['dataframes'])
        self.assertIn('saccades_left', result['dataframes'])
        self.assertIn('saccades_right', result['dataframes'])
        self.assertIn('blinks_left', result['dataframes'])
        self.assertIn('blinks_right', result['dataframes'])
        
        # Check summary counts
        self.assertEqual(result['summary']['samples'], 100)
        self.assertGreater(result['summary']['fixations'], 0)
        self.assertGreater(result['summary']['saccades'], 0)
        self.assertGreater(result['summary']['blinks'], 0)
        
        # Check we have frames
        self.assertIn('frames', result['summary'])
        self.assertEqual(result['summary']['frames'], 20)  # 100 rows / 5 per frame
    
    def test_extract_events_from_unified(self):
        """Test extracting events from the unified DataFrame"""
        # Test fixations
        fixations_left = extract_events_from_unified(self.test_df, 'is_fixation_left', 'left')
        self.assertGreater(len(fixations_left), 0)
        self.assertIn('start_time', fixations_left.columns)
        self.assertIn('end_time', fixations_left.columns)
        self.assertIn('duration', fixations_left.columns)
        self.assertIn('x', fixations_left.columns)
        self.assertIn('y', fixations_left.columns)
        self.assertIn('pupil', fixations_left.columns)
        
        # Test saccades
        saccades_left = extract_events_from_unified(self.test_df, 'is_saccade_left', 'left')
        self.assertGreater(len(saccades_left), 0)
        self.assertIn('start_time', saccades_left.columns)
        self.assertIn('end_time', saccades_left.columns)
        self.assertIn('duration', saccades_left.columns)
        self.assertIn('start_x', saccades_left.columns)
        self.assertIn('start_y', saccades_left.columns)
        self.assertIn('end_x', saccades_left.columns)
        self.assertIn('end_y', saccades_left.columns)
        self.assertIn('amplitude', saccades_left.columns)
        
        # Test blinks
        blinks_left = extract_events_from_unified(self.test_df, 'is_blink_left', 'left')
        self.assertGreater(len(blinks_left), 0)
        self.assertIn('start_time', blinks_left.columns)
        self.assertIn('end_time', blinks_left.columns)
        self.assertIn('duration', blinks_left.columns)
    
    def test_extract_features_from_unified(self):
        """Test extracting features from the unified DataFrame"""
        features_df = extract_features_from_unified(self.test_df, self.participant_id)
        
        # Check basic structure
        self.assertEqual(len(features_df), 1)
        self.assertIn('participant_id', features_df.columns)
        self.assertEqual(features_df['participant_id'][0], self.participant_id)
        
        # Check eye-specific features
        for eye in ['left', 'right']:
            # Pupil metrics
            self.assertIn(f'pupil_{eye}_mean', features_df.columns)
            self.assertIn(f'pupil_{eye}_std', features_df.columns)
            
            # Gaze metrics
            self.assertIn(f'gaze_{eye}_x_std', features_df.columns)
            self.assertIn(f'gaze_{eye}_y_std', features_df.columns)
            self.assertIn(f'gaze_{eye}_dispersion', features_df.columns)
            
            # Event counts and rates
            for event in ['fixation', 'saccade', 'blink']:
                self.assertIn(f'{event}_{eye}_count', features_df.columns)
                self.assertIn(f'{event}_{eye}_rate', features_df.columns)
        
        # Additional metrics
        self.assertIn('head_movement_mean', features_df.columns)
        self.assertIn('inter_pupil_distance_mean', features_df.columns)


if __name__ == '__main__':
    unittest.main()