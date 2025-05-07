"""
Advanced CSV Loading Tests
Author: Tal Alfi
Date: May 2025

This module tests advanced CSV loading functionality, including:
- Loading CSV files with different formats
- Handling missing or corrupted data
- Loading multiple CSV files
- Extracting features from CSV data
"""

import unittest
import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import CSV loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser import load_csv_file, load_multiple_csv_files


class TestAdvancedCSVLoading(unittest.TestCase):
    """
    Test suite for advanced CSV loading functionality.
    Tests handling of various CSV formats and data extraction.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.temp_files = []  # Keep track of temporary files
    
    def tearDown(self):
        """Clean up after each test."""
        # Delete temporary files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Delete temporary directory
        self.temp_dir.cleanup()
    
    def create_test_csv_file(self, data, filename=None):
        """Helper to create a temporary CSV file with the provided data."""
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            # Assume it's a dictionary format
            df = pd.DataFrame(data)
        
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix='_unified_eye_metrics.csv', delete=False) as temp_file:
                temp_path = temp_file.name
                self.temp_files.append(temp_path)
        else:
            temp_path = os.path.join(self.output_dir, filename)
            self.temp_files.append(temp_path)
        
        df.to_csv(temp_path, index=False)
        return temp_path
    
    def test_standard_csv_loading(self):
        """Test loading a standard unified eye metrics CSV file."""
        # Create a standard eye metrics CSV file
        data = {
            'timestamp': list(range(1000, 2000, 100)),
            'frame_number': list(range(10)),
            'x_left': [100 + i for i in range(10)],
            'y_left': [200 + i for i in range(10)],
            'x_right': [300 + i for i in range(10)],
            'y_right': [400 + i for i in range(10)],
            'pupil_left': [1000 + i for i in range(10)],
            'pupil_right': [1200 + i for i in range(10)]
        }
        
        csv_path = self.create_test_csv_file(data)
        
        # Load the CSV file
        result = load_csv_file(csv_path, output_dir=self.output_dir)
        
        # Verify the data was loaded correctly
        self.assertIsNotNone(result, "Loader should return a result")
        self.assertIn('data', result, "Result should include data")
        self.assertEqual(len(result['data']), 10, "Should load all 10 rows of data")
        
        # Check if the output file was created
        output_file = os.path.join(self.output_dir, os.path.basename(csv_path))
        self.assertTrue(os.path.exists(output_file), "Output file should be created")
        
        # Check if features were extracted
        if 'features' in result:
            features_df = result['features']
            self.assertIsInstance(features_df, pd.DataFrame, "Features should be a DataFrame")
    
    def test_missing_columns(self):
        """Test loading a CSV file with missing columns."""
        # Create a CSV file with missing columns
        data = {
            'timestamp': list(range(1000, 2000, 100)),
            'frame_number': list(range(10)),
            # Missing x_left, y_left
            'x_right': [300 + i for i in range(10)],
            'y_right': [400 + i for i in range(10)],
            # Missing pupil_left
            'pupil_right': [1200 + i for i in range(10)]
        }
        
        csv_path = self.create_test_csv_file(data)
        
        # Load the CSV file with missing columns
        result = load_csv_file(csv_path, output_dir=self.output_dir)
        
        # Verify the data was loaded despite missing columns
        self.assertIsNotNone(result, "Loader should return a result despite missing columns")
        self.assertIn('data', result, "Result should include data")
        self.assertEqual(len(result['data']), 10, "Should load all 10 rows of data")
        
        # Check if missing columns were handled
        loaded_data = result['data']
        self.assertTrue('x_left' in loaded_data.columns, "Missing column x_left should be added")
        self.assertTrue('y_left' in loaded_data.columns, "Missing column y_left should be added")
        self.assertTrue('pupil_left' in loaded_data.columns, "Missing column pupil_left should be added")
        
        # Missing columns should be filled with NaN
        self.assertTrue(loaded_data['x_left'].isna().all(), "Missing x_left should be NaN")
    
    def test_additional_columns(self):
        """Test loading a CSV file with additional columns."""
        # Create a CSV file with additional columns
        data = {
            'timestamp': list(range(1000, 2000, 100)),
            'frame_number': list(range(10)),
            'x_left': [100 + i for i in range(10)],
            'y_left': [200 + i for i in range(10)],
            'x_right': [300 + i for i in range(10)],
            'y_right': [400 + i for i in range(10)],
            'pupil_left': [1000 + i for i in range(10)],
            'pupil_right': [1200 + i for i in range(10)],
            # Additional columns
            'is_fixation_left': [i % 2 == 0 for i in range(10)],
            'is_fixation_right': [i % 2 == 0 for i in range(10)],
            'custom_column': ['value' + str(i) for i in range(10)]
        }
        
        csv_path = self.create_test_csv_file(data)
        
        # Load the CSV file with additional columns
        result = load_csv_file(csv_path, output_dir=self.output_dir)
        
        # Verify the data was loaded with additional columns
        self.assertIsNotNone(result, "Loader should return a result with additional columns")
        self.assertIn('data', result, "Result should include data")
        
        # Check if additional columns were preserved
        loaded_data = result['data']
        self.assertTrue('is_fixation_left' in loaded_data.columns, "is_fixation_left should be preserved")
        self.assertTrue('is_fixation_right' in loaded_data.columns, "is_fixation_right should be preserved")
        self.assertTrue('custom_column' in loaded_data.columns, "custom_column should be preserved")
    
    def test_varying_row_count(self):
        """Test loading multiple CSV files with varying row counts."""
        # Create multiple CSV files with different row counts
        data1 = {
            'timestamp': list(range(1000, 2000, 100)),
            'frame_number': list(range(10)),
            'x_left': [100 + i for i in range(10)],
            'y_left': [200 + i for i in range(10)]
        }
        
        data2 = {
            'timestamp': list(range(2000, 3000, 50)),
            'frame_number': list(range(20)),
            'x_left': [150 + i for i in range(20)],
            'y_left': [250 + i for i in range(20)]
        }
        
        data3 = {
            'timestamp': list(range(3000, 3500, 100)),
            'frame_number': list(range(5)),
            'x_left': [200 + i for i in range(5)],
            'y_left': [300 + i for i in range(5)]
        }
        
        # Create the CSV files
        csv_path1 = self.create_test_csv_file(data1, filename="file1_unified_eye_metrics.csv")
        csv_path2 = self.create_test_csv_file(data2, filename="file2_unified_eye_metrics.csv")
        csv_path3 = self.create_test_csv_file(data3, filename="file3_unified_eye_metrics.csv")
        
        # Load multiple CSV files with varying row counts
        result = load_multiple_csv_files(
            [csv_path1, csv_path2, csv_path3],
            output_dir=self.output_dir
        )
        
        # Verify the data was loaded and combined
        self.assertIsNotNone(result, "Loader should return a result for multiple files")
        self.assertIsInstance(result, pd.DataFrame, "Result should be a combined DataFrame")
        
        # Should combine all rows
        expected_rows = 10 + 20 + 5
        self.assertEqual(len(result), expected_rows, f"Should combine all {expected_rows} rows")
    
    def test_column_type_mismatches(self):
        """Test loading CSV files with column type mismatches."""
        # Create a CSV file with mixed column types
        data = {
            'timestamp': list(range(1000, 2000, 100)),
            'frame_number': [str(i) for i in range(10)],  # String instead of int
            'x_left': [str(100 + i) for i in range(10)],  # String instead of float
            'y_left': [200.5 + i for i in range(10)],
            'x_right': [300 + i for i in range(10)],
            'y_right': [400 + i for i in range(10)],
            'pupil_left': [1000 + i for i in range(10)],
            'pupil_right': ['large' if i % 2 == 0 else 1200 + i for i in range(10)]  # Mixed strings and numbers
        }
        
        csv_path = self.create_test_csv_file(data)
        
        # Load the CSV file with column type mismatches
        result = load_csv_file(csv_path, output_dir=self.output_dir)
        
        # Verify the data was loaded despite type mismatches
        self.assertIsNotNone(result, "Loader should return a result despite type mismatches")
        self.assertIn('data', result, "Result should include data")
        
        # Type conversion should be handled
        loaded_data = result['data']
        self.assertTrue(pd.api.types.is_numeric_dtype(loaded_data['frame_number']),
                       "frame_number should be converted to numeric")
        self.assertTrue(pd.api.types.is_numeric_dtype(loaded_data['x_left']),
                      "x_left should be converted to numeric")
        
        # Mixed string/number columns should have NaNs for non-numeric values
        self.assertTrue(loaded_data['pupil_right'].isna().any(),
                      "Non-numeric values in pupil_right should be NaN")
    
    def test_feature_extraction(self):
        """Test feature extraction from CSV data."""
        # Create a CSV file with data suitable for feature extraction
        # Generate random walk for eye positions with occasional fixations
        np.random.seed(42)  # For reproducibility
        n_samples = 1000
        timestamps = list(range(0, n_samples * 10, 10))
        frame_numbers = list(range(0, n_samples, 1))
        
        # Generate eye positions with fixations (low variance) and saccades (high variance)
        x_left = []
        y_left = []
        fixation_state = 0  # 0: saccade, 1: fixation
        fixation_centers = [(100, 100), (300, 200), (200, 300), (400, 400)]
        current_fixation = 0
        
        for i in range(n_samples):
            if i % 100 == 0:  # Switch state every 100 samples
                fixation_state = 1 - fixation_state
                if fixation_state == 1:  # Starting new fixation
                    current_fixation = (current_fixation + 1) % len(fixation_centers)
            
            if fixation_state == 1:  # In fixation
                center_x, center_y = fixation_centers[current_fixation]
                x_left.append(center_x + np.random.normal(0, 5))  # Low variance during fixation
                y_left.append(center_y + np.random.normal(0, 5))
            else:  # In saccade
                if i == 0:
                    x_left.append(0)
                    y_left.append(0)
                else:
                    dx = np.random.normal(0, 20)  # High variance during saccade
                    dy = np.random.normal(0, 20)
                    x_left.append(x_left[-1] + dx)
                    y_left.append(y_left[-1] + dy)
        
        # Create a dataset with eye tracking data
        data = {
            'timestamp': timestamps,
            'frame_number': frame_numbers,
            'x_left': x_left,
            'y_left': y_left,
            'x_right': [x + np.random.normal(0, 10) for x in x_left],  # Right eye with slight offset
            'y_right': [y + np.random.normal(0, 10) for y in y_left],
            'pupil_left': [1000 + np.random.normal(0, 50) for _ in range(n_samples)],
            'pupil_right': [1200 + np.random.normal(0, 50) for _ in range(n_samples)]
        }
        
        csv_path = self.create_test_csv_file(data)
        
        # Load the CSV file and extract features
        result = load_csv_file(csv_path, output_dir=self.output_dir, extract_features=True)
        
        # Verify features were extracted
        self.assertIsNotNone(result, "Loader should return a result")
        self.assertIn('features', result, "Result should include extracted features")
        
        features = result['features']
        self.assertIsInstance(features, pd.DataFrame, "Features should be a DataFrame")
        
        # Check for expected feature columns
        expected_features = [
            'pupil_left_mean', 'pupil_right_mean',
            'gaze_left_x_std', 'gaze_left_y_std',
            'gaze_right_x_std', 'gaze_right_y_std'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features.columns, f"Feature {feature} should be extracted")
        
        # Verify feature values are reasonable
        self.assertGreater(features['pupil_left_mean'].iloc[0], 0, "Pupil size mean should be positive")
        self.assertGreater(features['gaze_left_x_std'].iloc[0], 0, "Gaze position std should be positive")
    
    def test_multiple_csv_with_different_columns(self):
        """Test loading multiple CSV files with different column sets."""
        # Create CSV files with different column sets
        data1 = {
            'timestamp': list(range(1000, 2000, 100)),
            'frame_number': list(range(10)),
            'x_left': [100 + i for i in range(10)],
            'y_left': [200 + i for i in range(10)],
            'custom1': ['a' + str(i) for i in range(10)]
        }
        
        data2 = {
            'timestamp': list(range(2000, 3000, 100)),
            'frame_number': list(range(10, 20)),
            'x_right': [300 + i for i in range(10)],
            'y_right': [400 + i for i in range(10)],
            'custom2': ['b' + str(i) for i in range(10)]
        }
        
        # Create the CSV files
        csv_path1 = self.create_test_csv_file(data1, filename="file1_unified_eye_metrics.csv")
        csv_path2 = self.create_test_csv_file(data2, filename="file2_unified_eye_metrics.csv")
        
        # Load multiple CSV files with different column sets
        result = load_multiple_csv_files(
            [csv_path1, csv_path2],
            output_dir=self.output_dir
        )
        
        # Verify the data was loaded and columns were combined
        self.assertIsNotNone(result, "Loader should return a result for multiple files")
        self.assertIsInstance(result, pd.DataFrame, "Result should be a combined DataFrame")
        
        # Should have all columns from both files
        expected_columns = ['timestamp', 'frame_number', 
                          'x_left', 'y_left', 'custom1',
                          'x_right', 'y_right', 'custom2']
        
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Result should include {col} column")
        
        # Rows from first file should have NaN for columns unique to second file
        first_file_rows = result[result['frame_number'] < 10]
        self.assertTrue(first_file_rows['custom2'].isna().all(), 
                      "custom2 should be NaN for rows from first file")
        
        # Rows from second file should have NaN for columns unique to first file
        second_file_rows = result[result['frame_number'] >= 10]
        self.assertTrue(second_file_rows['custom1'].isna().all(),
                      "custom1 should be NaN for rows from second file")


if __name__ == '__main__':
    unittest.main()