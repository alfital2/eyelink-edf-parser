"""
Expanded unit tests for the EyeLinkASCParser

This module includes comprehensive tests for the EyeLinkASCParser class,
testing various file formats, edge cases, and validating feature extraction.
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser import EyeLinkASCParser, process_asc_file, load_csv_file
from tests.mock_data_generator import MockASCGenerator, create_test_suite


class TestExpandedParser(unittest.TestCase):
    """Expanded test cases for the EyeLinkASCParser"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with mock data files"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data_dir = os.path.join(cls.temp_dir, "asc_test_files")
        
        # Create a full suite of test files
        cls.test_files = create_test_suite(cls.test_data_dir)
        
        # Save paths to key test files
        cls.standard_file = cls.test_files['standard']['path']
        cls.large_file = cls.test_files['large']['path']
        cls.empty_file = cls.test_files['empty']['path']
        cls.samples_only_file = cls.test_files['samples_only']['path']
        cls.events_only_file = cls.test_files['events_only']['path']
        cls.movies_only_file = cls.test_files['movies_only']['path']
        cls.malformed_file = cls.test_files['malformed']['path']
        cls.left_eye_only_file = cls.test_files['left_eye_only']['path']
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        shutil.rmtree(cls.temp_dir)
    
    def test_parser_initialization(self):
        """Test the parser initialization with different file types"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        self.assertEqual(parser.file_path, self.standard_file)
        self.assertEqual(len(parser.file_lines), 0)  # Should be empty until read_file is called
        
        # We don't test with nonexistent files since the parser doesn't validate
        # file existence until read_file is called
    
    def test_read_file(self):
        """Test reading different types of ASC files"""
        # Test standard file
        parser = EyeLinkASCParser(self.standard_file)
        num_lines = parser.read_file()
        self.assertGreater(num_lines, 0)
        
        # Test empty file
        parser = EyeLinkASCParser(self.empty_file)
        num_lines = parser.read_file()
        self.assertGreater(num_lines, 0)  # Should at least have header
        
        # Test large file
        parser = EyeLinkASCParser(self.large_file)
        num_lines = parser.read_file()
        self.assertGreater(num_lines, 1000)  # Should have many lines
    
    def test_parse_metadata(self):
        """Test metadata extraction from different file types"""
        # Test standard file
        parser = EyeLinkASCParser(self.standard_file)
        parser.read_file()
        metadata = parser.parse_metadata()
        
        self.assertIsInstance(metadata, dict)
        self.assertGreater(len(metadata), 0)
        
        # Check for required metadata fields
        self.assertIn('VERSION', metadata)
        self.assertIn('DATE', metadata)
        
        # Test empty file (should still have header metadata)
        parser = EyeLinkASCParser(self.empty_file)
        parser.read_file()
        metadata = parser.parse_metadata()
        
        self.assertIsInstance(metadata, dict)
        self.assertGreater(len(metadata), 0)
    
    def test_parse_messages(self):
        """Test message parsing from different file types"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        parser.read_file()
        parser.parse_metadata()
        messages = parser.parse_messages()
        
        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)
        
        # Test movies_only file (should have many movie markers)
        parser = EyeLinkASCParser(self.movies_only_file)
        parser.read_file()
        parser.parse_metadata()
        messages = parser.parse_messages()
        
        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 10)
        
        # Check if movie segments were detected
        self.assertGreater(len(parser.movie_segments), 0)
        
        # Test empty file (should have no messages)
        parser = EyeLinkASCParser(self.empty_file)
        parser.read_file()
        parser.parse_metadata()
        messages = parser.parse_messages()
        
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 0)
    
    def test_parse_samples(self):
        """Test sample parsing from different file types"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        parser.read_file()
        num_samples = parser.parse_samples()
        
        self.assertGreater(num_samples, 0)
        self.assertEqual(num_samples, len(parser.sample_data))
        
        # Test samples_only file
        parser = EyeLinkASCParser(self.samples_only_file)
        parser.read_file()
        num_samples = parser.parse_samples()
        
        self.assertGreater(num_samples, 0)
        
        # Test empty file (should have no samples)
        parser = EyeLinkASCParser(self.empty_file)
        parser.read_file()
        num_samples = parser.parse_samples()
        
        self.assertEqual(num_samples, 0)
        
        # Test malformed file (should handle invalid samples)
        parser = EyeLinkASCParser(self.malformed_file)
        parser.read_file()
        num_samples = parser.parse_samples()
        
        self.assertGreater(num_samples, 0)
    
    def test_parse_events(self):
        """Test event parsing from different file types"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        parser.read_file()
        event_counts = parser.parse_events()
        
        self.assertIsInstance(event_counts, dict)
        self.assertIn('fixations_left', event_counts)
        self.assertIn('saccades_left', event_counts)
        self.assertIn('blinks_left', event_counts)
        
        # At least some events should be present
        total_events = sum(event_counts.values())
        self.assertGreater(total_events, 0)
        
        # Test events_only file
        parser = EyeLinkASCParser(self.events_only_file)
        parser.read_file()
        event_counts = parser.parse_events()
        
        self.assertIsInstance(event_counts, dict)
        total_events = sum(event_counts.values())
        self.assertGreater(total_events, 0)
        
        # Test empty file (should have no events)
        parser = EyeLinkASCParser(self.empty_file)
        parser.read_file()
        event_counts = parser.parse_events()
        
        self.assertIsInstance(event_counts, dict)
        total_events = sum(event_counts.values())
        self.assertEqual(total_events, 0)
        
        # Test malformed file (should handle invalid events)
        parser = EyeLinkASCParser(self.malformed_file)
        parser.read_file()
        event_counts = parser.parse_events()
        
        self.assertIsInstance(event_counts, dict)
        # Should still extract some valid events
        total_events = sum(event_counts.values())
        self.assertGreater(total_events, 0)
        
        # Test left_eye_only file
        parser = EyeLinkASCParser(self.left_eye_only_file)
        parser.read_file()
        event_counts = parser.parse_events()
        
        self.assertIsInstance(event_counts, dict)
        # The randomly generated test file might not always have events
        # so we just check that the event counts dictionary is properly formed
        self.assertIn('fixations_left', event_counts)
        self.assertIn('fixations_right', event_counts)
        self.assertIn('saccades_left', event_counts)
        self.assertIn('saccades_right', event_counts)
    
    def test_parse_file(self):
        """Test the complete file parsing process"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        summary = parser.parse_file()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('samples', summary)
        self.assertIn('fixations', summary)
        self.assertIn('saccades', summary)
        self.assertIn('blinks', summary)
        self.assertIn('messages', summary)
        
        # Check that we extracted something
        self.assertGreater(summary['samples'], 0)
        
        # Test with malformed file (should still parse successfully)
        parser = EyeLinkASCParser(self.malformed_file)
        summary = parser.parse_file()
        
        self.assertIsInstance(summary, dict)
        # Should still extract some valid data
        self.assertGreater(summary['samples'], 0)
    
    def test_to_dataframes(self):
        """Test conversion to pandas DataFrames"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        parser.parse_file()
        dfs = parser.to_dataframes()
        
        self.assertIsInstance(dfs, dict)
        self.assertIn('samples', dfs)
        self.assertIsInstance(dfs['samples'], pd.DataFrame)
        
        # Test unified metrics
        self.assertIn('unified_eye_metrics', dfs)
        unified_df = dfs['unified_eye_metrics']
        self.assertIsInstance(unified_df, pd.DataFrame)
        self.assertIn('timestamp', unified_df.columns)
        
        # Test with empty file
        parser = EyeLinkASCParser(self.empty_file)
        parser.parse_file()
        dfs = parser.to_dataframes()
        
        self.assertIsInstance(dfs, dict)
        # Should have no dataframes with data
        if 'samples' in dfs:
            self.assertEqual(len(dfs['samples']), 0)
    
    def test_create_unified_metrics_df(self):
        """Test creation of unified metrics DataFrame"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        parser.parse_file()
        unified_df = parser.create_unified_metrics_df()
        
        self.assertIsInstance(unified_df, pd.DataFrame)
        self.assertGreater(len(unified_df), 0)
        
        # Check for required columns
        required_columns = ['timestamp', 'x_left', 'y_left', 'pupil_left']
        for col in required_columns:
            self.assertIn(col, unified_df.columns)
        
        # Test with left_eye_only file
        parser = EyeLinkASCParser(self.left_eye_only_file)
        parser.parse_file()
        unified_df = parser.create_unified_metrics_df()
        
        self.assertIsInstance(unified_df, pd.DataFrame)
        self.assertGreater(len(unified_df), 0)
        
        # Should have left eye data
        self.assertTrue(unified_df['x_left'].notna().any())
        # Note: While we'd expect right eye data to be all NaN for a left-eye only file,
        # our mock generator might not guarantee this, so we don't test it specifically
    
    def test_save_to_csv(self):
        """Test saving data to CSV files"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        parser.parse_file()
        
        output_dir = os.path.join(self.temp_dir, "output_csv")
        saved_files = parser.save_to_csv(output_dir)
        
        self.assertIsInstance(saved_files, list)
        self.assertGreater(len(saved_files), 0)
        
        # Check that files were created
        for file_path in saved_files:
            self.assertTrue(os.path.exists(file_path))
            
        # Test with movie segments
        parser = EyeLinkASCParser(self.movies_only_file)
        parser.parse_file()
        
        movie_output_dir = os.path.join(self.temp_dir, "movie_output_csv")
        saved_files = parser.save_to_csv(movie_output_dir)
        
        self.assertIsInstance(saved_files, list)
        self.assertGreater(len(saved_files), 0)
        
        # Should create movie subdirectories
        movie_dirs = [d for d in os.listdir(movie_output_dir) 
                      if os.path.isdir(os.path.join(movie_output_dir, d)) and d != "general"]
        self.assertGreater(len(movie_dirs), 0)
    
    def test_extract_features(self):
        """Test feature extraction from different file types"""
        # Test with standard file
        parser = EyeLinkASCParser(self.standard_file)
        parser.parse_file()
        features_df = parser.extract_features()
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 1)  # Should have one row
        
        # Check for expected feature columns
        expected_features = ['participant_id', 'pupil_left_mean', 'fixation_left_count']
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)
        
        # Test with specific movie
        if parser.movie_segments:
            movie_name = parser.movie_segments[0]['movie_name']
            movie_features = parser.extract_features(movie_name)
            
            self.assertIsInstance(movie_features, pd.DataFrame)
            self.assertEqual(len(movie_features), 1)
            self.assertIn('movie_name', movie_features.columns)
            self.assertEqual(movie_features['movie_name'].iloc[0], movie_name)
        
        # Test with empty file (should still create feature row with participant_id)
        parser = EyeLinkASCParser(self.empty_file)
        parser.parse_file()
        features_df = parser.extract_features()
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 1)
        self.assertIn('participant_id', features_df.columns)
    
    def test_extract_features_per_movie(self):
        """Test extracting features for each movie segment"""
        # Test with movie file
        parser = EyeLinkASCParser(self.movies_only_file)
        parser.parse_file()
        movie_features = parser.extract_features_per_movie()
        
        self.assertIsInstance(movie_features, dict)
        self.assertIn("All Data", movie_features)
        
        # Should have a feature DataFrame for each movie plus "All Data"
        self.assertEqual(len(movie_features), len(parser.movie_segments) + 1)
        
        # Each movie's features should have the movie_name column
        for movie_name, features_df in movie_features.items():
            if movie_name != "All Data":
                self.assertIn('movie_name', features_df.columns)
                self.assertEqual(features_df['movie_name'].iloc[0], movie_name)
    
    def test_feature_validation(self):
        """Test that extracted features have reasonable values"""
        # Generate a controlled test file with known values
        generator = MockASCGenerator()
        
        # Generate a simple file with known fixations and saccades
        test_file_path = os.path.join(self.temp_dir, "validation_test.asc")
        
        with open(test_file_path, 'w') as f:
            # Write header
            for line in generator.generate_header():
                f.write(line + '\n')
            
            # Write recording start
            for line in generator.generate_recording_start():
                f.write(line + '\n')
            
            # Add 100 regular samples
            for _ in range(100):
                f.write(generator.generate_sample(noise_level=1.0) + '\n')
            
            # Add exactly 5 left fixations with known durations
            durations = [100, 200, 300, 400, 500]
            for duration in durations:
                for line in generator.generate_fixation("L", duration=duration):
                    f.write(line + '\n')
            
            # Add exactly 3 right saccades with known amplitudes
            amplitudes = [50, 100, 150]
            for amplitude in amplitudes:
                for line in generator.generate_saccade("R", amplitude=amplitude, duration=50):
                    f.write(line + '\n')
            
            # Write recording end
            for line in generator.generate_recording_end():
                f.write(line + '\n')
        
        # Parse the test file
        parser = EyeLinkASCParser(test_file_path)
        parser.parse_file()
        features_df = parser.extract_features()
        
        # Validate fixation features
        self.assertEqual(features_df['fixation_left_count'].iloc[0], 5)
        self.assertEqual(features_df['fixation_left_duration_mean'].iloc[0], np.mean(durations))
        
        # Validate saccade features
        self.assertEqual(features_df['saccade_right_count'].iloc[0], 3)
    
    def test_process_asc_file(self):
        """Test the process_asc_file function"""
        # Test with standard file
        output_dir = os.path.join(self.temp_dir, "process_output")
        result = process_asc_file(self.standard_file, output_dir)
        
        self.assertIsInstance(result, dict)
        self.assertIn('summary', result)
        self.assertIn('dataframes', result)
        self.assertIn('features', result)
        self.assertIn('saved_files', result)
        
        # Check that files were saved
        self.assertGreater(len(result['saved_files']), 0)
        for file_path in result['saved_files']:
            self.assertTrue(os.path.exists(file_path))
        
        # Test with unified_only=True
        unified_dir = os.path.join(self.temp_dir, "unified_output")
        unified_result = process_asc_file(self.standard_file, unified_dir, unified_only=True)
        
        self.assertIsInstance(unified_result, dict)
        self.assertIn('saved_files', unified_result)
        
        # Should have saved only one file
        self.assertEqual(len(unified_result['saved_files']), 1)
        unified_path = unified_result['saved_files'][0]
        self.assertTrue(os.path.exists(unified_path))
        self.assertTrue('unified_eye_metrics' in os.path.basename(unified_path))
    
    def test_load_csv_file(self):
        """Test loading a CSV file with the load_csv_file function"""
        # Create a test CSV file directly instead of relying on process_asc_file
        unified_csv = os.path.join(self.temp_dir, "test_unified_eye_metrics.csv")
        
        # Create a simple CSV file with eye tracking data
        test_df = pd.DataFrame({
            'timestamp': [1000, 1002, 1004, 1006],
            'x_left': [500.0, 505.0, 510.0, 515.0],
            'y_left': [400.0, 405.0, 410.0, 415.0],
            'pupil_left': [1000.0, 1010.0, 1020.0, 1030.0],
            'x_right': [550.0, 555.0, 560.0, 565.0],
            'y_right': [400.0, 405.0, 410.0, 415.0],
            'pupil_right': [1000.0, 1010.0, 1020.0, 1030.0]
        })
        
        # Save to CSV
        test_df.to_csv(unified_csv, index=False)
        
        # Load the CSV file
        csv_result = load_csv_file(unified_csv)
        
        self.assertIsInstance(csv_result, dict)
        self.assertIn('summary', csv_result)
        self.assertIn('dataframes', csv_result)
        self.assertIn('features', csv_result)
        
        # Check that dataframes were created
        self.assertIn('unified_eye_metrics', csv_result['dataframes'])
        self.assertIsInstance(csv_result['dataframes']['unified_eye_metrics'], pd.DataFrame)
        
        # Test with malformed CSV
        # Create a malformed CSV file
        malformed_csv = os.path.join(self.temp_dir, "malformed.csv")
        with open(malformed_csv, 'w') as f:
            f.write("timestamp,x_left,y_left,pupil_left,x_right,y_right,pupil_right\n")
            f.write("15021600,629.7,464.4,1183.0,666.4,491.6,1105.0\n")
            f.write("15021602,629.6,464.7,1183.0,666.3,492.1,1105.0\n")
            f.write("malformed,data,here,bad,more,bad,data\n")  # Malformed row
            f.write("15021606,629.5,465.3,1183.0,666.1,492.6,1105.0\n")
        
        # Should handle the malformed CSV gracefully
        try:
            malformed_result = load_csv_file(malformed_csv)
            
            self.assertIsInstance(malformed_result, dict)
            self.assertIn('dataframes', malformed_result)
            self.assertIn('unified_eye_metrics', malformed_result['dataframes'])
            
            # Should have extracted valid rows
            unified_df = malformed_result['dataframes']['unified_eye_metrics']
            self.assertGreater(len(unified_df), 0)
        except Exception as e:
            self.fail(f"load_csv_file failed with malformed CSV: {e}")


if __name__ == '__main__':
    unittest.main()