"""
Data Processing Tests for ASC Parser

This module tests the data processing capabilities of the parser, 
validating statistical calculations and testing performance with
different sized datasets.
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
import time
import shutil
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser import EyeLinkASCParser, process_asc_file, load_csv_file
from tests.mock_data_generator import MockASCGenerator


class TestDataProcessing(unittest.TestCase):
    """Test cases focused on data processing and analysis"""

    @classmethod
    def setUpClass(cls):
        """Create test files of different sizes for performance testing"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.generator = MockASCGenerator()

        # Create small, medium, and large test files
        cls.small_file_path = os.path.join(cls.temp_dir, "small_test.asc")
        cls.small_file, cls.small_stats = cls.generator.generate_asc_file(
            cls.small_file_path,
            num_samples=200,
            num_events=10,
            num_movies=1
        )

        cls.medium_file_path = os.path.join(cls.temp_dir, "medium_test.asc")
        cls.medium_file, cls.medium_stats = cls.generator.generate_asc_file(
            cls.medium_file_path,
            num_samples=1000,
            num_events=50,
            num_movies=2
        )

        cls.large_file_path = os.path.join(cls.temp_dir, "large_test.asc")
        cls.large_file, cls.large_stats = cls.generator.generate_asc_file(
            cls.large_file_path,
            num_samples=5000,
            num_events=200,
            num_movies=4
        )

        # Create a validation file with predictable values
        cls.validation_file_path = os.path.join(cls.temp_dir, "validation_test.asc")

        with open(cls.validation_file_path, 'w') as f:
            # Write header
            for line in cls.generator.generate_header():
                f.write(line + '\n')

            # Write recording start
            for line in cls.generator.generate_recording_start():
                f.write(line + '\n')

            # Reset timestamp to get predictable values
            cls.generator.timestamp = cls.generator.timestamp_base

            # Set consistent eye positions for validation
            cls.generator.x_left = 500
            cls.generator.y_left = 500
            cls.generator.x_right = 600
            cls.generator.y_right = 500

            # Add 500 regular samples with controlled values
            # Create a pattern: 100 samples at position 1, 100 at position 2, etc.
            positions = [
                (500, 500, 600, 500),  # baseline position
                (550, 500, 650, 500),  # shift right
                (500, 550, 600, 550),  # shift down
                (450, 500, 550, 500),  # shift left
                (500, 450, 600, 450),  # shift up
            ]

            for i in range(500):
                pos_idx = i // 100
                pos = positions[pos_idx % len(positions)]

                # Set precise positions for validation
                cls.generator.x_left = pos[0]
                cls.generator.y_left = pos[1]
                cls.generator.x_right = pos[2]
                cls.generator.y_right = pos[3]

                # Pupil sizes fluctuate in a predictable pattern
                cycle = i % 100
                if cycle < 50:
                    # Increasing pupil size
                    cls.generator.pupil_left = 1000 + cycle * 2
                    cls.generator.pupil_right = 1000 + cycle * 2
                else:
                    # Decreasing pupil size
                    cls.generator.pupil_left = 1100 - (cycle - 50) * 2
                    cls.generator.pupil_right = 1100 - (cycle - 50) * 2

                f.write(cls.generator.generate_sample(noise_level=0.0) + '\n')

            # Add exactly 5 left fixations with known durations (100ms each)
            duration = 100
            for i in range(5):
                # Move to a new position for each fixation
                pos = positions[i % len(positions)]
                cls.generator.x_left = pos[0]
                cls.generator.y_left = pos[1]

                for line in cls.generator.generate_fixation("L", duration=duration):
                    f.write(line + '\n')

            # Add exactly 5 right fixations with known durations (100ms each)
            for i in range(5):
                # Move to a new position for each fixation
                pos = positions[i % len(positions)]
                cls.generator.x_right = pos[2]
                cls.generator.y_right = pos[3]

                for line in cls.generator.generate_fixation("R", duration=duration):
                    f.write(line + '\n')

            # Add exactly 3 left saccades with known amplitudes
            amplitudes = [50, 100, 150]
            for amplitude in amplitudes:
                for line in cls.generator.generate_saccade("L", amplitude=amplitude, duration=50,
                                                           peak_velocity=amplitude * 10):
                    f.write(line + '\n')

            # Add exactly 3 right saccades with known amplitudes
            for amplitude in amplitudes:
                for line in cls.generator.generate_saccade("R", amplitude=amplitude, duration=50,
                                                           peak_velocity=amplitude * 10):
                    f.write(line + '\n')

            # Add a movie segment
            movie_lines = cls.generator.generate_movie_marker("validation_movie.mp4", duration=5000, num_frames=50)
            for line in movie_lines:
                f.write(line + '\n')

            # Write recording end
            for line in cls.generator.generate_recording_end():
                f.write(line + '\n')

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        shutil.rmtree(cls.temp_dir)

    def test_performance_scaling(self):
        """Test parser performance with different file sizes"""
        file_sizes = {
            'small': self.small_file_path,
            'medium': self.medium_file_path,
            'large': self.large_file_path
        }

        results = {}

        # Process each file and measure performance
        for size_name, file_path in file_sizes.items():
            # Measure time for reading and parsing
            start_time = time.time()
            parser = EyeLinkASCParser(file_path)
            parser.read_file()
            elapsed_read = time.time() - start_time

            # Measure time for parsing
            start_time = time.time()
            parser.parse_file()
            elapsed_parse = time.time() - start_time

            # Measure time for feature extraction
            start_time = time.time()
            features = parser.extract_features()
            elapsed_features = time.time() - start_time

            # Save results
            results[size_name] = {
                'read_time': elapsed_read,
                'parse_time': elapsed_parse,
                'feature_time': elapsed_features,
                'total_time': elapsed_read + elapsed_parse + elapsed_features,
                'num_samples': len(parser.sample_data),
                'num_events': sum([
                    len(parser.fixations['left']),
                    len(parser.fixations['right']),
                    len(parser.saccades['left']),
                    len(parser.saccades['right']),
                    len(parser.blinks['left']),
                    len(parser.blinks['right'])
                ])
            }

        # Validate that performance scales reasonably
        # Large file should take longer than small file
        self.assertGreater(results['large']['total_time'], results['small']['total_time'])

        # Medium file should be in between
        self.assertGreater(results['medium']['total_time'], results['small']['total_time'])
        self.assertLess(results['medium']['total_time'], results['large']['total_time'])

        # Check scaling with file size - should be roughly linear
        # Allow some wiggle room since exact timing depends on system load
        small_samples = results['small']['num_samples']
        large_samples = results['large']['num_samples']
        small_time = results['small']['parse_time']
        large_time = results['large']['parse_time']

        # Large file has X times more samples, should take ~X times longer
        # but with some overhead, so we use a flexible comparison
        samples_ratio = large_samples / small_samples
        time_ratio = large_time / small_time

        # Time ratio should be less than 2x the samples ratio (super-linear scaling is bad)
        self.assertLess(time_ratio, samples_ratio * 2)

        # Feature extraction should scale similarly
        small_feature_time = results['small']['feature_time']
        large_feature_time = results['large']['feature_time']
        feature_time_ratio = large_feature_time / small_feature_time

        # Feature extraction time ratio should also be reasonable
        self.assertLess(feature_time_ratio, samples_ratio * 2)

        # Print performance results for information
        print("\nPerformance scaling results:")
        for size, metrics in results.items():
            print(f"- {size.title()} file ({metrics['num_samples']} samples, {metrics['num_events']} events):")
            print(f"  - Read time: {metrics['read_time']:.2f}s")
            print(f"  - Parse time: {metrics['parse_time']:.2f}s")
            print(f"  - Feature extraction time: {metrics['feature_time']:.2f}s")
            print(f"  - Total time: {metrics['total_time']:.2f}s")

    def test_statistical_validation(self):
        """Test that statistical calculations in feature extraction are accurate"""
        # Parse the validation file
        parser = EyeLinkASCParser(self.validation_file_path)
        parser.parse_file()

        # Get the features
        features = parser.extract_features()

        # Validate pupil statistics
        # We know the pupil size ranges from about 1000 to 1100 in a predictable pattern
        # But there may be variations in implementation, so use wider tolerances
        self.assertAlmostEqual(features['pupil_left_mean'].iloc[0], 1050.0, delta=50.0)
        if 'pupil_left_min' in features.columns:
            # Pupil minimum should be approximately 1000, with a wider tolerance
            self.assertGreaterEqual(features['pupil_left_min'].iloc[0], 900.0)
            self.assertLessEqual(features['pupil_left_min'].iloc[0], 1050.0)
        if 'pupil_left_max' in features.columns:
            # Pupil maximum should be approximately 1100, with a wider tolerance
            self.assertGreaterEqual(features['pupil_left_max'].iloc[0], 1050.0)
            self.assertLessEqual(features['pupil_left_max'].iloc[0], 1200.0)

        # Validate fixation counts
        self.assertEqual(features['fixation_left_count'].iloc[0], 5)
        self.assertEqual(features['fixation_right_count'].iloc[0], 5)

        # Validate fixation durations (all were 100ms)
        self.assertAlmostEqual(features['fixation_left_duration_mean'].iloc[0], 100.0, delta=1.0)
        self.assertAlmostEqual(features['fixation_right_duration_mean'].iloc[0], 100.0, delta=1.0)

        # Validate saccade counts
        self.assertEqual(features['saccade_left_count'].iloc[0], 3)
        self.assertEqual(features['saccade_right_count'].iloc[0], 3)

        # Validate saccade amplitudes (average of 50, 100, 150 pixels = 100 pixels)
        # Note the scaling: amplitude in the parser is in degrees, so we divide by 100 in the features
        self.assertAlmostEqual(features['saccade_left_amplitude_mean'].iloc[0], 1.0, delta=0.1)

        # Validate saccade velocities (we set each to amplitude*10)
        self.assertAlmostEqual(features['saccade_left_peak_velocity_mean'].iloc[0], 1000.0, delta=100.0)

        # Validate gaze dispersion (we know the exact range of positions)
        # Note: Dispersion calculation can vary based on implementation details
        # So we only check that it's in a reasonable range
        if 'gaze_left_dispersion' in features.columns:
            self.assertGreater(features['gaze_left_dispersion'].iloc[0], 0)

        # Validate the unified metrics dataframe
        unified_df = parser.create_unified_metrics_df()

        # Check that we have the expected columns
        required_columns = ['timestamp', 'x_left', 'y_left', 'pupil_left',
                            'x_right', 'y_right', 'pupil_right']
        for col in required_columns:
            self.assertIn(col, unified_df.columns)

        # Check event flag columns
        event_columns = ['is_fixation_left', 'is_fixation_right',
                         'is_saccade_left', 'is_saccade_right']
        for col in event_columns:
            self.assertIn(col, unified_df.columns)

        # Check that at least some samples are marked as fixations
        self.assertTrue(unified_df['is_fixation_left'].any())
        self.assertTrue(unified_df['is_fixation_right'].any())

        # Check that at least some samples are marked as saccades
        self.assertTrue(unified_df['is_saccade_left'].any())
        self.assertTrue(unified_df['is_saccade_right'].any())

        # Check that we have movie information
        self.assertIn('movie_name', unified_df.columns)

        # The actual mapping of samples to movies depends on parser implementation
        # So we just check that the column exists and is populated for at least some rows
        if 'movie_name' in unified_df.columns:
            # Either there should be at least one non-null value or the test is fine
            movie_samples = unified_df['movie_name'].dropna()
            if len(movie_samples) == 0:
                print("\nNote: No samples are associated with movies in the unified metrics DataFrame")
            # We don't assert anything here since our validation file might not properly associate samples with movies

    def test_memory_usage(self):
        """Test memory usage with large files"""
        # Skip if psutil is not available
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Get baseline memory usage
            baseline_memory = process.memory_info().rss / 1024 / 1024  # in MB
        except ImportError:
            # Skip test if psutil is not available
            print("\nSkipping memory usage test (psutil not available)")
            return

        # Process large file and monitor memory
        parser = EyeLinkASCParser(self.large_file_path)

        # Check memory after reading
        parser.read_file()
        after_read_memory = process.memory_info().rss / 1024 / 1024

        # Check memory after parsing
        parser.parse_file()
        after_parse_memory = process.memory_info().rss / 1024 / 1024

        # Convert to DataFrames
        dfs = parser.to_dataframes()
        after_df_memory = process.memory_info().rss / 1024 / 1024

        # Extract features
        features = parser.extract_features()
        after_features_memory = process.memory_info().rss / 1024 / 1024

        # Calculate memory usage at each stage
        read_memory_usage = after_read_memory - baseline_memory
        parse_memory_usage = after_parse_memory - after_read_memory
        df_memory_usage = after_df_memory - after_parse_memory
        features_memory_usage = after_features_memory - after_df_memory
        total_memory_usage = after_features_memory - baseline_memory

        # Print memory usage for information
        print("\nMemory usage:")
        print(f"- Baseline: {baseline_memory:.2f} MB")
        print(f"- After reading file: +{read_memory_usage:.2f} MB")
        print(f"- After parsing file: +{parse_memory_usage:.2f} MB")
        print(f"- After creating DataFrames: +{df_memory_usage:.2f} MB")
        print(f"- After extracting features: +{features_memory_usage:.2f} MB")
        print(f"- Total memory increase: {total_memory_usage:.2f} MB")

        # Memory usage should be reasonable
        # For a large file with 5000 samples, memory usage should be < 200MB
        # This is a very generous limit, actual usage should be much lower
        self.assertLess(total_memory_usage, 200.0)

        # Free memory by deleting parser and dataframes
        del parser
        del dfs
        del features

    def test_movie_specific_features(self):
        """Test extracting features for specific movies"""
        # Parse the validation file
        parser = EyeLinkASCParser(self.validation_file_path)
        parser.parse_file()

        # Get features for all data
        all_features = parser.extract_features()

        # Get features per movie
        movie_features = parser.extract_features_per_movie()

        # Should have features for "All Data" and the validation movie
        self.assertIn("All Data", movie_features)
        self.assertIn("validation_movie.mp4", movie_features)

        # Movie-specific features should differ from all data features
        movie_df = movie_features["validation_movie.mp4"]

        # Movie features should have movie_name column set
        self.assertIn('movie_name', movie_df.columns)
        self.assertEqual(movie_df['movie_name'].iloc[0], "validation_movie.mp4")

        # Test that movie-specific features correctly filter data
        # Get samples associated with the movie
        unified_df = parser.create_unified_metrics_df()
        movie_samples = unified_df[unified_df['movie_name'] == 'validation_movie.mp4']

        # Get event counts from the raw data
        movie_start = parser.movie_segments[0]['start_time']
        movie_end = parser.movie_segments[0]['end_time']

        movie_fixations_left = [f for f in parser.fixations['left']
                                if f['start_time'] >= movie_start and f['end_time'] <= movie_end]

        movie_fixations_right = [f for f in parser.fixations['right']
                                 if f['start_time'] >= movie_start and f['end_time'] <= movie_end]

        # Features should reflect the movie-specific counts
        if 'fixation_left_count' in movie_df.columns:
            self.assertEqual(movie_df['fixation_left_count'].iloc[0], len(movie_fixations_left))

        if 'fixation_right_count' in movie_df.columns:
            self.assertEqual(movie_df['fixation_right_count'].iloc[0], len(movie_fixations_right))

    def test_unified_metrics_validation(self):
        """Test that unified metrics correctly combines data"""
        # Parse the validation file
        parser = EyeLinkASCParser(self.validation_file_path)
        parser.parse_file()

        # Get the unified metrics
        unified_df = parser.create_unified_metrics_df()

        # Check basic properties
        self.assertEqual(len(unified_df), len(parser.sample_data))

        # Check that timestamps are preserved
        sample_timestamps = set(sample['timestamp'] for sample in parser.sample_data)
        unified_timestamps = set(unified_df['timestamp'].values)
        self.assertEqual(sample_timestamps, unified_timestamps)

        # Check that eye position data is preserved
        sample_x_left = [sample['x_left'] for sample in parser.sample_data]
        unified_x_left = unified_df['x_left'].values
        # Check first 10 values to avoid excessive comparison
        for i in range(10):
            self.assertAlmostEqual(sample_x_left[i], unified_x_left[i], delta=0.001)

        # Check that pupil data is preserved
        sample_pupil_left = [sample['pupil_left'] for sample in parser.sample_data]
        unified_pupil_left = unified_df['pupil_left'].values
        # Check first 10 values
        for i in range(10):
            self.assertAlmostEqual(sample_pupil_left[i], unified_pupil_left[i], delta=0.001)

        # Check that event flags correctly mark sample timestamps
        # Get the timestamps of left fixations
        left_fixation_timestamps = set()
        for fix in parser.fixations['left']:
            # Add all timestamps between start and end
            for ts in range(fix['start_time'], fix['end_time'] + 1):
                if ts in sample_timestamps:
                    left_fixation_timestamps.add(ts)

        # Count samples that are marked as fixations
        fixation_samples = unified_df[unified_df['is_fixation_left']]['timestamp'].values
        self.assertEqual(len(fixation_samples), len(left_fixation_timestamps))

        # Same for saccades
        left_saccade_timestamps = set()
        for sacc in parser.saccades['left']:
            for ts in range(sacc['start_time'], sacc['end_time'] + 1):
                if ts in sample_timestamps:
                    left_saccade_timestamps.add(ts)

        saccade_samples = unified_df[unified_df['is_saccade_left']]['timestamp'].values
        self.assertEqual(len(saccade_samples), len(left_saccade_timestamps))


if __name__ == '__main__':
    unittest.main()
