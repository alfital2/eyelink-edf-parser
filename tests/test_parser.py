import unittest
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

# Add parent directory to path to import parser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser import EyeLinkASCParser, process_asc_file

TEST_FILE_PATH = os.path.join(os.path.dirname(__file__), "asc_files/sample_test.asc")

class TestEyeLinkASCParser(unittest.TestCase):
    """
    Comprehensive test suite for the EyeLinkASCParser.
    Tests validate exact counts and values from the ASC file.
    """
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Check if a specific ASC file was passed via environment variable
        # Parse the file once for all tests
        cls.parser = EyeLinkASCParser(TEST_FILE_PATH)
        cls.parser.parse_file()

        # Generate a gaze plot for visual verification
        cls.generate_gaze_plot()

    @classmethod
    def generate_gaze_plot(cls):
        """Generate a gaze path plot to visualize the data."""
        samples_df = pd.DataFrame(cls.parser.sample_data)

        # Create output directory
        tests_dir = os.path.dirname(__file__)
        output_dir = os.path.join(tests_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "gaze_positions.png")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot gaze positions for left eye if available
        if 'x_left' in samples_df.columns and 'y_left' in samples_df.columns:
            left_valid = ~samples_df['x_left'].isna() & ~samples_df['y_left'].isna()
            if any(left_valid):
                ax.scatter(samples_df.loc[left_valid, 'x_left'], samples_df.loc[left_valid, 'y_left'],
                           s=2, c='blue', alpha=0.5, label='Left eye')

        # Plot gaze positions for right eye if available
        if 'x_right' in samples_df.columns and 'y_right' in samples_df.columns:
            right_valid = ~samples_df['x_right'].isna() & ~samples_df['y_right'].isna()
            if any(right_valid):
                ax.scatter(samples_df.loc[right_valid, 'x_right'], samples_df.loc[right_valid, 'y_right'],
                           s=2, c='red', alpha=0.5, label='Right eye')

        # Highlight fixations for left eye
        if cls.parser.fixations['left']:
            fixations_left = pd.DataFrame(cls.parser.fixations['left'])
            if 'x' in fixations_left.columns and 'y' in fixations_left.columns:
                fix_valid = ~fixations_left['x'].isna() & ~fixations_left['y'].isna()
                if any(fix_valid):
                    ax.scatter(fixations_left.loc[fix_valid, 'x'], fixations_left.loc[fix_valid, 'y'],
                               s=50, marker='o', edgecolor='blue', facecolor='none', linewidth=2,
                               label='Left fixations')

        # Highlight fixations for right eye
        if cls.parser.fixations['right']:
            fixations_right = pd.DataFrame(cls.parser.fixations['right'])
            if 'x' in fixations_right.columns and 'y' in fixations_right.columns:
                fix_valid = ~fixations_right['x'].isna() & ~fixations_right['y'].isna()
                if any(fix_valid):
                    ax.scatter(fixations_right.loc[fix_valid, 'x'], fixations_right.loc[fix_valid, 'y'],
                               s=50, marker='o', edgecolor='red', facecolor='none', linewidth=2,
                               label='Right fixations')

        # Add frame markers if available
        if cls.parser.frame_markers:
            frame_df = pd.DataFrame(cls.parser.frame_markers)
            for _, row in frame_df.iterrows():
                ax.annotate(f"Frame #{int(row['frame'])}",
                            xy=(0.05, 0.95 - (row['frame'] - 1) * 0.05),
                            xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Set title and labels
        ax.set_title('Gaze Positions')
        ax.set_xlabel('X position (pixels)')
        ax.set_ylabel('Y position (pixels)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Invert Y-axis (0,0 is top-left in screen coordinates)
        ax.invert_yaxis()

        # Add metadata annotation
        metadata_text = f"File: {os.path.basename(TEST_FILE_PATH)}\n"
        metadata_text += f"Samples: {len(samples_df)}\n"
        metadata_text += f"Left fixations: {len(cls.parser.fixations['left'])}\n"
        metadata_text += f"Right fixations: {len(cls.parser.fixations['right'])}\n"
        metadata_text += f"Left saccades: {len(cls.parser.saccades['left'])}\n"
        metadata_text += f"Right saccades: {len(cls.parser.saccades['right'])}\n"
        metadata_text += f"Left blinks: {len(cls.parser.blinks['left'])}\n"
        metadata_text += f"Right blinks: {len(cls.parser.blinks['right'])}\n"

        ax.text(0.02, 0.02, metadata_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Generated gaze plot: {output_path}")

    def test_file_reading(self):
        """Test that the file is read correctly with the right number of lines."""
        parser = EyeLinkASCParser(TEST_FILE_PATH)
        num_lines = parser.read_file()

        # Count the lines in the actual file to compare
        with open(os.path.join(os.path.dirname(__file__), "asc_files/sample_test.asc"), 'r') as f:
            actual_lines = len(f.readlines())

        self.assertEqual(num_lines, actual_lines,
                         f"Parser reported {num_lines} lines but file has {actual_lines} lines")

    def test_metadata_extraction(self):
        """Test that metadata is extracted correctly from the file header."""
        metadata = self.parser.metadata

        # Check specific metadata values from the provided file
        self.assertEqual(metadata['TYPE'], 'EDF_FILE BINARY EVENT SAMPLE TAGGED',
                         "Incorrect TYPE metadata")
        self.assertEqual(metadata['VERSION'], 'EYELINK II 1',
                         "Incorrect VERSION metadata")
        self.assertEqual(metadata['CAMERA'], 'Eyelink GL Version 1.2 Sensor=AH7',
                         "Incorrect CAMERA metadata")
        self.assertEqual(metadata['SERIAL NUMBER'], 'CLG-TEST28',
                         "Incorrect SERIAL NUMBER metadata")

    def test_message_parsing(self):
        """Test that messages are parsed correctly with the right count and content."""
        messages = self.parser.messages

        # Check exact message counts
        all_messages = len(messages)
        self.assertEqual(all_messages, 64, f"Expected 49 messages, found {all_messages}")

        # Count specific message types
        cmd_messages = [m for m in messages if '!CMD' in m['content']]
        cal_messages = [m for m in messages if '!CAL' in m['content']]
        frame_messages = [m for m in messages if 'Play_Movie_Start FRAME' in m['content']]

        # Verify counts based on the provided file
        self.assertEqual(len(cmd_messages), 4,
                         f"Expected 4 CMD messages, found {len(cmd_messages)}")
        self.assertEqual(len(cal_messages), 15,
                         f"Expected 15 CAL messages, found {len(cal_messages)}")
        self.assertEqual(len(frame_messages), 3,
                         f"Expected 3 frame markers, found {len(frame_messages)}")

        # Verify the frame numbers in frame markers
        frame_numbers = sorted([int(m['content'].split('#')[1]) for m in frame_messages])
        expected_frames = [1, 2, 3]
        self.assertEqual(frame_numbers, expected_frames,
                         f"Expected frames {expected_frames}, found {frame_numbers}")

        # Check for specific messages
        movie_file_msg = next((m for m in messages if 'Movie File Name' in m['content']), None)
        self.assertIsNotNone(movie_file_msg, "Movie file name message not found")
        self.assertIn('Test_Movie.xvd', movie_file_msg['content'],
                      "Incorrect movie file name")

    def test_sample_parsing(self):
        """Test parsing of eye movement samples."""
        samples = self.parser.sample_data

        # Verify sample count from the provided file
        self.assertEqual(len(samples), 38, f"Expected 55 samples, found {len(samples)}")

        # Check first sample values
        first_sample = samples[0]
        self.assertEqual(first_sample['timestamp'], 15021600,
                         "First sample timestamp is incorrect")
        self.assertAlmostEqual(first_sample['x_left'], 629.7, 1,
                               "First sample x_left is incorrect")
        self.assertAlmostEqual(first_sample['y_left'], 464.4, 1,
                               "First sample y_left is incorrect")
        self.assertAlmostEqual(first_sample['pupil_left'], 1183.0, 1,
                               "First sample pupil_left is incorrect")
        self.assertAlmostEqual(first_sample['x_right'], 666.4, 1,
                               "First sample x_right is incorrect")
        self.assertAlmostEqual(first_sample['y_right'], 491.6, 1,
                               "First sample y_right is incorrect")
        self.assertAlmostEqual(first_sample['pupil_right'], 1105.0, 1,
                               "First sample pupil_right is incorrect")

    def test_fixation_parsing(self):
        """Test parsing of fixation events."""
        left_fixations = self.parser.fixations['left']
        right_fixations = self.parser.fixations['right']

        # Verify exact fixation counts from the provided file
        self.assertEqual(len(left_fixations), 2,
                         f"Expected 2 left fixations, found {len(left_fixations)}")
        self.assertEqual(len(right_fixations), 2,
                         f"Expected 2 right fixations, found {len(right_fixations)}")

        # Verify first left fixation
        if left_fixations:
            first_fix = left_fixations[0]
            self.assertEqual(first_fix['start_time'], 15021610,
                             "First left fixation start time is incorrect")
            self.assertEqual(first_fix['end_time'], 15021630,
                             "First left fixation end time is incorrect")
            self.assertEqual(first_fix['duration'], 20,
                             "First left fixation duration is incorrect")
            self.assertAlmostEqual(first_fix['x'], 630.5, 1,
                                   "First left fixation x position is incorrect")
            self.assertAlmostEqual(first_fix['y'], 465.7, 1,
                                   "First left fixation y position is incorrect")

        # Verify first right fixation
        if right_fixations:
            first_fix = right_fixations[0]
            self.assertEqual(first_fix['start_time'], 15021610,
                             "First right fixation start time is incorrect")
            self.assertEqual(first_fix['end_time'], 15021630,
                             "First right fixation end time is incorrect")
            self.assertEqual(first_fix['duration'], 20,
                             "First right fixation duration is incorrect")
            self.assertAlmostEqual(first_fix['x'], 666.0, 1,
                                   "First right fixation x position is incorrect")
            self.assertAlmostEqual(first_fix['y'], 490.5, 1,
                                   "First right fixation y position is incorrect")

    def test_saccade_parsing(self):
        """Test parsing of saccade events."""
        left_saccades = self.parser.saccades['left']
        right_saccades = self.parser.saccades['right']

        # Verify exact saccade counts from the provided file
        self.assertEqual(len(left_saccades), 1,
                         f"Expected 1 left saccade, found {len(left_saccades)}")
        self.assertEqual(len(right_saccades), 1,
                         f"Expected 1 right saccade, found {len(right_saccades)}")

        # Verify left saccade
        if left_saccades:
            saccade = left_saccades[0]
            self.assertEqual(saccade['start_time'], 15021650,
                             "Left saccade start time is incorrect")
            self.assertEqual(saccade['end_time'], 15021656,
                             "Left saccade end time is incorrect")
            self.assertEqual(saccade['duration'], 6,
                             "Left saccade duration is incorrect")
            self.assertAlmostEqual(saccade['start_x'], 628.0, 1,
                                   "Left saccade start_x is incorrect")
            self.assertAlmostEqual(saccade['start_y'], 464.0, 1,
                                   "Left saccade start_y is incorrect")
            self.assertAlmostEqual(saccade['end_x'], 618.0, 1,
                                   "Left saccade end_x is incorrect")
            self.assertAlmostEqual(saccade['end_y'], 458.0, 1,
                                   "Left saccade end_y is incorrect")
            self.assertAlmostEqual(saccade['amplitude'], 0.35, 2,
                                   "Left saccade amplitude is incorrect")

        # Verify right saccade
        if right_saccades:
            saccade = right_saccades[0]
            self.assertEqual(saccade['start_time'], 15021660,
                             "Right saccade start time is incorrect")
            self.assertEqual(saccade['end_time'], 15021664,
                             "Right saccade end time is incorrect")
            self.assertEqual(saccade['duration'], 4,
                             "Right saccade duration is incorrect")
            self.assertAlmostEqual(saccade['start_x'], 650.0, 1,
                                   "Right saccade start_x is incorrect")
            self.assertAlmostEqual(saccade['start_y'], 480.0, 1,
                                   "Right saccade start_y is incorrect")
            self.assertAlmostEqual(saccade['end_x'], 640.0, 1,
                                   "Right saccade end_x is incorrect")
            self.assertAlmostEqual(saccade['end_y'], 476.0, 1,
                                   "Right saccade end_y is incorrect")
            self.assertAlmostEqual(saccade['amplitude'], 0.25, 2,
                                   "Right saccade amplitude is incorrect")

    def test_blink_parsing(self):
        """Test parsing of blink events."""
        left_blinks = self.parser.blinks['left']
        right_blinks = self.parser.blinks['right']

        # Verify exact blink counts from the provided file
        self.assertEqual(len(left_blinks), 1,
                         f"Expected 1 left blink, found {len(left_blinks)}")
        self.assertEqual(len(right_blinks), 1,
                         f"Expected 1 right blink, found {len(right_blinks)}")

        # Verify left blink
        if left_blinks:
            blink = left_blinks[0]
            self.assertEqual(blink['start_time'], 15021670,
                             "Left blink start time is incorrect")
            self.assertEqual(blink['end_time'], 15021676,
                             "Left blink end time is incorrect")
            self.assertEqual(blink['duration'], 6,
                             "Left blink duration is incorrect")

        # Verify right blink
        if right_blinks:
            blink = right_blinks[0]
            self.assertEqual(blink['start_time'], 15021680,
                             "Right blink start time is incorrect")
            self.assertEqual(blink['end_time'], 15021684,
                             "Right blink end time is incorrect")
            self.assertEqual(blink['duration'], 4,
                             "Right blink duration is incorrect")

    def test_frame_marker_parsing(self):
        """Test parsing of video frame markers."""
        frame_markers = self.parser.frame_markers

        # Verify exact frame marker count from the provided file
        self.assertEqual(len(frame_markers), 3,
                         f"Expected 3 frame markers, found {len(frame_markers)}")

        # Verify frame numbers and timestamps
        expected_frames = [(15021620, 1), (15021640, 2), (15021690, 3)]

        for i, (expected_time, expected_frame) in enumerate(expected_frames):
            self.assertEqual(frame_markers[i]['timestamp'], expected_time,
                             f"Frame {i + 1} timestamp mismatch")
            self.assertEqual(frame_markers[i]['frame'], expected_frame,
                             f"Frame {i + 1} number mismatch")

    def test_dataframe_conversion(self):
        """Test conversion of parsed data to pandas DataFrames."""
        dataframes = self.parser.to_dataframes()

        # Check that key dataframes exist and have correct sizes
        self.assertIn('samples', dataframes, "Samples DataFrame not created")
        self.assertEqual(len(dataframes['samples']), 38,
                         "Samples DataFrame has incorrect size")

        self.assertIn('fixations_left', dataframes, "Left fixations DataFrame not created")
        self.assertEqual(len(dataframes['fixations_left']), 2,
                         "Left fixations DataFrame has incorrect size")

        self.assertIn('fixations_right', dataframes, "Right fixations DataFrame not created")
        self.assertEqual(len(dataframes['fixations_right']), 2,
                         "Right fixations DataFrame has incorrect size")

        self.assertIn('messages', dataframes, "Messages DataFrame not created")
        self.assertEqual(len(dataframes['messages']), 64,
                         "Messages DataFrame has incorrect size")

        self.assertIn('frames', dataframes, "Frames DataFrame not created")
        self.assertEqual(len(dataframes['frames']), 3,
                         "Frames DataFrame has incorrect size")

    def test_unified_metrics_df(self):
        """Test creation of the unified eye metrics DataFrame."""
        unified_df = self.parser.create_unified_metrics_df()

        # Check that unified_df has the correct size
        self.assertEqual(len(unified_df), 38,
                         "Unified metrics DataFrame has incorrect size")

        # Verify event markers in the unified df
        fixation_samples_left = unified_df['is_fixation_left'].sum()
        fixation_samples_right = unified_df['is_fixation_right'].sum()

        # Check that the number of samples marked as fixations is correct
        # First left fixation: 20ms, second: 6ms at 500Hz = ~13 samples
        self.assertGreaterEqual(fixation_samples_left, 12,
                                "Too few samples marked as left fixations")
        self.assertLessEqual(fixation_samples_left, 14,
                             "Too many samples marked as left fixations")

        # First right fixation: 20ms, second: 6ms at 500Hz = ~13 samples
        self.assertGreaterEqual(fixation_samples_right, 12,
                                "Too few samples marked as right fixations")
        self.assertLessEqual(fixation_samples_right, 14,
                             "Too many samples marked as right fixations")

    def test_feature_extraction(self):
        """Test extraction of aggregate features for machine learning."""
        features_df = self.parser.extract_features()

        # Check that the feature df has one row
        self.assertEqual(len(features_df), 1,
                         "Feature DataFrame should have exactly one row")

        # Check that key features have the correct values
        self.assertEqual(features_df['fixation_left_count'].iloc[0], 2,
                         "Incorrect left fixation count in features")
        self.assertEqual(features_df['fixation_right_count'].iloc[0], 2,
                         "Incorrect right fixation count in features")
        self.assertEqual(features_df['saccade_left_count'].iloc[0], 1,
                         "Incorrect left saccade count in features")
        self.assertEqual(features_df['saccade_right_count'].iloc[0], 1,
                         "Incorrect right saccade count in features")
        self.assertEqual(features_df['blink_left_count'].iloc[0], 1,
                         "Incorrect left blink count in features")
        self.assertEqual(features_df['blink_right_count'].iloc[0], 1,
                         "Incorrect right blink count in features")

    def test_movie_segment_parsing(self):
        """Test parsing of movie segments and related information."""
        # Check that the movie segments were parsed correctly
        if hasattr(self.parser, 'movie_segments'):
            # There should be 1 movie segment
            self.assertEqual(len(self.parser.movie_segments), 1,
                             "Incorrect number of movie segments")

            # Check the movie segment details
            segment = self.parser.movie_segments[0]
            self.assertEqual(segment['movie_name'], 'Test_Movie.xvd',
                             "Incorrect movie filename")
            self.assertEqual(segment['frame_count'], 3,
                             "Incorrect frame count in movie segment")

        # Verify frame markers in the movie segments
        if hasattr(self.parser, 'movie_segments') and self.parser.movie_segments:
            segment = self.parser.movie_segments[0]
            if 'frames' in segment:
                frames = segment['frames']
                self.assertEqual(len(frames), 3, "Incorrect number of frames in movie segment")

                # Check frame timestamps
                if 1 in frames and 2 in frames and 3 in frames:
                    self.assertEqual(frames[1], 15021620, "Incorrect timestamp for frame 1")
                    self.assertEqual(frames[2], 15021640, "Incorrect timestamp for frame 2")
                    self.assertEqual(frames[3], 15021690, "Incorrect timestamp for frame 3")

    def test_process_asc_file_function(self):
        """Test the process_asc_file helper function."""
        # Create a temporary directory for test outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            result = process_asc_file(TEST_FILE_PATH, temp_dir, extract_features=True)

            # Check that the function returns the expected structure
            self.assertIn('summary', result, "process_asc_file result missing 'summary'")
            self.assertIn('dataframes', result, "process_asc_file result missing 'dataframes'")
            self.assertIn('features', result, "process_asc_file result missing 'features'")

            # Check that the summary has the correct values
            summary = result['summary']
            self.assertEqual(summary['samples'], 38, "Incorrect sample count in summary")
            self.assertEqual(summary['fixations'], 4, "Incorrect fixation count in summary")
            self.assertEqual(summary['saccades'], 2, "Incorrect saccade count in summary")
            self.assertEqual(summary['blinks'], 2, "Incorrect blink count in summary")
            self.assertEqual(summary['messages'], 64, "Incorrect message count in summary")
            self.assertEqual(summary['frames'], 3, "Incorrect frame count in summary")


if __name__ == '__main__':
    unittest.main()
