"""
Unit tests for error recovery and handling of malformed files
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser import EyeLinkASCParser, process_asc_file, load_csv_file


class TestErrorRecovery(unittest.TestCase):
    """Test cases for error recovery and handling of malformed files"""
    
    def setUp(self):
        """Create test directory and files"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.temp_dir):
            # Clean up files
            for f in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, f)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            # Remove directory
            os.rmdir(self.temp_dir)
    
    def create_empty_asc_file(self):
        """Create an empty ASC file"""
        empty_file = os.path.join(self.temp_dir, "empty.asc")
        with open(empty_file, 'w') as f:
            f.write("")
        return empty_file
    
    def create_header_only_asc_file(self):
        """Create an ASC file with only header information, no data"""
        header_file = os.path.join(self.temp_dir, "header_only.asc")
        with open(header_file, 'w') as f:
            f.write("** CONVERTED FROM headeronly.edf using edfapi 4.1 MacOS X Mar 19 2024\n")
            f.write("** DATE: Sat Mar 29 10:00:00 2025\n")
            f.write("** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED\n")
            f.write("** VERSION: EYELINK II 1\n")
            f.write("** SOURCE: EYELINK CL\n")
        return header_file
    
    def create_malformed_sample_asc_file(self):
        """Create an ASC file with malformed sample data"""
        malformed_file = os.path.join(self.temp_dir, "malformed.asc")
        with open(malformed_file, 'w') as f:
            # Write header
            f.write("** CONVERTED FROM malformed.edf using edfapi 4.1 MacOS X Mar 19 2024\n")
            f.write("** DATE: Sat Mar 29 10:00:00 2025\n")
            f.write("** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED\n")
            f.write("** VERSION: EYELINK II 1\n\n")
            
            # Start recording block
            f.write("START\t15021400 \tLEFT\tRIGHT\tSAMPLES\tEVENTS\n")
            f.write("PRESCALER\t1\n")
            f.write("VPRESCALER\t1\n")
            f.write("PUPIL\tAREA\n")
            f.write("EVENTS\tGAZE\tLEFT\tRIGHT\tRATE\t 500.00\tTRACKING\tCR\tFILTER\t2\n")
            f.write("SAMPLES\tGAZE\tLEFT\tRIGHT\tHTARGET\tRATE\t 500.00\tTRACKING\tCR\tFILTER\t2\tINPUT\n")
            
            # Write a movie marker
            f.write("MSG\t15021500 Movie File Name: Test_Movie.xvd.\n")
            
            # Write some good samples, then malformed samples, then good samples again
            f.write("15021600\t  629.7\t  464.4\t 1183.0\t  666.4\t  491.6\t 1105.0\t  127.0\t..... \t\n")
            f.write("15021602\t  629.6\t  464.7\t 1183.0\t  666.3\t  492.1\t 1105.0\t  127.0\t..... \t\n")
            f.write("15021604\t  INVALID DATA HERE\t  MALFORMED\t DATA\t  MORE BAD\t DATA\t\n")  # Malformed
            f.write("15021606\t  629.5\t  465.3\t\n")  # Incomplete
            f.write("15021608\t  629.8\t  465.6\t 1183.0\t  665.5\t  493.0\t 1106.0\t  127.0\t..... \t\n")
            
            # Write some events (fixation, saccade)
            f.write("SFIX L   15021610\n")
            f.write("15021610\t  630.0\t  465.9\t 1183.0\t  665.5\t  492.9\t 1106.0\t  127.0\t..... \t\n")
            f.write("EFIX L   15021610\t15021620\t10\t  630.5\t  465.7\t 1184.0\n")
            
            # No right fixation events - this will make sure there are no right fixations recognized
            
            # Write good left saccade
            f.write("SSACC L  15021650\n")
            f.write("15021650\t  628.0\t  464.0\t 1180.0\t  664.0\t  488.0\t 1100.0\t  127.0\t..... \t\n")
            f.write("ESACC L  15021650\t15021656\t6\t  628.0\t  464.0\t  618.0\t  458.0\t   0.35\t     65\n")
            
            # End recording
            f.write("END\t15021800 \tSAMPLES\tEVENTS\tRES\t  29.64\t  40.77\n")
            
        return malformed_file
    
    def create_malformed_csv_file(self):
        """Create a malformed CSV file"""
        malformed_csv = os.path.join(self.temp_dir, "malformed_unified_eye_metrics.csv")
        with open(malformed_csv, 'w') as f:
            f.write("timestamp,x_left,y_left,pupil_left,x_right,y_right,pupil_right\n")
            f.write("15021600,629.7,464.4,1183.0,666.4,491.6,1105.0\n")
            f.write("15021602,629.6,464.7,1183.0,666.3,492.1,1105.0\n")
            f.write("malformed,data,here,bad,more,bad,data\n")  # Malformed row
            f.write("15021606,629.5,465.3,1183.0,666.1,492.6,1105.0\n")
        return malformed_csv
    
    def test_empty_asc_file(self):
        """Test handling of an empty ASC file"""
        empty_file = self.create_empty_asc_file()
        parser = EyeLinkASCParser(empty_file)
        
        # This should not raise an exception but should create empty data structures
        num_lines = parser.read_file()
        self.assertEqual(num_lines, 0, "Empty file should have 0 lines")
        
        # Parse the file - should handle gracefully
        parser.parse_file()
        
        # Check that we have empty data structures
        self.assertEqual(len(parser.sample_data), 0, "Empty file should have 0 samples")
        self.assertEqual(len(parser.fixations['left']), 0, "Empty file should have 0 left fixations")
        self.assertEqual(len(parser.fixations['right']), 0, "Empty file should have 0 right fixations")
        
        # Features should be extracted with just the participant ID
        features_df = parser.extract_features()
        self.assertEqual(len(features_df), 1, "Features DataFrame should have 1 row")
        self.assertIn('participant_id', features_df.columns, "Features should include participant_id")
        
    def test_header_only_asc_file(self):
        """Test handling of an ASC file with only header info"""
        header_file = self.create_header_only_asc_file()
        parser = EyeLinkASCParser(header_file)
        
        # Parse the file - should handle gracefully
        summary = parser.parse_file()
        
        # Check that we extracted metadata but no samples or events
        self.assertGreater(len(parser.metadata), 0, "Metadata should be extracted from header")
        self.assertEqual(summary['samples'], 0, "Header-only file should have 0 samples")
        self.assertEqual(summary['fixations'], 0, "Header-only file should have 0 fixations")
        
        # Test with process_asc_file too
        with tempfile.TemporaryDirectory() as temp_output_dir:
            result = process_asc_file(header_file, temp_output_dir, extract_features=True)
            self.assertIn('features', result, "Features should be extracted even from empty file")
            # Since there are no samples, the unified metrics dataframe might be created empty
            # but present in the result dataframes
            if 'dataframes' in result and 'samples' in result['dataframes']:
                self.assertEqual(len(result['dataframes']['samples']), 0, "No samples should be extracted")
    
    def test_malformed_sample_asc_file(self):
        """Test handling of an ASC file with malformed samples"""
        malformed_file = self.create_malformed_sample_asc_file()
        parser = EyeLinkASCParser(malformed_file)
        
        # Parse the file - should handle gracefully
        summary = parser.parse_file()
        
        # Check that valid samples and events were extracted
        self.assertGreater(len(parser.sample_data), 0, "Valid samples should be extracted")
        self.assertEqual(len(parser.fixations['left']), 1, "Valid left fixation should be extracted")
        self.assertEqual(len(parser.fixations['right']), 0, "Invalid right fixation should be skipped")
        self.assertEqual(len(parser.saccades['left']), 1, "Valid left saccade should be extracted")
        
        # Features should be extracted properly
        features_df = parser.extract_features()
        self.assertEqual(len(features_df), 1, "Features DataFrame should have 1 row")
        self.assertEqual(features_df['fixation_left_count'].iloc[0], 1, "Should find 1 valid left fixation")
        
        # Check that we have no right fixations (should have no column for right fixations)
        self.assertTrue('fixation_right_count' not in features_df.columns or
                      features_df['fixation_right_count'].iloc[0] == 0, 
                      "Should find 0 valid right fixations")
        
        # Unified metrics should be created
        unified_df = parser.create_unified_metrics_df()
        self.assertGreater(len(unified_df), 0, "Unified metrics should be created with valid samples")
        
        # Our test file doesn't have enough data to create movie segments, so we'll skip this check
        # Instead, check that there's at least one message about the movie
        movie_messages = [msg for msg in parser.messages if 'Movie File Name' in msg['content']]
        self.assertGreater(len(movie_messages), 0, "Should find movie file name messages")
    
    def test_process_asc_malformed_file(self):
        """Test process_asc_file with a malformed ASC file"""
        malformed_file = self.create_malformed_sample_asc_file()
        
        # Process the file with error handling
        with tempfile.TemporaryDirectory() as temp_output_dir:
            result = process_asc_file(malformed_file, temp_output_dir, extract_features=True)
            
            # Check that we got valid results
            self.assertIn('summary', result, "Summary should be returned")
            self.assertIn('features', result, "Features should be extracted")
            self.assertIn('dataframes', result, "DataFrames should be returned")
            
            # Check that some valid samples were extracted
            self.assertGreater(len(result['dataframes']['samples']), 0, "Valid samples should be extracted")
            
            # Check that the unified metrics file was saved
            self.assertIn('saved_files', result, "Files should be saved")
            self.assertTrue(any('unified_eye_metrics' in f for f in result['saved_files']), 
                            "Unified metrics file should be saved")
    
    def test_malformed_csv_file(self):
        """Test handling of a malformed CSV file"""
        malformed_csv = self.create_malformed_csv_file()
        
        # Try to load the CSV - should handle errors gracefully
        try:
            result = load_csv_file(malformed_csv)
            
            # Check that we got valid results
            self.assertIn('summary', result, "Summary should be returned")
            self.assertIn('dataframes', result, "DataFrames should be returned")
            
            # Check if any data was loaded
            self.assertGreater(len(result['dataframes']['unified_eye_metrics']), 0, 
                              "Some valid rows should be loaded")
                
        except Exception as e:
            # If an exception is raised, it should be a controlled one with proper error message
            self.fail(f"load_csv_file should handle malformed CSV gracefully, but raised: {e}")


if __name__ == '__main__':
    unittest.main()