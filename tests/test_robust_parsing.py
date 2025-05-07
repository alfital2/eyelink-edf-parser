"""
Robust Parsing Tests
Author: Tal Alfi
Date: May 2025

This module tests the parser's ability to handle various edge cases and errors in ASC files:
- Malformed lines
- Missing or corrupted data
- Inconsistent formats
- Recovery from errors
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path
import pandas as pd

# Add parent directory to path to import parser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser import process_asc_file


class TestRobustParsing(unittest.TestCase):
    """
    Test suite for robust parsing of ASC files.
    Tests the parser's ability to handle edge cases and recover from errors.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Clean up after each test."""
        # Delete temporary directory
        self.temp_dir.cleanup()
    
    def create_test_asc_file(self, content):
        """Helper to create a temporary ASC file with the provided content."""
        with tempfile.NamedTemporaryFile(suffix='.asc', delete=False) as temp_file:
            temp_file.write(content.encode('utf-8'))
            temp_path = temp_file.name
            return temp_path
    
    def test_incomplete_header(self):
        """Test handling of files with incomplete headers."""
        # Create a minimal ASC file with just some basic header info
        minimal_header = """** CONVERTED FROM minimal.edf
MSG	15000000 !CMD 0 select_parser_configuration 0
MSG	15000100 DISPLAY_COORDS 0 0 1279 1023
START	15021400 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2

15021600	  629.7	  464.4	 1183.0	  666.4	  491.6	 1105.0
15021602	  629.6	  464.7	 1183.0	  666.3	  492.1	 1105.0
15021604	  629.5	  465.0	 1183.0	  666.1	  492.6	 1105.0

SFIX L   15021610
15021610	  630.0	  465.9	 1183.0	  665.5	  492.9	 1106.0
15021612	  630.3	  466.0	 1183.0	  665.8	  492.4	 1106.0
EFIX L   15021610	15021630	20	  630.5	  465.7	 1184.0
"""
        
        test_file = self.create_test_asc_file(minimal_header)
        
        try:
            # Process the file with minimal header
            result = process_asc_file(test_file, output_dir=self.output_dir)
            
            # Verify parser handled the file
            self.assertIsNotNone(result, "Parser should return a result for files with minimal headers")
            self.assertIn('parser', result, "Result should include parser object")
            
            # Check that some data was extracted
            sample_data = result['parser'].sample_data
            self.assertGreater(len(sample_data), 0, "Should extract sample data despite minimal header")
            
            # Check for fixation events
            fixations = result['parser'].fixations.get('left', [])
            self.assertGreater(len(fixations), 0, "Should extract fixation events despite minimal header")
            
        finally:
            # Clean up test file
            os.unlink(test_file)
    
    def test_malformed_data_lines(self):
        """Test handling of files with malformed data lines."""
        # Create an ASC file with some malformed data lines
        malformed_content = """** CONVERTED FROM malformed.edf
MSG	15000000 !CMD 0 select_parser_configuration 0
MSG	15000100 DISPLAY_COORDS 0 0 1279 1023
START	15021400 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2

15021600	  629.7	  464.4	 1183.0	  666.4	  491.6	 1105.0
15021602	  bad data here
15021604	  629.5	  465.0	 1183.0	  666.1	  492.6	 1105.0
!@#$%^ this line is completely wrong
15021608	  629.8	  465.6	 1183.0	  665.5	  493.0	 1106.0

SFIX L   15021610
15021610	  630.0	  465.9	 1183.0	  665.5	  492.9	 1106.0
15021612	  630.3	  466.0	  *** missing data ***
15021614	  630.5	  466.1	 1183.0	  665.9	  491.7	 1106.0
EFIX L   15021610	15021630	20	  630.5	  465.7	 1184.0
"""
        
        test_file = self.create_test_asc_file(malformed_content)
        
        try:
            # Process the file with malformed data lines
            result = process_asc_file(test_file, output_dir=self.output_dir)
            
            # Verify parser handled the file
            self.assertIsNotNone(result, "Parser should return a result for files with malformed data lines")
            self.assertIn('parser', result, "Result should include parser object")
            
            # Check that valid data was extracted
            sample_data = result['parser'].sample_data
            self.assertGreater(len(sample_data), 0, "Should extract valid sample data despite malformed lines")
            
            # The parser should skip bad lines but process good ones
            self.assertLess(len(sample_data), 6, "Bad lines should be skipped")
            
        finally:
            # Clean up test file
            os.unlink(test_file)
    
    def test_inconsistent_event_formats(self):
        """Test handling of files with inconsistent event formats."""
        # Create an ASC file with inconsistent event formats
        inconsistent_events = """** CONVERTED FROM inconsistent.edf
MSG	15000000 !CMD 0 select_parser_configuration 0
MSG	15000100 DISPLAY_COORDS 0 0 1279 1023
START	15021400 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2

15021600	  629.7	  464.4	 1183.0	  666.4	  491.6	 1105.0
15021602	  629.6	  464.7	 1183.0	  666.3	  492.1	 1105.0
15021604	  629.5	  465.0	 1183.0	  666.1	  492.6	 1105.0

SFIX L   15021610
15021610	  630.0	  465.9	 1183.0	  665.5	  492.9	 1106.0
15021612	  630.3	  466.0	 1183.0	  665.8	  492.4	 1106.0
EFIX L   15021610	15021630	too few values
SFIX L   not a timestamp
15021614	  630.5	  466.1	 1183.0	  665.9	  491.7	 1106.0
EFIX L   15021614	15021620	nan	  invalid	 values	 here
SFIX R   15021622
15021622	  630.9	  465.9	 1184.0	  666.2	  489.8	 1107.0
EFIX R   15021622 15021632 10 666.0 490.0 1107.0 extra values here
"""
        
        test_file = self.create_test_asc_file(inconsistent_events)
        
        try:
            # Process the file with inconsistent event formats
            result = process_asc_file(test_file, output_dir=self.output_dir)
            
            # Verify parser handled the file
            self.assertIsNotNone(result, "Parser should return a result despite inconsistent event formats")
            self.assertIn('parser', result, "Result should include parser object")
            
            # Check that valid data was extracted
            sample_data = result['parser'].sample_data
            self.assertGreater(len(sample_data), 0, "Should extract valid sample data")
            
            # Check that valid events were extracted and invalid ones skipped
            fixations = result['parser'].fixations.get('left', []) + result['parser'].fixations.get('right', [])
            self.assertGreater(len(fixations), 0, "Should extract valid fixation events")
            
        finally:
            # Clean up test file
            os.unlink(test_file)
    
    def test_missing_message_timestamps(self):
        """Test handling of files with missing timestamps in messages."""
        # Create an ASC file with missing timestamps in messages
        missing_timestamps = """** CONVERTED FROM missing_timestamps.edf
MSG	15000000 !CMD 0 select_parser_configuration 0
MSG	DISPLAY_COORDS 0 0 1279 1023
MSG	!MODE RECORD
START	15021400 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2

MSG	15021500 Movie File Name: Test_Movie.xvd.
MSG	missing timestamp for movie frame
15021600	  629.7	  464.4	 1183.0	  666.4	  491.6	 1105.0
15021602	  629.6	  464.7	 1183.0	  666.3	  492.1	 1105.0

SFIX L   15021610
15021610	  630.0	  465.9	 1183.0	  665.5	  492.9	 1106.0
15021612	  630.3	  466.0	 1183.0	  665.8	  492.4	 1106.0
EFIX L   15021610	15021630	20	  630.5	  465.7	 1184.0

MSG	15021620 -12 Play_Movie_Start FRAME #1
MSG	invalid format
MSG
"""
        
        test_file = self.create_test_asc_file(missing_timestamps)
        
        try:
            # Process the file with missing message timestamps
            result = process_asc_file(test_file, output_dir=self.output_dir)
            
            # Verify parser handled the file
            self.assertIsNotNone(result, "Parser should return a result despite missing message timestamps")
            self.assertIn('parser', result, "Result should include parser object")
            
            # Check that valid messages were extracted
            messages = result['parser'].messages
            self.assertGreater(len(messages), 0, "Should extract valid messages")
            
            # Check if frame markers were detected
            frame_markers = result['parser'].frame_markers
            self.assertGreater(len(frame_markers), 0, "Should extract valid frame markers")
            
        finally:
            # Clean up test file
            os.unlink(test_file)
    
    def test_extreme_values(self):
        """Test handling of files with extreme values."""
        # Create an ASC file with extreme coordinate values
        extreme_values = """** CONVERTED FROM extreme_values.edf
MSG	15000000 !CMD 0 select_parser_configuration 0
MSG	15000100 DISPLAY_COORDS 0 0 1279 1023
START	15021400 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2

15021600	  9999.9	  9999.9	 1183.0	  -9999.9	  -9999.9	 1105.0
15021602	  629.6	  464.7	 1183.0	  666.3	  492.1	 1105.0
15021604	  629.5	  465.0	 1183.0	  666.1	  492.6	 1105.0

SFIX L   15021610
15021610	  630.0	  465.9	 1183.0	  665.5	  492.9	 1106.0
15021612	  -1000.0	  -1000.0	 1183.0	  665.8	  492.4	 1106.0
15021614	  630.5	  466.1	 1183.0	  665.9	  491.7	 1106.0
EFIX L   15021610	15021630	20	  -1000.0	  -1000.0	 1184.0
"""
        
        test_file = self.create_test_asc_file(extreme_values)
        
        try:
            # Process the file with extreme values
            result = process_asc_file(test_file, output_dir=self.output_dir)
            
            # Verify parser handled the file
            self.assertIsNotNone(result, "Parser should return a result despite extreme values")
            self.assertIn('parser', result, "Result should include parser object")
            
            # Check that sample data was extracted
            sample_data = result['parser'].sample_data
            self.assertGreater(len(sample_data), 0, "Should extract sample data despite extreme values")
            
            # Check that extreme values are handled appropriately
            # Specific behavior depends on implementation, but it shouldn't crash
            
        finally:
            # Clean up test file
            os.unlink(test_file)
    
    def test_empty_sections(self):
        """Test handling of files with empty sections."""
        # Create an ASC file with empty sections
        empty_sections = """** CONVERTED FROM empty_sections.edf
MSG	15000000 !CMD 0 select_parser_configuration 0
MSG	15000100 DISPLAY_COORDS 0 0 1279 1023
START	15021400 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2

15021600	  629.7	  464.4	 1183.0	  666.4	  491.6	 1105.0
15021602	  629.6	  464.7	 1183.0	  666.3	  492.1	 1105.0
15021604	  629.5	  465.0	 1183.0	  666.1	  492.6	 1105.0

END	15021700
START	15021800 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2
END	15021800

START	15021900 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2

15022000	  625.7	  460.4	 1183.0	  660.4	  490.6	 1105.0
15022002	  625.6	  460.7	 1183.0	  660.3	  490.1	 1105.0

SFIX L   15022010
15022010	  626.0	  461.9	 1183.0	  660.5	  489.9	 1106.0
15022012	  626.3	  462.0	 1183.0	  660.8	  489.4	 1106.0
EFIX L   15022010	15022030	20	  626.5	  461.7	 1184.0
"""
        
        test_file = self.create_test_asc_file(empty_sections)
        
        try:
            # Process the file with empty sections
            result = process_asc_file(test_file, output_dir=self.output_dir)
            
            # Verify parser handled the file
            self.assertIsNotNone(result, "Parser should return a result despite empty sections")
            self.assertIn('parser', result, "Result should include parser object")
            
            # Check that sample data was extracted from non-empty sections
            sample_data = result['parser'].sample_data
            self.assertGreater(len(sample_data), 0, "Should extract sample data from non-empty sections")
            
            # Check that events were extracted from non-empty sections
            fixations = result['parser'].fixations.get('left', [])
            self.assertGreater(len(fixations), 0, "Should extract fixation events from non-empty sections")
            
        finally:
            # Clean up test file
            os.unlink(test_file)
    
    def test_mixed_recording_modes(self):
        """Test handling of files with mixed recording modes."""
        # Create an ASC file with mixed recording modes
        mixed_modes = """** CONVERTED FROM mixed_modes.edf
MSG	15000000 !CMD 0 select_parser_configuration 0
MSG	15000100 DISPLAY_COORDS 0 0 1279 1023
START	15021400 	LEFT	RIGHT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RIGHT	RATE	 500.00	TRACKING	CR	FILTER	2

15021600	  629.7	  464.4	 1183.0	  666.4	  491.6	 1105.0
15021602	  629.6	  464.7	 1183.0	  666.3	  492.1	 1105.0

END	15021700
MSG	15021800 !MODE RECORD CR 250 2 1 L
START	15021800 	LEFT	SAMPLES	EVENTS
PRESCALER	1
VPRESCALER	1
EVENTS	GAZE	LEFT	RATE	 250.00	TRACKING	CR	FILTER	2

15022000	  625.7	  460.4	 1183.0
15022004	  625.6	  460.7	 1183.0

SFIX L   15022010
15022010	  626.0	  461.9	 1183.0
15022014	  626.3	  462.0	 1183.0
EFIX L   15022010	15022030	20	  626.5	  461.7	 1184.0
"""
        
        test_file = self.create_test_asc_file(mixed_modes)
        
        try:
            # Process the file with mixed recording modes
            result = process_asc_file(test_file, output_dir=self.output_dir)
            
            # Verify parser handled the file
            self.assertIsNotNone(result, "Parser should return a result despite mixed recording modes")
            self.assertIn('parser', result, "Result should include parser object")
            
            # Check that sample data was extracted from both recording modes
            sample_data = result['parser'].sample_data
            self.assertGreater(len(sample_data), 0, "Should extract sample data from both recording modes")
            
            # Check that events were extracted
            fixations = result['parser'].fixations.get('left', [])
            self.assertGreater(len(fixations), 0, "Should extract fixation events")
            
        finally:
            # Clean up test file
            os.unlink(test_file)


if __name__ == '__main__':
    unittest.main()