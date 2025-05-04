"""
Mock Data Generator for ASC Eye Tracking Files

This module generates synthetic ASC eye tracking data files for testing purposes.
It can create various scenarios including:
- Different numbers of samples
- Various eye movement events (fixations, saccades, blinks)
- Multiple movie segments
- Edge cases and invalid data
"""

import os
import random
import numpy as np
from datetime import datetime


class MockASCGenerator:
    """
    Generates synthetic EyeLink ASC files for testing purposes.
    """
    
    def __init__(self, screen_width=1920, screen_height=1080, sample_rate=500):
        """
        Initialize the generator with screen dimensions and sampling rate.
        
        Args:
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
            sample_rate: Sampling rate in Hz
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sample_rate = sample_rate
        self.timestamp_base = 10000000  # Starting timestamp
        self.timestamp = self.timestamp_base
        self.sample_interval = 1000 // sample_rate  # Time between samples in ms
        self.participant_id = "test_participant"
        self.file_version = "EYELINK II 1"
        self.tracker_version = "4.56"
        
        # Current eye positions
        self.x_left = screen_width / 2
        self.y_left = screen_height / 2
        self.x_right = screen_width / 2
        self.y_right = screen_height / 2
        self.pupil_left = 1000
        self.pupil_right = 1000
        
        # Lists to store generated events for later reference
        self.fixations = {"left": [], "right": []}
        self.saccades = {"left": [], "right": []}
        self.blinks = {"left": [], "right": []}
        self.movies = []
        self.frames = []
        self.messages = []
        
    def generate_header(self):
        """Generate a standard ASC file header"""
        now = datetime.now()
        date_str = now.strftime("%a %b %d %H:%M:%S %Y")
        
        header = []
        header.append(f"** CONVERTED FROM {self.participant_id}.edf using edfapi {self.tracker_version} {now.strftime('%b %d %Y')}")
        header.append(f"** DATE: {date_str}")
        header.append("** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED")
        header.append(f"** VERSION: {self.file_version}")
        header.append("** SOURCE: EYELINK CL")
        header.append("")
        return header
    
    def generate_recording_start(self):
        """Generate recording start markers"""
        lines = []
        lines.append(f"START\t{self.timestamp} \tLEFT\tRIGHT\tSAMPLES\tEVENTS")
        lines.append("PRESCALER\t1")
        lines.append("VPRESCALER\t1")
        lines.append("PUPIL\tAREA")
        lines.append("EVENTS\tGAZE\tLEFT\tRIGHT\tRATE\t 500.00\tTRACKING\tCR\tFILTER\t2")
        lines.append("SAMPLES\tGAZE\tLEFT\tRIGHT\tHTARGET\tRATE\t 500.00\tTRACKING\tCR\tFILTER\t2\tINPUT")
        return lines
    
    def generate_recording_end(self):
        """Generate recording end markers"""
        lines = []
        lines.append(f"END\t{self.timestamp} \tSAMPLES\tEVENTS\tRES\t  29.64\t  40.77")
        return lines
    
    def generate_sample(self, noise_level=2.0, random_invalid=False):
        """
        Generate a single eye tracking sample
        
        Args:
            noise_level: Amount of random noise to add to eye positions
            random_invalid: If True, occasionally generate invalid samples
        
        Returns:
            ASC formatted sample line as string
        """
        # Add some random movement/drift to eye positions
        self.x_left += random.uniform(-noise_level, noise_level)
        self.y_left += random.uniform(-noise_level, noise_level)
        self.x_right += random.uniform(-noise_level, noise_level)
        self.y_right += random.uniform(-noise_level, noise_level)
        
        # Keep positions within screen boundaries
        self.x_left = max(0, min(self.screen_width, self.x_left))
        self.y_left = max(0, min(self.screen_height, self.y_left))
        self.x_right = max(0, min(self.screen_width, self.x_right))
        self.y_right = max(0, min(self.screen_height, self.y_right))
        
        # Add minor random changes to pupil size
        self.pupil_left += random.uniform(-5, 5)
        self.pupil_right += random.uniform(-5, 5)
        
        # Keep pupil size in valid range
        self.pupil_left = max(500, min(2000, self.pupil_left))
        self.pupil_right = max(500, min(2000, self.pupil_right))
        
        # Increment timestamp
        self.timestamp += self.sample_interval
        
        # Occasionally generate invalid data
        if random_invalid and random.random() < 0.01:
            return f"{self.timestamp}\t  .\t  .\t .\t  .\t  .\t .\t  .\t..... \t"
        
        # Regular sample format
        return f"{self.timestamp}\t  {self.x_left:.1f}\t  {self.y_left:.1f}\t {self.pupil_left:.1f}\t  {self.x_right:.1f}\t  {self.y_right:.1f}\t {self.pupil_right:.1f}\t  0\t..... \t"
    
    def generate_fixation(self, eye="L", duration=200):
        """
        Generate a fixation event
        
        Args:
            eye: "L" for left eye, "R" for right eye
            duration: Fixation duration in milliseconds
        
        Returns:
            List of ASC formatted lines for the fixation event
        """
        lines = []
        eye_key = "left" if eye == "L" else "right"
        
        # Start time of fixation
        start_time = self.timestamp
        
        # Add fixation start
        lines.append(f"SFIX {eye}   {start_time}")
        
        # Generate a sample at the start of fixation
        lines.append(self.generate_sample(noise_level=0.5))
        
        # Set end time
        end_time = start_time + duration
        
        # Update timestamp to end of fixation
        self.timestamp = end_time
        
        # Create a stable position for this fixation
        fix_x = self.x_left if eye == "L" else self.x_right
        fix_y = self.y_left if eye == "L" else self.y_right
        fix_pupil = self.pupil_left if eye == "L" else self.pupil_right
        
        # Add fixation end
        lines.append(f"EFIX {eye}   {start_time}\t{end_time}\t{duration}\t  {fix_x:.1f}\t  {fix_y:.1f}\t {fix_pupil:.1f}")
        
        # Store fixation data for reference
        fixation_data = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'x': fix_x,
            'y': fix_y,
            'pupil': fix_pupil
        }
        self.fixations[eye_key].append(fixation_data)
        
        return lines
    
    def generate_saccade(self, eye="L", amplitude=50, duration=50, peak_velocity=300):
        """
        Generate a saccade event
        
        Args:
            eye: "L" for left eye, "R" for right eye
            amplitude: Saccade amplitude in pixels
            duration: Saccade duration in milliseconds
            peak_velocity: Peak velocity in pixels per second
            
        Returns:
            List of ASC formatted lines for the saccade event
        """
        lines = []
        eye_key = "left" if eye == "L" else "right"
        
        # Start position
        start_x = self.x_left if eye == "L" else self.x_right
        start_y = self.y_left if eye == "R" else self.y_right
        
        # Generate random direction for saccade
        angle = random.uniform(0, 2 * np.pi)
        
        # Calculate end position
        end_x = start_x + amplitude * np.cos(angle)
        end_y = start_y + amplitude * np.sin(angle)
        
        # Keep within screen bounds
        end_x = max(0, min(self.screen_width, end_x))
        end_y = max(0, min(self.screen_height, end_y))
        
        # Start time
        start_time = self.timestamp
        
        # Add saccade start
        lines.append(f"SSACC {eye}  {start_time}")
        
        # Update eye position to the end position for future samples
        if eye == "L":
            self.x_left = end_x
            self.y_left = end_y
        else:
            self.x_right = end_x
            self.y_right = end_y
            
        # Generate a sample during saccade
        lines.append(self.generate_sample(noise_level=5.0))
        
        # Set end time
        end_time = start_time + duration
        
        # Update timestamp
        self.timestamp = end_time
        
        # Add saccade end
        lines.append(f"ESACC {eye}  {start_time}\t{end_time}\t{duration}\t  {start_x:.1f}\t  {start_y:.1f}\t  {end_x:.1f}\t  {end_y:.1f}\t   {amplitude/100:.2f}\t     {peak_velocity}")
        
        # Store saccade data for reference
        saccade_data = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'amplitude': amplitude/100,
            'peak_velocity': peak_velocity
        }
        self.saccades[eye_key].append(saccade_data)
        
        return lines
    
    def generate_blink(self, eye="L", duration=100):
        """
        Generate a blink event
        
        Args:
            eye: "L" for left eye, "R" for right eye
            duration: Blink duration in milliseconds
            
        Returns:
            List of ASC formatted lines for the blink event
        """
        lines = []
        eye_key = "left" if eye == "L" else "right"
        
        # Start time
        start_time = self.timestamp
        
        # Add blink start
        lines.append(f"SBLINK {eye}  {start_time}")
        
        # Set end time
        end_time = start_time + duration
        
        # Update timestamp
        self.timestamp = end_time
        
        # Add blink end
        lines.append(f"EBLINK {eye}  {start_time}\t{end_time}\t{duration}")
        
        # Store blink data for reference
        blink_data = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        }
        self.blinks[eye_key].append(blink_data)
        
        return lines
    
    def generate_movie_marker(self, movie_name, duration=5000, num_frames=50):
        """
        Generate movie start and end markers, with frame markers in between
        
        Args:
            movie_name: Name of the movie
            duration: Movie duration in milliseconds
            num_frames: Number of frames in the movie
            
        Returns:
            List of ASC formatted lines for the movie
        """
        lines = []
        
        # Movie start marker
        start_time = self.timestamp
        lines.append(f"MSG\t{start_time} Movie File Name: {movie_name}.")
        
        # Frame markers
        frame_interval = duration // num_frames
        frames = []
        
        for frame in range(1, num_frames + 1):
            frame_time = start_time + frame * frame_interval
            frame_marker = f"MSG\t{frame_time} Play_Movie_Start FRAME #{frame}"
            lines.append(frame_marker)
            frames.append((frame, frame_time))
        
        # Update timestamp to after movie
        self.timestamp = start_time + duration
        
        # Movie end marker
        lines.append(f"MSG\t{self.timestamp} Movie File Name: {movie_name}. Displayed Frame Count: {num_frames}")
        
        # Store movie data for reference
        movie_data = {
            'movie_name': movie_name,
            'start_time': start_time,
            'end_time': self.timestamp,
            'duration': duration,
            'frames': frames,
            'frame_count': num_frames
        }
        self.movies.append(movie_data)
        self.frames.extend(frames)
        
        return lines
    
    def generate_custom_message(self, message):
        """
        Generate a custom message
        
        Args:
            message: Message text
            
        Returns:
            ASC formatted message line
        """
        msg = f"MSG\t{self.timestamp} {message}"
        self.messages.append((self.timestamp, message))
        return msg
    
    def generate_malformed_data(self, malformed_type="sample"):
        """
        Generate intentionally malformed data for testing error handling
        
        Args:
            malformed_type: Type of malformed data to generate 
                           ("sample", "fixation", "saccade", "blink", "message")
                           
        Returns:
            ASC formatted malformed line
        """
        if malformed_type == "sample":
            return f"{self.timestamp}\t  INVALID\t  DATA\t BAD\t  SAMPLE\t  HERE\t NOW\t"
        
        elif malformed_type == "fixation":
            if random.choice([True, False]):
                # Missing end fixation
                return f"SFIX L   {self.timestamp}"
            else:
                # Malformed end fixation
                return f"EFIX L   {self.timestamp}\tNOT_NUMBER\tBAD\t  DATA\t  HERE\t INVALID"
        
        elif malformed_type == "saccade":
            if random.choice([True, False]):
                # Missing end saccade
                return f"SSACC L  {self.timestamp}"
            else:
                # Malformed end saccade
                return f"ESACC L  {self.timestamp}\tNOT_NUMBER\tBAD\t  DATA\t  HERE\t INVALID"
        
        elif malformed_type == "blink":
            if random.choice([True, False]):
                # Missing end blink
                return f"SBLINK L  {self.timestamp}"
            else:
                # Malformed end blink
                return f"EBLINK L  {self.timestamp}\tNOT_NUMBER\tBAD"
        
        elif malformed_type == "message":
            return f"MSG\t{self.timestamp} CORRUPT MESSAGE FORMAT WITH MISSING\tENDING"
    
    def generate_asc_file(self, output_path, num_samples=1000, num_events=20, 
                         num_movies=2, include_malformed=False, malformed_ratio=0.05):
        """
        Generate a complete ASC file with specified parameters
        
        Args:
            output_path: Path to save the generated ASC file
            num_samples: Number of samples to generate
            num_events: Number of events (fixations, saccades, blinks) to generate
            num_movies: Number of movies to include
            include_malformed: Whether to include malformed data
            malformed_ratio: Ratio of samples that should be malformed (if include_malformed=True)
            
        Returns:
            Path to the generated file and a dict with statistics about what was generated
        """
        all_lines = []
        
        # Generate header
        all_lines.extend(self.generate_header())
        
        # Generate recording start
        all_lines.extend(self.generate_recording_start())
        
        # Reset timestamp to start of recording
        self.timestamp = self.timestamp_base
        
        # Track statistics
        stats = {
            'samples': 0,
            'fixations': {'left': 0, 'right': 0},
            'saccades': {'left': 0, 'right': 0},
            'blinks': {'left': 0, 'right': 0},
            'movies': [],
            'malformed': 0
        }
        
        # Generate movie markers interspersed throughout the recording
        if num_movies > 0:
            movie_intervals = num_samples // (num_movies + 1)
            for i in range(num_movies):
                # Generate some samples before movie
                for _ in range(movie_intervals // 2):
                    if include_malformed and random.random() < malformed_ratio:
                        all_lines.append(self.generate_malformed_data("sample"))
                        stats['malformed'] += 1
                    else:
                        all_lines.append(self.generate_sample())
                        stats['samples'] += 1
                
                # Generate a movie
                movie_name = f"test_movie_{i+1}.mp4"
                movie_lines = self.generate_movie_marker(movie_name, 
                                                       duration=random.randint(3000, 10000),
                                                       num_frames=random.randint(30, 100))
                all_lines.extend(movie_lines)
                stats['movies'].append(movie_name)
                
                # Generate some samples after movie
                for _ in range(movie_intervals // 2):
                    if include_malformed and random.random() < malformed_ratio:
                        all_lines.append(self.generate_malformed_data("sample"))
                        stats['malformed'] += 1
                    else:
                        all_lines.append(self.generate_sample())
                        stats['samples'] += 1
        
        # Fill in with additional samples and events
        samples_left = num_samples - stats['samples']
        samples_per_event = samples_left // (num_events + 1)
        
        for i in range(num_events):
            # Generate some regular samples
            for _ in range(samples_per_event):
                if include_malformed and random.random() < malformed_ratio:
                    all_lines.append(self.generate_malformed_data("sample"))
                    stats['malformed'] += 1
                else:
                    all_lines.append(self.generate_sample())
                    stats['samples'] += 1
            
            # Generate an event (fixation, saccade, or blink)
            event_type = random.choice(["fixation", "saccade", "blink"])
            eye = random.choice(["L", "R"])
            eye_key = "left" if eye == "L" else "right"
            
            if event_type == "fixation":
                if include_malformed and random.random() < malformed_ratio:
                    all_lines.append(self.generate_malformed_data("fixation"))
                    stats['malformed'] += 1
                else:
                    fixation_lines = self.generate_fixation(eye, duration=random.randint(100, 500))
                    all_lines.extend(fixation_lines)
                    stats['fixations'][eye_key] += 1
            
            elif event_type == "saccade":
                if include_malformed and random.random() < malformed_ratio:
                    all_lines.append(self.generate_malformed_data("saccade"))
                    stats['malformed'] += 1
                else:
                    saccade_lines = self.generate_saccade(eye, 
                                                        amplitude=random.randint(20, 200),
                                                        duration=random.randint(30, 100))
                    all_lines.extend(saccade_lines)
                    stats['saccades'][eye_key] += 1
            
            elif event_type == "blink":
                if include_malformed and random.random() < malformed_ratio:
                    all_lines.append(self.generate_malformed_data("blink"))
                    stats['malformed'] += 1
                else:
                    blink_lines = self.generate_blink(eye, duration=random.randint(50, 300))
                    all_lines.extend(blink_lines)
                    stats['blinks'][eye_key] += 1
        
        # Add remaining samples
        samples_left = num_samples - stats['samples']
        for _ in range(samples_left):
            if include_malformed and random.random() < malformed_ratio:
                all_lines.append(self.generate_malformed_data("sample"))
                stats['malformed'] += 1
            else:
                all_lines.append(self.generate_sample())
                stats['samples'] += 1
        
        # Generate recording end
        all_lines.extend(self.generate_recording_end())
        
        # Write to file
        with open(output_path, 'w') as f:
            for line in all_lines:
                f.write(line + '\n')
        
        return output_path, stats
    
    def create_test_suite(self, output_dir, include_edge_cases=True):
        """
        Generate a comprehensive suite of test files for different scenarios
        
        Args:
            output_dir: Directory to save the generated files
            include_edge_cases: Whether to include edge cases and malformed data
            
        Returns:
            Dict with paths to generated files and their descriptions
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        test_files = {}
        
        # Generate a standard file with good data
        standard_path = os.path.join(output_dir, "standard_test.asc")
        standard_file, standard_stats = self.generate_asc_file(
            standard_path,
            num_samples=500,
            num_events=20,
            num_movies=2,
            include_malformed=False
        )
        test_files['standard'] = {
            'path': standard_path,
            'description': "Standard ASC file with clean data",
            'stats': standard_stats
        }
        
        # Generate a large file
        large_path = os.path.join(output_dir, "large_test.asc")
        large_file, large_stats = self.generate_asc_file(
            large_path,
            num_samples=2000,
            num_events=50,
            num_movies=3,
            include_malformed=False
        )
        test_files['large'] = {
            'path': large_path,
            'description': "Large ASC file with more samples and events",
            'stats': large_stats
        }
        
        if include_edge_cases:
            # Empty file (just header)
            empty_path = os.path.join(output_dir, "empty_test.asc")
            with open(empty_path, 'w') as f:
                for line in self.generate_header():
                    f.write(line + '\n')
            test_files['empty'] = {
                'path': empty_path,
                'description': "Empty ASC file with only header information",
                'stats': {'samples': 0, 'events': 0, 'movies': 0}
            }
            
            # Sample-only file (no events)
            samples_path = os.path.join(output_dir, "samples_only_test.asc")
            samples_file, samples_stats = self.generate_asc_file(
                samples_path,
                num_samples=200,
                num_events=0,
                num_movies=0,
                include_malformed=False
            )
            test_files['samples_only'] = {
                'path': samples_path,
                'description': "ASC file with only samples, no events or movies",
                'stats': samples_stats
            }
            
            # Events-only file (minimal samples)
            events_path = os.path.join(output_dir, "events_only_test.asc")
            events_file, events_stats = self.generate_asc_file(
                events_path,
                num_samples=20,
                num_events=30,
                num_movies=0,
                include_malformed=False
            )
            test_files['events_only'] = {
                'path': events_path,
                'description': "ASC file with mostly events, minimal samples",
                'stats': events_stats
            }
            
            # Movies-only file (minimal samples)
            movies_path = os.path.join(output_dir, "movies_only_test.asc")
            movies_file, movies_stats = self.generate_asc_file(
                movies_path,
                num_samples=20,
                num_events=0,
                num_movies=5,
                include_malformed=False
            )
            test_files['movies_only'] = {
                'path': movies_path,
                'description': "ASC file with many movies, minimal samples/events",
                'stats': movies_stats
            }
            
            # Malformed data file
            malformed_path = os.path.join(output_dir, "malformed_test.asc")
            malformed_file, malformed_stats = self.generate_asc_file(
                malformed_path,
                num_samples=300,
                num_events=15,
                num_movies=1,
                include_malformed=True,
                malformed_ratio=0.1
            )
            test_files['malformed'] = {
                'path': malformed_path,
                'description': "ASC file with intentionally malformed data",
                'stats': malformed_stats
            }
            
            # Single eye file (left eye only)
            left_eye_path = os.path.join(output_dir, "left_eye_only_test.asc")
            # Save current generator state
            orig_x_right = self.x_right
            orig_y_right = self.y_right
            orig_pupil_right = self.pupil_right
            
            # Set right eye data to invalid
            self.x_right = float('nan')
            self.y_right = float('nan')
            self.pupil_right = 0.0
            
            left_eye_file, left_eye_stats = self.generate_asc_file(
                left_eye_path,
                num_samples=200,
                num_events=10,
                num_movies=1,
                include_malformed=False
            )
            
            # Restore generator state
            self.x_right = orig_x_right
            self.y_right = orig_y_right
            self.pupil_right = orig_pupil_right
            
            test_files['left_eye_only'] = {
                'path': left_eye_path,
                'description': "ASC file with left eye data only",
                'stats': left_eye_stats
            }
            
        return test_files


# Helper function to create a complete test suite
def create_test_suite(output_dir="test_data"):
    """
    Create a full suite of test files for ASC parsing
    
    Args:
        output_dir: Directory to save test files
        
    Returns:
        Dict with information about generated files
    """
    generator = MockASCGenerator()
    test_files = generator.create_test_suite(output_dir, include_edge_cases=True)
    
    # Print summary of what was generated
    print(f"Generated {len(test_files)} test files in {output_dir}:")
    for key, info in test_files.items():
        print(f"- {key}: {info['description']} ({os.path.basename(info['path'])})")
    
    return test_files


if __name__ == "__main__":
    # When run directly, generate a test suite in the current directory
    test_files = create_test_suite("asc_test_files")