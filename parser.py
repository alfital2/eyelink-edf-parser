import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Union, Optional


class EyeLinkASCParser:
    """
    Parser for EyeLink ASC files to extract eye tracking data for autism research.

    This parser extracts:
    - Basic experiment metadata
    - Sample data (position, pupil size, timestamp)
    - Event data (fixations, saccades, blinks)
    - Message data (experiment markers, video frame info)
    - Calibration information
    """

    def __init__(self, file_path: str):
        """
        Initialize the parser with the ASC file path.

        Args:
            file_path: Path to the ASC file
        """
        self.file_path = file_path
        self.file_lines = []
        self.metadata = {}
        self.sample_data = []
        self.fixations = {"left": [], "right": []}
        self.saccades = {"left": [], "right": []}
        self.blinks = {"left": [], "right": []}
        self.messages = []
        self.calibration_info = {}
        self.frame_markers = []

        # Sample headers
        self.sample_headers = [
            'timestamp', 'x_left', 'y_left', 'pupil_left',
            'x_right', 'y_right', 'pupil_right', 'input',
            'cr_info', 'cr_left', 'cr_right', 'cr_area'
        ]

        # Precompile regular expressions for better performance
        self.meta_pattern = re.compile(r'^\*\* (.+?):\s*(.+)')
        self.msg_pattern = re.compile(r'^MSG\s+(\d+)\s+(.+)')
        self.movie_start_pattern = re.compile(r'^Movie File Name:\s*([\w\d\._-]+)$')
        self.movie_end_pattern = re.compile(r'^Movie File Name:\s*([\w\d\._-]+)\.\s+Displayed Frame Count:\s+(\d+)')
        self.frame_pattern = re.compile(r'Play_Movie_Start FRAME #(\d+)')

        # Precompile event patterns
        self.fix_start_pattern = re.compile(r'^SFIX\s+([LR])\s+(\d+)')
        self.fix_end_pattern = re.compile(r'^EFIX\s+([LR])\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+(\d+)')
        self.sacc_start_pattern = re.compile(r'^SSACC\s+([LR])\s+(\d+)')
        self.sacc_end_pattern = re.compile(
            r'^ESACC\s+([LR])\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+(\d+)')
        self.blink_start_pattern = re.compile(r'^SBLINK\s+([LR])\s+(\d+)')
        self.blink_end_pattern = re.compile(r'^EBLINK\s+([LR])\s+(\d+)\s+(\d+)\s+(\d+)')

    def read_file(self):
        """Read the ASC file and store lines"""
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            self.file_lines = f.readlines()
        return len(self.file_lines)

    def parse_metadata(self):
        """Extract metadata from the file header"""
        for line in self.file_lines:
            if line.startswith('**'):
                match = self.meta_pattern.match(line)
                if match:
                    key, value = match.groups()
                    self.metadata[key.strip()] = value.strip()
            # Stop when we hit the end of the header
            elif not line.startswith('**') and not line.strip() == '':
                break

        return self.metadata

    def _collect_messages_and_frames(self):
        """First pass - collect all messages and frame markers"""
        has_frame_markers = False
        
        for line in self.file_lines:
            if line.startswith('MSG'):
                match = self.msg_pattern.match(line)
                if match:
                    timestamp, content = match.groups()
                    timestamp = int(timestamp)

                    # Store general message
                    self.messages.append({
                        'timestamp': timestamp,
                        'content': content
                    })

                    # Track frame markers
                    frame_match = self.frame_pattern.search(content)
                    if frame_match:
                        has_frame_markers = True
                        frame_num = int(frame_match.group(1))

                        # Create frame marker
                        frame_marker = {
                            'timestamp': timestamp,
                            'frame': frame_num,
                            'content': content,
                            'movie_name': None  # We'll fill this in later
                        }

                        # Add to general frame markers list
                        self.frame_markers.append(frame_marker)
                        
        return has_frame_markers
    
    def _identify_movie_markers(self):
        """Second pass - identify movie start and end markers"""
        start_markers = []
        end_markers = []
        potential_movies = []

        for msg in self.messages:
            content = msg['content']
            timestamp = msg['timestamp']

            # Check for movie file markers at start
            movie_start_match = self.movie_start_pattern.search(content)
            if movie_start_match:
                movie_name = movie_start_match.group(1)
                start_markers.append({
                    'name': movie_name,
                    'timestamp': timestamp
                })

            # Check for movie file markers at end (with frame count)
            movie_end_match = self.movie_end_pattern.search(content)
            if movie_end_match:
                movie_name = movie_end_match.group(1)
                frame_count = int(movie_end_match.group(2))
                end_markers.append({
                    'name': movie_name,
                    'timestamp': timestamp,
                    'frame_count': frame_count
                })
                
        return start_markers, end_markers, potential_movies
    
    def _process_paired_markers(self, start_markers, end_markers, potential_movies):
        """Process paired start/end markers to create movie segments"""
        for start in start_markers:
            # Find matching end marker
            matching_end = None
            for end in end_markers:
                if end['name'] == start['name']:
                    matching_end = end
                    break

            if matching_end:
                # Create a movie segment
                movie_name = start['name']
                start_time = start['timestamp']
                end_time = matching_end['timestamp']
                frame_count = matching_end.get('frame_count', 0)

                # Collect all frame markers that fall between these timestamps
                frames = {}
                for frame in self.frame_markers:
                    if start_time <= frame['timestamp'] <= end_time:
                        frames[frame['frame']] = frame['timestamp']
                        frame['movie_name'] = movie_name  # Update movie name in frame marker

                self.movie_segments.append({
                    'movie_name': movie_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'frames': frames,
                    'frame_count': frame_count or len(frames)
                })

                potential_movies.append({
                    'movie_name': movie_name,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
        return potential_movies
    
    def _process_unpaired_end_markers(self, end_markers, potential_movies):
        """Process end markers that don't have matching start markers"""
        for end in end_markers:
            # Check if this end marker has already been matched
            already_matched = False
            for movie in potential_movies:
                if movie['movie_name'] == end['name'] and movie['end_time'] == end['timestamp']:
                    already_matched = True
                    break

            if not already_matched:
                # This is an end marker without a start marker
                movie_name = end['name']
                end_time = end['timestamp']
                frame_count = end.get('frame_count', 0)

                # Try to find the start time based on frame markers
                # Look for the earliest frame that has not been assigned to any movie
                earliest_frame = None
                for frame in sorted(self.frame_markers, key=lambda x: x['timestamp']):
                    if frame['movie_name'] is None:  # Not yet assigned to a movie
                        earliest_frame = frame
                        break

                if earliest_frame:
                    start_time = earliest_frame['timestamp']

                    # Collect all frame markers that fall between these timestamps
                    frames = {}
                    for frame in self.frame_markers:
                        if start_time <= frame['timestamp'] <= end_time and frame['movie_name'] is None:
                            frames[frame['frame']] = frame['timestamp']
                            frame['movie_name'] = movie_name  # Update movie name in frame marker

                    self.movie_segments.append({
                        'movie_name': movie_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'frames': frames,
                        'frame_count': frame_count or len(frames)
                    })

                    potential_movies.append({
                        'movie_name': movie_name,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                    
        return potential_movies
    
    def _create_segments_from_frame_continuity(self, has_frame_markers, end_markers, default_movie_name):
        """Create segments based on frame number continuity if no explicit markers exist"""
        if not self.movie_segments and has_frame_markers:
            # Sort frames by timestamp
            sorted_frames = sorted(self.frame_markers, key=lambda x: x['timestamp'])

            # Group frames into segments based on frame number continuity
            segments = []
            current_segment = []

            for i, frame in enumerate(sorted_frames):
                if not current_segment or (frame['frame'] == current_segment[-1]['frame'] + 1 or
                                           (frame['frame'] == 1 and len(current_segment) > 0)):
                    # Continue current segment or start new one if frame is 1
                    if frame['frame'] == 1 and len(current_segment) > 0:
                        # Start of a new segment
                        segments.append(current_segment)
                        current_segment = [frame]
                    else:
                        # Continue current segment
                        current_segment.append(frame)
                else:
                    # Non-consecutive frame, start a new segment
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = [frame]

            # Add the last segment if it exists
            if current_segment:
                segments.append(current_segment)

            # Create movie segments from frame groupings
            for i, segment in enumerate(segments):
                if segment:
                    # Check if any end_markers align with the end of this segment
                    movie_name = None
                    frame_count = None

                    # Find the closest end marker after the last frame in this segment
                    last_frame_time = segment[-1]['timestamp']
                    closest_end = None
                    for end in end_markers:
                        if end['timestamp'] >= last_frame_time:
                            if closest_end is None or end['timestamp'] < closest_end['timestamp']:
                                closest_end = end

                    if closest_end and closest_end['timestamp'] - last_frame_time < 1000:  # Within 1 second
                        movie_name = closest_end['name']
                        frame_count = closest_end.get('frame_count', len(segment))

                    # Use default name if we didn't find a matching end marker
                    if movie_name is None:
                        movie_name = f"{default_movie_name}_{i + 1}"
                        frame_count = len(segment)

                    start_time = segment[0]['timestamp']
                    end_time = segment[-1]['timestamp']

                    # Create frame mapping
                    segment_frames = {frame['frame']: frame['timestamp'] for frame in segment}

                    # Update movie name in frame markers
                    for frame in segment:
                        frame['movie_name'] = movie_name

                    self.movie_segments.append({
                        'movie_name': movie_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'frames': segment_frames,
                        'frame_count': frame_count
                    })
    
    def _create_fallback_segment(self, end_markers, default_movie_name):
        """Create a fallback segment if no other segments were created but we have sample data"""
        if not self.movie_segments and self.sample_data:
            # Find first and last timestamps
            first_timestamp = self.sample_data[0]['timestamp']
            last_timestamp = self.sample_data[-1]['timestamp']

            # Check if we have any end markers
            if end_markers:
                movie_name = end_markers[0]['name']
                frame_count = end_markers[0].get('frame_count', 1)
            else:
                movie_name = default_movie_name
                frame_count = 1

            # Create a segment for the entire recording
            self.movie_segments.append({
                'movie_name': movie_name,
                'start_time': first_timestamp,
                'end_time': last_timestamp,
                'frames': {1: first_timestamp},  # Create at least one fake frame
                'frame_count': frame_count
            })
    
    def parse_messages(self):
        """Extract all message markers from the file"""
        # Initialize structures
        self.movie_segments = []
        default_movie_name = "unknown_movie"  # Default name when no movie name is found
        
        # Step 1: Collect all messages and frame markers
        has_frame_markers = self._collect_messages_and_frames()
        
        # Step 2: Identify movie markers
        start_markers, end_markers, potential_movies = self._identify_movie_markers()
        
        # Step 3: Process paired markers
        potential_movies = self._process_paired_markers(start_markers, end_markers, potential_movies)
        
        # Step 4: Process unpaired end markers
        potential_movies = self._process_unpaired_end_markers(end_markers, potential_movies)
        
        # Step 5: Create segments from frame continuity if needed
        self._create_segments_from_frame_continuity(has_frame_markers, end_markers, default_movie_name)
        
        # Step 6: Create fallback segment if needed
        self._create_fallback_segment(end_markers, default_movie_name)
        
        # Debug info
        print(f"Identified {len(self.movie_segments)} movie segments:")
        for i, segment in enumerate(self.movie_segments):
            print(f"  {i + 1}. {segment['movie_name']}: {segment['frame_count']} frames, "
                  f"{(segment['end_time'] - segment['start_time']) / 1000:.2f} seconds")

        return self.messages

    def parse_samples(self):
        """Extract eye movement samples (positions, pupil size, etc.)"""
        # Preallocate a reasonable sized list based on file size
        self.sample_data = []

        # Sample line pattern: timestamp  x_left  y_left  pupil_left  x_right  y_right  pupil_right  input ...
        for line in self.file_lines:
            # Skip non-data lines
            if not line[0].isdigit():
                continue

            parts = line.strip().split()
            if len(parts) < 8:  # Must have at least timestamp + basic eye data
                continue

            try:
                # Basic sample data
                sample = {
                    'timestamp': int(parts[0]),
                    'x_left': float(parts[1]) if parts[1] != '.' else np.nan,
                    'y_left': float(parts[2]) if parts[2] != '.' else np.nan,
                    'pupil_left': float(parts[3]) if parts[3] != '0.0' else np.nan,
                    'x_right': float(parts[4]) if parts[4] != '.' else np.nan,
                    'y_right': float(parts[5]) if parts[5] != '.' else np.nan,
                    'pupil_right': float(parts[6]) if parts[6] != '0.0' else np.nan,
                    'input': int(parts[7]) if parts[7].isdigit() else None
                }

                # Additional data if available
                if len(parts) > 8:
                    sample['cr_info'] = parts[8] if parts[8] != '.....' else None

                if len(parts) > 10:
                    sample['cr_left'] = float(parts[9]) if parts[9] != '.' else np.nan
                    sample['cr_right'] = float(parts[10]) if parts[10] != '.' else np.nan

                self.sample_data.append(sample)
            except (ValueError, IndexError) as e:
                # Skip problematic lines
                continue

        return len(self.sample_data)

    def parse_events(self):
        """Extract fixations, saccades, and blinks using optimized approach"""

        # Temporary storage for events with efficient dictionary operations
        temp_fixations = {'left': {}, 'right': {}}
        temp_saccades = {'left': {}, 'right': {}}
        temp_blinks = {'left': {}, 'right': {}}

        for line in self.file_lines:
            # Process each event type
            if line.startswith('SFIX'):
                match = self.fix_start_pattern.match(line)
                if match:
                    eye, timestamp = match.groups()
                    eye_key = 'left' if eye == 'L' else 'right'
                    temp_fixations[eye_key].setdefault(int(timestamp), {
                        'start_time': int(timestamp),
                        'end_time': None,
                        'duration': None,
                        'x': None,
                        'y': None,
                        'pupil': None
                    })

            elif line.startswith('EFIX'):
                match = self.fix_end_pattern.match(line)
                if match:
                    eye, start, end, duration, x, y, pupil = match.groups()
                    eye_key = 'left' if eye == 'L' else 'right'
                    start_time = int(start)

                    # Use dictionary update to efficiently merge data
                    entry = temp_fixations[eye_key].setdefault(start_time, {'start_time': start_time})
                    entry.update({
                        'end_time': int(end),
                        'duration': int(duration),
                        'x': float(x),
                        'y': float(y),
                        'pupil': float(pupil)
                    })

            elif line.startswith('SSACC'):
                match = self.sacc_start_pattern.match(line)
                if match:
                    eye, timestamp = match.groups()
                    eye_key = 'left' if eye == 'L' else 'right'
                    temp_saccades[eye_key].setdefault(int(timestamp), {
                        'start_time': int(timestamp),
                        'end_time': None,
                        'duration': None,
                        'start_x': None,
                        'start_y': None,
                        'end_x': None,
                        'end_y': None,
                        'amplitude': None,
                        'peak_velocity': None
                    })

            elif line.startswith('ESACC'):
                match = self.sacc_end_pattern.match(line)
                if match:
                    (eye, start, end, duration, start_x, start_y, end_x,
                     end_y, amplitude, peak_velocity) = match.groups()
                    eye_key = 'left' if eye == 'L' else 'right'
                    start_time = int(start)

                    # Use dictionary update for efficiency
                    entry = temp_saccades[eye_key].setdefault(start_time, {'start_time': start_time})
                    entry.update({
                        'end_time': int(end),
                        'duration': int(duration),
                        'start_x': float(start_x),
                        'start_y': float(start_y),
                        'end_x': float(end_x),
                        'end_y': float(end_y),
                        'amplitude': float(amplitude),
                        'peak_velocity': float(peak_velocity)
                    })

            elif line.startswith('SBLINK'):
                match = self.blink_start_pattern.match(line)
                if match:
                    eye, timestamp = match.groups()
                    eye_key = 'left' if eye == 'L' else 'right'
                    temp_blinks[eye_key].setdefault(int(timestamp), {
                        'start_time': int(timestamp),
                        'end_time': None,
                        'duration': None
                    })

            elif line.startswith('EBLINK'):
                match = self.blink_end_pattern.match(line)
                if match:
                    eye, start, end, duration = match.groups()
                    eye_key = 'left' if eye == 'L' else 'right'
                    start_time = int(start)

                    # Use dictionary update for efficiency
                    entry = temp_blinks[eye_key].setdefault(start_time, {'start_time': start_time})
                    entry.update({
                        'end_time': int(end),
                        'duration': int(duration)
                    })

        # Convert dictionaries to lists
        for eye in ['left', 'right']:
            self.fixations[eye] = list(temp_fixations[eye].values())
            self.saccades[eye] = list(temp_saccades[eye].values())
            self.blinks[eye] = list(temp_blinks[eye].values())

        return {
            'fixations_left': len(self.fixations['left']),
            'fixations_right': len(self.fixations['right']),
            'saccades_left': len(self.saccades['left']),
            'saccades_right': len(self.saccades['right']),
            'blinks_left': len(self.blinks['left']),
            'blinks_right': len(self.blinks['right'])
        }

    def parse_file(self):
        """Parse all data from the ASC file"""
        print(f"Reading file: {self.file_path}")
        self.read_file()
        print("Parsing metadata...")
        self.parse_metadata()
        print("Parsing messages...")
        self.parse_messages()
        print("Parsing samples...")
        num_samples = self.parse_samples()
        print("Parsing events...")
        event_counts = self.parse_events()

        print(f"Parsed {num_samples} samples")
        print(f"Events: {event_counts}")

        return {
            'metadata': self.metadata,
            'samples': len(self.sample_data),
            'fixations': event_counts['fixations_left'] + event_counts['fixations_right'],
            'saccades': event_counts['saccades_left'] + event_counts['saccades_right'],
            'blinks': event_counts['blinks_left'] + event_counts['blinks_right'],
            'messages': len(self.messages),
            'frames': len(self.frame_markers)
        }

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert parsed data to pandas DataFrames"""
        dfs = {}

        # Samples DataFrame
        if self.sample_data:
            dfs['samples'] = pd.DataFrame(self.sample_data)

        # Fixations DataFrames
        if self.fixations['left']:
            dfs['fixations_left'] = pd.DataFrame(self.fixations['left'])
        if self.fixations['right']:
            dfs['fixations_right'] = pd.DataFrame(self.fixations['right'])

        # Saccades DataFrames
        if self.saccades['left']:
            dfs['saccades_left'] = pd.DataFrame(self.saccades['left'])
        if self.saccades['right']:
            dfs['saccades_right'] = pd.DataFrame(self.saccades['right'])

        # Blinks DataFrames
        if self.blinks['left']:
            dfs['blinks_left'] = pd.DataFrame(self.blinks['left'])
        if self.blinks['right']:
            dfs['blinks_right'] = pd.DataFrame(self.blinks['right'])

        # Messages DataFrame
        if self.messages:
            dfs['messages'] = pd.DataFrame(self.messages)

        # Frame markers DataFrame
        if self.frame_markers:
            dfs['frames'] = pd.DataFrame(self.frame_markers)

        # Create a unified eye metrics dataframe
        if self.sample_data:
            dfs['unified_eye_metrics'] = self.create_unified_metrics_df()

        return dfs

    def create_unified_metrics_df(self) -> pd.DataFrame:
        """
        Create a unified dataframe with all eye and head movement metrics.
        Combines the relevant data from samples into a single, easy-to-analyze format.
        Returns:
            DataFrame with timestamps and all eye/head metrics.
        """

        def create_event_mask(events, timestamps):
            mask = np.zeros(len(timestamps), dtype=bool)
            for event in events:
                if 'start_time' in event and 'end_time' in event and event['end_time'] is not None:
                    start, end = event['start_time'], event['end_time']
                    event_mask = (timestamps >= start) & (timestamps <= end)
                    mask |= event_mask
            return mask

        unified_df = pd.DataFrame(self.sample_data)

        # Calculate head movement metrics
        if {'x_left', 'cr_left', 'x_right', 'cr_right'}.issubset(unified_df.columns):
            unified_df['head_movement_left_x'] = unified_df['x_left'] - unified_df['cr_left']
            unified_df['head_movement_right_x'] = unified_df['x_right'] - unified_df['cr_right']
            unified_df['head_movement_magnitude'] = np.sqrt(
                unified_df['head_movement_left_x'] ** 2 + unified_df['head_movement_right_x'] ** 2
            )

        # Calculate inter-pupil distance
        if {'x_left', 'x_right', 'y_left', 'y_right'}.issubset(unified_df.columns):
            unified_df['inter_pupil_distance'] = np.sqrt(
                (unified_df['x_right'] - unified_df['x_left']) ** 2 +
                (unified_df['y_right'] - unified_df['y_left']) ** 2
            )

        # Init event flags
        for eye in ['left', 'right']:
            unified_df[f'is_fixation_{eye}'] = False
            unified_df[f'is_saccade_{eye}'] = False
            unified_df[f'is_blink_{eye}'] = False

        # Calculate gaze velocity
        if len(unified_df) > 1:
            timestamps = unified_df['timestamp'].values
            for eye in ['left', 'right']:
                x_col, y_col = f'x_{eye}', f'y_{eye}'
                vel_col = f'gaze_velocity_{eye}'

                if {x_col, y_col}.issubset(unified_df.columns):
                    x_pos = unified_df[x_col].values
                    y_pos = unified_df[y_col].values
                    dt = np.diff(timestamps) / 1000.0
                    dx = np.diff(x_pos)
                    dy = np.diff(y_pos)

                    distances = np.sqrt(dx ** 2 + dy ** 2)
                    velocities = np.zeros_like(timestamps, dtype=float)
                    velocities[1:] = distances / dt
                    velocities[0] = np.nan

                    unified_df[vel_col] = np.where(np.isfinite(velocities), velocities, np.nan)

        # Assign event masks
        timestamps = unified_df['timestamp'].values
        for eye in ['left', 'right']:
            unified_df[f'is_fixation_{eye}'] = create_event_mask(self.fixations[eye], timestamps)
            unified_df[f'is_saccade_{eye}'] = create_event_mask(self.saccades[eye], timestamps)
            unified_df[f'is_blink_{eye}'] = create_event_mask(self.blinks[eye], timestamps)

        # Add movie_name and frame_number
        unified_df['movie_name'] = None
        unified_df['frame_number'] = None

        # Process each movie segment separately
        if hasattr(self, 'movie_segments') and self.movie_segments:
            for segment in self.movie_segments:
                movie_name = segment['movie_name']
                start_time = segment['start_time']
                end_time = segment['end_time']
                frames = segment.get('frames', {})

                # Assign movie name to all samples within this segment
                movie_mask = (unified_df['timestamp'] >= start_time) & (unified_df['timestamp'] <= end_time)
                unified_df.loc[movie_mask, 'movie_name'] = movie_name

                # Sort frames by timestamp for proper assignment
                if frames:
                    frame_timestamps = sorted(frames.items(), key=lambda x: x[1])

                    # If we only have one frame, assign it to all samples in this segment
                    if len(frame_timestamps) == 1:
                        unified_df.loc[movie_mask, 'frame_number'] = frame_timestamps[0][0]
                    else:
                        # Assign frames based on timestamp ranges
                        for i in range(len(frame_timestamps) - 1):
                            current_frame, current_ts = frame_timestamps[i]
                            next_ts = frame_timestamps[i + 1][1]
                            frame_mask = (unified_df['timestamp'] >= current_ts) & (unified_df['timestamp'] < next_ts)
                            unified_df.loc[frame_mask, 'frame_number'] = current_frame

                        # Handle the last frame
                        last_frame, last_ts = frame_timestamps[-1]
                        last_mask = (unified_df['timestamp'] >= last_ts) & (unified_df['timestamp'] <= end_time)
                        unified_df.loc[last_mask, 'frame_number'] = last_frame

        # Reorder columns
        desired_order = [
            'timestamp', 'movie_name', 'frame_number', 'x_left', 'y_left', 'pupil_left',
            'x_right', 'y_right', 'pupil_right', 'input', 'cr_info',
            'cr_left', 'cr_right', 'head_movement_left_x', 'head_movement_right_x',
            'head_movement_magnitude', 'inter_pupil_distance',
            'gaze_velocity_left', 'gaze_velocity_right',
            'is_fixation_left', 'is_fixation_right',
            'is_saccade_left', 'is_saccade_right',
            'is_blink_left', 'is_blink_right'
        ]
        actual_columns = list(unified_df.columns)
        ordered_columns = [col for col in desired_order if col in actual_columns]
        remaining_columns = [col for col in actual_columns if col not in desired_order]
        unified_df = unified_df[ordered_columns + remaining_columns]

        return unified_df

    def save_to_csv(self, output_dir: str = None):
        """
        Save all DataFrames to CSV files, with separate folders for each movie.

        Args:
            output_dir: Directory to save the files

        Returns:
            List of saved file paths
        """
        if output_dir is None:
            # Use same directory as the ASC file
            output_dir = os.path.dirname(self.file_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get participant ID from file name
        participant_id = os.path.splitext(os.path.basename(self.file_path))[0]

        # Create general directory for participant-wide data
        general_dir = os.path.join(output_dir, "general")
        os.makedirs(general_dir, exist_ok=True)

        # Get all dataframes
        dfs = self.to_dataframes()
        saved_files = []

        # Handle unified eye metrics specially - separate by movie
        if 'unified_eye_metrics' in dfs:
            unified_df = dfs['unified_eye_metrics']

            # If we have movie segments, split by movie
            if hasattr(self, 'movie_segments') and self.movie_segments:
                for movie_segment in self.movie_segments:
                    movie_name = movie_segment['movie_name']
                    start_time = movie_segment['start_time']
                    end_time = movie_segment['end_time']

                    # Clean movie name for folder and filename
                    clean_movie_name = re.sub(r'[^\w\d-]', '_',
                                              os.path.splitext(movie_name)[0] if '.' in movie_name else movie_name)

                    # Create movie directory at the top level
                    movie_dir = os.path.join(output_dir, clean_movie_name)
                    os.makedirs(movie_dir, exist_ok=True)

                    # Create plots directory inside movie directory
                    plots_dir = os.path.join(movie_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)

                    # Filter data for this movie segment
                    movie_mask = (unified_df['timestamp'] >= start_time) & (unified_df['timestamp'] <= end_time)
                    movie_data = unified_df[movie_mask].copy()

                    # We can drop the movie_name column since it's redundant with the folder
                    if 'movie_name' in movie_data.columns:
                        movie_data = movie_data.drop(columns=['movie_name'])

                    if not movie_data.empty:
                        # Save to movie-specific file
                        movie_file_path = os.path.join(movie_dir,
                                                       f"{participant_id}_unified_eye_metrics_{clean_movie_name}.csv")
                        movie_data.to_csv(movie_file_path, index=False)
                        saved_files.append(movie_file_path)
                        print(f"Saved unified eye metrics for movie {clean_movie_name} to {movie_file_path}")
            else:
                # No movie segments - save the whole unified dataframe
                print("Warning: No movie segments found. Creating a single unified metrics file.")
                output_path = os.path.join(output_dir, f"{participant_id}_unified_eye_metrics.csv")
                unified_df.to_csv(output_path, index=False)
                saved_files.append(output_path)
                print(f"Saved unified eye metrics to {output_path}")

        # Save other DataFrames to the general directory
        for df_name, df in dfs.items():
            if df_name != 'unified_eye_metrics':  # Skip unified metrics as it's handled specially
                output_path = os.path.join(general_dir, f"{participant_id}_{df_name}.csv")
                df.to_csv(output_path, index=False)
                saved_files.append(output_path)
                print(f"Saved {df_name} to {output_path}")

        # Save metadata as a separate file in the general directory
        if self.metadata:
            metadata_df = pd.DataFrame(list(self.metadata.items()), columns=['key', 'value'])
            metadata_path = os.path.join(general_dir, f"{participant_id}_metadata.csv")
            metadata_df.to_csv(metadata_path, index=False)
            saved_files.append(metadata_path)
            print(f"Saved metadata to {metadata_path}")

        return saved_files

    def extract_features(self) -> pd.DataFrame:
        """
        Extract key features for machine learning analysis focused on autism research.
        Optimized using aggregated statistics calculations.

        Returns:
            DataFrame with aggregate features that might be relevant for autism classification
        """
        features = {}

        # Basic metadata
        features['participant_id'] = os.path.splitext(os.path.basename(self.file_path))[0]

        # Sample statistics
        if len(self.sample_data) > 0:
            # Create DataFrame only once
            samples_df = pd.DataFrame(self.sample_data)

            # Calculate pupil size statistics using aggregation
            for eye in ['left', 'right']:
                pupil_col = f'pupil_{eye}'
                if pupil_col in samples_df.columns:
                    # Calculate all statistics at once
                    pupil_stats = samples_df[pupil_col].agg(['mean', 'std', 'min', 'max'])
                    features[f'pupil_{eye}_mean'] = pupil_stats['mean']
                    features[f'pupil_{eye}_std'] = pupil_stats['std']
                    features[f'pupil_{eye}_min'] = pupil_stats['min']
                    features[f'pupil_{eye}_max'] = pupil_stats['max']

            # Gaze position variability (reflects scan patterns)
            for eye in ['left', 'right']:
                x_col, y_col = f'x_{eye}', f'y_{eye}'
                if x_col in samples_df.columns and y_col in samples_df.columns:
                    # Calculate standard deviations
                    gaze_stats_x = samples_df[x_col].agg(['std', 'min', 'max'])
                    gaze_stats_y = samples_df[y_col].agg(['std', 'min', 'max'])

                    features[f'gaze_{eye}_x_std'] = gaze_stats_x['std']
                    features[f'gaze_{eye}_y_std'] = gaze_stats_y['std']

                    # Calculate dispersion (total area covered by gaze)
                    x_range = gaze_stats_x['max'] - gaze_stats_x['min']
                    y_range = gaze_stats_y['max'] - gaze_stats_y['min']
                    features[f'gaze_{eye}_dispersion'] = x_range * y_range if not np.isnan(x_range) and not np.isnan(
                        y_range) else np.nan

            # Head movement features - vectorized calculations
            if 'cr_left' in samples_df.columns and 'x_left' in samples_df.columns:
                # Calculate head movement metrics vectorized
                samples_df['head_movement_left_x'] = samples_df['x_left'] - samples_df['cr_left']
                samples_df['head_movement_right_x'] = samples_df['x_right'] - samples_df['cr_right']

                # Calculate magnitudes
                samples_df['head_movement_magnitude'] = np.sqrt(
                    samples_df['head_movement_left_x'] ** 2 + samples_df['head_movement_right_x'] ** 2
                )

                # Extract features with aggregation
                head_movement_stats = samples_df['head_movement_magnitude'].agg(['mean', 'std', 'max'])
                features['head_movement_mean'] = head_movement_stats['mean']
                features['head_movement_std'] = head_movement_stats['std']
                features['head_movement_max'] = head_movement_stats['max']

                # Calculate movement frequency - number of direction changes
                head_dir_changes = ((samples_df['head_movement_magnitude'].diff() > 0) !=
                                    (samples_df['head_movement_magnitude'].shift().diff() > 0)).sum()
                features['head_movement_frequency'] = head_dir_changes / (
                        len(samples_df) / 500)  # Assuming 500Hz sampling

            # Inter-pupil distance (can indicate depth changes or vergence)
            if 'x_left' in samples_df.columns and 'x_right' in samples_df.columns:
                # Vectorized calculation
                samples_df['inter_pupil_distance'] = np.sqrt(
                    (samples_df['x_right'] - samples_df['x_left']) ** 2 +
                    (samples_df['y_right'] - samples_df['y_left']) ** 2
                )

                ipd_stats = samples_df['inter_pupil_distance'].agg(['mean', 'std'])
                features['inter_pupil_distance_mean'] = ipd_stats['mean']
                features['inter_pupil_distance_std'] = ipd_stats['std']

        # Fixation features - with more efficient calculations
        for eye in ['left', 'right']:
            if self.fixations[eye]:
                fix_df = pd.DataFrame(self.fixations[eye])
                if not fix_df.empty and 'duration' in fix_df.columns:
                    features[f'fixation_{eye}_count'] = len(fix_df)

                    # Calculate multiple statistics at once
                    duration_stats = fix_df['duration'].agg(['mean', 'std'])
                    features[f'fixation_{eye}_duration_mean'] = duration_stats['mean']
                    features[f'fixation_{eye}_duration_std'] = duration_stats['std']

                    # Calculate fixation rate
                    if self.sample_data:
                        # Calculate recording duration in seconds
                        recording_duration = (self.sample_data[-1]['timestamp'] - self.sample_data[0][
                            'timestamp']) / 1000
                        if recording_duration > 0:
                            features[f'fixation_{eye}_rate'] = len(fix_df) / recording_duration
                        else:
                            features[f'fixation_{eye}_rate'] = np.nan
                    else:
                        features[f'fixation_{eye}_rate'] = np.nan

        # Saccade features - with more efficient calculations
        for eye in ['left', 'right']:
            if self.saccades[eye]:
                sacc_df = pd.DataFrame(self.saccades[eye])
                if not sacc_df.empty:
                    features[f'saccade_{eye}_count'] = len(sacc_df)

                    # Calculate amplitude statistics
                    if 'amplitude' in sacc_df.columns:
                        amp_stats = sacc_df['amplitude'].agg(['mean', 'std'])
                        features[f'saccade_{eye}_amplitude_mean'] = amp_stats['mean']
                        features[f'saccade_{eye}_amplitude_std'] = amp_stats['std']

                    # Calculate duration and velocity
                    if 'duration' in sacc_df.columns:
                        features[f'saccade_{eye}_duration_mean'] = sacc_df['duration'].mean()

                    if 'peak_velocity' in sacc_df.columns:
                        features[f'saccade_{eye}_peak_velocity_mean'] = sacc_df['peak_velocity'].mean()

        # Blink features - with more efficient calculations
        for eye in ['left', 'right']:
            if self.blinks[eye]:
                blink_df = pd.DataFrame(self.blinks[eye])
                if not blink_df.empty and 'duration' in blink_df.columns:
                    features[f'blink_{eye}_count'] = len(blink_df)
                    features[f'blink_{eye}_duration_mean'] = blink_df['duration'].mean()

                    # Calculate blink rate
                    if self.sample_data:
                        # Calculate recording duration in seconds
                        recording_duration = (self.sample_data[-1]['timestamp'] - self.sample_data[0][
                            'timestamp']) / 1000
                        if recording_duration > 0:
                            features[f'blink_{eye}_rate'] = len(blink_df) / recording_duration
                        else:
                            features[f'blink_{eye}_rate'] = np.nan
                    else:
                        features[f'blink_{eye}_rate'] = np.nan

        # Create a single-row DataFrame
        features_df = pd.DataFrame([features])
        return features_df


def process_asc_file(file_path: str, output_dir: str = None, extract_features: bool = True,
                     unified_only: bool = False) -> dict:
    """
    Process a single ASC file and return all parsed data.

    Args:
        file_path: Path to the ASC file
        output_dir: Directory to save output CSV files
        extract_features: Whether to extract aggregate features for ML
        unified_only: If True, only save the unified eye metrics CSV file

    Returns:
        Dictionary with parsing results
    """
    parser = EyeLinkASCParser(file_path)
    summary = parser.parse_file()

    # Get all dataframes
    dataframes = parser.to_dataframes()

    results = {
        'summary': summary,
        'dataframes': dataframes
    }

    if output_dir:
        if unified_only:
            # Only save the unified metrics file
            if 'unified_eye_metrics' in dataframes:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_unified_eye_metrics.csv")

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                dataframes['unified_eye_metrics'].to_csv(output_path, index=False)
                print(f"Saved unified eye metrics to {output_path}")
                results['saved_files'] = [output_path]
        else:
            # Save all dataframes
            results['saved_files'] = parser.save_to_csv(output_dir)

    if extract_features:
        results['features'] = parser.extract_features()
        print("\nExtracted features:")
        print(results['features'])

    return results


def process_multiple_files(file_paths: List[str], output_dir: str = None,
                           unified_only: bool = False) -> pd.DataFrame:
    """
    Process multiple ASC files and combine their features.

    Args:
        file_paths: List of paths to ASC files
        output_dir: Directory to save output CSV files
        unified_only: If True, only save the unified eye metrics CSV files

    Returns:
        DataFrame with combined features from all files
    """
    all_features = []
    all_unified_metrics = []

    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}")
        result = process_asc_file(file_path, output_dir, unified_only=unified_only)

        if 'features' in result:
            all_features.append(result['features'])

        # Also collect unified metrics dataframes if requested
        if unified_only and 'dataframes' in result and 'unified_eye_metrics' in result['dataframes']:
            # Add a participant ID column
            participant_id = os.path.splitext(os.path.basename(file_path))[0]
            df = result['dataframes']['unified_eye_metrics'].copy()
            df['participant_id'] = participant_id
            all_unified_metrics.append(df)

    # Combine all features into a single DataFrame
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)

        # Save combined features if output_dir is provided
        if output_dir:
            combined_path = os.path.join(output_dir, "combined_features.csv")
            combined_features.to_csv(combined_path, index=False)
            print(f"\nSaved combined features to {combined_path}")

    # Combine all unified metrics if requested
    if unified_only and all_unified_metrics:
        combined_metrics = pd.concat(all_unified_metrics, ignore_index=True)

        # Save combined unified metrics if output_dir is provided
        if output_dir:
            combined_metrics_path = os.path.join(output_dir, "all_participants_unified_metrics.csv")
            combined_metrics.to_csv(combined_metrics_path, index=False)
            print(f"\nSaved combined unified metrics from all participants to {combined_metrics_path}")

    # Return the combined features (for ML/DL analysis)
    if all_features:
        return pd.concat(all_features, ignore_index=True)

    return pd.DataFrame()


def load_csv_file(file_path: str, output_dir: str = None, extract_features: bool = True) -> dict:
    """
    Load a unified eye metrics CSV file and return it in the same format as process_asc_file.
    
    This function allows for directly loading a previously processed eye metrics CSV file,
    bypassing the slow ASC parsing process.
    
    Args:
        file_path: Path to the unified eye metrics CSV file
        output_dir: Directory to save output files (if needed)
        extract_features: Whether to extract aggregate features for ML
    
    Returns:
        Dictionary with loaded data in the same format as process_asc_file
    """
    print(f"Loading CSV file: {file_path}")
    
    # Check if the file exists and has the right extension
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    if not file_path.lower().endswith('.csv'):
        raise ValueError(f"Expected a CSV file, got: {file_path}")
    
    # Load the unified eye metrics DataFrame
    unified_df = pd.read_csv(file_path)
    
    # Get the participant ID from the filename
    participant_id = os.path.splitext(os.path.basename(file_path))[0].replace('_unified_eye_metrics', '')
    
    # Initialize empty dataframes for the standard outputs
    # The main value is in the unified_eye_metrics DataFrame, but we'll create placeholders for the others
    dataframes = {
        'unified_eye_metrics': unified_df,
        'samples': pd.DataFrame(),
        'messages': pd.DataFrame(),
        'frames': pd.DataFrame()
    }
    
    # If the unified DataFrame has fixation, saccade, or blink flags, we can recreate those events
    event_dfs = {}
    
    # Extract fixation events if we have the appropriate columns
    if {'is_fixation_left', 'timestamp'}.issubset(unified_df.columns):
        fixations_left = extract_events_from_unified(unified_df, 'is_fixation_left', 'left')
        if not fixations_left.empty:
            event_dfs['fixations_left'] = fixations_left
            
    if {'is_fixation_right', 'timestamp'}.issubset(unified_df.columns):
        fixations_right = extract_events_from_unified(unified_df, 'is_fixation_right', 'right')
        if not fixations_right.empty:
            event_dfs['fixations_right'] = fixations_right
    
    # Extract saccade events
    if {'is_saccade_left', 'timestamp'}.issubset(unified_df.columns):
        saccades_left = extract_events_from_unified(unified_df, 'is_saccade_left', 'left')
        if not saccades_left.empty:
            event_dfs['saccades_left'] = saccades_left
            
    if {'is_saccade_right', 'timestamp'}.issubset(unified_df.columns):
        saccades_right = extract_events_from_unified(unified_df, 'is_saccade_right', 'right')
        if not saccades_right.empty:
            event_dfs['saccades_right'] = saccades_right
    
    # Extract blink events
    if {'is_blink_left', 'timestamp'}.issubset(unified_df.columns):
        blinks_left = extract_events_from_unified(unified_df, 'is_blink_left', 'left')
        if not blinks_left.empty:
            event_dfs['blinks_left'] = blinks_left
            
    if {'is_blink_right', 'timestamp'}.issubset(unified_df.columns):
        blinks_right = extract_events_from_unified(unified_df, 'is_blink_right', 'right')
        if not blinks_right.empty:
            event_dfs['blinks_right'] = blinks_right
    
    # Add the event dataframes to our results
    dataframes.update(event_dfs)
    
    # Create a basic summary from what we have
    summary = {
        'samples': len(unified_df),
        'fixations': sum([len(df) for name, df in event_dfs.items() if 'fixation' in name]),
        'saccades': sum([len(df) for name, df in event_dfs.items() if 'saccade' in name]),
        'blinks': sum([len(df) for name, df in event_dfs.items() if 'blink' in name]),
        'messages': 0,
        'frames': 0
    }
    
    # Count unique frames if we have frame numbers
    if 'frame_number' in unified_df.columns:
        summary['frames'] = unified_df['frame_number'].nunique()
    
    # Prepare the results dictionary
    results = {
        'summary': summary,
        'dataframes': dataframes
    }
    
    # Extract features if requested
    if extract_features:
        results['features'] = extract_features_from_unified(unified_df, participant_id)
        print("\nExtracted features:")
        print(results['features'])
    
    # Save to output directory if requested
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Copy the unified metrics file to the output directory
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        if output_path != file_path:  # Don't copy if it's the same file
            unified_df.to_csv(output_path, index=False)
            results['saved_files'] = [output_path]
            print(f"Saved unified eye metrics to {output_path}")
    
    return results


def extract_events_from_unified(unified_df: pd.DataFrame, flag_column: str, eye: str) -> pd.DataFrame:
    """
    Extract event data (fixations, saccades, blinks) from the unified DataFrame based on flag columns.
    
    Args:
        unified_df: The unified eye metrics DataFrame
        flag_column: The column with boolean flags for this event type (e.g., 'is_fixation_left')
        eye: 'left' or 'right' - which eye this data is for
        
    Returns:
        DataFrame containing the extracted events
    """
    # Get consecutive runs of True values in the flag column
    event_df = unified_df[unified_df[flag_column]]
    
    if event_df.empty:
        return pd.DataFrame()
    
    # Group consecutive events
    event_df = event_df.copy()
    event_df['group'] = (event_df['timestamp'].diff() > 100).cumsum()
    
    # For each group, create an event
    events = []
    
    for group_id, group_df in event_df.groupby('group'):
        # Calculate basic metrics
        start_time = group_df['timestamp'].min()
        end_time = group_df['timestamp'].max()
        duration = end_time - start_time
        
        event = {
            'start_time': start_time, 
            'end_time': end_time,
            'duration': duration
        }
        
        # For fixations, add position and pupil size
        if 'fixation' in flag_column:
            x_col, y_col, pupil_col = f'x_{eye}', f'y_{eye}', f'pupil_{eye}'
            if {x_col, y_col, pupil_col}.issubset(group_df.columns):
                event['x'] = group_df[x_col].mean()
                event['y'] = group_df[y_col].mean()
                event['pupil'] = group_df[pupil_col].mean()
        
        # For saccades, add start/end positions and amplitude
        if 'saccade' in flag_column:
            x_col, y_col = f'x_{eye}', f'y_{eye}'
            if {x_col, y_col}.issubset(group_df.columns):
                event['start_x'] = group_df[x_col].iloc[0]
                event['start_y'] = group_df[y_col].iloc[0]
                event['end_x'] = group_df[x_col].iloc[-1]
                event['end_y'] = group_df[y_col].iloc[-1]
                
                # Calculate amplitude
                dx = event['end_x'] - event['start_x']
                dy = event['end_y'] - event['start_y']
                event['amplitude'] = np.sqrt(dx**2 + dy**2)
                
                # Add peak velocity if available
                vel_col = f'gaze_velocity_{eye}'
                if vel_col in group_df.columns:
                    event['peak_velocity'] = group_df[vel_col].max()
        
        events.append(event)
    
    return pd.DataFrame(events)


def extract_features_from_unified(unified_df: pd.DataFrame, participant_id: str) -> pd.DataFrame:
    """
    Extract features from a unified eye metrics DataFrame.
    This is similar to the extract_features method in EyeLinkASCParser but works directly with a DataFrame.
    
    Args:
        unified_df: The unified eye metrics DataFrame
        participant_id: The ID of the participant
        
    Returns:
        DataFrame with extracted features
    """
    features = {}
    
    # Basic metadata
    features['participant_id'] = participant_id
    
    # Calculate pupil size statistics
    for eye in ['left', 'right']:
        pupil_col = f'pupil_{eye}'
        if pupil_col in unified_df.columns:
            # Calculate all statistics at once
            pupil_stats = unified_df[pupil_col].agg(['mean', 'std', 'min', 'max'])
            features[f'pupil_{eye}_mean'] = pupil_stats['mean']
            features[f'pupil_{eye}_std'] = pupil_stats['std']
            features[f'pupil_{eye}_min'] = pupil_stats['min']
            features[f'pupil_{eye}_max'] = pupil_stats['max']
    
    # Gaze position variability (reflects scan patterns)
    for eye in ['left', 'right']:
        x_col, y_col = f'x_{eye}', f'y_{eye}'
        if x_col in unified_df.columns and y_col in unified_df.columns:
            # Calculate standard deviations
            gaze_stats_x = unified_df[x_col].agg(['std', 'min', 'max'])
            gaze_stats_y = unified_df[y_col].agg(['std', 'min', 'max'])

            features[f'gaze_{eye}_x_std'] = gaze_stats_x['std']
            features[f'gaze_{eye}_y_std'] = gaze_stats_y['std']

            # Calculate dispersion (total area covered by gaze)
            x_range = gaze_stats_x['max'] - gaze_stats_x['min']
            y_range = gaze_stats_y['max'] - gaze_stats_y['min']
            features[f'gaze_{eye}_dispersion'] = x_range * y_range if not np.isnan(x_range) and not np.isnan(
                y_range) else np.nan
    
    # Head movement features
    if 'head_movement_magnitude' in unified_df.columns:
        head_movement_stats = unified_df['head_movement_magnitude'].agg(['mean', 'std', 'max'])
        features['head_movement_mean'] = head_movement_stats['mean']
        features['head_movement_std'] = head_movement_stats['std']
        features['head_movement_max'] = head_movement_stats['max']
    
    # Inter-pupil distance statistics
    if 'inter_pupil_distance' in unified_df.columns:
        ipd_stats = unified_df['inter_pupil_distance'].agg(['mean', 'std'])
        features['inter_pupil_distance_mean'] = ipd_stats['mean']
        features['inter_pupil_distance_std'] = ipd_stats['std']
    
    # Event statistics (count, rate, duration)
    for event_type in ['fixation', 'saccade', 'blink']:
        for eye in ['left', 'right']:
            flag_col = f'is_{event_type}_{eye}'
            if flag_col in unified_df.columns:
                # Count events (transitions from False to True)
                event_starts = (unified_df[flag_col] & ~unified_df[flag_col].shift(1).fillna(False)).sum()
                features[f'{event_type}_{eye}_count'] = event_starts
                
                # Calculate event rate (events per second)
                if len(unified_df) > 1:
                    duration_sec = (unified_df['timestamp'].max() - unified_df['timestamp'].min()) / 1000
                    features[f'{event_type}_{eye}_rate'] = event_starts / duration_sec if duration_sec > 0 else np.nan
                
                # Calculate mean duration of events
                # First, label each event with a group number
                is_event = unified_df[flag_col]
                if is_event.sum() > 0:
                    event_df = unified_df[is_event].copy()
                    event_df['group'] = (event_df['timestamp'].diff() > 100).cumsum()
                    
                    # For each group, calculate duration
                    durations = []
                    for _, group in event_df.groupby('group'):
                        duration = group['timestamp'].max() - group['timestamp'].min()
                        durations.append(duration)
                    
                    if durations:
                        features[f'{event_type}_{eye}_duration_mean'] = np.mean(durations)
    
    # Create a single-row DataFrame
    features_df = pd.DataFrame([features])
    return features_df


def load_multiple_csv_files(file_paths: List[str], output_dir: str = None) -> pd.DataFrame:
    """
    Load multiple unified eye metrics CSV files and combine their features.
    
    Args:
        file_paths: List of paths to unified eye metrics CSV files
        output_dir: Directory to save output files (if needed)
    
    Returns:
        DataFrame with combined features from all files
    """
    all_features = []
    all_unified_metrics = []
    
    for file_path in file_paths:
        print(f"\nLoading CSV file: {file_path}")
        try:
            result = load_csv_file(file_path, output_dir)
            
            if 'features' in result:
                all_features.append(result['features'])
            
            # Collect unified metrics dataframes
            if 'dataframes' in result and 'unified_eye_metrics' in result['dataframes']:
                # Add a participant ID column
                participant_id = os.path.splitext(os.path.basename(file_path))[0].replace('_unified_eye_metrics', '')
                df = result['dataframes']['unified_eye_metrics'].copy()
                if 'participant_id' not in df.columns:
                    df['participant_id'] = participant_id
                all_unified_metrics.append(df)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    # Combine all features into a single DataFrame
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Save combined features if output_dir is provided
        if output_dir:
            combined_path = os.path.join(output_dir, "combined_features.csv")
            combined_features.to_csv(combined_path, index=False)
            print(f"\nSaved combined features to {combined_path}")
    else:
        combined_features = pd.DataFrame()
    
    # Combine all unified metrics
    if all_unified_metrics:
        combined_metrics = pd.concat(all_unified_metrics, ignore_index=True)
        
        # Save combined unified metrics if output_dir is provided
        if output_dir:
            combined_metrics_path = os.path.join(output_dir, "all_participants_unified_metrics.csv")
            combined_metrics.to_csv(combined_metrics_path, index=False)
            print(f"\nSaved combined unified metrics from all participants to {combined_metrics_path}")
    
    # Return the combined features (for ML/DL analysis)
    return combined_features
