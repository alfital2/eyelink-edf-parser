import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Union, Optional


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

    def read_file(self):
        """Read the ASC file and store lines"""
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            self.file_lines = f.readlines()
        return len(self.file_lines)

    def parse_metadata(self):
        """Extract metadata from the file header"""
        meta_pattern = re.compile(r'^\*\* (.+?):\s*(.+)')
        for line in self.file_lines:
            if line.startswith('**'):
                match = meta_pattern.match(line)
                if match:
                    key, value = match.groups()
                    self.metadata[key.strip()] = value.strip()
            # Stop when we hit the end of the header
            elif not line.startswith('**') and not line.strip() == '':
                break

        return self.metadata

    def parse_messages(self):
        """Extract all message markers from the file"""
        msg_pattern = re.compile(r'^MSG\s+(\d+)\s+(.+)')

        for line in self.file_lines:
            if line.startswith('MSG'):
                match = msg_pattern.match(line)
                if match:
                    timestamp, content = match.groups()

                    # Collect video frame markers
                    frame_match = re.search(r'Play_Movie_Start FRAME #(\d+)', content)
                    if frame_match:
                        frame_num = int(frame_match.group(1))
                        self.frame_markers.append({
                            'timestamp': int(timestamp),
                            'frame': frame_num,
                            'content': content
                        })

                    # Extract calibration info
                    if '!CAL' in content:
                        # Store calibration data in a separate structure
                        if 'CALIBRATION' in content:
                            calibration_type = re.search(r'CALIBRATION\s+(\w+)', content)
                            if calibration_type:
                                self.calibration_info['type'] = calibration_type.group(1)
                        # Calibration validation results
                        validation_match = re.search(r'VALIDATION.+?(\w+)\s+ERROR\s+([\d.]+)\s+avg\.\s+([\d.]+)',
                                                     content)
                        if validation_match:
                            quality, avg_error, max_error = validation_match.groups()
                            self.calibration_info['quality'] = quality
                            self.calibration_info['avg_error'] = float(avg_error)

                    # Store general message
                    self.messages.append({
                        'timestamp': int(timestamp),
                        'content': content
                    })

        return self.messages

    def parse_samples(self):
        """Extract eye movement samples (positions, pupil size, etc.)"""
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
        """Extract fixations, saccades, and blinks"""
        # Event patterns
        fix_start_pattern = re.compile(r'^SFIX\s+([LR])\s+(\d+)')
        fix_end_pattern = re.compile(r'^EFIX\s+([LR])\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+(\d+)')

        sacc_start_pattern = re.compile(r'^SSACC\s+([LR])\s+(\d+)')
        sacc_end_pattern = re.compile(
            r'^ESACC\s+([LR])\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+(\d+)')

        blink_start_pattern = re.compile(r'^SBLINK\s+([LR])\s+(\d+)')
        blink_end_pattern = re.compile(r'^EBLINK\s+([LR])\s+(\d+)\s+(\d+)\s+(\d+)')

        for line in self.file_lines:
            # Fixation start
            match = fix_start_pattern.match(line)
            if match:
                eye, timestamp = match.groups()
                eye_key = 'left' if eye == 'L' else 'right'
                self.fixations[eye_key].append({
                    'start_time': int(timestamp),
                    'end_time': None,
                    'duration': None,
                    'x': None,
                    'y': None,
                    'pupil': None
                })
                continue

            # Fixation end
            match = fix_end_pattern.match(line)
            if match:
                eye, start, end, duration, x, y, pupil = match.groups()
                eye_key = 'left' if eye == 'L' else 'right'
                # Add full fixation data
                self.fixations[eye_key].append({
                    'start_time': int(start),
                    'end_time': int(end),
                    'duration': int(duration),
                    'x': float(x),
                    'y': float(y),
                    'pupil': float(pupil)
                })
                continue

            # Saccade start
            match = sacc_start_pattern.match(line)
            if match:
                eye, timestamp = match.groups()
                eye_key = 'left' if eye == 'L' else 'right'
                self.saccades[eye_key].append({
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
                continue

            # Saccade end
            match = sacc_end_pattern.match(line)
            if match:
                eye, start, end, duration, start_x, start_y, end_x, end_y, amplitude, peak_velocity = match.groups()
                eye_key = 'left' if eye == 'L' else 'right'
                self.saccades[eye_key].append({
                    'start_time': int(start),
                    'end_time': int(end),
                    'duration': int(duration),
                    'start_x': float(start_x),
                    'start_y': float(start_y),
                    'end_x': float(end_x),
                    'end_y': float(end_y),
                    'amplitude': float(amplitude),
                    'peak_velocity': float(peak_velocity)
                })
                continue

            # Blink start
            match = blink_start_pattern.match(line)
            if match:
                eye, timestamp = match.groups()
                eye_key = 'left' if eye == 'L' else 'right'
                self.blinks[eye_key].append({
                    'start_time': int(timestamp),
                    'end_time': None,
                    'duration': None
                })
                continue

            # Blink end
            match = blink_end_pattern.match(line)
            if match:
                eye, start, end, duration = match.groups()
                eye_key = 'left' if eye == 'L' else 'right'
                self.blinks[eye_key].append({
                    'start_time': int(start),
                    'end_time': int(end),
                    'duration': int(duration)
                })

        event_counts = {
            'fixations_left': len(self.fixations['left']),
            'fixations_right': len(self.fixations['right']),
            'saccades_left': len(self.saccades['left']),
            'saccades_right': len(self.saccades['right']),
            'blinks_left': len(self.blinks['left']),
            'blinks_right': len(self.blinks['right'])
        }

        return event_counts

    def parse_file(self):
        """Parse all data from the ASC file"""
        print(f"Reading file: {self.file_path}")
        self.read_file()
        print(f"Parsing metadata...")
        self.parse_metadata()
        print(f"Parsing messages...")
        self.parse_messages()
        print(f"Parsing samples...")
        num_samples = self.parse_samples()
        print(f"Parsing events...")
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
        This combines the relevant data from samples into a single, easy-to-analyze format.

        Returns:
            DataFrame with timestamps and all eye/head metrics
        """
        # Start with the sample data which has timestamps
        unified_df = pd.DataFrame(self.sample_data)

        # Calculate the head movement metrics
        if 'x_left' in unified_df.columns and 'cr_left' in unified_df.columns:
            # Calculate head movement for each eye
            # The distance between pupil center and corneal reflection changes with head movement
            unified_df['head_movement_left_x'] = unified_df['x_left'] - unified_df['cr_left']
            unified_df['head_movement_right_x'] = unified_df['x_right'] - unified_df['cr_right']

            # Calculate overall head movement magnitude (Euclidean distance)
            unified_df['head_movement_magnitude'] = np.sqrt(
                unified_df['head_movement_left_x'] ** 2 + unified_df['head_movement_right_x'] ** 2
            )

        # Calculate inter-pupil distance (can indicate depth changes)
        if 'x_left' in unified_df.columns and 'x_right' in unified_df.columns:
            unified_df['inter_pupil_distance'] = np.sqrt(
                (unified_df['x_right'] - unified_df['x_left']) ** 2 +
                (unified_df['y_right'] - unified_df['y_left']) ** 2
            )

        # Calculate gaze velocity for each eye
        unified_df['gaze_velocity_left'] = np.nan
        unified_df['gaze_velocity_right'] = np.nan

        # Calculate velocity (degree/second) based on position changes
        if len(unified_df) > 1:
            for eye in ['left', 'right']:
                x_col, y_col = f'x_{eye}', f'y_{eye}'
                vel_col = f'gaze_velocity_{eye}'

                # Calculate position difference
                x_diff = unified_df[x_col].diff()
                y_diff = unified_df[y_col].diff()

                # Calculate time difference in seconds
                time_diff = unified_df['timestamp'].diff() / 1000.0

                # Calculate Euclidean distance
                distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

                # Calculate velocity (pixels/second)
                velocity_values = distance / time_diff

                # Replace infinite values with NaN (fix for FutureWarning)
                velocity_values = velocity_values.replace([np.inf, -np.inf], np.nan)
                unified_df[vel_col] = velocity_values

        # Add indicators for events
        unified_df['is_fixation_left'] = False
        unified_df['is_fixation_right'] = False
        unified_df['is_saccade_left'] = False
        unified_df['is_saccade_right'] = False
        unified_df['is_blink_left'] = False
        unified_df['is_blink_right'] = False

        # Mark fixation periods
        for eye in ['left', 'right']:
            for fix in self.fixations[eye]:
                if 'start_time' in fix and 'end_time' in fix:
                    # Mark samples within fixation period
                    mask = (unified_df['timestamp'] >= fix['start_time']) & (unified_df['timestamp'] <= fix['end_time'])
                    unified_df.loc[mask, f'is_fixation_{eye}'] = True

        # Mark saccade periods
        for eye in ['left', 'right']:
            for sacc in self.saccades[eye]:
                if 'start_time' in sacc and 'end_time' in sacc:
                    # Mark samples within saccade period
                    mask = (unified_df['timestamp'] >= sacc['start_time']) & (
                                unified_df['timestamp'] <= sacc['end_time'])
                    unified_df.loc[mask, f'is_saccade_{eye}'] = True

        # Mark blink periods
        for eye in ['left', 'right']:
            for blink in self.blinks[eye]:
                if 'start_time' in blink and 'end_time' in blink:
                    # Mark samples within blink period
                    mask = (unified_df['timestamp'] >= blink['start_time']) & (
                                unified_df['timestamp'] <= blink['end_time'])
                    unified_df.loc[mask, f'is_blink_{eye}'] = True

        return unified_df

    def save_to_csv(self, output_dir: str = None):
        """Save all DataFrames to CSV files"""
        if output_dir is None:
            # Use same directory as the ASC file
            output_dir = os.path.dirname(self.file_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_name = os.path.splitext(os.path.basename(self.file_path))[0]

        # Get all dataframes
        dfs = self.to_dataframes()

        # Save each DataFrame to a separate CSV file
        saved_files = []
        for df_name, df in dfs.items():
            output_path = os.path.join(output_dir, f"{base_name}_{df_name}.csv")
            df.to_csv(output_path, index=False)
            saved_files.append(output_path)
            print(f"Saved {df_name} to {output_path}")

        # Save metadata as a separate file
        if self.metadata:
            metadata_df = pd.DataFrame(list(self.metadata.items()), columns=['key', 'value'])
            metadata_path = os.path.join(output_dir, f"{base_name}_metadata.csv")
            metadata_df.to_csv(metadata_path, index=False)
            saved_files.append(metadata_path)
            print(f"Saved metadata to {metadata_path}")

        return saved_files

    def extract_features(self) -> pd.DataFrame:
        """
        Extract key features for machine learning analysis focused on autism research.

        Returns:
            DataFrame with aggregate features that might be relevant for autism classification
        """
        features = {}

        # Basic metadata
        features['participant_id'] = os.path.splitext(os.path.basename(self.file_path))[0]

        # Sample statistics
        if len(self.sample_data) > 0:
            samples_df = pd.DataFrame(self.sample_data)

            # Pupil size features
            for eye in ['left', 'right']:
                pupil_col = f'pupil_{eye}'
                if pupil_col in samples_df.columns:
                    features[f'pupil_{eye}_mean'] = samples_df[pupil_col].mean()
                    features[f'pupil_{eye}_std'] = samples_df[pupil_col].std()
                    features[f'pupil_{eye}_min'] = samples_df[pupil_col].min()
                    features[f'pupil_{eye}_max'] = samples_df[pupil_col].max()

            # Gaze position variability (reflects scan patterns)
            for eye in ['left', 'right']:
                x_col, y_col = f'x_{eye}', f'y_{eye}'
                if x_col in samples_df.columns and y_col in samples_df.columns:
                    features[f'gaze_{eye}_x_std'] = samples_df[x_col].std()
                    features[f'gaze_{eye}_y_std'] = samples_df[y_col].std()

                    # Calculate dispersion (total area covered by gaze)
                    x_range = samples_df[x_col].max() - samples_df[x_col].min()
                    y_range = samples_df[y_col].max() - samples_df[y_col].min()
                    features[f'gaze_{eye}_dispersion'] = x_range * y_range if not np.isnan(x_range) and not np.isnan(
                        y_range) else np.nan

            # Head movement features
            if 'cr_left' in samples_df.columns and 'x_left' in samples_df.columns:
                # Calculate head movement metrics
                samples_df['head_movement_left_x'] = samples_df['x_left'] - samples_df['cr_left']
                samples_df['head_movement_right_x'] = samples_df['x_right'] - samples_df['cr_right']

                # Calculate magnitudes
                samples_df['head_movement_magnitude'] = np.sqrt(
                    samples_df['head_movement_left_x'] ** 2 + samples_df['head_movement_right_x'] ** 2
                )

                # Extract features
                features['head_movement_mean'] = samples_df['head_movement_magnitude'].mean()
                features['head_movement_std'] = samples_df['head_movement_magnitude'].std()
                features['head_movement_max'] = samples_df['head_movement_magnitude'].max()

                # Calculate movement frequency - number of direction changes
                head_dir_changes = ((samples_df['head_movement_magnitude'].diff() > 0) !=
                                    (samples_df['head_movement_magnitude'].shift().diff() > 0)).sum()
                features['head_movement_frequency'] = head_dir_changes / (
                            len(samples_df) / 500)  # Assuming 500Hz sampling

            # Inter-pupil distance (can indicate depth changes or vergence)
            if 'x_left' in samples_df.columns and 'x_right' in samples_df.columns:
                samples_df['inter_pupil_distance'] = np.sqrt(
                    (samples_df['x_right'] - samples_df['x_left']) ** 2 +
                    (samples_df['y_right'] - samples_df['y_left']) ** 2
                )

                features['inter_pupil_distance_mean'] = samples_df['inter_pupil_distance'].mean()
                features['inter_pupil_distance_std'] = samples_df['inter_pupil_distance'].std()

        # Fixation features
        for eye in ['left', 'right']:
            if self.fixations[eye]:
                fix_df = pd.DataFrame(self.fixations[eye])
                if not fix_df.empty and 'duration' in fix_df.columns:
                    features[f'fixation_{eye}_count'] = len(fix_df)
                    features[f'fixation_{eye}_duration_mean'] = fix_df['duration'].mean()
                    features[f'fixation_{eye}_duration_std'] = fix_df['duration'].std()
                    features[f'fixation_{eye}_rate'] = len(fix_df) / (
                                max(self.sample_data[-1]['timestamp'] - self.sample_data[0]['timestamp'],
                                    1) / 1000) if self.sample_data else np.nan

        # Saccade features
        for eye in ['left', 'right']:
            if self.saccades[eye]:
                sacc_df = pd.DataFrame(self.saccades[eye])
                if not sacc_df.empty:
                    if 'amplitude' in sacc_df.columns:
                        features[f'saccade_{eye}_count'] = len(sacc_df)
                        features[f'saccade_{eye}_amplitude_mean'] = sacc_df['amplitude'].mean()
                        features[f'saccade_{eye}_amplitude_std'] = sacc_df['amplitude'].std()
                    if 'duration' in sacc_df.columns:
                        features[f'saccade_{eye}_duration_mean'] = sacc_df['duration'].mean()
                    if 'peak_velocity' in sacc_df.columns:
                        features[f'saccade_{eye}_peak_velocity_mean'] = sacc_df['peak_velocity'].mean()

        # Blink features
        for eye in ['left', 'right']:
            if self.blinks[eye]:
                blink_df = pd.DataFrame(self.blinks[eye])
                if not blink_df.empty and 'duration' in blink_df.columns:
                    features[f'blink_{eye}_count'] = len(blink_df)
                    features[f'blink_{eye}_duration_mean'] = blink_df['duration'].mean()
                    features[f'blink_{eye}_rate'] = len(blink_df) / (
                                max(self.sample_data[-1]['timestamp'] - self.sample_data[0]['timestamp'],
                                    1) / 1000) if self.sample_data else np.nan

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

