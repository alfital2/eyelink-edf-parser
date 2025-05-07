"""
Feature Model

This module contains the FeatureModel class which is responsible for handling
eye movement features and their calculations in the Eye Movement Analysis application.
"""

from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd


class FeatureModel(QObject):
    """
    Model class representing eye movement features.
    
    This class manages the extraction, calculation, and organization of eye movement 
    features for analysis. It serves as the data model for the feature tables in the GUI.
    
    Attributes:
        features_df (pd.DataFrame): DataFrame containing calculated features
        feature_categories (list): List of feature categories and their organization
        feature_explanations (dict): Dictionary of explanations for each feature
    """
    # Define signals for feature data changes
    features_calculated = pyqtSignal(pd.DataFrame)  # Emitted when features are calculated
    features_updated = pyqtSignal(pd.DataFrame)     # Emitted when features are updated
    
    def __init__(self, feature_categories=None):
        """
        Initialize the FeatureModel.
        
        Args:
            feature_categories (list, optional): Predefined feature categories structure.
                If None, default categories will be loaded.
        """
        super().__init__()
        self.features_df = None
        self.feature_categories = feature_categories
        self.feature_explanations = self._initialize_feature_explanations()
    
    def load_feature_categories(self, feature_categories):
        """
        Load feature categories structure from an external source.
        
        Args:
            feature_categories (list): Feature categories definition
        """
        self.feature_categories = feature_categories
        
    def get_feature_categories(self):
        """
        Get the current feature categories structure.
        
        Returns:
            list: The feature categories structure
        """
        return self.feature_categories
    
    def calculate_features(self, eye_tracking_data):
        """
        Calculate eye movement features from eye tracking data.
        
        Args:
            eye_tracking_data (EyeTrackingData): The eye tracking data to analyze
            
        Returns:
            pd.DataFrame: DataFrame containing all calculated features
        """
        # Initialize an empty DataFrame to store all features
        features = {}
        
        # Store participant ID if available
        if hasattr(eye_tracking_data, 'participant_id'):
            features['participant_id'] = eye_tracking_data.participant_id
        
        # Calculate basic pupil size metrics
        self._calculate_pupil_size_features(eye_tracking_data, features)
        
        # Calculate gaze position metrics
        self._calculate_gaze_features(eye_tracking_data, features)
        
        # Calculate fixation metrics
        self._calculate_fixation_features(eye_tracking_data, features)
        
        # Calculate saccade metrics
        self._calculate_saccade_features(eye_tracking_data, features)
        
        # Calculate blink metrics
        self._calculate_blink_features(eye_tracking_data, features)
        
        # Calculate head movement metrics if available
        self._calculate_head_movement_features(eye_tracking_data, features)
        
        # Convert features dictionary to DataFrame
        self.features_df = pd.DataFrame([features])
        
        # Emit signal that features have been calculated
        self.features_calculated.emit(self.features_df)
        
        return self.features_df
    
    def _calculate_pupil_size_features(self, eye_tracking_data, features):
        """
        Calculate pupil size metrics.
        
        Args:
            eye_tracking_data (EyeTrackingData): The eye tracking data
            features (dict): Dictionary to store calculated features
        """
        # Get pupil data from the eye tracking data object
        if hasattr(eye_tracking_data, 'pupil_data'):
            pupil_data = eye_tracking_data.pupil_data
            
            # Calculate pupil size metrics for left eye
            if 'left' in pupil_data:
                left_pupil = pupil_data['left']
                features['pupil_left_mean'] = left_pupil.mean() if len(left_pupil) > 0 else float('nan')
                features['pupil_left_std'] = left_pupil.std() if len(left_pupil) > 0 else float('nan')
                features['pupil_left_min'] = left_pupil.min() if len(left_pupil) > 0 else float('nan')
                features['pupil_left_max'] = left_pupil.max() if len(left_pupil) > 0 else float('nan')
            
            # Calculate pupil size metrics for right eye
            if 'right' in pupil_data:
                right_pupil = pupil_data['right']
                features['pupil_right_mean'] = right_pupil.mean() if len(right_pupil) > 0 else float('nan')
                features['pupil_right_std'] = right_pupil.std() if len(right_pupil) > 0 else float('nan')
                features['pupil_right_min'] = right_pupil.min() if len(right_pupil) > 0 else float('nan')
                features['pupil_right_max'] = right_pupil.max() if len(right_pupil) > 0 else float('nan')
    
    def _calculate_gaze_features(self, eye_tracking_data, features):
        """
        Calculate gaze position metrics.
        
        Args:
            eye_tracking_data (EyeTrackingData): The eye tracking data
            features (dict): Dictionary to store calculated features
        """
        # Get gaze data from the eye tracking data object
        if hasattr(eye_tracking_data, 'gaze_data'):
            gaze_data = eye_tracking_data.gaze_data
            
            # Calculate gaze metrics for left eye
            if 'left' in gaze_data:
                left_gaze = gaze_data['left']
                if 'x' in left_gaze and 'y' in left_gaze:
                    features['gaze_left_x_std'] = left_gaze['x'].std() if len(left_gaze['x']) > 0 else float('nan')
                    features['gaze_left_y_std'] = left_gaze['y'].std() if len(left_gaze['y']) > 0 else float('nan')
                    
                    # Calculate dispersion (spatial variance) - approximate measure of gaze stability
                    if len(left_gaze['x']) > 0 and len(left_gaze['y']) > 0:
                        features['gaze_left_dispersion'] = (
                            (left_gaze['x'].max() - left_gaze['x'].min()) +
                            (left_gaze['y'].max() - left_gaze['y'].min())
                        )
                    else:
                        features['gaze_left_dispersion'] = float('nan')
            
            # Calculate gaze metrics for right eye
            if 'right' in gaze_data:
                right_gaze = gaze_data['right']
                if 'x' in right_gaze and 'y' in right_gaze:
                    features['gaze_right_x_std'] = right_gaze['x'].std() if len(right_gaze['x']) > 0 else float('nan')
                    features['gaze_right_y_std'] = right_gaze['y'].std() if len(right_gaze['y']) > 0 else float('nan')
                    
                    # Calculate dispersion (spatial variance)
                    if len(right_gaze['x']) > 0 and len(right_gaze['y']) > 0:
                        features['gaze_right_dispersion'] = (
                            (right_gaze['x'].max() - right_gaze['x'].min()) +
                            (right_gaze['y'].max() - right_gaze['y'].min())
                        )
                    else:
                        features['gaze_right_dispersion'] = float('nan')
    
    def _calculate_fixation_features(self, eye_tracking_data, features):
        """
        Calculate fixation metrics.
        
        Args:
            eye_tracking_data (EyeTrackingData): The eye tracking data
            features (dict): Dictionary to store calculated features
        """
        # Get fixation data from the eye tracking data object
        if hasattr(eye_tracking_data, 'fixations'):
            fixations = eye_tracking_data.fixations
            
            # Calculate fixation metrics for left eye
            if 'left' in fixations:
                left_fixations = fixations['left']
                features['fixation_left_count'] = len(left_fixations)
                
                if len(left_fixations) > 0:
                    # Extract fixation durations and calculate statistics
                    left_durations = [fix.get('duration', 0) for fix in left_fixations]
                    features['fixation_left_duration_mean'] = sum(left_durations) / len(left_durations)
                    
                    # Calculate standard deviation if we have more than one fixation
                    if len(left_durations) > 1:
                        mean = features['fixation_left_duration_mean']
                        features['fixation_left_duration_std'] = (
                            sum((d - mean) ** 2 for d in left_durations) / len(left_durations)
                        ) ** 0.5
                    else:
                        features['fixation_left_duration_std'] = 0
                    
                    # Calculate fixation rate (fixations per second)
                    if hasattr(eye_tracking_data, 'duration_sec') and eye_tracking_data.duration_sec > 0:
                        features['fixation_left_rate'] = len(left_fixations) / eye_tracking_data.duration_sec
                    else:
                        features['fixation_left_rate'] = float('nan')
                else:
                    features['fixation_left_duration_mean'] = float('nan')
                    features['fixation_left_duration_std'] = float('nan')
                    features['fixation_left_rate'] = float('nan')
            
            # Calculate fixation metrics for right eye
            if 'right' in fixations:
                right_fixations = fixations['right']
                features['fixation_right_count'] = len(right_fixations)
                
                if len(right_fixations) > 0:
                    # Extract fixation durations and calculate statistics
                    right_durations = [fix.get('duration', 0) for fix in right_fixations]
                    features['fixation_right_duration_mean'] = sum(right_durations) / len(right_durations)
                    
                    # Calculate standard deviation if we have more than one fixation
                    if len(right_durations) > 1:
                        mean = features['fixation_right_duration_mean']
                        features['fixation_right_duration_std'] = (
                            sum((d - mean) ** 2 for d in right_durations) / len(right_durations)
                        ) ** 0.5
                    else:
                        features['fixation_right_duration_std'] = 0
                    
                    # Calculate fixation rate (fixations per second)
                    if hasattr(eye_tracking_data, 'duration_sec') and eye_tracking_data.duration_sec > 0:
                        features['fixation_right_rate'] = len(right_fixations) / eye_tracking_data.duration_sec
                    else:
                        features['fixation_right_rate'] = float('nan')
                else:
                    features['fixation_right_duration_mean'] = float('nan')
                    features['fixation_right_duration_std'] = float('nan')
                    features['fixation_right_rate'] = float('nan')
    
    def _calculate_saccade_features(self, eye_tracking_data, features):
        """
        Calculate saccade metrics.
        
        Args:
            eye_tracking_data (EyeTrackingData): The eye tracking data
            features (dict): Dictionary to store calculated features
        """
        # Get saccade data from the eye tracking data object
        if hasattr(eye_tracking_data, 'saccades'):
            saccades = eye_tracking_data.saccades
            
            # Calculate saccade metrics for left eye
            if 'left' in saccades:
                left_saccades = saccades['left']
                features['saccade_left_count'] = len(left_saccades)
                
                if len(left_saccades) > 0:
                    # Extract saccade amplitudes and durations
                    left_amplitudes = [sacc.get('amplitude', 0) for sacc in left_saccades]
                    left_durations = [sacc.get('duration', 0) for sacc in left_saccades]
                    
                    # Calculate amplitude statistics
                    features['saccade_left_amplitude_mean'] = sum(left_amplitudes) / len(left_amplitudes)
                    if len(left_amplitudes) > 1:
                        mean = features['saccade_left_amplitude_mean']
                        features['saccade_left_amplitude_std'] = (
                            sum((a - mean) ** 2 for a in left_amplitudes) / len(left_amplitudes)
                        ) ** 0.5
                    else:
                        features['saccade_left_amplitude_std'] = 0
                    
                    # Calculate duration statistics
                    features['saccade_left_duration_mean'] = sum(left_durations) / len(left_durations)
                else:
                    features['saccade_left_amplitude_mean'] = float('nan')
                    features['saccade_left_amplitude_std'] = float('nan')
                    features['saccade_left_duration_mean'] = float('nan')
            
            # Calculate saccade metrics for right eye
            if 'right' in saccades:
                right_saccades = saccades['right']
                features['saccade_right_count'] = len(right_saccades)
                
                if len(right_saccades) > 0:
                    # Extract saccade amplitudes and durations
                    right_amplitudes = [sacc.get('amplitude', 0) for sacc in right_saccades]
                    right_durations = [sacc.get('duration', 0) for sacc in right_saccades]
                    
                    # Calculate amplitude statistics
                    features['saccade_right_amplitude_mean'] = sum(right_amplitudes) / len(right_amplitudes)
                    if len(right_amplitudes) > 1:
                        mean = features['saccade_right_amplitude_mean']
                        features['saccade_right_amplitude_std'] = (
                            sum((a - mean) ** 2 for a in right_amplitudes) / len(right_amplitudes)
                        ) ** 0.5
                    else:
                        features['saccade_right_amplitude_std'] = 0
                    
                    # Calculate duration statistics
                    features['saccade_right_duration_mean'] = sum(right_durations) / len(right_durations)
                else:
                    features['saccade_right_amplitude_mean'] = float('nan')
                    features['saccade_right_amplitude_std'] = float('nan')
                    features['saccade_right_duration_mean'] = float('nan')
    
    def _calculate_blink_features(self, eye_tracking_data, features):
        """
        Calculate blink metrics.
        
        Args:
            eye_tracking_data (EyeTrackingData): The eye tracking data
            features (dict): Dictionary to store calculated features
        """
        # Get blink data from the eye tracking data object
        if hasattr(eye_tracking_data, 'blinks'):
            blinks = eye_tracking_data.blinks
            
            # Calculate blink metrics for left eye
            if 'left' in blinks:
                left_blinks = blinks['left']
                features['blink_left_count'] = len(left_blinks)
                
                if len(left_blinks) > 0:
                    # Extract blink durations
                    left_durations = [blink.get('duration', 0) for blink in left_blinks]
                    features['blink_left_duration_mean'] = sum(left_durations) / len(left_durations)
                    
                    # Calculate blink rate (blinks per minute)
                    if hasattr(eye_tracking_data, 'duration_sec') and eye_tracking_data.duration_sec > 0:
                        features['blink_left_rate'] = (len(left_blinks) / eye_tracking_data.duration_sec) * 60
                    else:
                        features['blink_left_rate'] = float('nan')
                else:
                    features['blink_left_duration_mean'] = float('nan')
                    features['blink_left_rate'] = float('nan')
            
            # Calculate blink metrics for right eye
            if 'right' in blinks:
                right_blinks = blinks['right']
                features['blink_right_count'] = len(right_blinks)
                
                if len(right_blinks) > 0:
                    # Extract blink durations
                    right_durations = [blink.get('duration', 0) for blink in right_blinks]
                    features['blink_right_duration_mean'] = sum(right_durations) / len(right_durations)
                    
                    # Calculate blink rate (blinks per minute)
                    if hasattr(eye_tracking_data, 'duration_sec') and eye_tracking_data.duration_sec > 0:
                        features['blink_right_rate'] = (len(right_blinks) / eye_tracking_data.duration_sec) * 60
                    else:
                        features['blink_right_rate'] = float('nan')
                else:
                    features['blink_right_duration_mean'] = float('nan')
                    features['blink_right_rate'] = float('nan')
    
    def _calculate_head_movement_features(self, eye_tracking_data, features):
        """
        Calculate head movement metrics if available.
        
        Args:
            eye_tracking_data (EyeTrackingData): The eye tracking data
            features (dict): Dictionary to store calculated features
        """
        # Get head movement data from the eye tracking data object if available
        if hasattr(eye_tracking_data, 'head_movement'):
            head_data = eye_tracking_data.head_movement
            
            if len(head_data) > 0:
                # Calculate basic head movement statistics
                features['head_movement_mean'] = sum(head_data) / len(head_data)
                
                # Calculate standard deviation
                if len(head_data) > 1:
                    mean = features['head_movement_mean']
                    features['head_movement_std'] = (
                        sum((h - mean) ** 2 for h in head_data) / len(head_data)
                    ) ** 0.5
                else:
                    features['head_movement_std'] = 0
                
                # Calculate maximum head movement
                features['head_movement_max'] = max(head_data)
                
                # Calculate head movement frequency as changes per second
                # A change is defined as consecutive values where difference exceeds threshold
                if hasattr(eye_tracking_data, 'duration_sec') and eye_tracking_data.duration_sec > 0:
                    threshold = 0.1  # Example threshold for detecting significant movement
                    changes = sum(1 for i in range(1, len(head_data)) 
                                 if abs(head_data[i] - head_data[i-1]) > threshold)
                    features['head_movement_frequency'] = changes / eye_tracking_data.duration_sec
                else:
                    features['head_movement_frequency'] = float('nan')
            else:
                features['head_movement_mean'] = float('nan')
                features['head_movement_std'] = float('nan')
                features['head_movement_max'] = float('nan')
                features['head_movement_frequency'] = float('nan')
    
    def update_features(self, additional_features):
        """
        Update the features DataFrame with additional features.
        
        Args:
            additional_features (dict): Dictionary of additional features to add
            
        Returns:
            pd.DataFrame: Updated features DataFrame
        """
        if self.features_df is None:
            self.features_df = pd.DataFrame([additional_features])
        else:
            # Update existing features with new values
            for key, value in additional_features.items():
                self.features_df[key] = value
        
        # Emit signal that features have been updated
        self.features_updated.emit(self.features_df)
        
        return self.features_df
    
    def get_features(self):
        """
        Get the current features DataFrame.
        
        Returns:
            pd.DataFrame: The features DataFrame or None if not calculated
        """
        return self.features_df
    
    def _initialize_feature_explanations(self):
        """
        Initialize explanations for each feature.
        
        Returns:
            dict: Dictionary mapping feature keys to their explanations
        """
        # Create a dictionary of explanations for each feature
        return {
            # Pupil size metrics
            "pupil_left_mean": "Average pupil size for the left eye during the recording",
            "pupil_right_mean": "Average pupil size for the right eye during the recording",
            "pupil_left_std": "Standard deviation of pupil size for the left eye, indicating variability",
            "pupil_right_std": "Standard deviation of pupil size for the right eye, indicating variability",
            "pupil_left_min": "Minimum pupil size for the left eye during the recording",
            "pupil_right_min": "Minimum pupil size for the right eye during the recording",
            "pupil_left_max": "Maximum pupil size for the left eye during the recording",
            "pupil_right_max": "Maximum pupil size for the right eye during the recording",
            
            # Gaze position metrics
            "gaze_left_x_std": "Standard deviation of horizontal (X) gaze position for the left eye",
            "gaze_right_x_std": "Standard deviation of horizontal (X) gaze position for the right eye",
            "gaze_left_y_std": "Standard deviation of vertical (Y) gaze position for the left eye",
            "gaze_right_y_std": "Standard deviation of vertical (Y) gaze position for the right eye",
            "gaze_left_dispersion": "Spatial dispersion of gaze points for the left eye (max-min X + max-min Y)",
            "gaze_right_dispersion": "Spatial dispersion of gaze points for the right eye (max-min X + max-min Y)",
            
            # Fixation metrics
            "fixation_left_count": "Number of fixations detected for the left eye",
            "fixation_right_count": "Number of fixations detected for the right eye",
            "fixation_left_duration_mean": "Average duration of fixations for the left eye in milliseconds",
            "fixation_right_duration_mean": "Average duration of fixations for the right eye in milliseconds",
            "fixation_left_duration_std": "Standard deviation of fixation durations for the left eye",
            "fixation_right_duration_std": "Standard deviation of fixation durations for the right eye",
            "fixation_left_rate": "Number of fixations per second for the left eye",
            "fixation_right_rate": "Number of fixations per second for the right eye",
            
            # Saccade metrics
            "saccade_left_count": "Number of saccades detected for the left eye",
            "saccade_right_count": "Number of saccades detected for the right eye",
            "saccade_left_amplitude_mean": "Average amplitude (size) of saccades for the left eye in degrees",
            "saccade_right_amplitude_mean": "Average amplitude (size) of saccades for the right eye in degrees",
            "saccade_left_amplitude_std": "Standard deviation of saccade amplitudes for the left eye",
            "saccade_right_amplitude_std": "Standard deviation of saccade amplitudes for the right eye",
            "saccade_left_duration_mean": "Average duration of saccades for the left eye in milliseconds",
            "saccade_right_duration_mean": "Average duration of saccades for the right eye in milliseconds",
            
            # Blink metrics
            "blink_left_count": "Number of blinks detected for the left eye",
            "blink_right_count": "Number of blinks detected for the right eye",
            "blink_left_duration_mean": "Average duration of blinks for the left eye in milliseconds",
            "blink_right_duration_mean": "Average duration of blinks for the right eye in milliseconds",
            "blink_left_rate": "Number of blinks per minute for the left eye",
            "blink_right_rate": "Number of blinks per minute for the right eye",
            
            # Head movement metrics
            "head_movement_mean": "Average head movement magnitude during the recording",
            "head_movement_std": "Standard deviation of head movement, indicating variability",
            "head_movement_max": "Maximum head movement magnitude during the recording",
            "head_movement_frequency": "Frequency of significant head movements per second"
        }