"""
Eye Tracking Data Model

This module defines the EyeTrackingData class, which encapsulates eye tracking data
and provides methods for data access and manipulation.
"""

import os
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal


class EyeTrackingData(QObject):
    """
    Model class representing eye tracking data.
    
    This class encapsulates the raw eye tracking data (samples, fixations, 
    saccades, etc.) and provides methods for data access, filtering, and manipulation.
    It implements the Observable pattern using Qt signals to notify observers
    of data changes.
    """
    
    # Define signals for data changes
    data_loaded = pyqtSignal(object)  # Emitted when data is loaded
    data_processed = pyqtSignal(dict)  # Emitted when data is processed
    
    def __init__(self):
        """Initialize the eye tracking data model."""
        super().__init__()
        
        # Data storage
        self.raw_data = None  # Original raw data frame
        self.processed_data = None  # Processed data frame
        self.file_paths = []  # Source file paths
        self.file_type = None  # Type of files (ASC or CSV)
        self.participant_id = None  # ID of the participant
        
        # Summary statistics
        self.summary = {
            'samples': 0,
            'fixations': 0,
            'saccades': 0,
            'blinks': 0,
            'frames': 0
        }
        
        # Output paths
        self.output_dir = None
        
    def load_from_files(self, file_paths, file_type):
        """
        Load eye tracking data from files.
        
        Args:
            file_paths (list): Paths to data files
            file_type (str): Type of files ('ASC Files' or 'CSV Files')
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        self.file_paths = file_paths
        self.file_type = file_type
        
        try:
            # Set participant ID based on the first file
            base_name = os.path.basename(file_paths[0])
            self.participant_id = os.path.splitext(base_name)[0]
            
            # For CSV files, remove '_unified_eye_metrics' from the participant ID if present
            if self.file_type == "CSV Files" and self.participant_id.endswith('_unified_eye_metrics'):
                self.participant_id = self.participant_id.replace('_unified_eye_metrics', '')
            
            # Simple check if files exist
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    return False
            
            # Note: Actual data loading is delegated to processing methods
            # to avoid blocking the UI thread
            
            # Signal that files are ready for processing
            self.data_loaded.emit(self)
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def get_file_info(self):
        """
        Get information about the loaded files.
        
        Returns:
            dict: Dictionary with file information
        """
        return {
            'file_paths': self.file_paths,
            'file_type': self.file_type,
            'participant_id': self.participant_id
        }
    
    def set_output_directory(self, output_dir):
        """
        Set the output directory for processed data.
        
        Args:
            output_dir (str): Path to output directory
        """
        self.output_dir = output_dir
    
    def get_fixation_data(self):
        """
        Get fixation data for visualization.
        
        Returns:
            pd.DataFrame: DataFrame containing fixation data
        """
        if self.processed_data is None:
            return None
        
        # Return only fixation data
        return self.processed_data[self.processed_data['is_fixation_left'] | 
                                  self.processed_data['is_fixation_right']]
    
    def get_saccade_data(self):
        """
        Get saccade data for visualization.
        
        Returns:
            pd.DataFrame: DataFrame containing saccade data
        """
        if self.processed_data is None:
            return None
        
        # Return only saccade data
        return self.processed_data[self.processed_data['is_saccade_left'] | 
                                  self.processed_data['is_saccade_right']]
    
    def get_blink_data(self):
        """
        Get blink data for visualization.
        
        Returns:
            pd.DataFrame: DataFrame containing blink data
        """
        if self.processed_data is None:
            return None
        
        # Return only blink data
        return self.processed_data[self.processed_data['is_blink_left'] | 
                                  self.processed_data['is_blink_right']]
    
    def get_summary(self):
        """
        Get summary statistics about the data.
        
        Returns:
            dict: Dictionary with summary statistics
        """
        return self.summary
    
    def update_summary(self, summary_data):
        """
        Update summary statistics with processed data.
        
        Args:
            summary_data (dict): Dictionary with summary statistics
        """
        self.summary.update(summary_data)
    
    def set_processed_data(self, data, summary=None):
        """
        Set processed data and update summary.
        
        Args:
            data (pd.DataFrame): Processed data
            summary (dict, optional): Summary statistics
        """
        self.processed_data = data
        
        if summary:
            self.update_summary(summary)
        
        # Calculate summary statistics if not provided
        if not summary and data is not None:
            self.summary['samples'] = len(data)
            self.summary['fixations'] = data['is_fixation_left'].sum() + data['is_fixation_right'].sum() 
            self.summary['saccades'] = data['is_saccade_left'].sum() + data['is_saccade_right'].sum()
            self.summary['blinks'] = data['is_blink_left'].sum() + data['is_blink_right'].sum()
            if 'frame_number' in data.columns:
                self.summary['frames'] = data['frame_number'].nunique()
        
        # Emit signal with processed data
        self.data_processed.emit({
            'data': self.processed_data,
            'summary': self.summary
        })
    
    def get_movie_names(self):
        """
        Get list of movie names in the data.
        
        Returns:
            list: List of movie names
        """
        if self.processed_data is None or 'movie' not in self.processed_data.columns:
            return ['All Data']
        
        return ['All Data'] + self.processed_data['movie'].unique().tolist()
    
    def get_data_for_movie(self, movie_name):
        """
        Get data for a specific movie.
        
        Args:
            movie_name (str): Name of the movie
            
        Returns:
            pd.DataFrame: DataFrame filtered for the specified movie
        """
        if self.processed_data is None:
            return None
        
        if movie_name == 'All Data':
            return self.processed_data
        
        return self.processed_data[self.processed_data['movie'] == movie_name]