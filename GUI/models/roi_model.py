"""
ROI (Region of Interest) Model

This module defines the ROIModel class, which encapsulates data and functionality
related to Regions of Interest (ROIs) in eye tracking studies.
"""

import os
import json
from collections import defaultdict
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class ROIModel(QObject):
    """
    Model class representing Regions of Interest (ROIs) data.
    
    This class encapsulates ROI definitions, handles loading ROI data from files,
    and provides methods for ROI analysis and classification. It implements the 
    Observable pattern using Qt signals to notify observers of ROI data changes.
    """
    
    # Define signals for ROI data changes
    roi_loaded = pyqtSignal(object)  # Emitted when ROI data is loaded
    roi_analyzed = pyqtSignal(dict)  # Emitted when ROI analysis is completed
    
    def __init__(self):
        """Initialize the ROI model."""
        super().__init__()
        
        # ROI data storage
        self.roi_data = {}  # Dictionary mapping frame numbers to ROI definitions
        self.roi_file_path = None  # Path to the loaded ROI file
        self.frame_keys = {}  # Frame keys for faster lookup
        self.frame_range_map = {}  # Frame range mapping for faster nearest frame lookup
        
        # ROI analysis results
        self.roi_durations = {}  # Time spent on each ROI
        self.first_fixation_times = {}  # Time to first fixation for each ROI
        self.roi_revisits = {}  # Number of revisits to each ROI
        self.transition_matrix = defaultdict(lambda: defaultdict(int))  # ROI transition counts
        
        # ROI categories (social vs. non-social)
        self.social_rois = ['Face', 'Hand', 'Eyes', 'Mouth', 'Person', 'Body']
        self.nonsocial_rois = ['Background', 'Object', 'Bed', 'Couch', 'Torso', 'Floor', 'Wall', 'Toy']
        self.roi_categories = {}  # Mapping of ROI labels to categories (social/non-social)
        
    def load_from_file(self, file_path):
        """
        Load ROI data from a JSON file.
        
        Args:
            file_path (str): Path to ROI JSON file
            
        Returns:
            bool: True if ROI data was loaded successfully, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                raw_roi_data = json.load(f)
            
            # Check for new format with "annotations" key
            if "annotations" in raw_roi_data:
                self.roi_data = raw_roi_data["annotations"]
            else:
                # Legacy format
                self.roi_data = raw_roi_data
            
            self.roi_file_path = file_path
            
            # Preprocess ROI data for faster lookups
            self._preprocess_roi_data()
            
            # Signal that ROI data has been loaded
            self.roi_loaded.emit(self)
            return True
            
        except Exception as e:
            print(f"Error loading ROI data: {str(e)}")
            return False
    
    def _preprocess_roi_data(self):
        """
        Preprocess ROI data for faster lookups during analysis.
        
        This method:
        1. Converts frame keys to integers
        2. Creates a frame range map for quickly finding nearest frames
        3. Categorizes ROIs as social or non-social
        """
        # Convert frame keys to integers
        self.frame_keys = {}
        for key in self.roi_data.keys():
            try:
                frame_num = int(key)
                self.frame_keys[frame_num] = self.roi_data[key]
            except ValueError:
                continue
        
        # Create frame range map for faster nearest frame lookups
        frame_numbers = sorted(self.frame_keys.keys())
        self.frame_range_map = {}
        
        if frame_numbers:
            for i, frame in enumerate(frame_numbers):
                if i == 0:
                    # For the first frame, use it for anything less than the midpoint to the next frame
                    next_frame = frame_numbers[i + 1] if i + 1 < len(frame_numbers) else frame + 1000
                    self.frame_range_map[(0, (frame + next_frame) // 2)] = frame
                elif i == len(frame_numbers) - 1:
                    # For the last frame, use it for anything greater than the midpoint from the previous frame
                    prev_frame = frame_numbers[i - 1]
                    self.frame_range_map[((prev_frame + frame) // 2, float('inf'))] = frame
                else:
                    # For middle frames, use the midpoints between adjacent frames
                    prev_frame = frame_numbers[i - 1]
                    next_frame = frame_numbers[i + 1]
                    self.frame_range_map[((prev_frame + frame) // 2, (frame + next_frame) // 2)] = frame
        
        # Categorize ROIs
        self._categorize_rois()
    
    def _categorize_rois(self):
        """
        Categorize ROIs as social or non-social based on their labels.
        """
        # Collect all unique ROI labels
        all_roi_labels = set()
        for frame, rois in self.frame_keys.items():
            for roi in rois:
                if 'label' in roi:
                    all_roi_labels.add(roi['label'])
        
        # Categorize each ROI
        self.roi_categories = {}
        for roi in all_roi_labels:
            # Try to categorize based on exact matches first
            if roi in self.social_rois:
                self.roi_categories[roi] = 'Social'
            elif roi in self.nonsocial_rois:
                self.roi_categories[roi] = 'Non-Social'
            else:
                # If no exact match, try partial matching
                if any(social_term in roi.lower() for social_term in 
                       ['face', 'hand', 'eye', 'mouth', 'person', 'body']):
                    self.roi_categories[roi] = 'Social'
                else:
                    self.roi_categories[roi] = 'Non-Social'
    
    def is_point_in_roi(self, x, y, roi_coords, cache=None):
        """
        Check if a point (x, y) is inside a polygon defined by roi_coords.
        
        Args:
            x (float): Normalized x-coordinate (0-1)
            y (float): Normalized y-coordinate (0-1)
            roi_coords (list): List of coordinate dictionaries with 'x' and 'y' keys
            cache (dict, optional): Cache for polygon checks
            
        Returns:
            bool: True if the point is inside the ROI polygon, False otherwise
        """
        # Use cached result if available
        if cache is not None:
            cache_key = (tuple((coord['x'], coord['y']) for coord in roi_coords), x, y)
            if cache_key in cache:
                return cache[cache_key]
        
        # Extract points from coordinates
        points = [(coord['x'], coord['y']) for coord in roi_coords]
        
        # Need at least 3 points to form a polygon
        if len(points) < 3:
            return False
        
        # Ray casting algorithm
        inside = False
        j = len(points) - 1
        
        for i in range(len(points)):
            xi, yi = points[i]
            xj, yj = points[j]
            
            # Check if point is on an edge or vertex (exact match)
            if (yi == y and xi == x) or (yj == y and xj == x):
                return True
            
            # Check if the point is on a horizontal edge
            if (abs(yi - yj) < 1e-9) and (abs(yi - y) < 1e-9) and (min(xi, xj) <= x <= max(xi, xj)):
                return True
            
            # Ray casting - check if ray crosses this edge
            # Using a small epsilon for floating point comparison
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        # Cache the result if cache is provided
        if cache is not None:
            cache[cache_key] = inside
        
        return inside
    
    def find_roi_for_fixation(self, frame_num, x, y, cache=None):
        """
        Find which ROI a fixation point falls within.
        
        Args:
            frame_num (int): Frame number
            x (float): Normalized x-coordinate (0-1)
            y (float): Normalized y-coordinate (0-1)
            cache (dict, optional): Cache for polygon checks
            
        Returns:
            str: Label of the ROI containing the point, or None if not found
        """
        # Find the nearest frame in ROI data
        nearest_frame = None
        
        # Try the frame range map first
        for (start, end), frame in self.frame_range_map.items():
            if start <= frame_num < end:
                nearest_frame = frame
                break
        
        # If no match in the range map, fall back to nearest neighbor approach
        if nearest_frame is None:
            try:
                nearest_frame = min(self.frame_keys.keys(), key=lambda x: abs(x - frame_num))
            except:
                return None
        
        # Skip if the frame distance is too large
        frame_distance = abs(nearest_frame - frame_num)
        if frame_distance > 1000:  # Use a reasonable threshold
            return None
        
        # Get the ROIs for this frame
        rois_in_frame = self.frame_keys[nearest_frame]
        
        # Check each ROI in this frame
        for roi in rois_in_frame:
            if 'label' not in roi or 'coordinates' not in roi:
                continue
            
            label = roi['label']
            coords = roi['coordinates']
            
            # Check if point is inside this ROI
            if self.is_point_in_roi(x, y, coords, cache):
                return label
        
        return None
    
    def analyze_fixations(self, fixation_data, screen_width, screen_height):
        """
        Analyze fixation data to compute ROI statistics.
        
        Args:
            fixation_data (pd.DataFrame): DataFrame with fixation data
            screen_width (int): Width of the screen in pixels
            screen_height (int): Height of the screen in pixels
            
        Returns:
            dict: Dictionary with ROI analysis results
        """
        if fixation_data is None or fixation_data.empty or not self.frame_keys:
            return {}
        
        # Reset analysis results
        self.roi_durations = {}
        self.first_fixation_times = {}
        self.roi_revisits = {}
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        
        # Cache for polygon checks to avoid redundant calculations
        polygon_check_cache = {}
        
        # Track ROIs seen for revisit counting
        seen_rois = set()
        
        # Track last ROI for transition matrix
        last_roi = None
        
        # First timestamp for relative timing
        start_timestamp = fixation_data['timestamp'].min()
        
        # Process each fixation
        for idx, row in fixation_data.iterrows():
            if pd.isna(row['frame_number']):
                continue
            
            frame_num = int(row['frame_number'])
            
            # Get normalized coordinates
            if row['x_left'] > 1.0 or row['y_left'] > 1.0:
                x_norm = row['x_left'] / screen_width
                y_norm = row['y_left'] / screen_height
            else:
                x_norm = row['x_left']
                y_norm = row['y_left']
            
            # Find which ROI the fixation is in
            current_roi = self.find_roi_for_fixation(frame_num, x_norm, y_norm, polygon_check_cache)
            
            if current_roi:
                # Calculate time relative to start
                timestamp_sec = (row['timestamp'] - start_timestamp) / 1000.0  # Convert to seconds
                
                # Update ROI durations
                if current_roi not in self.roi_durations:
                    self.roi_durations[current_roi] = 0
                self.roi_durations[current_roi] += 1  # Each fixation counts as one unit
                
                # First fixation time
                if current_roi not in self.first_fixation_times or self.first_fixation_times[current_roi] is None:
                    self.first_fixation_times[current_roi] = timestamp_sec
                    seen_rois.add(current_roi)
                elif current_roi in seen_rois:
                    # This ROI has been seen before, count as revisit
                    if current_roi not in self.roi_revisits:
                        self.roi_revisits[current_roi] = 0
                    self.roi_revisits[current_roi] += 1
                
                # Transition matrix
                if last_roi is not None and last_roi != current_roi:
                    self.transition_matrix[last_roi][current_roi] += 1
                
                last_roi = current_roi
        
        # Compute social vs non-social attention
        social_time = 0
        nonsocial_time = 0
        other_time = 0
        
        for roi, duration in self.roi_durations.items():
            if roi in self.roi_categories:
                if self.roi_categories[roi] == 'Social':
                    social_time += duration
                elif self.roi_categories[roi] == 'Non-Social':
                    nonsocial_time += duration
            else:
                other_time += duration
        
        # Calculate percentages
        total_time = social_time + nonsocial_time + other_time
        social_pct = (social_time / total_time * 100) if total_time > 0 else 0
        nonsocial_pct = (nonsocial_time / total_time * 100) if total_time > 0 else 0
        other_pct = (other_time / total_time * 100) if total_time > 0 else 0
        
        # Prepare analysis results
        analysis_results = {
            'roi_durations': self.roi_durations,
            'first_fixation_times': self.first_fixation_times,
            'roi_revisits': self.roi_revisits,
            'transition_matrix': dict(self.transition_matrix),  # Convert defaultdict to dict for serialization
            'roi_categories': self.roi_categories,
            'social_attention': {
                'social_time': social_time,
                'nonsocial_time': nonsocial_time,
                'other_time': other_time,
                'social_pct': social_pct,
                'nonsocial_pct': nonsocial_pct,
                'other_pct': other_pct
            }
        }
        
        # Signal that ROI analysis is complete
        self.roi_analyzed.emit(analysis_results)
        
        return analysis_results
    
    def get_sorted_rois_by_duration(self):
        """
        Get ROIs sorted by duration (descending).
        
        Returns:
            list: List of tuples (roi_label, duration) sorted by duration
        """
        return sorted(self.roi_durations.items(), key=lambda x: x[1], reverse=True)
    
    def get_social_nonsocial_balance(self):
        """
        Get the balance between social and non-social attention.
        
        Returns:
            dict: Dictionary with social/non-social attention metrics
        """
        social_time = 0
        nonsocial_time = 0
        other_time = 0
        
        for roi, duration in self.roi_durations.items():
            if roi in self.roi_categories:
                if self.roi_categories[roi] == 'Social':
                    social_time += duration
                elif self.roi_categories[roi] == 'Non-Social':
                    nonsocial_time += duration
            else:
                other_time += duration
        
        # Calculate total and percentages
        total_time = social_time + nonsocial_time + other_time
        
        return {
            'social_time': social_time,
            'nonsocial_time': nonsocial_time,
            'other_time': other_time,
            'total_time': total_time,
            'social_pct': (social_time / total_time * 100) if total_time > 0 else 0,
            'nonsocial_pct': (nonsocial_time / total_time * 100) if total_time > 0 else 0,
            'other_pct': (other_time / total_time * 100) if total_time > 0 else 0
        }