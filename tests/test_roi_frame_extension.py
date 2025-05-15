#!/usr/bin/env python3
"""
Test script for ROI frame extension functionality
Author: Tal Alfi
Date: April 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the ROI manager class
from roi_manager import ROIManager

def create_test_roi_data(num_frames, output_path):
    """
    Create a simple test ROI file with fewer frames than the movie
    
    Args:
        num_frames: Number of frames to create in the ROI file
        output_path: Path to save the test ROI file
        
    Returns:
        Path to the created file
    """
    # Create a dictionary representing ROI data
    roi_data = {"frames": {}}
    
    # Create simple ROI data for each frame
    for i in range(num_frames):
        frame_key = str(i)
        
        # Create a square ROI in the center
        center_roi = {
            "label": "Center",
            "object_id": f"center_{i}",
            "coordinates": [
                {"x": 0.4, "y": 0.4},
                {"x": 0.6, "y": 0.4},
                {"x": 0.6, "y": 0.6},
                {"x": 0.4, "y": 0.6}
            ]
        }
        
        # Create a circle-like ROI in the top-right
        tr_points = []
        for angle in range(0, 360, 45):
            x = 0.8 + 0.1 * np.cos(np.radians(angle))
            y = 0.2 + 0.1 * np.sin(np.radians(angle))
            tr_points.append({"x": x, "y": y})
        
        tr_roi = {
            "label": "TopRight",
            "object_id": f"tr_{i}",
            "coordinates": tr_points
        }
        
        # Add ROIs to the frame
        roi_data["frames"][frame_key] = [center_roi, tr_roi]
    
    # Save to a JSON file
    import json
    with open(output_path, 'w') as f:
        json.dump(roi_data, f, indent=2)
    
    print(f"Created test ROI file with {num_frames} frames: {output_path}")
    return output_path

def create_test_movie_data(num_frames, output_path):
    """
    Create test movie data with more frames than the ROI file
    
    Args:
        num_frames: Number of frames to create in the movie data
        output_path: Path to save the test movie data
        
    Returns:
        DataFrame with movie data
    """
    # Create a DataFrame representing eye tracking data
    data = {
        "timestamp": list(range(0, num_frames * 33, 33)),  # Assuming 30fps
        "frame_number": list(range(num_frames)),
        "x_left": np.random.uniform(0.3, 0.7, num_frames) * 1280,  # Pixel values
        "y_left": np.random.uniform(0.3, 0.7, num_frames) * 1024,
        "x_right": np.random.uniform(0.3, 0.7, num_frames) * 1280,
        "y_right": np.random.uniform(0.3, 0.7, num_frames) * 1024
    }
    
    df = pd.DataFrame(data)
    
    # Save to a CSV file
    df.to_csv(output_path, index=False)
    
    print(f"Created test movie data with {num_frames} frames: {output_path}")
    return df

def test_roi_frame_extension():
    """
    Test the ROI frame extension functionality
    """
    # Create test directory if it doesn't exist
    test_dir = "test_results/roi_extension"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test data with different frame counts
    roi_frames = 50
    movie_frames = 200
    
    roi_path = os.path.join(test_dir, "test_roi.json")
    movie_path = os.path.join(test_dir, "test_movie.csv")
    
    # Create test data files
    create_test_roi_data(roi_frames, roi_path)
    movie_data = create_test_movie_data(movie_frames, movie_path)
    
    # Load ROI data
    roi_manager = ROIManager()
    success = roi_manager.load_roi_file(roi_path)
    
    if not success:
        print("Failed to load ROI data")
        return
    
    # Print initial ROI frame information
    print(f"Original ROI frame count: {len(roi_manager.frame_numbers)}")
    print(f"First few ROI frames: {roi_manager.frame_numbers[:5]}...")
    print(f"Last few ROI frames: {roi_manager.frame_numbers[-5:]}...")
    
    # Verify we have fewer ROI frames than movie frames
    print(f"Movie frame count: {movie_frames}")
    
    # Test frame extension
    extended = roi_manager.extend_roi_frames(movie_frames)
    
    if not extended:
        print("Failed to extend ROI frames")
        return
    
    # Print extended ROI frame information
    print(f"Extended ROI frame count: {len(roi_manager.frame_numbers)}")
    print(f"First few extended ROI frames: {roi_manager.frame_numbers[:5]}...")
    print(f"Last few extended ROI frames: {roi_manager.frame_numbers[-5:]}...")
    
    # Verify that the extended frame count matches the movie frame count
    if len(roi_manager.frame_numbers) == movie_frames:
        print("SUCCESS: Extended ROI frame count matches movie frame count")
    else:
        print(f"ERROR: Extended ROI frame count ({len(roi_manager.frame_numbers)}) does not match movie frame count ({movie_frames})")
    
    # Visualize some extended frames to verify that we get ROIs for frames that didn't exist originally
    visualize_extended_frames(roi_manager, movie_data, test_dir)
    
def visualize_extended_frames(roi_manager, movie_data, output_dir):
    """
    Visualize some extended frames to verify the ROI extension
    
    Args:
        roi_manager: ROIManager instance with extended frames
        movie_data: DataFrame with movie data
        output_dir: Directory to save visualizations
    """
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Choose some test frames
    # 1. Original frame that should be preserved
    # 2. Middle frame that is new in the extended range
    # 3. Last frame in the extended range
    test_frames = [0, 100, 199]
    
    for frame in test_frames:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Set up axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Invert y-axis to match screen coordinates
        
        # Draw ROIs for this frame
        roi_manager.draw_rois_on_axis(ax, frame)
        
        # Get eye data for this frame
        frame_data = movie_data[movie_data['frame_number'] == frame]
        
        if not frame_data.empty:
            # Convert eye positions to normalized coordinates
            x_left_norm = frame_data['x_left'].values[0] / 1280
            y_left_norm = frame_data['y_left'].values[0] / 1024
            x_right_norm = frame_data['x_right'].values[0] / 1280
            y_right_norm = frame_data['y_right'].values[0] / 1024
            
            # Plot eye positions
            ax.scatter(x_left_norm, y_left_norm, color='blue', s=80, label='Left Eye')
            ax.scatter(x_right_norm, y_right_norm, color='red', s=80, label='Right Eye')
            
            # Check if the gaze point is in any ROI
            roi = roi_manager.find_roi_at_gaze(frame, x_left_norm, y_left_norm)
            if roi:
                roi_label = roi.get('label', 'Unknown')
                ax.set_title(f"Frame {frame}: Gaze in ROI '{roi_label}'", fontsize=14)
            else:
                ax.set_title(f"Frame {frame}: Gaze not in any ROI", fontsize=14)
        else:
            ax.set_title(f"Frame {frame}", fontsize=14)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        # Save figure
        save_path = os.path.join(vis_dir, f"frame_{frame}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved visualization for frame {frame}: {save_path}")

if __name__ == "__main__":
    test_roi_frame_extension()