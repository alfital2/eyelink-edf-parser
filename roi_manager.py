"""
ROI (Region of Interest) Manager for Eye Movement Analysis
Author: Tal Alfi
Date: April 2025
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class ROIManager:
    """
    Manager for handling Region of Interest (ROI) data from JSON files.
    This class provides functionality to load, parse, and query ROI data for eye tracking analysis.
    """

    def __init__(self):
        """Initialize the ROI manager."""
        self.roi_data = {}  # Frame -> List of ROIs
        self.loaded_file = None
        self.frame_numbers = []

    def load_roi_file(self, file_path: str) -> bool:
        """
        Load ROI data from a JSON file.

        Args:
            file_path: Path to the JSON file containing ROI data

        Returns:
            True if loading was successful, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"Error: ROI file not found: {file_path}")
            return False

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Clear existing data
            self.roi_data = {}

            # Check for different possible data structures
            if "annotations" in data:
                # Handle case where data is nested under "annotations"
                print("Found annotations key in JSON")
                data = data["annotations"]

            # Handle direct mapping of frame -> ROIs
            for frame_key, rois in data.items():
                try:
                    # Try to convert the frame key to integer
                    frame_num = int(frame_key)
                    self.roi_data[frame_num] = rois
                    print(f"Added {len(rois)} ROIs for frame {frame_num}")
                except ValueError:
                    # Skip non-integer keys but log them
                    print(f"Warning: Skipping non-integer frame key: {frame_key}")

            # Store sorted frame numbers for easy access
            self.frame_numbers = sorted(self.roi_data.keys())
            self.loaded_file = file_path

            print(f"Successfully loaded ROI data from {file_path}")
            print(f"Found ROI data for {len(self.frame_numbers)} frames")

            # Print some basic statistics about the loaded data
            if self.frame_numbers:
                roi_counts = [len(self.roi_data[frame]) for frame in self.frame_numbers]
                print(f"Average ROIs per frame: {np.mean(roi_counts):.1f}")
                print(f"ROI labels found: {self.get_unique_labels()}")
            else:
                print("No valid frame data found. Dumping file structure:")
                self._print_json_structure(data)

            return len(self.frame_numbers) > 0

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file: {file_path}")
            return False
        except Exception as e:
            print(f"Error loading ROI file: {str(e)}")
            return False


    def get_unique_labels(self) -> List[str]:
        """Get a list of unique ROI labels in the loaded data."""
        labels = set()
        for frame, rois in self.roi_data.items():
            for roi in rois:
                if "label" in roi:
                    labels.add(roi["label"])
        return sorted(list(labels))

    def get_frame_rois(self, frame_number: int) -> List[Dict[str, Any]]:
        """
        Get all ROIs for a specific frame.

        Args:
            frame_number: The frame number to get ROIs for

        Returns:
            List of ROI dictionaries for the specified frame, or empty list if none exist
        """
        return self.roi_data.get(frame_number, [])

    def point_in_polygon(self, x: float, y: float, coordinates: List[Dict[str, float]]) -> bool:
        """
        Test if a point is inside a polygon using the ray casting algorithm.

        Args:
            x: X-coordinate of the test point
            y: Y-coordinate of the test point
            coordinates: List of coordinate dictionaries with 'x' and 'y' keys

        Returns:
            True if the point is inside the polygon, False otherwise
        """
        # Extract polygon points
        poly_points = [(point['x'], point['y']) for point in coordinates]
        num_points = len(poly_points)

        # Need at least 3 points to form a polygon
        if num_points < 3:
            return False

        # Ray casting algorithm
        inside = False
        j = num_points - 1

        for i in range(num_points):
            xi, yi = poly_points[i]
            xj, yj = poly_points[j]

            # Check if point is on an edge
            if (yi == y and xi == x) or (yj == y and xj == x):
                return True

            # Check if the point is on a horizontal edge
            if (yi == yj) and (yi == y) and (min(xi, xj) <= x <= max(xi, xj)):
                return True

            # Cast ray
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def is_gaze_in_roi(self, x: float, y: float, roi: Dict[str, Any]) -> bool:
        """
        Test if a gaze point falls within an ROI.

        Args:
            x: X-coordinate of the gaze point, normalized to [0,1]
            y: Y-coordinate of the gaze point, normalized to [0,1]
            roi: ROI dictionary containing 'coordinates' list

        Returns:
            True if the gaze is inside the ROI, False otherwise
        """
        if "coordinates" not in roi:
            return False

        return self.point_in_polygon(x, y, roi["coordinates"])

    def find_roi_at_point(self, frame_number: int, x: float, y: float) -> Optional[Dict[str, Any]]:
        """
        Find the ROI at a specific gaze point for a given frame.

        Args:
            frame_number: Frame number to check
            x: X-coordinate of the gaze point, normalized to [0,1]
            y: Y-coordinate of the gaze point, normalized to [0,1]

        Returns:
            ROI dictionary if the point is inside an ROI, None otherwise
        """
        rois = self.get_frame_rois(frame_number)

        for roi in rois:
            if self.is_gaze_in_roi(x, y, roi):
                return roi

        return None

    def get_nearest_frame(self, frame_number: int) -> Optional[int]:
        """
        Get the nearest available frame number to the requested frame.
        Useful when ROI data doesn't have entries for every video frame.

        Args:
            frame_number: The requested frame number

        Returns:
            The nearest available frame number, or None if no frames are loaded
        """
        if not self.frame_numbers:
            return None

        if frame_number in self.frame_numbers:
            return frame_number

        # Find the closest frame number
        closest_frame = min(self.frame_numbers, key=lambda x: abs(x - frame_number))
        return closest_frame

    def find_roi_at_gaze(self, frame_number: int, x: float, y: float,
                         use_nearest_frame: bool = True) -> Optional[Dict[str, Any]]:
        """
        High-level function to find an ROI at a gaze point.

        Args:
            frame_number: Frame number to check
            x: X-coordinate of the gaze point, normalized to [0,1]
            y: Y-coordinate of the gaze point, normalized to [0,1]
            use_nearest_frame: Whether to use the nearest available frame if exact frame not found

        Returns:
            ROI dictionary if found, None otherwise
        """
        if frame_number in self.roi_data:
            return self.find_roi_at_point(frame_number, x, y)

        if use_nearest_frame:
            nearest_frame = self.get_nearest_frame(frame_number)
            if nearest_frame is not None:
                return self.find_roi_at_point(nearest_frame, x, y)

        return None

    def _print_json_structure(self, data, prefix="", max_depth=2, current_depth=0):
        """Helper to print the structure of the JSON data for debugging."""
        if current_depth > max_depth:
            print(f"{prefix}...")
            return

        if isinstance(data, dict):
            print(f"{prefix}{{")
            for key, value in list(data.items())[:5]:  # Limit to first 5 items
                print(f"{prefix}  {key}:")
                self._print_json_structure(value, prefix + "  ", max_depth, current_depth + 1)
            if len(data) > 5:
                print(f"{prefix}  ... ({len(data) - 5} more keys)")
            print(f"{prefix}}}")
        elif isinstance(data, list):
            print(f"{prefix}[{len(data)} items]")
            if data and current_depth < max_depth:
                self._print_json_structure(data[0], prefix + "  ", max_depth, current_depth + 1)
        else:
            print(f"{prefix}{data}")

    def draw_rois_on_axis(self, ax, frame_number: int,
                          color_map: Dict[str, str] = None,
                          alpha: float = 0.3,
                          show_labels: bool = True,
                          highlighted_roi: Optional[str] = None):
        """
        Draw ROIs for a specific frame on a matplotlib axis.

        Args:
            ax: Matplotlib axis to draw on
            frame_number: Frame number to draw ROIs for
            color_map: Dictionary mapping ROI labels to colors
            alpha: Transparency of the ROI polygons
            show_labels: Whether to show ROI labels
            highlighted_roi: Optional ROI ID to highlight
        """
        import matplotlib.patches as patches

        # Get ROIs for this frame, using the nearest frame if necessary
        if frame_number in self.roi_data:
            rois = self.roi_data[frame_number]
        else:
            nearest_frame = self.get_nearest_frame(frame_number)
            if nearest_frame is None:
                # No data found for this frame, silently return
                return
            rois = self.roi_data[nearest_frame]

        if not rois:
            # No ROIs found for this frame, silently return
            return

        # Default color map if none provided
        if color_map is None:
            color_map = {
                "Face": "red",
                "Hand": "green",
                "Torso": "blue",
                "Couch": "gray",
                "Bed": "brown",
                # Default for any other label
                "default": "purple"
            }

        # Draw each ROI
        roi_count = 0
        for roi in rois:
            if "coordinates" not in roi or "label" not in roi:
                # Silently skip ROIs missing required properties
                continue

            # Get ROI properties
            coords = roi["coordinates"]
            label = roi["label"]
            object_id = roi.get("object_id", "")

            # Skip if no coordinates
            if not coords:
                # Silently skip ROIs with empty coordinates
                continue

            # Get polygon points
            poly_points = [(point['x'], point['y']) for point in coords]

            # Determine color
            color = color_map.get(label, color_map.get("default", "purple"))

            # Set properties for highlighted ROI
            if highlighted_roi and object_id == highlighted_roi:
                line_width = 3
                edge_color = "yellow"
                fill_alpha = alpha + 0.2
            else:
                line_width = 1
                edge_color = "black"
                fill_alpha = alpha

            # Create polygon patch
            poly = patches.Polygon(
                poly_points,
                closed=True,
                fill=True,
                color=color,
                alpha=fill_alpha,
                edgecolor=edge_color,
                linewidth=line_width
            )

            # Add polygon to axis
            ax.add_patch(poly)
            roi_count += 1

            # Add label if requested
            if show_labels:
                # Find centroid of polygon for label placement
                x_coords = [p[0] for p in poly_points]
                y_coords = [p[1] for p in poly_points]
                centroid_x = sum(x_coords) / len(poly_points)
                centroid_y = sum(y_coords) / len(poly_points)

                # Draw label with white background for readability
                ax.text(
                    centroid_x, centroid_y,
                    label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc="white",
                        ec="black",
                        alpha=0.8
                    )
                )

        print(f"Successfully drew {roi_count} ROIs")
        
    def create_test_visualization(self, frame_number: int = None, save_path: Optional[str] = None):
        """
        Create a test visualization of ROIs for a specific frame.

        Args:
            frame_number: Frame number to visualize, defaults to first available frame
            save_path: Optional path to save the visualization

        Returns:
            Matplotlib figure and axis
        """
        import matplotlib.pyplot as plt

        # If no frame specified, use the first available
        if frame_number is None and self.frame_numbers:
            frame_number = self.frame_numbers[0]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Set axis limits to normalized coordinates
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Invert y-axis to match screen coordinates

        # Draw ROIs
        self.draw_rois_on_axis(ax, frame_number)

        # Add title and labels
        ax.set_title(f"ROIs for Frame {frame_number}", fontsize=14)
        ax.set_xlabel("X Position (normalized)", fontsize=12)
        ax.set_ylabel("Y Position (normalized)", fontsize=12)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        return fig, ax


# Simple standalone test
if __name__ == "__main__":
    roi_manager = ROIManager()
    test_file = "roi_data.json"  # Replace with your test file

    if os.path.exists(test_file):
        print(f"\nAttempting to load ROI file: {test_file}")
        success = roi_manager.load_roi_file(test_file)

        if success:
            print("\nROI data loaded successfully!")

            # Test ROI hit detection with a sample point
            test_frame = roi_manager.frame_numbers[0]

            # Test a few sample points
            test_points = [
                (0.5, 0.5, "center of the screen"),
                (0.25, 0.25, "top-left quadrant"),
                (0.75, 0.75, "bottom-right quadrant")
            ]

            print("\nTesting ROI hit detection:")
            for x, y, desc in test_points:
                roi = roi_manager.find_roi_at_gaze(test_frame, x, y)
                if roi:
                    print(f"Point at {desc} ({x:.2f}, {y:.2f}) is inside ROI: {roi.get('label', 'unknown')}")
                else:
                    print(f"Point at {desc} ({x:.2f}, {y:.2f}) is not inside any ROI")

            # Create and show visualization
            print("\nCreating visualization for the first frame...")
            roi_manager.create_test_visualization(save_path="roi_visualization.png")
            import matplotlib.pyplot as plt

            plt.show()
        else:
            print("\nFailed to load usable ROI data.")
            print("Please check the JSON format or provide a sample of the correct structure.")