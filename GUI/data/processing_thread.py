"""
Processing Thread Module

Implements a QThread-based worker for processing eye tracking data files
without freezing the GUI. Handles both ASC (EyeLink raw data) and CSV
(pre-processed data) file formats, supports data extraction, feature calculation,
and visualization generation.
"""

# Standard library imports
import os
import datetime
import traceback

# PyQt5 imports
from PyQt5.QtCore import QThread, pyqtSignal

# Local application imports - updated for new package structure
from .parser import (
    process_asc_file, process_multiple_files, 
    load_csv_file, load_multiple_csv_files
)
from ..visualization.eyelink_visualizer import MovieEyeTrackingVisualizer


class ProcessingThread(QThread):
    """Thread for running processing operations without freezing the GUI"""
    update_progress = pyqtSignal(int)
    status_update = pyqtSignal(str)  # Signal for status updates
    processing_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_paths, output_dir, visualize, extract_features, generate_report=False,
                 file_type="ASC Files"):
        """
        Initialize the processing thread
        
        Args:
            file_paths: List of paths to files to process
            output_dir: Directory to save output files
            visualize: Whether to generate visualizations
            extract_features: Whether to extract eye movement features
            generate_report: Whether to generate an HTML report
            file_type: Type of files to process ("ASC Files" or "CSV Files")
        """
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.visualize = visualize
        self.extract_features = extract_features
        self.generate_report = generate_report
        self.file_type = file_type

    def run(self):
        """Main processing method that runs in a separate thread"""
        try:
            # Create timestamped directory
            self.status_update.emit("Creating output directories...")
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.timestamped_dir = os.path.join(self.output_dir, timestamp)
            self.data_dir = os.path.join(self.timestamped_dir, 'data')
            self.viz_dir = os.path.join(self.timestamped_dir, 'plots')
            self.feature_dir = os.path.join(self.timestamped_dir, 'features')

            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.viz_dir, exist_ok=True)
            os.makedirs(self.feature_dir, exist_ok=True)

            # Process the files using the appropriate functions based on file type
            self.update_progress.emit(5)

            file_type_display = "ASC" if self.file_type == "ASC Files" else "CSV"
            num_files = len(self.file_paths)
            file_paths_display = self.file_paths[0] if num_files == 1 else f"{num_files} files"

            self.status_update.emit(
                f"Reading {file_type_display} file{'s' if num_files > 1 else ''}: {os.path.basename(file_paths_display) if num_files == 1 else ''}")
            self.update_progress.emit(10)

            if self.file_type == "ASC Files":
                # Process ASC files
                if num_files == 1:
                    self.status_update.emit(f"Parsing eye tracking data from {os.path.basename(self.file_paths[0])}...")
                    self.update_progress.emit(15)

                    self.status_update.emit("Extracting sample data and events...")
                    self.update_progress.emit(20)

                    self.status_update.emit("Processing fixations, saccades, and blinks...")
                    self.update_progress.emit(30)

                    if self.extract_features:
                        self.status_update.emit("Calculating eye movement metrics...")
                        self.update_progress.emit(40)

                    result = process_asc_file(
                        self.file_paths[0],
                        output_dir=self.data_dir,
                        extract_features=self.extract_features
                    )
                    self.status_update.emit("Eye tracking data processing complete.")
                    self.update_progress.emit(50)
                else:
                    self.status_update.emit(f"Processing {num_files} ASC files...")
                    self.update_progress.emit(20)

                    for i, file_path in enumerate(self.file_paths):
                        progress = 20 + int((i / num_files) * 30)  # Progress from 20% to 50%
                        self.status_update.emit(f"Processing file {i + 1}/{num_files}: {os.path.basename(file_path)}")
                        self.update_progress.emit(progress)

                    combined_features = process_multiple_files(
                        self.file_paths,
                        output_dir=self.data_dir
                    )
                    result = {"features": combined_features}
                    self.status_update.emit("All files processed successfully.")
                    self.update_progress.emit(50)
            else:
                # Process CSV files
                if num_files == 1:
                    self.status_update.emit(f"Loading data from CSV file: {os.path.basename(self.file_paths[0])}...")
                    self.update_progress.emit(30)

                    if self.extract_features:
                        self.status_update.emit("Calculating eye movement metrics from CSV data...")
                        self.update_progress.emit(40)

                    result = load_csv_file(
                        self.file_paths[0],
                        output_dir=self.data_dir,
                        extract_features=self.extract_features
                    )
                    self.status_update.emit("CSV data processing complete.")
                    self.update_progress.emit(50)
                else:
                    self.status_update.emit(f"Processing {num_files} CSV files...")
                    self.update_progress.emit(20)

                    for i, file_path in enumerate(self.file_paths):
                        progress = 20 + int((i / num_files) * 30)  # Progress from 20% to 50%
                        self.status_update.emit(f"Processing file {i + 1}/{num_files}: {os.path.basename(file_path)}")
                        self.update_progress.emit(progress)

                    combined_features = load_multiple_csv_files(
                        self.file_paths,
                        output_dir=self.data_dir
                    )
                    result = {"features": combined_features}
                    self.status_update.emit("All CSV files processed successfully.")
                    self.update_progress.emit(50)

            # Generate visualizations if requested
            vis_results = {}
            if self.visualize:
                self.status_update.emit("Initializing visualization engine...")
                self.update_progress.emit(55)

                visualizer = MovieEyeTrackingVisualizer(
                    base_dir=self.data_dir,
                    screen_size=(1280, 1024)
                )

                participant_id = None
                if len(self.file_paths) == 1:
                    participant_id = os.path.splitext(os.path.basename(self.file_paths[0]))[0]

                    # For CSV files, remove '_unified_eye_metrics' from the participant ID if present
                    if self.file_type == "CSV Files" and participant_id.endswith('_unified_eye_metrics'):
                        participant_id = participant_id.replace('_unified_eye_metrics', '')

                self.status_update.emit("Generating visualizations for eye tracking data...")
                self.update_progress.emit(60)

                # Start processing visualizations
                vis_results = visualizer.process_all_movies(participant_id)

                # Count how many visualizations were created
                total_visualizations = sum(len(plots) for plots in vis_results.values())
                self.status_update.emit(
                    f"Created {total_visualizations} visualizations across {len(vis_results)} movies.")
                self.update_progress.emit(80)

                # Generate HTML report if requested
                if self.generate_report and vis_results:
                    self.status_update.emit("Generating HTML report of all visualizations...")
                    self.update_progress.emit(85)

                    report_dir = os.path.join(self.viz_dir, 'report')
                    report_path = visualizer.generate_report(vis_results, report_dir)
                    result['report_path'] = report_path
                    self.status_update.emit("HTML report generated successfully.")

                self.update_progress.emit(90)

            # Return the complete results
            self.status_update.emit("Finalizing results...")
            result['visualizations'] = vis_results
            result['output_dir'] = self.timestamped_dir

            # Add summary information
            if 'summary' not in result and 'parser' in result:
                # Extract some basic counts
                parser = result['parser']
                summary = {
                    'samples': len(parser.sample_data) if hasattr(parser, 'sample_data') else 0,
                    'fixations': sum(len(fixs) for fixs in parser.fixations.values()) if hasattr(parser,
                                                                                               'fixations') else 0,
                    'saccades': sum(len(saccs) for saccs in parser.saccades.values()) if hasattr(parser,
                                                                                               'saccades') else 0,
                    'blinks': sum(len(blinks) for blinks in parser.blinks.values()) if hasattr(parser, 'blinks') else 0,
                    'frames': len(parser.frame_markers) if hasattr(parser, 'frame_markers') else 0
                }
                result['summary'] = summary

            self.processing_complete.emit(result)
            self.status_update.emit("Processing complete!")
            self.update_progress.emit(100)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.status_update.emit(error_msg)
            self.error_occurred.emit(f"{error_msg}\n{traceback.format_exc()}")
        finally:
            # Proper cleanup to prevent phantom threads
            self.quit()
            self.wait()