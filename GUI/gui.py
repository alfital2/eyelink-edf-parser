import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QCheckBox, QTabWidget, QSplitter,
                             QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import datetime
from parser import process_asc_file, process_multiple_files
from eyelink_visualizer import MovieEyeTrackingVisualizer


class ProcessingThread(QThread):
    """Thread for running processing operations without freezing the GUI"""
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_paths, output_dir, visualize, extract_features, generate_report=False):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.visualize = visualize
        self.extract_features = extract_features
        self.generate_report = generate_report

    def run(self):
        try:
            # Create timestamped directory
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.timestamped_dir = os.path.join(self.output_dir, timestamp)
            self.data_dir = os.path.join(self.timestamped_dir, 'data')
            self.viz_dir = os.path.join(self.timestamped_dir, 'plots')
            self.feature_dir = os.path.join(self.timestamped_dir, 'features')

            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.viz_dir, exist_ok=True)
            os.makedirs(self.feature_dir, exist_ok=True)

            # Process the files using your existing functions
            self.update_progress.emit(10)

            if len(self.file_paths) == 1:
                result = process_asc_file(
                    self.file_paths[0],
                    output_dir=self.data_dir,
                    extract_features=self.extract_features
                )
                self.update_progress.emit(50)
            else:
                combined_features = process_multiple_files(
                    self.file_paths,
                    output_dir=self.data_dir
                )
                result = {"features": combined_features}
                self.update_progress.emit(50)

            # Generate visualizations if requested
            vis_results = {}
            if self.visualize:
                visualizer = MovieEyeTrackingVisualizer(
                    base_dir=self.data_dir,
                    screen_size=(1280, 1024)
                )

                participant_id = None
                if len(self.file_paths) == 1:
                    participant_id = os.path.splitext(os.path.basename(self.file_paths[0]))[0]

                self.update_progress.emit(60)
                vis_results = visualizer.process_all_movies(participant_id)

                # Generate HTML report if requested
                if self.generate_report and vis_results:
                    report_dir = os.path.join(self.viz_dir, 'report')
                    report_path = visualizer.generate_report(vis_results, report_dir)
                    result['report_path'] = report_path

                self.update_progress.emit(90)

            # Return the complete results
            result['visualizations'] = vis_results
            result['output_dir'] = self.timestamped_dir
            self.processing_complete.emit(result)

            self.update_progress.emit(100)

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class EyeMovementAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Movement Analysis for Autism Classification")
        self.setGeometry(100, 100, 1200, 800)

        # File paths and settings
        self.file_paths = []
        self.output_dir = None
        self.visualization_results = {}
        self.movie_visualizations = {}

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Create tabs for different sections
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Tab 1: Data Processing
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)

        # File selection section
        file_section = QWidget()
        file_layout = QHBoxLayout(file_section)

        self.file_label = QLabel("No files selected")
        select_file_btn = QPushButton("Select ASC File(s)")
        select_file_btn.clicked.connect(self.select_files)

        file_layout.addWidget(select_file_btn)
        file_layout.addWidget(self.file_label, 1)

        processing_layout.addWidget(file_section)

        # Output directory section
        output_section = QWidget()
        output_layout = QHBoxLayout(output_section)

        self.output_label = QLabel("No output directory selected")
        select_output_btn = QPushButton("Select Output Directory")
        select_output_btn.clicked.connect(self.select_output_dir)

        output_layout.addWidget(select_output_btn)
        output_layout.addWidget(self.output_label, 1)

        processing_layout.addWidget(output_section)

        # Options section
        options_section = QWidget()
        options_layout = QHBoxLayout(options_section)

        self.visualize_cb = QCheckBox("Generate Visualizations")
        self.visualize_cb.setChecked(True)
        self.extract_features_cb = QCheckBox("Extract Features")
        self.extract_features_cb.setChecked(True)
        self.generate_report_cb = QCheckBox("Generate HTML Report")
        self.generate_report_cb.setChecked(True)

        options_layout.addWidget(self.visualize_cb)
        options_layout.addWidget(self.extract_features_cb)
        options_layout.addWidget(self.generate_report_cb)
        options_layout.addStretch()

        processing_layout.addWidget(options_section)

        # Process button section
        process_section = QWidget()
        process_layout = QHBoxLayout(process_section)

        self.process_btn = QPushButton("Process Data")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        process_layout.addWidget(self.process_btn)
        process_layout.addWidget(self.progress_bar, 1)

        processing_layout.addWidget(process_section)

        # Status section
        self.status_label = QLabel("Ready")
        processing_layout.addWidget(self.status_label)

        processing_layout.addStretch()

        # Tab 2: Results & Visualization
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # Visualization controls and display
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        # Movie selection
        movie_section = QWidget()
        movie_layout = QHBoxLayout(movie_section)

        movie_layout.addWidget(QLabel("Select Movie:"))
        self.movie_combo = QComboBox()
        self.movie_combo.setEnabled(False)
        self.movie_combo.currentIndexChanged.connect(self.movie_selected)
        movie_layout.addWidget(self.movie_combo, 1)

        # Visualization type selection
        viz_type_layout = QHBoxLayout()
        viz_type_layout.addWidget(QLabel("Visualization Type:"))
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.setEnabled(False)
        self.viz_type_combo.currentIndexChanged.connect(self.visualization_type_selected)
        viz_type_layout.addWidget(self.viz_type_combo, 1)

        movie_layout.addLayout(viz_type_layout)
        viz_layout.addWidget(movie_section)

        # Visualization area
        self.image_label = QLabel("Visualization will be shown here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        viz_layout.addWidget(self.image_label)

        # Open report button
        self.report_btn = QPushButton("Open HTML Report")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self.open_report)
        viz_layout.addWidget(self.report_btn)

        results_layout.addWidget(viz_widget)

        # Add tabs to the main tab widget
        tabs.addTab(processing_tab, "Data Processing")
        tabs.addTab(results_tab, "Results & Visualization")

        # Set the central widget
        self.setCentralWidget(central_widget)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select ASC Files", "", "ASC Files (*.asc)"
        )
        if files:
            self.file_paths = files
            if len(files) == 1:
                self.file_label.setText(f"Selected: {files[0]}")
            else:
                self.file_label.setText(f"Selected {len(files)} files")
            self.update_process_button()

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(f"Output: {dir_path}")
            self.update_process_button()

    def update_process_button(self):
        self.process_btn.setEnabled(
            bool(self.file_paths) and bool(self.output_dir)
        )

    def process_data(self):
        # Disable UI elements during processing
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")

        # Start processing thread
        self.processing_thread = ProcessingThread(
            self.file_paths,
            self.output_dir,
            self.visualize_cb.isChecked(),
            self.extract_features_cb.isChecked(),
            self.generate_report_cb.isChecked()
        )

        # Connect signals
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)

        # Start processing
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def processing_finished(self, results):
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing complete!")

        # Re-enable UI elements
        self.process_btn.setEnabled(True)

        # Show results summary
        if "summary" in results:
            summary = results["summary"]
            msg = (f"Processed data:\n"
                   f"- {summary['samples']} samples\n"
                   f"- {summary['fixations']} fixations\n"
                   f"- {summary['saccades']} saccades\n"
                   f"- {summary['blinks']} blinks\n"
                   f"- {summary['frames']} frames")
            QMessageBox.information(self, "Processing Complete", msg)

        # Update visualization controls if visualizations were generated
        if "visualizations" in results and results["visualizations"]:
            self.output_dir = results["output_dir"]
            self.visualization_results = results["visualizations"]

            # Clear previous data
            self.movie_combo.clear()
            self.movie_visualizations = {}

            # Populate movie combo box
            self.movie_combo.addItems(list(self.visualization_results.keys()))
            self.movie_combo.setEnabled(True)

            # If we have at least one movie, select it
            if self.movie_combo.count() > 0:
                self.movie_combo.setCurrentIndex(0)

        # Enable the report button if a report was generated
        if 'report_path' in results and os.path.exists(results['report_path']):
            self.report_path = results['report_path']
            self.report_btn.setEnabled(True)

    def processing_error(self, error_msg):
        self.status_label.setText("Error occurred during processing")
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Processing Error", error_msg)

    def movie_selected(self, index):
        """Handle movie selection and dynamically discover available visualizations"""
        if index < 0 or self.movie_combo.count() == 0:
            return

        movie = self.movie_combo.currentText()
        print(f"Movie selected: {movie}")

        self.viz_type_combo.clear()

        # Check if we have visualization data for this movie
        if movie not in self.visualization_results:
            self.viz_type_combo.setEnabled(False)
            self.image_label.setText(f"No visualizations available for {movie}")
            return

        # Get all plot files available for this movie
        available_visualizations = {}

        for category, plot_paths in self.visualization_results[movie].items():
            for plot_path in plot_paths:
                if os.path.exists(plot_path):
                    # Extract the base filename without participant prefix
                    basename = os.path.basename(plot_path)
                    # Skip any hidden files
                    if basename.startswith('.'):
                        continue

                    # Remove participant prefix if present (anything before the underscore)
                    parts = basename.split('_')
                    if len(parts) > 1:
                        # If there's a prefix, join everything after the first underscore
                        cleaned_name = '_'.join(parts[1:])
                    else:
                        cleaned_name = basename

                    # Remove extension for display name
                    display_name = os.path.splitext(cleaned_name)[0]

                    # Convert to friendly display name (replace underscores with spaces and capitalize)
                    display_name = display_name.replace('_', ' ').title()

                    # Store mapping between display name and actual file path
                    available_visualizations[display_name] = plot_path

        # Add available visualizations to combo box
        if available_visualizations:
            self.movie_visualizations[movie] = available_visualizations
            self.viz_type_combo.addItems(sorted(available_visualizations.keys()))
            self.viz_type_combo.setEnabled(True)

            # Select the first visualization
            self.viz_type_combo.setCurrentIndex(0)
        else:
            self.viz_type_combo.setEnabled(False)
            self.image_label.setText(f"No visualizations found for movie: {movie}")

    def visualization_type_selected(self, index):
        """Show the selected visualization"""
        if index < 0 or self.viz_type_combo.count() == 0:
            return

        self.show_visualization()

    def show_visualization(self):
        """Load and display the selected visualization"""
        movie = self.movie_combo.currentText()
        viz_type = self.viz_type_combo.currentText()

        # Clear current display
        self.image_label.clear()

        if movie not in self.movie_visualizations or viz_type not in self.movie_visualizations[movie]:
            self.image_label.setText(f"Visualization not found: {viz_type}")
            return

        # Get the path to the visualization
        plot_path = self.movie_visualizations[movie][viz_type]

        if not os.path.exists(plot_path):
            self.image_label.setText(f"File not found: {plot_path}")
            return

        try:
            # Display the image
            pixmap = QPixmap(plot_path)
            if not pixmap.isNull():
                # Scale pixmap to fit the label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
            else:
                self.image_label.setText(f"Failed to load image: {plot_path}")
        except Exception as e:
            self.image_label.setText(f"Error displaying image: {str(e)}")

    def open_report(self):
        """Open the HTML report in the default web browser"""
        if hasattr(self, 'report_path') and os.path.exists(self.report_path):
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(self.report_path)}")
        else:
            QMessageBox.warning(self, "Report Not Found",
                                "The visualization report could not be found.")

    def resizeEvent(self, event):
        """Handle window resize event to update image scaling"""
        super().resizeEvent(event)
        if hasattr(self, 'image_label') and self.image_label.pixmap() is not None:
            # Re-scale the current image if there is one
            self.show_visualization()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeMovementAnalysisGUI()
    window.show()
    sys.exit(app.exec_())