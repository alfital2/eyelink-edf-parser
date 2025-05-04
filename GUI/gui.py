import sys
import os

# Add parent directory to path so we can import modules from the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Configure matplotlib for thread safety BEFORE importing any matplotlib-related modules
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QCheckBox, QTabWidget, QSplitter,
                             QProgressBar, QMessageBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGroupBox,
                             QTextBrowser, QToolTip, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QCursor, QPalette
import datetime
import pandas as pd
from parser import process_asc_file, process_multiple_files, load_csv_file, load_multiple_csv_files
from eyelink_visualizer import MovieEyeTrackingVisualizer
# Import documentation module
from documentation import (get_feature_explanations, get_visualization_explanations,
                           get_formatted_feature_documentation, get_formatted_visualization_documentation)
from animated_roi_scanpath import AnimatedROIScanpathWidget


class ProcessingThread(QThread):
    """Thread for running processing operations without freezing the GUI"""
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_paths, output_dir, visualize, extract_features, generate_report=False, file_type="ASC Files"):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.visualize = visualize
        self.extract_features = extract_features
        self.generate_report = generate_report
        self.file_type = file_type

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

            # Process the files using the appropriate functions based on file type
            self.update_progress.emit(10)

            if self.file_type == "ASC Files":
                # Process ASC files
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
            else:
                # Process CSV files
                if len(self.file_paths) == 1:
                    result = load_csv_file(
                        self.file_paths[0],
                        output_dir=self.data_dir,
                        extract_features=self.extract_features
                    )
                    self.update_progress.emit(50)
                else:
                    combined_features = load_multiple_csv_files(
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
                    
                    # For CSV files, remove '_unified_eye_metrics' from the participant ID if present
                    if self.file_type == "CSV Files" and participant_id.endswith('_unified_eye_metrics'):
                        participant_id = participant_id.replace('_unified_eye_metrics', '')

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
        finally:
            # Proper cleanup to prevent phantom threads
            self.quit()
            self.wait()


class AnimatedROIScanpathTab(QWidget):
    """Tab widget for animated scanpath visualization with ROI overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.roi_file_path = None

    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)

        # Create the animated ROI scanpath widget
        self.scanpath_widget = AnimatedROIScanpathWidget()

        # Add widget to layout
        layout.addWidget(self.scanpath_widget)


    def load_data(self, data, movie_name, screen_width=1280, screen_height=1024):
        """Load data into the animated ROI scanpath widget."""
        # Store data for future reference
        self.data = data
        self.movie_name = movie_name
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Pass ROI file path to the widget ONLY if explicitly set by user
        if hasattr(self, 'roi_file_path') and self.roi_file_path and os.path.exists(self.roi_file_path):
            # Update the widget's ROI file label
            self.scanpath_widget.roi_file_label.setText(f"ROI File: {os.path.basename(self.roi_file_path)}")
            
            # Load data with ROI path
            return self.scanpath_widget.load_data(data, self.roi_file_path, movie_name, screen_width, screen_height)
        else:
            # If no ROI file is explicitly selected, load eye data without ROI
            # Status label was removed to save space
            return self.scanpath_widget.load_data(data, None, movie_name, screen_width, screen_height)


class EyeMovementAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Movement Analysis for Autism Classification (ASC/CSV)")
        self.setGeometry(100, 100, 1200, 800)

        # File paths and settings
        self.file_paths = []
        self.output_dir = None
        self.visualization_results = {}
        self.movie_visualizations = {}
        self.features_data = None
        self.selected_file_type = "ASC Files"  # Default file type

        # Get feature and visualization explanations from the documentation module
        self.feature_explanations = get_feature_explanations()
        self.visualization_explanations = get_visualization_explanations()

        # Check if dark mode is enabled
        self.is_dark_mode = self.is_dark_theme()

        # Install event filter to detect theme changes
        app = QApplication.instance()
        app.installEventFilter(self)

        # Initialize UI
        self.init_ui()

    def eventFilter(self, obj, event):
        """Event filter to detect system theme changes"""
        if obj == QApplication.instance() and event.type() == event.ApplicationPaletteChange:
            # Theme has changed - update dark mode status
            new_dark_mode = self.is_dark_theme()
            if new_dark_mode != self.is_dark_mode:
                self.is_dark_mode = new_dark_mode
                self.refresh_theme()
        return super().eventFilter(obj, event)

    def refresh_theme(self):
        """Refresh UI with current theme"""
        # Update the stylesheet
        self.centralWidget().setStyleSheet(self.get_theme_style())

        # Refresh feature tables
        for category_name, table_info in self.feature_tables.items():
            table = table_info["table"]
            if not self.is_dark_mode:
                table.setStyleSheet("QTableWidget { background-color: white; border: 1px solid #ddd; }")
            else:
                table.setStyleSheet("")  # Default dark mode styling from main style

        # Refresh visualization explanation text area
        if not self.is_dark_mode:
            self.viz_explanation.setStyleSheet("background-color: #f8f8f8; border: 1px solid #e0e0e0;")
        else:
            self.viz_explanation.setStyleSheet("")  # Default dark mode styling

        # Re-display current visualization if any
        if hasattr(self, 'image_label') and self.image_label.pixmap() is not None:
            self.show_visualization()

    def is_dark_theme(self):
        """Detect if dark theme is active by checking background color"""
        app = QApplication.instance()
        palette = app.palette()
        background_color = palette.color(QPalette.Window)
        # If the background color is dark, assume dark theme
        return background_color.lightness() < 128

    def get_theme_style(self):
        """Get style sheet based on current theme"""
        if self.is_dark_mode:
            return """
            QGroupBox { 
                font-weight: bold; 
                font-size: 14px;
                color: #f0f0f0;
            }
            QTableWidget { 
                gridline-color: #555;
                background-color: rgba(60, 60, 60, 120);
                border: 1px solid #555;
            }
            QTableWidget::item {
                color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #444;
                color: #f0f0f0;
                border: 1px solid #555;
            }
            QTextBrowser {
                background-color: rgba(60, 60, 60, 120);
                border: 1px solid #555;
            }
            """
        else:
            return """
            QGroupBox { 
                font-weight: bold; 
                font-size: 14px; 
            }
            QTableWidget { 
                gridline-color: #ccc;
                background-color: white;
            }
            QTableWidget::item {
                background-color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
            }
            QTextBrowser {
                background-color: white;
            }
            """

    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Apply theme-specific styles
        central_widget.setStyleSheet(self.get_theme_style())

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
        
        # Create a single button for loading source files (both ASC and CSV)
        select_file_btn = QPushButton("Load Source File(s)")
        select_file_btn.setToolTip(
            "Load eye tracking data files:\n"
            "• ASC Files: Raw EyeLink eye tracking data\n"
            "• CSV Files: Preprocessed unified_eye_metrics files"
        )
        select_file_btn.clicked.connect(self.select_files)
        select_file_btn.setMinimumWidth(150)

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
        self.visualize_cb.setToolTip("Generate visualization plots for eye tracking data")
        
        self.extract_features_cb = QCheckBox("Extract Features")
        self.extract_features_cb.setChecked(True)
        self.extract_features_cb.setToolTip("Extract statistical features from eye tracking data")
        
        self.generate_report_cb = QCheckBox("Generate HTML Report")
        self.generate_report_cb.setChecked(True)
        self.generate_report_cb.setToolTip("Create an HTML report with visualization results")
        
        # Create a help button for file format information
        file_format_help = QPushButton("?")
        file_format_help.setFixedSize(25, 25)
        file_format_help.setToolTip(
            "Supported File Formats:\n\n"
            "ASC Files: Raw EyeLink eye tracking data files.\n"
            "These are the original files from the eye tracker.\n\n"
            "CSV Files: Preprocessed unified eye metrics files.\n"
            "These are generated after processing ASC files and contain\n"
            "already extracted eye tracking data in CSV format.\n\n"
            "The program automatically detects the file type based on the extension."
        )
        file_format_help.clicked.connect(self.show_file_format_help)
        
        options_layout.addWidget(self.visualize_cb)
        options_layout.addWidget(self.extract_features_cb)
        options_layout.addWidget(self.generate_report_cb)
        options_layout.addStretch()
        options_layout.addWidget(file_format_help)

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

        # Visualization area - Make it fill available space
        self.image_label = QLabel("Visualization will be shown here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        viz_layout.addWidget(self.image_label)

        # Visualization explanation area
        self.viz_explanation = QTextBrowser()
        self.viz_explanation.setMaximumHeight(150)
        if not self.is_dark_mode:
            self.viz_explanation.setStyleSheet("background-color: #f8f8f8; border: 1px solid #e0e0e0;")
        self.viz_explanation.setText("Select a visualization to see explanation")
        viz_layout.addWidget(self.viz_explanation)

        # Open report button
        self.report_btn = QPushButton("Open HTML Report")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self.open_report)
        viz_layout.addWidget(self.report_btn)

        results_layout.addWidget(viz_widget)

        # Tab 3: Features Display
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)

        # Features header and overview
        features_header = QLabel("Eye Movement Features for Autism Research")
        features_header.setFont(QFont("Arial", 12, QFont.Bold))
        features_layout.addWidget(features_header)

        features_overview = QTextBrowser()
        features_overview.setMaximumHeight(100)
        features_overview.setHtml("""
        <p>This tab displays extracted eye movement features that may serve as biomarkers for autism spectrum disorder classification.
        Research suggests individuals with ASD exhibit distinct patterns of visual attention, particularly when viewing social stimuli.</p>
        <p>Move your mouse over any feature name to see a detailed explanation of how it's calculated and its potential relevance to autism research.</p>
        """)
        features_layout.addWidget(features_overview)

        # Create tables for different feature categories
        self.create_feature_tables(features_layout)

        # Add save features button
        save_features_btn = QPushButton("Export Features to CSV")
        save_features_btn.clicked.connect(self.save_features)
        features_layout.addWidget(save_features_btn)

        # Tab 4: Documentation
        documentation_tab = QWidget()
        documentation_layout = QVBoxLayout(documentation_tab)

        # Create documentation browser with tabs for features and visualizations
        doc_tabs = QTabWidget()
        documentation_layout.addWidget(doc_tabs)

        # Features documentation
        feature_doc = QWidget()
        feature_doc_layout = QVBoxLayout(feature_doc)

        feature_doc_text = QTextBrowser()
        feature_doc_text.setOpenExternalLinks(True)
        feature_doc_text.setStyleSheet("font-size: 14px;")
        feature_doc_text.setHtml(get_formatted_feature_documentation())

        feature_doc_layout.addWidget(feature_doc_text)

        # Visualization documentation
        viz_doc = QWidget()
        viz_doc_layout = QVBoxLayout(viz_doc)

        viz_doc_text = QTextBrowser()
        viz_doc_text.setOpenExternalLinks(True)
        viz_doc_text.setStyleSheet("font-size: 14px;")
        viz_doc_text.setHtml(get_formatted_visualization_documentation())

        viz_doc_layout.addWidget(viz_doc_text)

        # Add documentation tabs
        doc_tabs.addTab(feature_doc, "Feature Documentation")
        doc_tabs.addTab(viz_doc, "Visualization Documentation")

        # Add tabs to the main tab widget
        tabs.addTab(processing_tab, "Data Processing")
        tabs.addTab(results_tab, "Results & Visualization")
        self.animated_viz_tab = AnimatedROIScanpathTab()
        tabs.addTab(self.animated_viz_tab, "Animated Scanpath")
        tabs.addTab(features_tab, "Extracted Features")
        tabs.addTab(documentation_tab, "Documentation")

        # Set the central widget
        self.setCentralWidget(central_widget)
        
        # Initialize the ROI file path attribute
        self.roi_file_path = None
        self.roi_data = None


    def create_feature_tables(self, parent_layout):
        """Create organized tables for different categories of features"""
        # Create a scrollable widget for all tables
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        features_scroll_layout = QVBoxLayout(scroll_content)
        features_scroll_layout.setSpacing(10)  # Add spacing between feature groups

        if not self.is_dark_mode:
            # Fix the background color for light mode
            scroll_content.setStyleSheet("background-color: #f5f5f5;")

        # Create feature category sections
        categories = [
            ("Basic Information", ["participant_id"]),
            ("Pupil Size Features", ["pupil_left_mean", "pupil_left_std", "pupil_left_min", "pupil_left_max",
                                     "pupil_right_mean", "pupil_right_std", "pupil_right_min", "pupil_right_max"]),
            ("Gaze Position Features", ["gaze_left_x_std", "gaze_left_y_std", "gaze_left_dispersion",
                                        "gaze_right_x_std", "gaze_right_y_std", "gaze_right_dispersion"]),
            ("Fixation Features",
             ["fixation_left_count", "fixation_left_duration_mean", "fixation_left_duration_std", "fixation_left_rate",
              "fixation_right_count", "fixation_right_duration_mean", "fixation_right_duration_std",
              "fixation_right_rate"]),
            ("Saccade Features", ["saccade_left_count", "saccade_left_amplitude_mean", "saccade_left_amplitude_std",
                                  "saccade_left_duration_mean",
                                  "saccade_right_count", "saccade_right_amplitude_mean", "saccade_right_amplitude_std",
                                  "saccade_right_duration_mean"]),
            ("Blink Features", ["blink_left_count", "blink_left_duration_mean", "blink_left_rate",
                                "blink_right_count", "blink_right_duration_mean", "blink_right_rate"]),
            ("Head Movement Features",
             ["head_movement_mean", "head_movement_std", "head_movement_max", "head_movement_frequency"])
        ]

        # Create a table for each category
        self.feature_tables = {}
        for category_name, feature_keys in categories:
            # Create a group box for each category
            group_box = QGroupBox(category_name)
            group_layout = QVBoxLayout(group_box)

            # Create table
            table = QTableWidget(0, 2)  # Rows will be added dynamically, 2 columns (Feature, Value)
            table.setHorizontalHeaderLabels(["Feature", "Value"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)

            # Additional styling for light mode
            if not self.is_dark_mode:
                table.setStyleSheet("QTableWidget { background-color: white; border: 1px solid #ddd; }")

            # Enable tooltips for the table
            table.setMouseTracking(True)
            table.cellEntered.connect(lambda row, col, t=table, features=feature_keys:
                                      self.show_feature_tooltip(row, col, t, features))

            # Store the table and its associated feature keys
            self.feature_tables[category_name] = {
                "table": table,
                "features": feature_keys
            }

            group_layout.addWidget(table)
            features_scroll_layout.addWidget(group_box)

        scroll_area.setWidget(scroll_content)
        parent_layout.addWidget(scroll_area)

    def show_feature_tooltip(self, row, col, table, features):
        """Show a tooltip with feature explanation when hovering over a feature name"""
        if col == 0 and row < len(features):  # Only show tooltips for feature names column
            feature_key = features[row]
            if feature_key in self.feature_explanations:
                explanation = self.feature_explanations[feature_key]
                QToolTip.showText(QCursor.pos(), explanation)

    def update_feature_tables(self, features_df):
        """Update all feature tables with data from the features DataFrame"""
        if features_df is None or features_df.empty:
            return

        # Store the features data
        self.features_data = features_df

        # For each category table, update the values
        for category_name, table_info in self.feature_tables.items():
            table = table_info["table"]
            feature_keys = table_info["features"]

            # Clear the table
            table.setRowCount(0)

            # Add rows for each feature in this category
            for i, feature_key in enumerate(feature_keys):
                if feature_key in features_df.columns:
                    row_position = table.rowCount()
                    table.insertRow(row_position)

                    # Format the feature name to be more readable
                    display_name = feature_key.replace('_', ' ').title()

                    # Create items
                    name_item = QTableWidgetItem(display_name)

                    # Set tooltip with explanation if available
                    if feature_key in self.feature_explanations:
                        name_item.setToolTip(self.feature_explanations[feature_key])

                    # Format the value based on type
                    value = features_df[feature_key].iloc[0]

                    # Handle NaN values properly
                    if pd.isna(value):
                        value_text = "N/A"
                    elif isinstance(value, (int, float)):
                        # Format number with appropriate precision
                        try:
                            if float(value).is_integer():
                                value_text = str(int(value))
                            else:
                                value_text = f"{value:.4f}"
                        except:
                            # If conversion fails, use the value as is
                            value_text = str(value)
                    else:
                        value_text = str(value)

                    value_item = QTableWidgetItem(value_text)

                    # Add items to table
                    table.setItem(row_position, 0, name_item)
                    table.setItem(row_position, 1, value_item)

    def select_files(self):
        # Combined file filter for both ASC and CSV files
        file_filter = "Eye Tracking Files (*.asc *.csv);;ASC Files (*.asc);;CSV Files (*.csv);;All Files (*.*)"
        dialog_title = "Select Eye Tracking Data Files"
            
        files, selected_filter = QFileDialog.getOpenFileNames(
            self, dialog_title, "", file_filter
        )
        
        if files:
            self.file_paths = files
            
            # Update the file label
            if len(files) == 1:
                self.file_label.setText(f"Selected: {os.path.basename(files[0])}")
            else:
                self.file_label.setText(f"Selected {len(files)} files")
            
            # Automatically determine file type based on extension
            is_csv = True
            for file_path in files:
                if file_path.lower().endswith('.asc'):
                    is_csv = False
                    break
            
            # Set the file type based on extension
            self.selected_file_type = "CSV Files" if is_csv else "ASC Files"
            file_type_display = "CSV" if is_csv else "ASC"
            self.status_label.setText(f"Using {file_type_display} files. Click 'Process Data' to continue.")
            
            # Enable process button
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
        
    def update_file_type_info(self, index):
        """Update status label with information about the selected file type"""
        if index == 0:  # ASC Files
            self.status_label.setText("ASC Files: Original EyeLink data files. Processing may take longer but includes all raw data.")
        else:  # CSV Files
            self.status_label.setText("CSV Files: Pre-processed data files. Much faster loading for visualizations and analysis.")

    def process_data(self):
        # Disable UI elements during processing
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")

        # Start processing thread with the selected file type
        self.processing_thread = ProcessingThread(
            self.file_paths,
            self.output_dir,
            self.visualize_cb.isChecked(),
            self.extract_features_cb.isChecked(),
            self.generate_report_cb.isChecked(),
            self.selected_file_type
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

        # Update the features display if features were extracted
        if "features" in results and not results["features"].empty:
            self.update_feature_tables(results["features"])

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
                
            # ROI files must be explicitly selected by the user, 
            # not automatically loaded - no automatic ROI detection here
            
            # Set screen dimensions for visualizations if not already set
            if not hasattr(self, 'screen_width') or not hasattr(self, 'screen_height'):
                self.screen_width = 1280
                self.screen_height = 1024
            
            # Load data for animated tabs from all movies
            for movie in self.visualization_results.keys():
                self._load_animation_data_for_movie(movie)

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
            
        # Update the animated visualization tabs to show the same movie
        self._load_animation_data_for_movie(movie)

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

    def _load_animation_data_for_movie(self, movie):
        """Load data for the given movie into both animation tabs"""
        # Prevent reentrant calls that could cause duplicate processes
        if hasattr(self, '_loading_animation_data') and self._loading_animation_data:
            print("Animation data loading already in progress, skipping duplicate call")
            return
            
        # Set a flag to prevent reentrant calls
        self._loading_animation_data = True
        
        try:
            # Find the data directory for this movie
            data_dir = None
            data_path = None

            # First check if we have an output directory
            if hasattr(self, 'output_dir') and self.output_dir:
                # Look for data directory
                data_dir = os.path.join(self.output_dir, 'data')
                if os.path.exists(data_dir):
                    # First, check for exact match with movie name
                    for file in os.listdir(data_dir):
                        if 'unified_eye_metrics' in file and movie in file and file.endswith('.csv'):
                            data_path = os.path.join(data_dir, file)
                            print(f"Found data file with exact match: {data_path}")
                            break

                    # If no exact match, check for any unified_eye_metrics file
                    if not data_path:
                        for file in os.listdir(data_dir):
                            if 'unified_eye_metrics' in file and file.endswith('.csv'):
                                data_path = os.path.join(data_dir, file)
                                print(f"Found data file without exact match: {data_path}")
                                # Extract real movie name from filename if possible
                                parts = file.split('_unified_eye_metrics_')
                                if len(parts) > 1 and '_' in parts[1]:
                                    real_movie_name = parts[1].split('.')[0]  
                                    print(f"Detected movie name from file: {real_movie_name}")
                                break

            # If we still don't have a data directory, try to extract it from visualization paths
            if not data_dir or not data_path:
                for paths in self.visualization_results[movie].values():
                    if paths:  # If there's at least one path
                        # Get directory containing the visualization
                        viz_dir = os.path.dirname(paths[0])
                        # The movie directory is usually the parent of the plots directory
                        potential_data_dir = os.path.dirname(viz_dir)
                        if os.path.exists(potential_data_dir):
                            data_dir = potential_data_dir
                            # First look for CSV files with exact movie name match
                            for file in os.listdir(data_dir):
                                if 'unified_eye_metrics' in file and movie in file and file.endswith('.csv'):
                                    data_path = os.path.join(data_dir, file)
                                    print(f"Found data file with exact match: {data_path}")
                                    break
                                    
                            # If no exact match, try any unified_eye_metrics file
                            if not data_path:
                                for file in os.listdir(data_dir):
                                    if 'unified_eye_metrics' in file and file.endswith('.csv'):
                                        data_path = os.path.join(data_dir, file)
                                        print(f"Found data file without exact match: {data_path}")
                                        # Extract real movie name from filename if possible
                                        parts = file.split('_unified_eye_metrics_')
                                        if len(parts) > 1 and '_' in parts[1]:
                                            real_movie_name = parts[1].split('.')[0]
                                            print(f"Detected movie name from file: {real_movie_name}")
                                        break
                            
                            # If we found a data file, stop searching
                            if data_path:
                                break

            # Make sure we have the screen dimensions initialized
            if not hasattr(self, 'screen_width') or not hasattr(self, 'screen_height'):
                self.screen_width = 1280
                self.screen_height = 1024

            # Load the data if found
            if data_path and os.path.exists(data_path):
                print(f"Loading animation data from: {data_path}")
                import pandas as pd  # Ensure pandas is imported here
                data = pd.read_csv(data_path)

                # Determine the actual movie name to use
                display_movie_name = movie
                
                # Try to extract the real movie name from the file path if not found above
                if not 'real_movie_name' in locals():
                    parts = os.path.basename(data_path).split('_unified_eye_metrics_')
                    if len(parts) > 1 and '.' in parts[1]:
                        real_movie_name = parts[1].split('.')[0]
                        display_movie_name = real_movie_name
                        print(f"Extracted movie name from path: {display_movie_name}")
                else:
                    display_movie_name = real_movie_name
                    print(f"Using previously detected movie name: {display_movie_name}")

                # Check if necessary columns are present
                required_cols = ['timestamp', 'x_left', 'y_left', 'x_right', 'y_right']
                missing_cols = [col for col in required_cols if col not in data.columns]

                if missing_cols:
                    print(f"Animation data missing required columns: {missing_cols}")
                    if hasattr(self, 'animated_viz_tab') and hasattr(self.animated_viz_tab, 'scanpath_widget'):
                        self.animated_viz_tab.scanpath_widget.status_label.setText(
                            f"Error: Data missing columns: {', '.join(missing_cols)}")
                elif data.empty:
                    print("Animation data is empty")
                    if hasattr(self, 'animated_viz_tab') and hasattr(self.animated_viz_tab, 'scanpath_widget'):
                        self.animated_viz_tab.scanpath_widget.status_label.setText("Error: Empty data file")
                else:
                    # Load data into the animated scanpath widget
                    if hasattr(self, 'animated_viz_tab'):
                        # If the widget isn't already a combined scanpath widget, 
                        # import the module and confirm it's the correct import
                        from animated_scanpath import AnimatedScanpathWidget
                        
                        # Load the data - this will populate both the normal scanpath
                        # view and any ROI data if available - use the actual movie name from the file
                        success = self.animated_viz_tab.load_data(data, display_movie_name, self.screen_width, self.screen_height)
                        if success:
                            print(f"Loaded {len(data)} samples for animated visualization of {display_movie_name}")
            else:
                print(f"No data file found for movie: {movie}")
                print(f"Searched in directory: {data_dir}")
                if data_dir:
                    print(f"Files in directory: {os.listdir(data_dir)}")
                
                if hasattr(self, 'animated_viz_tab') and hasattr(self.animated_viz_tab, 'scanpath_widget'):
                    # Status label was removed to save space, silently handle the error
                    pass

        except Exception as e:
            print(f"Error loading data for animated scanpath: {e}")
            import traceback
            traceback.print_exc()  # Print detailed error for debugging
            if hasattr(self, 'animated_viz_tab') and hasattr(self.animated_viz_tab, 'scanpath_widget'):
                # Status label was removed to save space, silently handle the error
                pass
        finally:
            # Always clear the loading flag to prevent deadlocks
            self._loading_animation_data = False

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

                # Update the explanation text for this visualization
                viz_name = viz_type.lower().replace(' ', '_')
                for key in self.visualization_explanations:
                    if key in viz_name:
                        self.viz_explanation.setHtml(self.visualization_explanations[key])
                        break
                else:
                    self.viz_explanation.setText(f"No detailed explanation available for {viz_type}.")
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
    
    def show_file_format_help(self):
        """Show detailed help about file formats"""
        help_text = (
            "<h3>Supported File Formats</h3>"
            "<p>The application automatically detects the file type based on the file extension.</p>"
            "<p><b>ASC Files:</b> Raw EyeLink eye tracking data files</p>"
            "<ul>"
            "<li>These are the original files exported from the EyeLink eye tracker</li>"
            "<li>They contain raw gaze data, events (fixations, saccades, blinks), and messages</li>"
            "<li>Processing these files takes longer but provides access to all raw data</li>"
            "<li>File extension: .asc</li>"
            "</ul>"
            "<p><b>CSV Files:</b> Preprocessed unified eye metrics files</p>"
            "<ul>"
            "<li>These are generated after processing ASC files</li>"
            "<li>They contain already extracted eye tracking data in a structured CSV format</li>"
            "<li>Loading these files is faster than processing raw ASC files</li>"
            "<li>Look for files containing 'unified_eye_metrics' in their name</li>"
            "<li>File extension: .csv</li>"
            "</ul>"
            "<p>For most visualization purposes, either file format will work. Use CSV files for faster loading "
            "when you've already processed the data once.</p>"
        )
        QMessageBox.information(self, "File Format Help", help_text)

    def save_features(self):
        """Save the extracted features to a CSV file"""
        if self.features_data is None or self.features_data.empty:
            QMessageBox.warning(self, "No Features Available",
                                "There are no features available to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Features", "", "CSV Files (*.csv)"
        )

        if file_path:
            try:
                self.features_data.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Successful",
                                        f"Features successfully exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error",
                                     f"Failed to export features: {str(e)}")

    def resizeEvent(self, event):
        """Handle window resize event to update image scaling"""
        super().resizeEvent(event)
        if hasattr(self, 'image_label') and self.image_label.pixmap() is not None:
            # Re-scale the current image if there is one
            self.show_visualization()

    def showEvent(self, event):
        """Handle window show event - make sure the window is maximized on startup"""
        super().showEvent(event)
        # Maximize window when first shown
        self.showMaximized()


def parse_args():
    """Parse command line arguments for the GUI."""
    import argparse
    parser = argparse.ArgumentParser(description='Eye Movement Analysis GUI')
    parser.add_argument('--test_mode', action='store_true', 
                      help='Run in test mode with predefined files')
    parser.add_argument('--source_file', type=str, 
                      help='Path to source file (.asc or .csv) for test mode')
    parser.add_argument('--destination_folder', type=str, 
                      help='Output folder for test mode')
    
    # For backward compatibility
    parser.add_argument('--csv_file', type=str, 
                      help='Path to CSV file for test mode (alternative to --source_file)')
    parser.add_argument('--use_csv', action='store_true',
                      help='[Deprecated] File type is now automatically detected')
    
    args = parser.parse_args()
    
    # If source_file is not set but csv_file is, use csv_file as source_file
    if not args.source_file and args.csv_file:
        args.source_file = args.csv_file
        
    return args


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeMovementAnalysisGUI()
    
    window.screen_width = 1280
    window.screen_height = 1024
    
    # Parse command line arguments
    args = parse_args()
    
    # If in test mode, automatically set up the files
    if args.test_mode:
        print("Running in test mode")
        
        # Set file if provided - this needs to happen before we create and show the window
        source_file_path = None
        if args.source_file:
            # Try to convert to absolute path if needed
            file_path = os.path.abspath(args.source_file)
            print(f"Checking file path: {file_path}")
            
            if os.path.exists(file_path):
                source_file_path = file_path
                window.file_paths = [file_path]
                # Update the file label (this will be visible once the window is shown)
                window.file_label.setText(f"Selected: {os.path.basename(file_path)}")
                
                # Automatically determine file type based on extension
                is_csv = file_path.lower().endswith('.csv')
                window.selected_file_type = "CSV Files" if is_csv else "ASC Files"
                file_type = "CSV" if is_csv else "ASC"
                print(f"Detected file type: {file_type}")
                
                # Update status label
                window.status_label.setText(f"Using {file_type} file: {os.path.basename(file_path)}")
                
                # Process events to try to update UI
                app.processEvents()
                print(f"Using file: {file_path}")
            else:
                print(f"WARNING: File not found at path: {file_path}")
        
        # Set output directory if provided
        dest_path = None
        if args.destination_folder:
            # Try to convert to absolute path if needed
            dest_path = os.path.abspath(args.destination_folder)
            
            if not os.path.exists(dest_path):
                os.makedirs(dest_path, exist_ok=True)
                
            window.output_dir = dest_path
            window.output_label.setText(f"Output: {dest_path}")
            # Force update the UI to ensure the label is refreshed
            app.processEvents()
            print(f"Using output directory: {dest_path}")
        
        # Update process button state if both file and directory are specified
        if source_file_path and dest_path:
            window.update_process_button()
            # Force update the UI to ensure the button is enabled
            app.processEvents()
            
            # If both file and output directory are specified, automatically process
            print("Auto-processing the specified file...")
            
            # Use a QTimer for delayed processing but also try immediate processing as backup
            from PyQt5.QtCore import QTimer
            
            # Show the window first to make sure all UI elements are visible
            window.show()
            app.processEvents()  # Process events to render the window
            
            # Use QTimer to process data after the UI has settled 
            # This avoids duplicate processing and ensures the UI is ready
            QTimer.singleShot(200, window.process_data)
    else:
        # Only show the window here if not in test mode
        window.show()
        
    sys.exit(app.exec_())
