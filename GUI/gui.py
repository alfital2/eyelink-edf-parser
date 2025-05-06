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
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QCheckBox, QTabWidget, QSplitter, QStackedWidget,
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
    status_update = pyqtSignal(str)  # New signal for status updates
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
            
            self.status_update.emit(f"Reading {file_type_display} file{'s' if num_files > 1 else ''}: {os.path.basename(file_paths_display) if num_files == 1 else ''}")
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
                        self.status_update.emit(f"Processing file {i+1}/{num_files}: {os.path.basename(file_path)}")
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
                        self.status_update.emit(f"Processing file {i+1}/{num_files}: {os.path.basename(file_path)}")
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
                self.status_update.emit(f"Created {total_visualizations} visualizations across {len(vis_results)} movies.")
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
                    'fixations': sum(len(fixs) for fixs in parser.fixations.values()) if hasattr(parser, 'fixations') else 0,
                    'saccades': sum(len(saccs) for saccs in parser.saccades.values()) if hasattr(parser, 'saccades') else 0,
                    'blinks': sum(len(blinks) for blinks in parser.blinks.values()) if hasattr(parser, 'blinks') else 0,
                    'frames': len(parser.frame_markers) if hasattr(parser, 'frame_markers') else 0
                }
                result['summary'] = summary
            
            self.processing_complete.emit(result)
            self.status_update.emit("Processing complete!")
            self.update_progress.emit(100)

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}"
            self.status_update.emit(error_msg)
            self.error_occurred.emit(f"{error_msg}\n{traceback.format_exc()}")
        finally:
            # Proper cleanup to prevent phantom threads
            self.quit()
            self.wait()


# The AnimatedROIScanpathTab class has been removed since we've integrated this functionality
# directly into the Results & Visualization tab


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
        
        # Screen dimensions for visualizations
        self.screen_width = 1280
        self.screen_height = 1024

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
            
        # Update feature header color based on theme
        if hasattr(self, 'features_header'):
            if self.is_dark_mode:
                self.features_header.setStyleSheet("color: #58b0ff;")
            else:
                self.features_header.setStyleSheet("color: #0078d7;")

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
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 16px;
                background-color: rgba(40, 40, 40, 150);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #333;
            }
            QTableWidget { 
                gridline-color: #555;
                background-color: rgba(60, 60, 60, 120);
                border: 1px solid #555;
                border-radius: 3px;
                alternate-background-color: rgba(70, 70, 70, 120);
            }
            QTableWidget::item {
                color: #f0f0f0;
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #2a82da;
            }
            QHeaderView::section {
                background-color: #444;
                color: #f0f0f0;
                border: 1px solid #555;
                padding: 4px;
                font-weight: bold;
            }
            QTextBrowser {
                background-color: rgba(60, 60, 60, 120);
                border: 1px solid #555;
                border-radius: 3px;
            }
            """
        else:
            return """
            QGroupBox { 
                font-weight: bold; 
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 16px;
                background-color: rgba(245, 245, 245, 150);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #f5f5f5;
            }
            QTableWidget { 
                gridline-color: #ccc;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 3px;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #0078d7;
                color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                padding: 4px;
                font-weight: bold;
            }
            QTextBrowser {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 3px;
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

        # Status section with detailed information
        status_section = QWidget()
        status_layout = QVBoxLayout(status_section)
        
        # Current status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        # Processing log (shows history of status updates)
        self.status_log = QTextBrowser()
        self.status_log.setMaximumHeight(150)
        self.status_log.setPlaceholderText("Processing log will appear here...")
        status_layout.addWidget(self.status_log)
        
        processing_layout.addWidget(status_section)

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

        # Create selector label with fixed width
        movie_label = QLabel("Select Movie:")
        movie_label.setFixedWidth(100)
        movie_layout.addWidget(movie_label)
        
        # Create movie combo with fixed width
        self.movie_combo = QComboBox()
        self.movie_combo.setEnabled(False)
        self.movie_combo.setMinimumWidth(200)
        self.movie_combo.currentIndexChanged.connect(self.movie_selected)
        movie_layout.addWidget(self.movie_combo)
        
        # Add spacer for better alignment
        movie_layout.addSpacing(20)

        # Visualization type selection
        viz_type_label = QLabel("Visualization Type:")
        viz_type_label.setFixedWidth(120)
        movie_layout.addWidget(viz_type_label)
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.setEnabled(False)
        self.viz_type_combo.setMinimumWidth(200)
        self.viz_type_combo.currentIndexChanged.connect(self.visualization_type_selected)
        movie_layout.addWidget(self.viz_type_combo)
        
        # Add spacer at the end
        movie_layout.addStretch(1)
        viz_layout.addWidget(movie_section)
        
        # ROI Controls Section
        roi_section = QWidget()
        roi_layout = QHBoxLayout(roi_section)
        
        # Load ROI button
        self.load_roi_btn = QPushButton("Load ROI")
        self.load_roi_btn.clicked.connect(self.select_roi_file)
        roi_layout.addWidget(self.load_roi_btn)
        
        # ROI file label
        self.roi_label = QLabel("No ROI file selected")
        roi_layout.addWidget(self.roi_label, 1)
        
        # Generate Social Attention Plots button (initially disabled)
        self.generate_social_btn = QPushButton("Generate Social Attention Plots")
        self.generate_social_btn.setEnabled(False)
        self.generate_social_btn.clicked.connect(self.generate_social_attention_plots)
        roi_layout.addWidget(self.generate_social_btn)
        
        viz_layout.addWidget(roi_section)

        # Create a stacked widget to switch between static image and animated visualization
        self.viz_stack = QStackedWidget()
        self.viz_stack.setMinimumSize(800, 500)
        self.viz_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add static image label as first widget in stack
        self.image_label = QLabel("Visualization will be shown here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.viz_stack.addWidget(self.image_label)
        
        # Create and add animated scanpath widget as second widget in stack
        self.animated_scanpath = AnimatedROIScanpathWidget()
        self.viz_stack.addWidget(self.animated_scanpath)
        
        # Add the stacked widget to the layout
        viz_layout.addWidget(self.viz_stack)

        # Visualization explanation has been removed as requested
        # Users can find descriptions in the documentation tab

        # Open report button
        self.report_btn = QPushButton("Open HTML Report")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self.open_report)
        viz_layout.addWidget(self.report_btn)

        results_layout.addWidget(viz_widget)

        # Tab 3: Features Display
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)

        # Features header
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(10, 10, 10, 0)
        
        self.features_header = QLabel("Eye Movement Features for Autism Research")
        self.features_header.setFont(QFont("Arial", 14, QFont.Bold))
        # Set header color based on theme
        if self.is_dark_mode:
            self.features_header.setStyleSheet("color: #58b0ff;")
        else:
            self.features_header.setStyleSheet("color: #0078d7;")
        header_layout.addWidget(self.features_header)
        
        # Add movie selector dropdown
        header_layout.addStretch()
        
        # Create a widget to hold the movie selector
        selector_container = QWidget()
        selector_layout = QHBoxLayout(selector_container)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create label with fixed width
        movie_label = QLabel("Select Movie:")
        movie_label.setFixedWidth(100)
        selector_layout.addWidget(movie_label)
        
        # Create combo with fixed width
        self.feature_movie_combo = QComboBox()
        self.feature_movie_combo.setMinimumWidth(200)
        self.feature_movie_combo.setToolTip("Select a movie to view its specific features, or 'All Data' to view aggregate features")
        self.feature_movie_combo.addItem("All Data")
        self.feature_movie_combo.setEnabled(False)
        self.feature_movie_combo.currentIndexChanged.connect(self.feature_movie_selected)
        selector_layout.addWidget(self.feature_movie_combo)
        
        header_layout.addWidget(selector_container)
        
        features_layout.addWidget(header_container)

        features_overview = QTextBrowser()
        features_overview.setMaximumHeight(100)
        features_overview.setHtml("""
        <p><b>Eye Movement Features for Autism Research</b> - This tab displays extracted eye movement features that may serve as biomarkers for autism spectrum disorder classification.
        Research suggests individuals with ASD exhibit distinct patterns of visual attention, particularly when viewing social stimuli.</p>
        <p><b>Data Organization:</b> Features are organized into categories, with left and right eye measurements displayed side by side for easy comparison.
        Hover over any feature name to see a detailed explanation of how it's calculated and its potential relevance to autism research.</p>
        <p><b>Movie Selection:</b> Use the dropdown menu to view features for specific movies or "All Data" for aggregate metrics across the entire recording session. 
        This allows you to compare eye movement patterns across different stimuli and identify context-specific effects.</p>
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
        tabs.addTab(features_tab, "Extracted Features")
        tabs.addTab(documentation_tab, "Documentation")

        # Set the central widget
        self.setCentralWidget(central_widget)
        
        # Initialize the ROI file path attribute
        self.roi_file_path = None
        self.roi_data = None


    def create_feature_tables(self, parent_layout):
        """Create organized tables for different categories of features in a grid layout"""
        # Create a scrollable widget for all tables
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        
        # Use a grid layout instead of vertical layout for better space utilization
        features_grid_layout = QGridLayout(scroll_content)
        features_grid_layout.setSpacing(15)  # Add more spacing between feature groups
        features_grid_layout.setContentsMargins(15, 15, 15, 15)  # Add margins around the grid

        if not self.is_dark_mode:
            # Fix the background color for light mode
            scroll_content.setStyleSheet("background-color: #f5f5f5;")

        # Define categories but reorganize them to better handle left/right eye metrics
        categories = [
            ("Basic Information", ["participant_id"], 0, 0),  # row 0, col 0
            
            # Combined eye metrics tables with left/right columns
            ("Pupil Size", [
                {"name": "Mean Pupil Size", "left": "pupil_left_mean", "right": "pupil_right_mean"},
                {"name": "Pupil Size Std", "left": "pupil_left_std", "right": "pupil_right_std"},
                {"name": "Min Pupil Size", "left": "pupil_left_min", "right": "pupil_right_min"},
                {"name": "Max Pupil Size", "left": "pupil_left_max", "right": "pupil_right_max"}
            ], 0, 1),  # row 0, col 1
            
            ("Gaze Position", [
                {"name": "X Standard Deviation", "left": "gaze_left_x_std", "right": "gaze_right_x_std"},
                {"name": "Y Standard Deviation", "left": "gaze_left_y_std", "right": "gaze_right_y_std"},
                {"name": "Gaze Dispersion", "left": "gaze_left_dispersion", "right": "gaze_right_dispersion"}
            ], 0, 2),  # row 0, col 2
            
            ("Fixation Metrics", [
                {"name": "Fixation Count", "left": "fixation_left_count", "right": "fixation_right_count"},
                {"name": "Mean Duration (ms)", "left": "fixation_left_duration_mean", "right": "fixation_right_duration_mean"},
                {"name": "Duration Std (ms)", "left": "fixation_left_duration_std", "right": "fixation_right_duration_std"},
                {"name": "Fixation Rate", "left": "fixation_left_rate", "right": "fixation_right_rate"}
            ], 1, 0),  # row 1, col 0
            
            ("Saccade Metrics", [
                {"name": "Saccade Count", "left": "saccade_left_count", "right": "saccade_right_count"},
                {"name": "Mean Amplitude (°)", "left": "saccade_left_amplitude_mean", "right": "saccade_right_amplitude_mean"},
                {"name": "Amplitude Std (°)", "left": "saccade_left_amplitude_std", "right": "saccade_right_amplitude_std"},
                {"name": "Mean Duration (ms)", "left": "saccade_left_duration_mean", "right": "saccade_right_duration_mean"}
            ], 1, 1),  # row 1, col 1
            
            ("Blink Metrics", [
                {"name": "Blink Count", "left": "blink_left_count", "right": "blink_right_count"},
                {"name": "Mean Duration (ms)", "left": "blink_left_duration_mean", "right": "blink_right_duration_mean"},
                {"name": "Blink Rate", "left": "blink_left_rate", "right": "blink_right_rate"}
            ], 1, 2),  # row 1, col 2
            
            ("Head Movement", [
                {"name": "Mean", "key": "head_movement_mean"},
                {"name": "Standard Deviation", "key": "head_movement_std"},
                {"name": "Maximum", "key": "head_movement_max"},
                {"name": "Frequency", "key": "head_movement_frequency"}
            ], 2, 0)  # row 2, col 0
        ]

        # Create a table for each category
        self.feature_tables = {}
        
        # Track all original feature keys for tooltip lookup
        all_feature_keys = {}
        
        # Process each category
        for category_info in categories:
            category_name = category_info[0]
            feature_data = category_info[1]
            row_pos = category_info[2]
            col_pos = category_info[3]
            
            # Create a group box for each category
            group_box = QGroupBox(category_name)
            group_layout = QVBoxLayout(group_box)
            group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Determine if this is a combined left/right table or a regular table
            is_combined = isinstance(feature_data[0], dict) and "left" in feature_data[0]
            
            if is_combined:
                # Combined left/right table with 3 columns
                table = QTableWidget(0, 3)  # Metric, Left Eye, Right Eye
                table.setHorizontalHeaderLabels(["Metric", "Left Eye", "Right Eye"])
                
                # Equal column widths
                header = table.horizontalHeader()
                header.setSectionResizeMode(0, QHeaderView.Stretch)
                header.setSectionResizeMode(1, QHeaderView.Stretch)
                header.setSectionResizeMode(2, QHeaderView.Stretch)
                
                # Collect all the original feature keys for tooltips
                feature_keys = []
                for item in feature_data:
                    if "left" in item:
                        feature_keys.append(item["left"])
                    if "right" in item:
                        feature_keys.append(item["right"])
                
                all_feature_keys[category_name] = feature_keys
                
            elif "key" in feature_data[0]:
                # Non-combined table with single value column
                table = QTableWidget(0, 2)  # Metric, Value
                table.setHorizontalHeaderLabels(["Metric", "Value"])
                
                # Equal column widths
                header = table.horizontalHeader()
                header.setSectionResizeMode(0, QHeaderView.Stretch)
                header.setSectionResizeMode(1, QHeaderView.Stretch)
                
                # Collect feature keys for tooltips
                feature_keys = [item["key"] for item in feature_data]
                all_feature_keys[category_name] = feature_keys
                
            else:
                # Regular table (e.g., Basic Information)
                table = QTableWidget(0, 2)  # Feature, Value
                table.setHorizontalHeaderLabels(["Feature", "Value"])
                
                # Equal column widths
                header = table.horizontalHeader()
                header.setSectionResizeMode(0, QHeaderView.Stretch)
                header.setSectionResizeMode(1, QHeaderView.Stretch)
                
                all_feature_keys[category_name] = feature_data
            
            # Common table settings
            table.verticalHeader().setVisible(False)
            table.setAlternatingRowColors(True)  # Enable alternating row colors
            table.setSelectionMode(QTableWidget.SingleSelection)  # Allow selecting entire rows
            table.setSelectionBehavior(QTableWidget.SelectRows)
            
            # Additional styling for light mode
            if not self.is_dark_mode:
                table.setStyleSheet("QTableWidget { background-color: white; border: 1px solid #ddd; }")
            
            # Enable tooltips for the table
            table.setMouseTracking(True)
            table.cellEntered.connect(lambda row, col, t=table, cat=category_name: 
                                     self.show_feature_tooltip(row, col, t, all_feature_keys.get(cat, [])))

            # Store the table and feature configuration
            self.feature_tables[category_name] = {
                "table": table,
                "features": feature_data,
                "is_combined": is_combined
            }
            
            # Add table to the group box
            group_layout.addWidget(table)
            
            # Add to grid layout at specified position
            features_grid_layout.addWidget(group_box, row_pos, col_pos)

        # Set column and row stretch factors to distribute space evenly
        for i in range(3):  # 3 columns
            features_grid_layout.setColumnStretch(i, 1)
        for i in range(3):  # 3 rows
            features_grid_layout.setRowStretch(i, 1)
        
        scroll_area.setWidget(scroll_content)
        parent_layout.addWidget(scroll_area)

    def show_feature_tooltip(self, row, col, table, features):
        """Show a tooltip with feature explanation when hovering over a feature name"""
        # Only show tooltips for feature names column (first column)
        if col == 0 and row < table.rowCount():
            # Get the tooltip directly from the table item if it exists
            cell_item = table.item(row, col)
            if cell_item and cell_item.toolTip():
                QToolTip.showText(QCursor.pos(), cell_item.toolTip())
            # If no tooltip in the item, try to find it from the feature list
            elif features and row < len(features):
                # Handle different feature list formats
                if isinstance(features, list):
                    if row < len(features):
                        # Get the feature key based on the type of item in the list
                        if isinstance(features[row], dict):
                            # If the features are dictionaries with keys
                            if "key" in features[row]:
                                feature_key = features[row]["key"]
                            elif "left" in features[row]:
                                # For left/right pairs, show tooltip for the left key
                                feature_key = features[row]["left"]
                            else:
                                return
                        else:
                            # Simple string keys
                            feature_key = features[row]
                        
                        # Show the tooltip if we have an explanation
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
            feature_data = table_info["features"]
            is_combined = table_info.get("is_combined", False)

            # Clear the table
            table.setRowCount(0)

            # Handle different table types
            if is_combined:
                # This is a combined left/right eye table with 3 columns
                self._update_combined_table(table, feature_data, features_df)
            elif isinstance(feature_data[0], dict) and "key" in feature_data[0]:
                # This is a regular table with named metrics
                self._update_named_table(table, feature_data, features_df)
            else:
                # This is a simple table with direct feature keys
                self._update_simple_table(table, feature_data, features_df)
                
    def _update_combined_table(self, table, feature_data, features_df):
        """Update a table with left/right eye metrics in separate columns"""
        for i, item in enumerate(feature_data):
            # Skip if either left or right key is missing from the feature data
            if not all(key in features_df.columns for key in [item["left"], item["right"]]):
                continue
                
            row_position = table.rowCount()
            table.insertRow(row_position)
            
            # Create items
            name_item = QTableWidgetItem(item["name"])
            
            # Set tooltips with explanation if available
            tooltips = []
            if item["left"] in self.feature_explanations:
                tooltips.append(self.feature_explanations[item["left"]])
            if item["right"] in self.feature_explanations:
                tooltips.append(self.feature_explanations[item["right"]])
                
            if tooltips:
                name_item.setToolTip("\n\n".join(tooltips))
                
            # Get and format left eye value
            left_value = features_df[item["left"]].iloc[0]
            left_value_text = self._format_value(left_value)
            left_item = QTableWidgetItem(left_value_text)
            
            # Get and format right eye value
            right_value = features_df[item["right"]].iloc[0]
            right_value_text = self._format_value(right_value)
            right_item = QTableWidgetItem(right_value_text)
            
            # Add items to table
            table.setItem(row_position, 0, name_item)
            table.setItem(row_position, 1, left_item)
            table.setItem(row_position, 2, right_item)
    
    def _update_named_table(self, table, feature_data, features_df):
        """Update a table with named metrics (non-left/right)"""
        for i, item in enumerate(feature_data):
            if item["key"] not in features_df.columns:
                continue
                
            row_position = table.rowCount()
            table.insertRow(row_position)
            
            # Create items
            name_item = QTableWidgetItem(item["name"])
            
            # Set tooltip with explanation if available
            if item["key"] in self.feature_explanations:
                name_item.setToolTip(self.feature_explanations[item["key"]])
                
            # Get and format value
            value = features_df[item["key"]].iloc[0]
            value_text = self._format_value(value)
            value_item = QTableWidgetItem(value_text)
            
            # Add items to table
            table.setItem(row_position, 0, name_item)
            table.setItem(row_position, 1, value_item)
    
    def _update_simple_table(self, table, feature_keys, features_df):
        """Update a simple table with direct feature keys"""
        for i, feature_key in enumerate(feature_keys):
            if feature_key not in features_df.columns:
                continue
                
            row_position = table.rowCount()
            table.insertRow(row_position)
            
            # Format the feature name to be more readable
            display_name = feature_key.replace('_', ' ').title()
            
            # Create items
            name_item = QTableWidgetItem(display_name)
            
            # Set tooltip with explanation if available
            if feature_key in self.feature_explanations:
                name_item.setToolTip(self.feature_explanations[feature_key])
                
            # Get and format value
            value = features_df[feature_key].iloc[0]
            value_text = self._format_value(value)
            value_item = QTableWidgetItem(value_text)
            
            # Add items to table
            table.setItem(row_position, 0, name_item)
            table.setItem(row_position, 1, value_item)
    
    def _format_value(self, value):
        """Format a value for display in the table"""
        # Handle NaN values properly
        if pd.isna(value):
            return "N/A"
        elif isinstance(value, (int, float)):
            # Format number with appropriate precision
            try:
                if float(value).is_integer():
                    return str(int(value))
                else:
                    return f"{value:.4f}"
            except:
                # If conversion fails, use the value as is
                return str(value)
        else:
            return str(value)

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
        
        # Clear the status log before starting new processing
        self.status_log.clear()

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
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)

        # Start processing
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """Update the status label with detailed processing information"""
        # Update current status label
        self.status_label.setText(message)
        
        # Add message to the processing log with timestamp
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        # Add new message to the log
        self.status_log.append(formatted_message)
        
        # Scroll to bottom to show the latest message
        self.status_log.verticalScrollBar().setValue(self.status_log.verticalScrollBar().maximum())

    def processing_finished(self, results):
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing complete!")

        # Re-enable UI elements
        self.process_btn.setEnabled(True)

        # Show results summary
        if "summary" in results:
            summary = results["summary"]
            
            # Format summary message
            msg = (f"Processed data:\n"
                   f"- {summary['samples']} samples\n"
                   f"- {summary['fixations']} fixations\n"
                   f"- {summary['saccades']} saccades\n"
                   f"- {summary['blinks']} blinks\n"
                   f"- {summary['frames']} frames")
            
            # Add summary to the status log with highlighting
            self.status_log.setTextColor(Qt.blue if not self.is_dark_mode else Qt.cyan)
            self.status_log.append("\n--- PROCESSING SUMMARY ---")
            self.status_log.append(f"• Processed {summary['samples']} eye tracking samples")
            self.status_log.append(f"• Detected {summary['fixations']} fixations")
            self.status_log.append(f"• Detected {summary['saccades']} saccades")
            self.status_log.append(f"• Detected {summary['blinks']} blinks")
            self.status_log.append(f"• Processed {summary['frames']} video frames")
            
            # If there are visualizations, add count
            if "visualizations" in results and results["visualizations"]:
                viz_count = sum(len(viz) for viz in results["visualizations"].values())
                self.status_log.append(f"• Generated {viz_count} visualizations")
                
            # Add output directory
            if "output_dir" in results:
                self.status_log.append(f"• Output saved to: {results['output_dir']}")
            
            self.status_log.append("--- END OF SUMMARY ---")
            self.status_log.setTextColor(Qt.black if not self.is_dark_mode else Qt.white)
                
            # Summary is already shown in the status log, no need for a popup message

        # Update the features display if features were extracted
        if "features" in results and not results["features"].empty:
            # Store movie-specific features if available
            if "movie_features" in results:
                self.movie_features = results["movie_features"]
                
                # Clear and populate the feature movie combo box
                self.feature_movie_combo.clear()
                self.feature_movie_combo.addItem("All Data")
                
                # Add movie names (excluding "All Data" which we already added)
                for movie_name in self.movie_features.keys():
                    if movie_name != "All Data":
                        self.feature_movie_combo.addItem(movie_name)
                
                # Enable the combo box if we have multiple movies
                self.feature_movie_combo.setEnabled(self.feature_movie_combo.count() > 1)
                
                # Select "All Data" by default
                self.feature_movie_combo.setCurrentIndex(0)
            else:
                # If no movie features, just update with the overall features
                self.movie_features = {"All Data": results["features"]}
                self.feature_movie_combo.setEnabled(False)
            
            # Display the features (initially shows "All Data")
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
            
            # No need to preload animation data anymore - it will be loaded when requested

        # Enable the report button if a report was generated
        if 'report_path' in results and os.path.exists(results['report_path']):
            self.report_path = results['report_path']
            self.report_btn.setEnabled(True)
            
    def feature_movie_selected(self, index):
        """Handle movie selection in the features tab"""
        if index < 0 or not hasattr(self, 'movie_features'):
            return
            
        # Get the selected movie name
        movie_name = self.feature_movie_combo.currentText()
        
        # Update the features display with the selected movie's features
        if movie_name in self.movie_features:
            # Update feature tables with the selected movie's features
            self.update_feature_tables(self.movie_features[movie_name])
            
            # Update the header to indicate which movie's features are displayed
            if movie_name == "All Data":
                header_text = "Eye Movement Features for Autism Research"
            else:
                header_text = f"Eye Movement Features: {movie_name}"
            
            self.features_header.setText(header_text)

    def processing_error(self, error_msg):
        self.status_label.setText("Error occurred during processing")
        self.process_btn.setEnabled(True)
        
        # Log the error with a different format to make it stand out
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        # Add a short version to the log
        short_error = error_msg.split('\n')[0]  # Just the first line
        formatted_error = f"[{timestamp}] ❌ ERROR: {short_error}"
        self.status_log.append(formatted_error)
        
        # Make the error message red to stand out
        self.status_log.setTextColor(Qt.red)
        self.status_log.append("See details in the error message box")
        self.status_log.setTextColor(Qt.black if not self.is_dark_mode else Qt.white)
        
        # Scroll to bottom
        self.status_log.verticalScrollBar().setValue(self.status_log.verticalScrollBar().maximum())
        
        # Show error in message box
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
            
        # No need to preload animation data anymore - it will be loaded when requested in show_visualization

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
            
            # Add static visualizations
            self.viz_type_combo.addItems(sorted(available_visualizations.keys()))
            
            # Add animated scanpath visualization option
            self.viz_type_combo.addItem("Animated Scanpath")
            
            self.viz_type_combo.setEnabled(True)

            # Select the first visualization
            self.viz_type_combo.setCurrentIndex(0)
        else:
            self.viz_type_combo.setEnabled(False)
            self.image_label.setText(f"No visualizations found for movie: {movie}")

    # This method has been removed since we now handle animated visualization directly in the show_visualization method

    def visualization_type_selected(self, index):
        """Show the selected visualization"""
        if index < 0 or self.viz_type_combo.count() == 0:
            return

        self.show_visualization()

    def show_visualization(self):
        """Load and display the selected visualization"""
        movie = self.movie_combo.currentText()
        viz_type = self.viz_type_combo.currentText()
        
        # Special case for animated scanpath
        if viz_type == "Animated Scanpath":
            # Switch to the animated scanpath widget
            self.viz_stack.setCurrentIndex(1)
            
            # Load the data for this movie into the animated scanpath widget
            try:
                # Find the data path for this movie - reusing _load_animation_data_for_movie logic
                data = self._get_movie_data(movie)
                if data is not None:
                    # Extract real movie name from the data path if possible
                    data_path = data["data_path"]
                    parts = os.path.basename(data_path).split('_unified_eye_metrics_')
                    if len(parts) > 1 and '.' in parts[1]:
                        real_movie_name = parts[1].split('.')[0]
                    else:
                        real_movie_name = movie
                    
                    # Load the data into the animated scanpath widget
                    self.animated_scanpath.load_data(data["data"], None, real_movie_name, 
                                                     self.screen_width, self.screen_height)
                else:
                    self.image_label.setText(f"No data available for movie: {movie}")
                    self.viz_stack.setCurrentIndex(0)
            except Exception as e:
                self.image_label.setText(f"Error loading animated scanpath: {str(e)}")
                self.viz_stack.setCurrentIndex(0)
            return

        # For all other visualization types, switch to the image label widget
        self.viz_stack.setCurrentIndex(0)
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
            
    def _get_movie_data(self, movie):
        """Helper method to get the data for a specific movie"""
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
                                    break
                        
                        # If we found a data file, stop searching
                        if data_path:
                            break

        # Load the data if found
        if data_path and os.path.exists(data_path):
            import pandas as pd  # Ensure pandas is imported here
            data = pd.read_csv(data_path)
            
            # Check if necessary columns are present
            required_cols = ['timestamp', 'x_left', 'y_left', 'x_right', 'y_right']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols or data.empty:
                return None
            else:
                return {"data": data, "data_path": data_path}
        
        return None

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
            
    def select_roi_file(self):
        """Open file dialog to select ROI JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ROI File", "", "JSON Files (*.json)"
        )
        
        if file_path and os.path.exists(file_path):
            # Store the path and update the UI
            self.roi_file_path = file_path
            self.roi_label.setText(f"ROI File: {os.path.basename(file_path)}")
            
            # Enable the generate social attention plots button
            self.generate_social_btn.setEnabled(True)
    
    def generate_social_attention_plots(self):
        """Generate social attention plots based on loaded ROI file"""
        if not self.roi_file_path:
            QMessageBox.warning(
                self, 
                "ROI File Required", 
                "Please load an ROI file first."
            )
            return
            
        # Get the currently selected movie
        if self.movie_combo.count() == 0 or self.movie_combo.currentIndex() < 0:
            QMessageBox.warning(
                self,
                "No Movie Selected",
                "Please select a movie for analysis."
            )
            return
            
        movie = self.movie_combo.currentText()
        
        # Get the movie data
        movie_data = self._get_movie_data(movie)
        if movie_data is None or movie_data.get("data") is None:
            QMessageBox.warning(
                self,
                "No Data Available",
                f"Could not find eye tracking data for {movie}."
            )
            return
            
        # Load the ROI file
        import json
        try:
            with open(self.roi_file_path, 'r') as f:
                raw_roi_data = json.load(f)
            
            # Check for the new format with "annotations" key
            if "annotations" in raw_roi_data:
                print(f"DEBUG: Found 'annotations' key in ROI file, using new format")
                roi_data = raw_roi_data["annotations"]
            else:
                print(f"DEBUG: Using legacy ROI format")
                roi_data = raw_roi_data
                
            print(f"DEBUG: Processed ROI data contains {len(roi_data)} frame entries")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading ROI File",
                f"Failed to load ROI data: {str(e)}"
            )
            return
            
        # Show processing message
        progress_dialog = QMessageBox(
            QMessageBox.Information,
            "Processing",
            f"Generating social attention plots for {movie}...",
            QMessageBox.NoButton,
            self
        )
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # Create a plot showing time spent on each ROI
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            
            data = movie_data["data"]
            
            print(f"DEBUG: Starting social attention plot generation")
            print(f"DEBUG: ROI data keys: {list(roi_data.keys())}")
            print(f"DEBUG: Data shape: {data.shape}, columns: {data.columns}")
            
            # Get frame numbers from both data and ROI file
            if 'frame_number' not in data.columns:
                raise ValueError("Eye tracking data does not contain frame numbers")
                
            print(f"DEBUG: Frame numbers in data: {data['frame_number'].min()} to {data['frame_number'].max()}")
            
            # Count time spent on each ROI
            roi_durations = {}
            
            # First, convert frame keys to integers if they're strings
            frame_keys = {}
            for key in roi_data.keys():
                try:
                    frame_keys[int(key)] = roi_data[key]
                    # Print a sample of ROI data for the first few frames to debug
                    if len(frame_keys) <= 3:  # Only show first 3 frames for debugging
                        roi_sample = roi_data[key]
                        if roi_sample:  # If the frame has ROIs defined
                            roi_labels = [roi['label'] for roi in roi_sample if 'label' in roi]
                            print(f"DEBUG: Frame {key} has {len(roi_sample)} ROIs: {roi_labels}")
                except ValueError:
                    print(f"DEBUG: Skipping non-integer key: {key}")
                    continue  # Skip non-integer keys
                    
            if frame_keys:    
                print(f"DEBUG: Converted {len(frame_keys)} frame keys, range: {min(frame_keys.keys())} to {max(frame_keys.keys())}")
                
                # Print a histogram of frame number distribution to help understand spacing
                frame_numbers = sorted(frame_keys.keys())
                if len(frame_numbers) > 1:
                    intervals = [frame_numbers[i+1] - frame_numbers[i] for i in range(len(frame_numbers)-1)]
                    if intervals:
                        avg_interval = sum(intervals) / len(intervals)
                        print(f"DEBUG: Average interval between frames: {avg_interval:.2f}")
                        print(f"DEBUG: First 10 frame numbers: {frame_numbers[:10]}")
            else:
                print(f"DEBUG: No valid frame keys found in ROI data")
            
            # Get fixation data
            fixation_data = data[data['is_fixation_left'] | data['is_fixation_right']]
            print(f"DEBUG: Found {len(fixation_data)} fixation data points")
            
            # Process each fixation
            processed_count = 0
            hit_count = 0
            
            for _, row in fixation_data.iterrows():
                if pd.isna(row['frame_number']):
                    continue
                    
                frame_num = int(row['frame_number'])
                processed_count += 1
                
                # Find the nearest frame in ROI data
                if not frame_keys:
                    print(f"DEBUG: No frame keys available")
                    break
                
                try:
                    # We want to process all frames to find ROI hits
                    # only uncomment for debugging if output gets too verbose
                    # if processed_count > 200:
                    #     print(f"DEBUG: Limiting to first 200 frames for debugging")
                    #     break
                        
                    nearest_frame = min(frame_keys.keys(), key=lambda x: abs(x - frame_num))
                    frame_distance = abs(nearest_frame - frame_num)
                    
                    print(f"DEBUG: Frame {frame_num} closest match is ROI frame {nearest_frame}, distance: {frame_distance}")
                    
                    # Significantly increase the frame distance tolerance 
                    # ROI data often has large gaps between defined frames
                    if frame_distance > 1000:  # Increased to 1000 to handle sparse ROI data
                        print(f"DEBUG: Frame distance too large ({frame_distance}), skipping")
                        continue
                        
                    if frame_distance > 0:
                        print(f"DEBUG: Using ROI frame {nearest_frame} for eye tracking frame {frame_num} (distance: {frame_distance})")
                except Exception as e:
                    print(f"DEBUG: Error finding nearest frame: {e}")
                    continue
                    
                # Check if the gaze is in any ROI
                rois_in_frame = frame_keys[nearest_frame]
                print(f"DEBUG: Frame {frame_num} -> nearest ROI frame {nearest_frame} with {len(rois_in_frame)} ROIs")
                
                # Get normalized coordinates
                if pd.isna(row['x_left']) or pd.isna(row['y_left']):
                    continue
                
                # Print original coordinates for debugging
                print(f"DEBUG: Original coordinates - x: {row['x_left']}, y: {row['y_left']}")
                
                # Check if coordinates need normalization (>1.0 means they're in pixels)
                if row['x_left'] > 1.0 or row['y_left'] > 1.0:
                    x_norm = row['x_left'] / self.screen_width
                    y_norm = row['y_left'] / self.screen_height
                    print(f"DEBUG: Normalized from pixels - x_norm: {x_norm}, y_norm: {y_norm}")
                else:
                    x_norm = row['x_left']
                    y_norm = row['y_left']
                    print(f"DEBUG: Already normalized - x_norm: {x_norm}, y_norm: {y_norm}")
                    
                # Check each ROI in this frame
                for roi in rois_in_frame:
                    if 'label' not in roi or 'coordinates' not in roi:
                        print(f"DEBUG: Missing label or coordinates in ROI: {roi.keys()}")
                        continue
                        
                    label = roi['label']
                    coords = roi['coordinates']
                    
                    print(f"DEBUG: Checking ROI '{label}' with {len(coords)} coordinate points")
                    for i, coord in enumerate(coords):
                        print(f"DEBUG: ROI vertex {i}: x={coord['x']}, y={coord['y']}")
                    
                    # Check if point is inside polygon
                    is_inside = self._point_in_polygon(x_norm, y_norm, coords)
                    print(f"DEBUG: Point ({x_norm}, {y_norm}) is {'inside' if is_inside else 'outside'} ROI '{label}'")
                    
                    if is_inside:
                        # Add time spent to this ROI
                        if label not in roi_durations:
                            roi_durations[label] = 0
                        roi_durations[label] += 1  # Each fixation counts as one time unit
                        hit_count += 1
                        print(f"DEBUG: Hit count increased to {hit_count}, duration for '{label}' = {roi_durations[label]}")
                        break  # Only count one ROI per fixation
            
            print(f"DEBUG: Processed {processed_count} fixations, found {hit_count} ROI hits")
            print(f"DEBUG: ROI durations: {roi_durations}")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Initialize bars variable to avoid reference errors
            bars = None
            
            # Check if we found any ROI hits
            if not roi_durations:
                print(f"DEBUG: No ROI hits found! Creating empty plot with message.")
                # Display a message on the plot
                ax.text(0.5, 0.5, "No ROI fixations detected.\nCheck ROI file and eye tracking data alignment.",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'No ROI Hits Found in {movie}')
                # Set some reasonable axis limits for empty plot
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(-0.5, 0.5)
            else:
                # Sort ROIs by duration
                sorted_rois = sorted(roi_durations.items(), key=lambda x: x[1], reverse=True)
                labels = [item[0] for item in sorted_rois]
                durations = [item[1] for item in sorted_rois]
                
                print(f"DEBUG: Plotting with labels: {labels}")
                print(f"DEBUG: Plotting with durations: {durations}")
                
                # Plot bar chart
                bars = ax.bar(labels, durations, color='skyblue')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height}',
                            ha='center', va='bottom')
                
                # Add title and labels
                ax.set_title(f'Time Spent on Each ROI in {movie}')
                ax.set_xlabel('ROI Label')
                ax.set_ylabel('Fixation Count')
                
                # Rotate x-axis labels if many ROIs
                if len(labels) > 5:
                    plt.xticks(rotation=45, ha='right')
                
            # Tight layout
            plt.tight_layout()
            
            # Save the plot to the same directory as other plots
            # First find the plots directory for this movie
            plots_dir = None
            
            # Check if the movie exists in visualization_results
            if movie in self.visualization_results:
                # Get any existing plot path for this movie
                for category, plot_paths in self.visualization_results[movie].items():
                    if plot_paths:
                        # Get directory containing the visualization
                        plots_dir = os.path.dirname(plot_paths[0])
                        break
            
            # If no plots directory found, create one in the output directory
            if plots_dir is None:
                # Use the data directory from movie_data
                data_dir = os.path.dirname(movie_data["data_path"])
                # Plots directory is typically parallel to data directory
                plots_dir = os.path.join(data_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
            
            # Create the filename
            plot_filename = f"social_attention_roi_time_{movie.replace(' ', '_')}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            
            # Save the figure
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Add the plot to visualization results for this movie
            if movie not in self.visualization_results:
                self.visualization_results[movie] = {}
                
            if 'social' not in self.visualization_results[movie]:
                self.visualization_results[movie]['social'] = []
                
            self.visualization_results[movie]['social'].append(plot_path)
            
            # Add the plot to the movie visualizations dictionary
            if movie not in self.movie_visualizations:
                self.movie_visualizations[movie] = {}
                
            # Add with a display name
            display_name = "ROI Attention Time"
            self.movie_visualizations[movie][display_name] = plot_path
            
            # Update the visualization dropdown
            current_index = self.viz_type_combo.currentIndex()
            self.movie_selected(self.movie_combo.currentIndex())
            
            # Set the dropdown to the new visualization
            for i in range(self.viz_type_combo.count()):
                if self.viz_type_combo.itemText(i) == display_name:
                    self.viz_type_combo.setCurrentIndex(i)
                    break
            
            # Close progress dialog
            progress_dialog.close()
            
            # Show success message
            QMessageBox.information(
                self,
                "Plot Generated",
                f"Social attention plot for {movie} has been generated and added to the visualization dropdown."
            )
            
        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(
                self,
                "Error Generating Plot",
                f"Failed to generate social attention plot: {str(e)}"
            )
            
    def _point_in_polygon(self, x, y, coordinates):
        """Check if a point is inside a polygon defined by coordinates"""
        # Extract points from coordinates
        points = [(coord['x'], coord['y']) for coord in coordinates]
        
        print(f"DEBUG: Point-in-polygon check for point ({x}, {y}) against polygon with {len(points)} points")
        print(f"DEBUG: Polygon vertices: {points}")
        
        # Need at least 3 points to form a polygon
        if len(points) < 3:
            print(f"DEBUG: Polygon has fewer than 3 points, returning False")
            return False
            
        # Ray casting algorithm
        inside = False
        j = len(points) - 1
        
        for i in range(len(points)):
            xi, yi = points[i]
            xj, yj = points[j]
            
            # Check if point is on an edge
            if (yi == y and xi == x) or (yj == y and xj == x):
                print(f"DEBUG: Point is exactly on vertex ({xi}, {yi}) or ({xj}, {yj}), returning True")
                return True
                
            # Check if the point is on a horizontal edge
            if (yi == yj) and (yi == y) and (min(xi, xj) <= x <= max(xi, xj)):
                print(f"DEBUG: Point is on horizontal edge between ({xi}, {yi}) and ({xj}, {yj}), returning True")
                return True
                
            # Ray casting
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
                print(f"DEBUG: Ray crossed edge between ({xi}, {yi}) and ({xj}, {yj}), inside = {inside}")
                
            j = i
            
        print(f"DEBUG: Point-in-polygon result: {inside}")
        return inside

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

# run the gui on test mode with this syntax : --test_mode
#       --source_file
#       /Users/talalfi/Desktop/tmp/1017735502_unified_eye_metrics_Dinstein_Girls_90_SecX.csv
#       --destination_folder /Users/talalfi/Desktop/tmp/dst_files
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeMovementAnalysisGUI()
    
    # Screen dimensions are now set in __init__
    
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
