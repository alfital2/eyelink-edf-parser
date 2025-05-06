import sys
import os

# Add parent directory to path so we can import modules from the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Configure matplotlib for thread safety BEFORE importing any matplotlib-related modules
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues

# Global variable for plot progress tracking
current_plot_progress = "0/0"  # Will be updated during plot generation

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
        
        # First add any visualizations from our movie_visualizations dictionary
        # which might include the advanced plots
        if movie in self.movie_visualizations:
            for display_name, plot_path in self.movie_visualizations[movie].items():
                # Skip the Social vs Non-Social Balance plot
                if "Social vs Non-Social Balance" in display_name:
                    print(f"DEBUG: Skipping disabled plot: {display_name}")
                    continue
                    
                if os.path.exists(plot_path):
                    available_visualizations[display_name] = plot_path
                    print(f"DEBUG: Using existing visualization: {display_name} -> {os.path.basename(plot_path)}")

        # Then search through visualization_results
        for category, plot_paths in self.visualization_results[movie].items():
            print(f"DEBUG: Processing {len(plot_paths)} plots in category '{category}'")
            for plot_path in plot_paths:
                if os.path.exists(plot_path):
                    # Extract the base filename without participant prefix
                    basename = os.path.basename(plot_path)
                    # Skip any hidden files
                    if basename.startswith('.'):
                        continue
                        
                    # Skip paths we've already added
                    already_added = False
                    for existing_path in available_visualizations.values():
                        if plot_path == existing_path:
                            already_added = True
                            break
                    if already_added:
                        continue

                    # For ROI plots, use special handling to create better display names
                    if "roi_" in basename.lower() or "social_attention" in basename.lower():
                        if "roi_fixation_sequence" in basename:
                            display_name = "ROI Fixation Sequence"
                        elif "roi_transition_matrix" in basename:
                            display_name = "ROI Transition Matrix"
                        elif "roi_first_fixation_latency" in basename:
                            display_name = "ROI First Fixation Latency"
                        # ROI Dwell Time Comparison removed (merged with ROI Attention Time)
                        # ROI Revisitation removed as per user request
                        elif "roi_fixation_duration_distribution" in basename:
                            display_name = "ROI Fixation Duration Distribution"
                        elif "roi_temporal_heatmap" in basename:
                            display_name = "ROI Temporal Heatmap"
                        elif "social_attention_roi_time" in basename:
                            display_name = "ROI Attention Time"
                        else:
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
                    else:
                        # Standard handling for non-ROI plots
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
                    print(f"DEBUG: Added visualization: {display_name} -> {os.path.basename(plot_path)}")

        # Add available visualizations to combo box
        if available_visualizations:
            # Update movie_visualizations
            self.movie_visualizations[movie] = available_visualizations
            
            # Print available visualizations for debugging
            print(f"DEBUG: Adding {len(available_visualizations)} visualizations to dropdown:")
            for viz_name in sorted(available_visualizations.keys()):
                print(f"DEBUG:   - {viz_name}: {os.path.basename(available_visualizations[viz_name])}")
            
            # Clear and add static visualizations
            self.viz_type_combo.clear()
            self.viz_type_combo.addItems(sorted(available_visualizations.keys()))
            
            # Add animated scanpath visualization option
            self.viz_type_combo.addItem("Animated Scanpath")
            
            self.viz_type_combo.setEnabled(True)
            
            # Make sure special ROI visualizations are included
            roi_entries = [name for name in available_visualizations.keys() 
                          if "roi" in name.lower() or "fixation sequence" in name.lower() 
                          or "transition matrix" in name.lower() or "revisitation" in name.lower()]
            
            if roi_entries:
                print(f"DEBUG: Found {len(roi_entries)} ROI-related visualization entries:")
                for entry in roi_entries:
                    print(f"DEBUG:   - {entry}")

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
    
    def create_advanced_roi_plots(self, movie, roi_durations, fixation_data, plots_dir, 
                               frame_keys, frame_range_map, polygon_check_cache, status_label, progress_bar,
                               update_progress_func=None):
        """
        Generate advanced ROI-based social attention plots
        
        Args:
            movie: Name of the movie being analyzed
            roi_durations: Dictionary with ROI labels as keys and fixation counts as values
            fixation_data: DataFrame with fixation data
            plots_dir: Directory to save the plots
            frame_keys: Dictionary mapping frame numbers to ROI data
            frame_range_map: Dictionary for fast frame lookup
            polygon_check_cache: Cache for polygon checks 
            status_label: Label to update with status
            progress_bar: Progress bar to update
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator
        from collections import defaultdict
        
        if not roi_durations:
            print("WARNING: No ROI hits found, cannot generate advanced plots")
            return
            
        print(f"DEBUG: Found {len(roi_durations)} ROIs with the following durations:")
        for roi, duration in sorted(roi_durations.items(), key=lambda x: x[1], reverse=True):
            print(f"DEBUG:   - {roi}: {duration} fixations")
            
        # Sort the roi_durations by value (descending) for consistent ordering in plots
        sorted_rois = sorted(roi_durations.items(), key=lambda x: x[1], reverse=True)
        roi_labels = [item[0] for item in sorted_rois]
        
        # Create a sequential fixation history to track ROI sequence over time
        # We'll use this for multiple plots
        status_label.setText("Analyzing fixation sequence...")
        progress_bar.setValue(97)
        QApplication.processEvents()
        
        # Initialize data structures for all plots
        roi_sequence = []  # For Fixation Sequence
        first_fixation_times = {roi: None for roi in roi_labels}  # For First Fixation Latency
        roi_revisits = {roi: 0 for roi in roi_labels}  # For Revisitation
        seen_rois = set()  # Track which ROIs have been seen
        
        # For duration distributions (new plot)
        fixation_durations = {roi: [] for roi in roi_labels}
        last_fixation_data = {'roi': None, 'start_time': None}
        
        # For Temporal Heatmap (new plot)
        # Create time bins (100ms bins across the entire stimulus)
        time_bin_size = 0.1  # 100ms bins
        max_time = fixation_data['timestamp'].max()
        min_time = fixation_data['timestamp'].min()
        total_duration = (max_time - min_time) / 1000.0  # in seconds
        num_bins = int(total_duration / time_bin_size) + 1
        
        # Create a 2D array: rows = ROIs, columns = time bins
        temporal_heatmap = {roi: np.zeros(num_bins) for roi in roi_labels}
        
        # For Transition Matrix
        transition_matrix = defaultdict(lambda: defaultdict(int))
        last_roi = None
        
        # Re-process fixations to collect data for all plots in a single pass
        fixation_count = len(fixation_data)
        fixation_data = fixation_data.sort_values(by='timestamp')  # Ensure time ordering
        
        # First timestamp for relative timing
        start_timestamp = fixation_data['timestamp'].min()
        
        # Process each fixation
        for idx, row in fixation_data.iterrows():
            if pd.isna(row['frame_number']):
                continue
                
            frame_num = int(row['frame_number'])
            
            # Find the nearest frame 
            nearest_frame = None
            for (start, end), frame in frame_range_map.items():
                if start <= frame_num < end:
                    nearest_frame = frame
                    break
                    
            if nearest_frame is None:
                try:
                    nearest_frame = min(frame_keys.keys(), key=lambda x: abs(x - frame_num))
                except:
                    continue
                    
            frame_distance = abs(nearest_frame - frame_num)
            if frame_distance > 1000:  
                continue
                
            # Get the ROIs for this frame
            rois_in_frame = frame_keys[nearest_frame]
            
            # Get normalized coordinates
            if row['x_left'] > 1.0 or row['y_left'] > 1.0:
                x_norm = row['x_left'] / self.screen_width
                y_norm = row['y_left'] / self.screen_height
            else:
                x_norm = row['x_left']
                y_norm = row['y_left']
                
            # Find which ROI the fixation is in, if any
            current_roi = None
            for roi in rois_in_frame:
                if 'label' not in roi or 'coordinates' not in roi:
                    continue
                
                label = roi['label']
                coords = roi['coordinates']
                
                if label not in roi_labels:
                    continue  # Skip ROIs that didn't make it into the main plot
                
                # Use cached polygon checks
                cache_key = (tuple((coord['x'], coord['y']) for coord in coords), x_norm, y_norm)
                if cache_key in polygon_check_cache:
                    is_inside = polygon_check_cache[cache_key]
                else:
                    is_inside = self._point_in_polygon(x_norm, y_norm, coords)
                    polygon_check_cache[cache_key] = is_inside
                
                if is_inside:
                    current_roi = label
                    break
                    
            if current_roi:
                # Add to sequence
                timestamp_sec = (row['timestamp'] - start_timestamp) / 1000.0  # Convert to seconds
                roi_sequence.append((timestamp_sec, current_roi))
                
                # Track fixation durations for the new ROI Fixation Duration Distribution plot
                if last_fixation_data['roi'] == current_roi and last_fixation_data['start_time'] is not None:
                    # Same ROI as last fixation - continue the duration
                    pass
                else:
                    # Different ROI or first fixation on this ROI
                    # If we were tracking a previous ROI, save its duration
                    if last_fixation_data['roi'] is not None and last_fixation_data['start_time'] is not None:
                        duration = timestamp_sec - last_fixation_data['start_time']
                        if duration > 0 and duration < 10:  # Filter out unreasonable durations
                            fixation_durations[last_fixation_data['roi']].append(duration)
                            if len(fixation_durations[last_fixation_data['roi']]) % 20 == 0:
                                print(f"DEBUG: Recorded fixation duration of {duration:.2f}s on {last_fixation_data['roi']}")
                    
                    # Start tracking the new ROI
                    last_fixation_data['roi'] = current_roi
                    last_fixation_data['start_time'] = timestamp_sec
                
                # Debug - occasional sample of ROI hits
                if len(roi_sequence) % 100 == 0:
                    print(f"DEBUG: Found ROI hit on '{current_roi}' at time {timestamp_sec:.2f}s (fixation #{len(roi_sequence)})")
                
                # First fixation time
                if first_fixation_times[current_roi] is None:
                    first_fixation_times[current_roi] = timestamp_sec
                    seen_rois.add(current_roi)
                    print(f"DEBUG: First fixation on '{current_roi}' at {timestamp_sec:.2f}s")
                elif current_roi in seen_rois:
                    # This ROI has been seen before, count as revisit
                    roi_revisits[current_roi] += 1
                    
                # Update temporal heatmap
                # Calculate which time bin this fixation belongs to
                time_bin = int((timestamp_sec * 1000) / (time_bin_size * 1000))
                if time_bin < num_bins:
                    # Increment the count for this ROI at this time bin
                    temporal_heatmap[current_roi][time_bin] += 1
                
                # Transition matrix
                if last_roi is not None and last_roi != current_roi:
                    transition_matrix[last_roi][current_roi] += 1
                    
                last_roi = current_roi
            else:
                # No ROI in focus - if we were tracking a fixation on an ROI, save its duration and reset
                timestamp_sec = (row['timestamp'] - start_timestamp) / 1000.0  # Convert to seconds
                if last_fixation_data['roi'] is not None and last_fixation_data['start_time'] is not None:
                    duration = timestamp_sec - last_fixation_data['start_time']
                    if duration > 0 and duration < 10:  # Filter out unreasonable durations
                        fixation_durations[last_fixation_data['roi']].append(duration)
                    last_fixation_data['roi'] = None
                    last_fixation_data['start_time'] = None
                
        # 1. ROI Social vs Non-Social Attention Balance Plot - SKIPPED
        # Variables needed for compatibility with later code
        social_rois = ['Face', 'Hand', 'Eyes', 'Mouth', 'Person', 'Body']
        nonsocial_rois = ['Background', 'Object', 'Bed', 'Couch', 'Torso', 'Floor', 'Wall', 'Toy']
        
        # Move directly to next plot in sequence
        if update_progress_func:
            update_progress_func(2, "Moving to next plot...")
        else:
            status_label.setText("Moving to next plot...")
            QApplication.processEvents()
        
        # Create mapping for all ROIs in the data
        roi_categories = {}
        for roi in roi_labels:
            # Try to categorize based on exact matches first
            if roi in social_rois:
                roi_categories[roi] = 'Social'
            elif roi in nonsocial_rois:
                roi_categories[roi] = 'Non-Social'
            else:
                # If no exact match, try partial matching
                if any(social_term in roi for social_term in ['face', 'hand', 'eye', 'mouth', 'person', 'body']):
                    roi_categories[roi] = 'Social'
                else:
                    roi_categories[roi] = 'Non-Social'
        
        print(f"DEBUG: ROI Categories: {roi_categories}")
        
        # Compute time spent looking at each ROI category
        social_time = 0
        nonsocial_time = 0
        other_time = 0
        
        for roi, duration in roi_durations.items():
            if roi in roi_categories:
                if roi_categories[roi] == 'Social':
                    social_time += duration
                elif roi_categories[roi] == 'Non-Social':
                    nonsocial_time += duration
            else:
                other_time += duration
        
        # Calculate total looking time and percentages
        total_time = social_time + nonsocial_time + other_time
        social_pct = (social_time / total_time * 100) if total_time > 0 else 0
        nonsocial_pct = (nonsocial_time / total_time * 100) if total_time > 0 else 0
        other_pct = (other_time / total_time * 100) if total_time > 0 else 0
        
        print(f"DEBUG: Social time: {social_time:.2f}s ({social_pct:.1f}%)")
        print(f"DEBUG: Non-social time: {nonsocial_time:.2f}s ({nonsocial_pct:.1f}%)")
        
        # Skip creating the social balance plot as requested
        # Define the path variables needed for compatibility with later code
        balance_filename = f"roi_social_balance_{movie.replace(' ', '_')}.png"
        balance_path = os.path.join(plots_dir, balance_filename)
        
        # Create the figure and axes to avoid NameError with ax2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Skip all pie chart visualization code
        # But keep the axes for the next plot
        # Sort ROIs by duration
        sorted_rois = sorted([(roi, time) for roi, time in roi_durations.items()], 
                             key=lambda x: x[1], reverse=True)
        
        # Extract data for plotting
        roi_names = [item[0] for item in sorted_rois]
        roi_times = [item[1] for item in sorted_rois]
        
        # Define colors based on ROI category
        bar_colors = [
            '#ff9999' if roi in roi_categories and roi_categories[roi] == 'Social' else
            '#66b3ff' if roi in roi_categories and roi_categories[roi] == 'Non-Social' else
            '#c2c2f0'
            for roi in roi_names
        ]
        
        # Create the bar chart
        bars = ax2.barh(roi_names, roi_times, color=bar_colors)
        
        # Add social/non-social labels to the bars
        for i, (roi, time) in enumerate(zip(roi_names, roi_times)):
            category = roi_categories.get(roi, 'Other')
            ax2.text(time + 0.1, i, f"{category} ({time:.1f}s)", 
                    va='center', fontsize=8, alpha=0.7)
        
        # Add title and labels
        # Skip creating and saving the social balance plot
        print(f"DEBUG: Skipping social balance plot generation (disabled)")
        
        # Skip adding to visualization results
        print(f"DEBUG: Skipping adding ROI Social Balance plot to visualization options")
            
        
        # 2. ROI Transition Matrix Plot
        if update_progress_func:
            update_progress_func(3, "Generating ROI Transition Matrix plot...")
        else:
            status_label.setText("Generating ROI Transition Matrix plot...")
            QApplication.processEvents()
        
        print(f"DEBUG: Creating ROI Transition Matrix with {sum(sum(v.values()) for v in transition_matrix.values())} transitions")
        print(f"DEBUG: Transition matrix has {len(transition_matrix)} source ROIs")
        
        if transition_matrix:
            # Create a dense representation for the heatmap
            transition_array = np.zeros((len(roi_labels), len(roi_labels)))
            
            # Fill the transition array
            for i, from_roi in enumerate(roi_labels):
                for j, to_roi in enumerate(roi_labels):
                    transition_array[i, j] = transition_matrix[from_roi][to_roi]
                    
            # Create the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use log scale if values range is large
            try:
                if np.max(transition_array) > 0:
                    if np.max(transition_array) / (np.min(transition_array[transition_array > 0]) or 1) > 10:
                        # Use log normalization for widely varying values
                        from matplotlib.colors import LogNorm
                        norm = LogNorm(vmin=max(1, np.min(transition_array[transition_array > 0])), 
                                     vmax=max(2, np.max(transition_array)))
                        sns.heatmap(transition_array, cmap="YlOrRd", ax=ax, 
                                  xticklabels=roi_labels, yticklabels=roi_labels,
                                  norm=norm, annot=True, fmt=".0f", linewidths=0.5)
                    else:
                        # Use regular normalization for more uniform values
                        sns.heatmap(transition_array, cmap="YlOrRd", ax=ax, 
                                  xticklabels=roi_labels, yticklabels=roi_labels,
                                  annot=True, fmt=".0f", linewidths=0.5)
                else:
                    # Fallback for empty matrix
                    sns.heatmap(transition_array, cmap="YlOrRd", ax=ax, 
                              xticklabels=roi_labels, yticklabels=roi_labels,
                              annot=True, fmt=".0f", linewidths=0.5)
            except Exception as e:
                print(f"ERROR in transition matrix heatmap: {e}")
                # Use a simpler approach if seaborn fails
                ax.imshow(transition_array, cmap="YlOrRd")
                ax.set_xticks(np.arange(len(roi_labels)))
                ax.set_yticks(np.arange(len(roi_labels)))
                ax.set_xticklabels(roi_labels)
                ax.set_yticklabels(roi_labels)
            
            # Add title and labels
            ax.set_title(f'ROI Transition Matrix for {movie}')
            ax.set_xlabel('To ROI')
            ax.set_ylabel('From ROI')
            
            # Adjust labels for readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Tight layout
            plt.tight_layout()
            
            # Save the plot
            matrix_filename = f"roi_transition_matrix_{movie.replace(' ', '_')}.png"
            matrix_path = os.path.join(plots_dir, matrix_filename)
            plt.savefig(matrix_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Add to visualization results
            self.visualization_results[movie]['social'].append(matrix_path)
            self.movie_visualizations[movie]["ROI Transition Matrix"] = matrix_path
            
        
        # 3. ROI First Fixation Latency Plot
        if update_progress_func:
            update_progress_func(4, "Generating ROI First Fixation Latency plot...")
        else:
            status_label.setText("Generating ROI First Fixation Latency plot...")
            QApplication.processEvents()
        
        if first_fixation_times:
            # Filter out ROIs that were never fixated
            valid_first_times = {roi: time for roi, time in first_fixation_times.items() if time is not None}
            
            if valid_first_times:
                # Sort by first fixation time
                sorted_latencies = sorted(valid_first_times.items(), key=lambda x: x[1])
                latency_rois = [item[0] for item in sorted_latencies]
                latency_times = [item[1] for item in sorted_latencies]
                
                # Create the bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot horizontal bars for better readability with many ROIs
                bars = ax.barh(latency_rois, latency_times, color='skyblue')
                
                # Add value labels on the bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{width:.2f}s', va='center')
                
                # Add title and labels
                ax.set_title(f'Time to First Fixation on Each ROI in {movie}')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('ROI')
                
                # Add gridlines
                ax.grid(True, linestyle='--', alpha=0.6, axis='x')
                
                # Tight layout
                plt.tight_layout()
                
                # Save the plot
                latency_filename = f"roi_first_fixation_latency_{movie.replace(' ', '_')}.png"
                latency_path = os.path.join(plots_dir, latency_filename)
                plt.savefig(latency_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # Add to visualization results
                self.visualization_results[movie]['social'].append(latency_path)
                self.movie_visualizations[movie]["ROI First Fixation Latency"] = latency_path
                
        
        # ROI Dwell Time Comparison - Removed as it's now merged with ROI Attention Time plot
        # Skip this step in the overall progress
        # update_overall_progress(5, "Generating ROI Dwell Time Comparison plot...")
        
        # We don't need to modify total_plots here, that's handled in the calling function
        # This line was causing an UnboundLocalError
        # total_plots -= 1
                
        
        # ROI Revisitation Plot (Removed as per user request)
        # We still track roi_revisits data for potential future use, but don't generate the plot
        
        # Collect some statistics for debugging only
        if roi_revisits:
            # Filter to ROIs that were fixated at least once
            valid_revisits = {roi: count for roi, count in roi_revisits.items() 
                             if first_fixation_times.get(roi) is not None}
            
            # Log some statistics but don't create the plot
            if valid_revisits:
                print(f"DEBUG: Revisit counts collected but plot generation skipped")
                for roi, count in sorted(valid_revisits.items(), key=lambda x: x[1], reverse=True):
                    print(f"DEBUG:   - {roi}: {count} revisits")
                
        # 6. ROI Fixation Duration Distribution plot (NEW)
        if update_progress_func:
            update_progress_func(6, "Generating ROI Fixation Duration Distribution plot...")
        else:
            status_label.setText("Generating ROI Fixation Duration Distribution plot...")
            QApplication.processEvents()
        
        print(f"DEBUG: Creating ROI Fixation Duration Distribution with data for {sum(1 for durations in fixation_durations.values() if durations)} ROIs")
        
        # Filter out ROIs with no duration data
        valid_durations = {roi: durs for roi, durs in fixation_durations.items() if durs}
        
        if valid_durations:
            # Print statistics about each ROI's durations
            for roi, durations in valid_durations.items():
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    print(f"DEBUG: {roi} has {len(durations)} fixations with average duration {avg_duration:.2f}s")
            
            # Create a violin plot or boxplot for each ROI's fixation durations
            try:
                # Create the figure
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Prepare data for the violin plot
                data_to_plot = []
                labels = []
                
                # Sort ROIs by median duration for better visualization
                sorted_rois = sorted(valid_durations.items(), 
                                    key=lambda x: np.median(x[1]) if x[1] else 0, 
                                    reverse=True)
                
                for roi, durations in sorted_rois:
                    if durations:  # Skip empty durations
                        data_to_plot.append(durations)
                        labels.append(roi)
                
                if data_to_plot:
                    # Select plot type based on number of data points
                    if all(len(d) >= 5 for d in data_to_plot):
                        # Use violin plot for sufficient data
                        sns.violinplot(data=data_to_plot, ax=ax, inner="box", 
                                      palette="pastel", cut=0)
                        
                        # Add individual points for more detail
                        sns.stripplot(data=data_to_plot, ax=ax, size=3, color=".3", 
                                     alpha=0.4, jitter=True)
                    else:
                        # Use boxplot for less data
                        sns.boxplot(data=data_to_plot, ax=ax, 
                                   palette="pastel", whis=1.5)
                        
                        # Add individual points 
                        sns.stripplot(data=data_to_plot, ax=ax, size=4, color=".3", 
                                     alpha=0.6, jitter=True)
                    
                    # Set x-tick labels
                    ax.set_xticklabels(labels)
                    
                    # Add title and labels
                    ax.set_title(f'Distribution of Fixation Durations by ROI in {movie}')
                    ax.set_xlabel('ROI')
                    ax.set_ylabel('Fixation Duration (seconds)')
                    
                    # Rotate labels for better readability
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Add a legend for quartiles
                    from matplotlib.patches import Patch
                    from matplotlib.lines import Line2D
                    
                    # Create custom legend elements
                    legend_elements = [
                        Line2D([0], [0], color='k', lw=2, label='Median'),
                        Patch(facecolor='b', alpha=0.3, label='Durations Distribution')
                    ]
                    
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    # Add gridlines
                    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
                    
                    # Tight layout
                    plt.tight_layout()
                    
                    # Save the plot
                    duration_filename = f"roi_fixation_duration_distribution_{movie.replace(' ', '_')}.png"
                    duration_path = os.path.join(plots_dir, duration_filename)
                    
                    # Make sure the directory exists
                    os.makedirs(os.path.dirname(duration_path), exist_ok=True)
                    print(f"DEBUG: Saving ROI Fixation Duration Distribution plot to: {duration_path}")
                    
                    plt.savefig(duration_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Verify file was created
                    if os.path.exists(duration_path):
                        print(f"DEBUG: Successfully saved duration distribution plot to {duration_path}")
                    else:
                        print(f"ERROR: Failed to save duration distribution plot to {duration_path}")
                    
                    # Add to visualization results
                    self.visualization_results[movie]['social'].append(duration_path)
                    print(f"DEBUG: Added ROI Fixation Duration Distribution plot to visualization_results")
                    
                    # Make sure the movie is in the dictionary
                    if movie not in self.movie_visualizations:
                        self.movie_visualizations[movie] = {}
                    
                    # Add to movie_visualizations with display name
                    self.movie_visualizations[movie]["ROI Fixation Duration Distribution"] = duration_path
                    print(f"DEBUG: Added ROI Fixation Duration Distribution plot to movie_visualizations")
            except Exception as e:
                print(f"ERROR creating ROI Fixation Duration Distribution plot: {e}")
                import traceback
                traceback.print_exc()
                
        # 7. ROI Temporal Heatmap (NEW)
        if update_progress_func:
            update_progress_func(7, "Generating ROI Temporal Heatmap...")
            progress_bar.setValue(0)  # Reset for new task
        else:
            status_label.setText("Generating ROI Temporal Heatmap...")
            progress_bar.setValue(0)  # Reset for new task
            QApplication.processEvents()
        
        print(f"DEBUG: Creating ROI Temporal Heatmap with {num_bins} time bins")
        
        # Check if we have data for the temporal heatmap
        if temporal_heatmap:
            try:
                # Filter out empty ROIs (those with no fixations)
                active_rois = [roi for roi in roi_labels 
                              if np.sum(temporal_heatmap[roi]) > 0]
                
                if active_rois:
                    # Create the figure
                    fig, ax = plt.subplots(figsize=(15, 8))
                    
                    # Prepare data for the heatmap
                    heatmap_data = np.array([temporal_heatmap[roi] for roi in active_rois])
                    
                    # Apply smoothing to the heatmap data for better visualization
                    from scipy.ndimage import gaussian_filter1d
                    smoothed_data = np.copy(heatmap_data)
                    for i in range(len(smoothed_data)):
                        # Apply moderate smoothing
                        smoothed_data[i] = gaussian_filter1d(smoothed_data[i], sigma=2)
                    
                    # Create the heatmap
                    im = ax.imshow(smoothed_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Fixation Intensity')
                    
                    # Set y-tick labels (ROI names)
                    ax.set_yticks(np.arange(len(active_rois)))
                    ax.set_yticklabels(active_rois)
                    
                    # Set x-tick labels (time in seconds)
                    # Place ticks every 1 second
                    seconds_per_tick = 1.0  # 1 second between ticks
                    ticks_per_bin = int(seconds_per_tick / time_bin_size)
                    tick_positions = np.arange(0, num_bins, ticks_per_bin)
                    tick_labels = [f"{t * time_bin_size:.1f}" for t in tick_positions]
                    
                    # Only show a subset of ticks if there are too many
                    if len(tick_positions) > 20:
                        tick_positions = tick_positions[::len(tick_positions)//20]
                        tick_labels = [f"{t * time_bin_size:.1f}" for t in tick_positions]
                    
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels)
                    
                    # Add title and labels
                    ax.set_title(f'ROI Attention Over Time in {movie}')
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('ROI')
                    
                    # Add gridlines
                    ax.grid(False)
                    
                    # Add time markers for significant events
                    # Calculate mean fixation time per ROI for annotation
                    for i, roi in enumerate(active_rois):
                        if first_fixation_times[roi] is not None:
                            # Mark the first fixation time
                            first_time_bin = int((first_fixation_times[roi] * 1000) / (time_bin_size * 1000))
                            if first_time_bin < num_bins:
                                ax.plot([first_time_bin, first_time_bin], [i-0.4, i+0.4], 'g-', linewidth=1)
                                ax.text(first_time_bin, i+0.5, 'First', color='green', 
                                       ha='center', va='bottom', fontsize=8)
                    
                    # Tight layout
                    plt.tight_layout()
                    
                    # Save the plot
                    temporal_filename = f"roi_temporal_heatmap_{movie.replace(' ', '_')}.png"
                    temporal_path = os.path.join(plots_dir, temporal_filename)
                    
                    # Make sure the directory exists
                    os.makedirs(os.path.dirname(temporal_path), exist_ok=True)
                    print(f"DEBUG: Saving ROI Temporal Heatmap to: {temporal_path}")
                    
                    plt.savefig(temporal_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Verify file was created
                    if os.path.exists(temporal_path):
                        print(f"DEBUG: Successfully saved temporal heatmap to {temporal_path}")
                    else:
                        print(f"ERROR: Failed to save temporal heatmap to {temporal_path}")
                    
                    # Add to visualization results
                    self.visualization_results[movie]['social'].append(temporal_path)
                    print(f"DEBUG: Added ROI Temporal Heatmap to visualization_results")
                    
                    # Make sure the movie is in the dictionary
                    if movie not in self.movie_visualizations:
                        self.movie_visualizations[movie] = {}
                    
                    # Add to movie_visualizations with display name
                    self.movie_visualizations[movie]["ROI Temporal Heatmap"] = temporal_path
                    print(f"DEBUG: Added ROI Temporal Heatmap to movie_visualizations")
            except Exception as e:
                print(f"ERROR creating ROI Temporal Heatmap: {e}")
                import traceback
                traceback.print_exc()

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
            
        # Create a progress dialog with a progress bar
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Generating Social Attention Plots")
        progress_dialog.setFixedSize(400, 150)
        
        dialog_layout = QVBoxLayout(progress_dialog)
        
        # Overall progress status
        overall_status_label = QLabel(f"Generating plots for {movie}...")
        overall_status_label.setStyleSheet("font-weight: bold;")
        dialog_layout.addWidget(overall_status_label)
        
        # Overall progress bar
        overall_progress_bar = QProgressBar()
        overall_progress_bar.setRange(0, 100)
        overall_progress_bar.setValue(0)
        overall_progress_bar.setStyleSheet("QProgressBar {height: 15px;}")
        dialog_layout.addWidget(overall_progress_bar)
        
        # Current task status
        status_label = QLabel("Preparing data...")
        dialog_layout.addWidget(status_label)
        
        # Current task progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        dialog_layout.addWidget(progress_bar)
        
        # Define the total number of plot generation steps
        # 1. ROI Attention Time (basic)
        # 2. ROI Social vs Non-Social Attention Balance (disabled)
        # 3. ROI Transition Matrix
        # 4. ROI First Fixation Latency
        # 5. ROI Dwell Time Comparison (merged with Attention Time)
        # 6. ROI Fixation Duration Distribution
        # 7. ROI Temporal Heatmap
        total_plots = 6  # Excluding the removed ROI Revisitation plot
        current_plot = 0
        
        # Helper function to update overall progress
        def update_overall_progress(plot_num, status_text):
            nonlocal current_plot
            current_plot = plot_num
            overall_progress = int((plot_num / total_plots) * 100)
            overall_progress_bar.setValue(overall_progress)
            overall_status_label.setText(f"Generating plots for {movie}... ({plot_num}/{total_plots})")
            status_label.setText(status_text)
            progress_bar.setValue(0)  # Reset progress for new task
            QApplication.processEvents()
            
            # Add a global progress indicator variable to be used in each plot
            global current_plot_progress
            current_plot_progress = f"{plot_num}/{total_plots}"
        
        # Show the dialog (non-modal)
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
            status_label.setText("Processing ROI data...")
            progress_bar.setValue(10)
            QApplication.processEvents()
            
            # Count time spent on each ROI
            roi_durations = {}
            
            # First, convert frame keys to integers if they're strings - OPTIMIZATION: Do this once and cache
            frame_keys = {}
            for key in roi_data.keys():
                try:
                    frame_keys[int(key)] = roi_data[key]
                    # Only print a sample for debugging
                    if len(frame_keys) <= 3:
                        roi_sample = roi_data[key]
                        if roi_sample:
                            roi_labels = [roi['label'] for roi in roi_sample if 'label' in roi]
                            print(f"DEBUG: Frame {key} has {len(roi_sample)} ROIs: {roi_labels}")
                except ValueError:
                    print(f"DEBUG: Skipping non-integer key: {key}")
                    continue
            
            if not frame_keys:
                print(f"DEBUG: No valid frame keys found in ROI data")
                raise ValueError("No valid frame keys found in ROI data")
            
            status_label.setText("Analyzing fixations...")
            progress_bar.setValue(20)
            QApplication.processEvents()
                
            # OPTIMIZATION: Pre-process ROI data for faster lookup
            # Create a frame range map to quickly find the nearest frame
            frame_numbers = sorted(frame_keys.keys())
            frame_range_map = {}
            
            if frame_numbers:
                # Print some statistics about the frame distribution
                print(f"DEBUG: Frame key range: {min(frame_numbers)} to {max(frame_numbers)}")
                if len(frame_numbers) > 1:
                    # Calculate average interval between frames
                    intervals = [frame_numbers[i+1] - frame_numbers[i] for i in range(len(frame_numbers)-1)]
                    if intervals:
                        avg_interval = sum(intervals) / len(intervals)
                        print(f"DEBUG: Average interval between frames: {avg_interval:.2f}")
                
                # Create a map of frame ranges for faster nearest frame lookups
                for i, frame in enumerate(frame_numbers):
                    if i == 0:
                        # For the first frame, use it for anything less than the midpoint to the next frame
                        next_frame = frame_numbers[i+1] if i+1 < len(frame_numbers) else frame + 1000
                        frame_range_map[(0, (frame + next_frame)//2)] = frame
                    elif i == len(frame_numbers) - 1:
                        # For the last frame, use it for anything greater than the midpoint from the previous frame
                        prev_frame = frame_numbers[i-1]
                        frame_range_map[((prev_frame + frame)//2, float('inf'))] = frame
                    else:
                        # For middle frames, use the midpoints between adjacent frames
                        prev_frame = frame_numbers[i-1]
                        next_frame = frame_numbers[i+1]
                        frame_range_map[((prev_frame + frame)//2, (frame + next_frame)//2)] = frame
            
            # Get fixation data - OPTIMIZATION: Filter for valid frames
            fixation_data = data[data['is_fixation_left'] | data['is_fixation_right']]
            fixation_data = fixation_data.dropna(subset=['frame_number', 'x_left', 'y_left'])
            fixation_count = len(fixation_data)
            print(f"DEBUG: Found {fixation_count} valid fixation data points")
            
            # Process fixations in batches for better progress reporting
            processed_count = 0
            hit_count = 0
            batch_size = max(1, fixation_count // 50)  # ~50 progress updates
            
            # OPTIMIZATION: Cache polygon checks to avoid redundant calculations
            polygon_check_cache = {}
            
            # Process each fixation
            for idx, row in fixation_data.iterrows():
                frame_num = int(row['frame_number'])
                processed_count += 1
                
                # Update progress every batch
                if processed_count % batch_size == 0:
                    progress = 20 + int(75 * processed_count / fixation_count)
                    progress_bar.setValue(progress)
                    status_label.setText(f"Processing fixations: {processed_count}/{fixation_count}")
                    QApplication.processEvents()
                
                # Find the nearest frame in ROI data - OPTIMIZATION: Use frame range map for faster lookup
                nearest_frame = None
                
                # Try the frame range map first
                for (start, end), frame in frame_range_map.items():
                    if start <= frame_num < end:
                        nearest_frame = frame
                        break
                        
                # If no match in the range map, fall back to the slower nearest neighbor approach
                if nearest_frame is None:
                    try:
                        nearest_frame = min(frame_keys.keys(), key=lambda x: abs(x - frame_num))
                    except Exception as e:
                        print(f"DEBUG: Error finding nearest frame: {e}")
                        continue
                
                frame_distance = abs(nearest_frame - frame_num)
                
                # Skip if the frame distance is too large
                if frame_distance > 1000:  # Use a threshold based on your data
                    continue
                
                # Get the ROIs for this frame
                rois_in_frame = frame_keys[nearest_frame]
                
                # Get normalized coordinates
                if row['x_left'] > 1.0 or row['y_left'] > 1.0:
                    x_norm = row['x_left'] / self.screen_width
                    y_norm = row['y_left'] / self.screen_height
                else:
                    x_norm = row['x_left']
                    y_norm = row['y_left']
                
                # Check each ROI in this frame
                for roi in rois_in_frame:
                    if 'label' not in roi or 'coordinates' not in roi:
                        continue
                    
                    label = roi['label']
                    coords = roi['coordinates']
                    
                    # OPTIMIZATION: Use cached results for polygon checks
                    cache_key = (tuple((coord['x'], coord['y']) for coord in coords), x_norm, y_norm)
                    if cache_key in polygon_check_cache:
                        is_inside = polygon_check_cache[cache_key]
                    else:
                        # Check if point is inside polygon
                        is_inside = self._point_in_polygon(x_norm, y_norm, coords)
                        polygon_check_cache[cache_key] = is_inside
                    
                    if is_inside:
                        # Add time spent to this ROI
                        if label not in roi_durations:
                            roi_durations[label] = 0
                        roi_durations[label] += 1  # Each fixation counts as one time unit
                        hit_count += 1
                        break  # Only count one ROI per fixation
            
            print(f"DEBUG: Processed {processed_count} fixations, found {hit_count} ROI hits")
            print(f"DEBUG: ROI durations: {roi_durations}")
            
            # Update progress
            update_overall_progress(1, "Creating ROI Attention Time plot...")
            progress_bar.setValue(95)
            
            # Create the plot - taller to accommodate the explanation text
            fig, ax = plt.subplots(figsize=(10, 7))
            
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
                
                # Calculate total fixation time for percentage
                total_duration = sum(durations)
                percentages = [(count / total_duration) * 100 for count in durations]
                
                # Plot bar chart
                bars = ax.bar(labels, durations, color='skyblue')
                
                # Add value labels with both count and percentage on top of bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    percentage = percentages[i]
                    # Position the raw count on top of the bar
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=10)
                    # Add percentage text in the middle of the bar
                    ax.text(bar.get_x() + bar.get_width()/2., height/2,
                            f'{percentage:.1f}%',
                            ha='center', va='center', fontsize=10, 
                            color='black', fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, pad=2, boxstyle='round'))
                
                # Add title and labels with explanation
                ax.set_title(f'Time Spent on Each ROI in {movie}')
                ax.set_xlabel('Region of Interest (ROI)')
                ax.set_ylabel('Fixation Count')
                
                # Remove progress indicator and footnote as requested
                # This keeps the plot cleaner without pagination indicators or explanatory notes
                
                # Rotate x-axis labels if many ROIs
                if len(labels) > 5:
                    plt.xticks(rotation=45, ha='right')
                
            # Tight layout with padding for the explanation text if needed
            if not roi_durations:
                plt.tight_layout()
            else:
                plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Add extra space at bottom and top for footnote and progress indicators
            
            # Save the plot to the same directory as other plots
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
            status_label.setText("Saving plot...")
            progress_bar.setValue(98)
            QApplication.processEvents()
            
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
            
            # Make sure the movie is in visualization_results and has a social category
            if movie not in self.visualization_results:
                self.visualization_results[movie] = {}
            if 'social' not in self.visualization_results[movie]:
                self.visualization_results[movie]['social'] = []
                
            # Print current visualization_results structure before generating advanced plots
            print(f"DEBUG: Before advanced plots, visualization_results structure:")
            for m in self.visualization_results:
                print(f"DEBUG:   Movie: {m}")
                for category, paths in self.visualization_results[m].items():
                    print(f"DEBUG:     Category: {category}, Plots: {len(paths)}")
                    for path in paths:
                        print(f"DEBUG:       - {os.path.basename(path)}")
                        
            # Generate advanced ROI plots
            status_label.setText("Generating advanced ROI plots...")
            progress_bar.setValue(99)
            QApplication.processEvents()
            try:
                # Pass the update_overall_progress function to the advanced ROI plots method
                self.create_advanced_roi_plots(movie, roi_durations, fixation_data, plots_dir, 
                                             frame_keys, frame_range_map, polygon_check_cache, 
                                             status_label, progress_bar, update_overall_progress)
                print(f"DEBUG: Successfully generated advanced ROI plots")
                # Print how many plots we now have for this movie
                if movie in self.visualization_results and 'social' in self.visualization_results[movie]:
                    print(f"DEBUG: Total social plots for {movie}: {len(self.visualization_results[movie]['social'])}")
                    for i, plot_path in enumerate(self.visualization_results[movie]['social']):
                        print(f"DEBUG: Plot {i+1}: {os.path.basename(plot_path)}")
                        
                # Debug check if plots actually exist on disk
                plot_found = False
                for pattern in ["roi_fixation_sequence_*.png", "roi_transition_matrix_*.png"]:
                    try:
                        import glob
                        matching_files = glob.glob(os.path.join(plots_dir, pattern))
                        if matching_files:
                            print(f"DEBUG: Found {len(matching_files)} files matching {pattern} in {plots_dir}")
                            for f in matching_files:
                                print(f"DEBUG:   - {os.path.basename(f)}")
                                plot_found = True
                        else:
                            print(f"DEBUG: No files matching {pattern} in {plots_dir}")
                    except Exception as e:
                        print(f"DEBUG: Error searching for {pattern}: {e}")
                        
                if not plot_found:
                    print(f"WARNING: No advanced plot files found on disk after generation!")
            except Exception as e:
                print(f"Error generating advanced ROI plots: {e}")
                import traceback
                traceback.print_exc()
                
            # Update the visualization dropdown
            update_overall_progress(total_plots, "Completing plot generation... Updating UI")
            progress_bar.setValue(99)
            
            # Store the current selection to restore after refresh
            current_viz = self.viz_type_combo.currentText() if self.viz_type_combo.currentIndex() >= 0 else None
            
            # Refresh the visualization dropdown to include all plots (basic and advanced)
            self.movie_selected(self.movie_combo.currentIndex())
            
            # Print available visualizations
            print(f"DEBUG: Available visualizations after refresh:")
            for vizname in [self.viz_type_combo.itemText(i) for i in range(self.viz_type_combo.count())]:
                print(f"DEBUG:   - {vizname}")
                
            # First try to show the initial plot we just created
            plot_shown = False
            for i in range(self.viz_type_combo.count()):
                viz_name = self.viz_type_combo.itemText(i)
                if viz_name == display_name:
                    self.viz_type_combo.setCurrentIndex(i)
                    plot_shown = True
                    break
                    
            # If the dropdown has been updated, also show one of the advanced plots
            if not plot_shown and self.viz_type_combo.count() > 0:
                # Show the first item in the dropdown
                self.viz_type_combo.setCurrentIndex(0)
            
            # Close progress dialog
            progress_dialog.close()
            
            # Regenerate HTML report with the new plot if a report exists
            if hasattr(self, 'report_path') and self.report_path and os.path.exists(os.path.dirname(self.report_path)):
                try:
                    status_label.setText("Updating HTML report...")
                    progress_bar.setValue(100)
                    QApplication.processEvents()
                    
                    # Get visualizer instance
                    from eyelink_visualizer import MovieEyeTrackingVisualizer
                    
                    # Determine the base directory for the visualizer - use the parent of the plots directory
                    # or alternatively the directory containing the movie data file
                    if movie_data and "data_path" in movie_data:
                        base_dir = os.path.dirname(os.path.dirname(movie_data["data_path"]))
                    else:
                        # Fallback to current output directory
                        base_dir = self.output_dir
                        
                    print(f"DEBUG: Using base directory for report regeneration: {base_dir}")
                    visualizer = MovieEyeTrackingVisualizer(base_dir=base_dir, screen_size=(self.screen_width, self.screen_height))
                    
                    # Generate a new report
                    report_dir = os.path.dirname(self.report_path)
                    visualizer.generate_report(self.visualization_results, report_dir)
                    print(f"Regenerated HTML report to include new social attention plots")
                    
                    # Print out total number of plots included in the report
                    total_plots = 0
                    for m, categories in self.visualization_results.items():
                        for category, plots in categories.items():
                            total_plots += len(plots)
                    print(f"DEBUG: Report contains {total_plots} total plots")
                    
                    # Show success message with report information
                    QMessageBox.information(
                        self,
                        "Plot Generated",
                        f"Social attention plot for {movie} has been generated and added to the visualization dropdown.\n\n"
                        f"The HTML report has been updated to include the new plot."
                    )
                except Exception as e:
                    # Show success message without report information
                    QMessageBox.information(
                        self,
                        "Plot Generated",
                        f"Social attention plot for {movie} has been generated and added to the visualization dropdown.\n\n"
                        f"Note: Could not update HTML report: {str(e)}"
                    )
            else:
                # Show regular success message
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
        """Check if a point is inside a polygon defined by coordinates using an optimized ray casting algorithm"""
        # Extract points from coordinates
        points = [(coord['x'], coord['y']) for coord in coordinates]
        
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
