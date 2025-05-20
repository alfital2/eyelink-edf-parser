"""
Main GUI Module for Eye Movement Analysis
"""

# Standard library imports
import sys
import os
import datetime
import webbrowser

# Add parent directory to path, so we can import modules from the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Configure matplotlib for thread safety BEFORE importing any matplotlib-related modules
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues

# Third-party imports
import pandas as pd

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    # Layout widgets
    QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QStackedWidget, QScrollArea,
    # Interactive widgets
    QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox, QProgressBar, QMessageBox, 
    # Container widgets
    QTabWidget, QGroupBox,
    # Data display widgets
    QTableWidget, QTableWidgetItem, QHeaderView, QTextBrowser, QToolTip,
    # Size policy
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QCursor, QPalette

# Local application imports
from GUI.data.parser import (
    process_asc_file, process_multiple_files, 
    load_csv_file, load_multiple_csv_files
)
from GUI.visualization.eyelink_visualizer import MovieEyeTrackingVisualizer
from GUI.utils.documentation import (
    get_feature_explanations, get_visualization_explanations,
    get_formatted_feature_documentation, get_formatted_visualization_documentation
)
from animated_roi_scanpath import AnimatedROIScanpathWidget
from GUI.theme_manager import ThemeManager
from GUI.feature_table_manager import FeatureTableManager
from GUI.visualization.plot_generator import PlotGenerator
from GUI.data.processing_thread import ProcessingThread
from GUI.utils.settings_manager import SettingsManager
from GUI.utils.visualization_helper import VisualizationHelper

# Global variable for plot progress tracking
current_plot_progress = "0/0"  # Will be updated during plot generation


class EyeMovementAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Movement Analysis for Autism Classification (ASC/CSV)")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # File paths and settings
        self.file_paths = []
        self.output_dir = None
        self.visualization_results = {}
        self.movie_visualizations = {}
        self.features_data = None
        self.selected_file_type = "ASC Files"  # Default file type
        
        # Load screen dimensions from settings
        self.screen_width, self.screen_height = self.settings_manager.get_screen_dimensions()
        print(f"Loaded screen dimensions: {self.screen_width}x{self.screen_height}")
        
        # Get feature and visualization explanations
        self.feature_explanations = get_feature_explanations()
        self.visualization_explanations = get_visualization_explanations()

        # Initialize managers
        self.theme_manager = ThemeManager(self)
        self.plot_generator = PlotGenerator(
            self.screen_width, 
            self.screen_height, 
            self.visualization_results, 
            self.movie_visualizations
        )
        self.feature_table_manager = FeatureTableManager(
            self, 
            self.theme_manager, 
            self.feature_explanations
        )
        
        # Initialize feature_tables attribute
        self.feature_tables = self.feature_table_manager.feature_tables
        
        # Initialize ROI attributes
        self.roi_file_path = None
        self.roi_data = None

        # Initialize UI
        self.init_ui()

    def refresh_theme(self):
        """Refresh UI with current theme"""
        # Update the stylesheet
        self.centralWidget().setStyleSheet(self.theme_manager.get_theme_style())

        # Refresh feature tables
        for category_name, table_info in self.feature_table_manager.feature_tables.items():
            table = table_info["table"]
            if not self.theme_manager.is_dark_mode:
                table.setStyleSheet("QTableWidget { background-color: white; border: 1px solid #ddd; }")
            else:
                table.setStyleSheet("")  # Default dark mode styling from main style

        # Update feature header color based on theme
        if hasattr(self, 'features_header'):
            if self.theme_manager.is_dark_mode:
                self.features_header.setStyleSheet("color: #58b0ff;")
            else:
                self.features_header.setStyleSheet("color: #0078d7;")

        # Re-display current visualization if any
        if hasattr(self, 'image_label') and self.image_label.pixmap() is not None:
            self.show_visualization()

    def init_ui(self):
        """Initialize the main user interface"""
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Apply theme-specific styles
        central_widget.setStyleSheet(self.theme_manager.get_theme_style())

        # Create tabs for different sections
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Create the four main tabs
        processing_tab = self._create_processing_tab()
        results_tab = self._create_results_tab()
        features_tab = self._create_features_tab()
        documentation_tab = self._create_documentation_tab()

        # Add tabs to the main tab widget
        tabs.addTab(processing_tab, "Data Processing")
        tabs.addTab(results_tab, "Results & Visualization")
        tabs.addTab(features_tab, "Extracted Features")
        tabs.addTab(documentation_tab, "Documentation")

        # Set the central widget
        self.setCentralWidget(central_widget)

    def _create_processing_tab(self):
        """Create the data processing tab"""
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)

        # Add file selection section
        file_section = QWidget()
        file_layout = QHBoxLayout(file_section)
        
        self.file_label = QLabel("No files selected")
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

        # Add output directory section
        output_section = QWidget()
        output_layout = QHBoxLayout(output_section)

        self.output_label = QLabel("No output directory selected")
        select_output_btn = QPushButton("Select Output Directory")
        select_output_btn.clicked.connect(self.select_output_dir)

        output_layout.addWidget(select_output_btn)
        output_layout.addWidget(self.output_label, 1)

        processing_layout.addWidget(output_section)

        # Add options section
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
        
        # Add screen aspect ratio section
        aspect_section = self._create_aspect_ratio_section()
        processing_layout.addWidget(aspect_section)

        # Add process button section
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

        # Add status section
        status_section = QWidget()
        status_layout = QVBoxLayout(status_section)
        
        # Current status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        # Processing log
        self.status_log = QTextBrowser()
        self.status_log.setMaximumHeight(150)
        self.status_log.setPlaceholderText("Processing log will appear here...")
        status_layout.addWidget(self.status_log)
        
        processing_layout.addWidget(status_section)
        processing_layout.addStretch()
        
        return processing_tab
        
    def _create_aspect_ratio_section(self):
        """Create the aspect ratio selection section"""
        aspect_section = QWidget()
        aspect_layout = QHBoxLayout(aspect_section)
        
        aspect_label = QLabel("Screen Aspect Ratio:")
        aspect_layout.addWidget(aspect_label)
        
        self.aspect_ratio_combo = QComboBox()
        # Add common aspect ratios (width x height)
        self.aspect_ratio_combo.addItem("1280 x 1024 (5:4)", (1280, 1024))  # Default
        self.aspect_ratio_combo.addItem("1920 x 1080 (16:9)", (1920, 1080))
        self.aspect_ratio_combo.addItem("1366 x 768 (16:9)", (1366, 768))
        self.aspect_ratio_combo.addItem("1440 x 900 (16:10)", (1440, 900))
        self.aspect_ratio_combo.addItem("1024 x 768 (4:3)", (1024, 768))
        self.aspect_ratio_combo.addItem("3840 x 2160 (4K 16:9)", (3840, 2160))
        self.aspect_ratio_combo.setToolTip("Select the screen resolution used during the eye tracking experiment")
        aspect_layout.addWidget(self.aspect_ratio_combo)
        
        # Set the combo box selection to match the saved screen dimensions
        saved_index = self.settings_manager.get_aspect_ratio_index()
        if saved_index >= 0 and saved_index < self.aspect_ratio_combo.count():
            # Restore by index
            self.aspect_ratio_combo.setCurrentIndex(saved_index)
        else:
            # Find closest match by dimensions
            index = self.aspect_ratio_combo.findData((self.screen_width, self.screen_height))
            if index >= 0:
                self.aspect_ratio_combo.setCurrentIndex(index)
                
        # Connect signal after setting the initial value
        self.aspect_ratio_combo.currentIndexChanged.connect(self.update_screen_dimensions)
        
        return aspect_section

    def _create_results_tab(self):
        """Create the results & visualization tab"""
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # Visualization controls and display
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        # Add movie selection section
        movie_section = QWidget()
        movie_layout = QHBoxLayout(movie_section)

        # Movie selector
        movie_label = QLabel("Select Movie:")
        movie_label.setFixedWidth(100)
        movie_layout.addWidget(movie_label)
        
        self.movie_combo = QComboBox()
        self.movie_combo.setEnabled(False)
        self.movie_combo.setMinimumWidth(200)
        self.movie_combo.currentIndexChanged.connect(self.movie_selected)
        movie_layout.addWidget(self.movie_combo)
        
        movie_layout.addSpacing(20)

        # Visualization type selector
        viz_type_label = QLabel("Visualization Type:")
        viz_type_label.setFixedWidth(120)
        movie_layout.addWidget(viz_type_label)
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.setEnabled(False)
        self.viz_type_combo.setMinimumWidth(200)
        self.viz_type_combo.currentIndexChanged.connect(self.visualization_type_selected)
        movie_layout.addWidget(self.viz_type_combo)
        
        movie_layout.addStretch(1)
        viz_layout.addWidget(movie_section)
        
        # Add ROI controls section
        roi_section = QWidget()
        roi_layout = QHBoxLayout(roi_section)
        
        self.load_roi_btn = QPushButton("Load ROI")
        self.load_roi_btn.clicked.connect(self.select_roi_file)
        roi_layout.addWidget(self.load_roi_btn)
        
        self.roi_label = QLabel("No ROI file selected")
        roi_layout.addWidget(self.roi_label, 1)
        
        self.generate_social_btn = QPushButton("Generate Social Attention Plots")
        self.generate_social_btn.setEnabled(False)
        self.generate_social_btn.clicked.connect(self.generate_social_attention_plots)
        roi_layout.addWidget(self.generate_social_btn)
        
        viz_layout.addWidget(roi_section)

        # Create stacked widget for visualizations
        self.viz_stack = QStackedWidget()
        self.viz_stack.setMinimumSize(800, 500)
        self.viz_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add static image widget
        self.image_label = QLabel("Visualization will be shown here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.viz_stack.addWidget(self.image_label)
        
        # Add animated scanpath widget
        self.animated_scanpath = AnimatedROIScanpathWidget()
        self.viz_stack.addWidget(self.animated_scanpath)
        
        viz_layout.addWidget(self.viz_stack)

        # Add report button
        self.report_btn = QPushButton("Open HTML Report")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self.open_report)
        viz_layout.addWidget(self.report_btn)

        results_layout.addWidget(viz_widget)
        
        return results_tab

    def _create_features_tab(self):
        """Create the features display tab"""
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)

        # Features header section
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(10, 10, 10, 0)
        
        # Features title
        self.features_header = QLabel("Eye Movement Features for Autism Research")
        self.features_header.setFont(QFont("Arial", 14, QFont.Bold))
        if self.theme_manager.is_dark_mode:
            self.features_header.setStyleSheet("color: #58b0ff;")
        else:
            self.features_header.setStyleSheet("color: #0078d7;")
        header_layout.addWidget(self.features_header)
        
        header_layout.addStretch()
        
        # Movie selector for features
        selector_container = QWidget()
        selector_layout = QHBoxLayout(selector_container)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        
        movie_label = QLabel("Select Movie:")
        movie_label.setFixedWidth(100)
        selector_layout.addWidget(movie_label)
        
        self.feature_movie_combo = QComboBox()
        self.feature_movie_combo.setMinimumWidth(200)
        self.feature_movie_combo.setToolTip("Select a movie to view its specific features, or 'All Data' to view aggregate features")
        self.feature_movie_combo.addItem("All Data")
        self.feature_movie_combo.setEnabled(False)
        self.feature_movie_combo.currentIndexChanged.connect(self.feature_movie_selected)
        selector_layout.addWidget(self.feature_movie_combo)
        
        header_layout.addWidget(selector_container)
        features_layout.addWidget(header_container)

        # Features overview
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

        # Create feature tables
        self.feature_table_manager.create_feature_tables(features_layout)

        # Add save features button
        save_features_btn = QPushButton("Export Features to CSV")
        save_features_btn.clicked.connect(self.save_features)
        features_layout.addWidget(save_features_btn)
        
        return features_tab

    def _create_documentation_tab(self):
        """Create the documentation tab"""
        documentation_tab = QWidget()
        documentation_layout = QVBoxLayout(documentation_tab)

        # Create documentation browser with tabs
        doc_tabs = QTabWidget()
        documentation_layout.addWidget(doc_tabs)

        # Features documentation tab
        feature_doc = QWidget()
        feature_doc_layout = QVBoxLayout(feature_doc)

        feature_doc_text = QTextBrowser()
        feature_doc_text.setOpenExternalLinks(True)
        feature_doc_text.setStyleSheet("font-size: 14px;")
        feature_doc_text.setHtml(get_formatted_feature_documentation())

        feature_doc_layout.addWidget(feature_doc_text)

        # Visualization documentation tab
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
        
        return documentation_tab

    def update_feature_tables(self, features_df):
        """Update all feature tables with data from the features DataFrame"""
        if features_df is None or features_df.empty:
            return

        # Store the features data
        self.features_data = features_df
        
        # Update tables using the feature table manager
        self.feature_table_manager.update_feature_tables(features_df)

    def select_files(self):
        """Open file selection dialog for eye tracking data files"""
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
        """Open directory selection dialog for output folder"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(f"Output: {dir_path}")
            self.update_process_button()

    def update_process_button(self):
        """Enable the process button when both files and output directory are selected"""
        self.process_btn.setEnabled(
            bool(self.file_paths) and bool(self.output_dir)
        )
        
    def update_file_type_info(self, index):
        """Update status label with information about the selected file type"""
        if index == 0:  # ASC Files
            self.status_label.setText("ASC Files: Original (converted) EyeLink data files. Processing may take longer "
                                      "but includes all raw data.")
        else:  # CSV Files
            self.status_label.setText("CSV Files: Pre-processed data files. Much faster loading for visualizations "
                                      "and analysis.")
                                      
    def update_screen_dimensions(self, index):
        """Update screen dimensions based on the selected aspect ratio"""
        if index < 0:
            return
            
        # Get the selected dimensions from the combo box data
        dimensions = self.aspect_ratio_combo.currentData()
        if dimensions:
            self.screen_width = dimensions[0]
            self.screen_height = dimensions[1]
            
            # Update the plot generator
            if hasattr(self, 'plot_generator'):
                self.plot_generator.screen_width = self.screen_width
                self.plot_generator.screen_height = self.screen_height
                
            # Log the change
            self.status_log.append(f"Screen dimensions updated to {self.screen_width} x {self.screen_height}")
            
            # Update any active visualizations
            if hasattr(self, 'animated_scanpath'):
                self.animated_scanpath.screen_width = self.screen_width
                self.animated_scanpath.screen_height = self.screen_height
                # Redraw if data is loaded
                if self.animated_scanpath.data is not None:
                    self.animated_scanpath.redraw()
            
            # Save the setting to QSettings
            self.settings_manager.save_screen_dimensions(
                self.screen_width, 
                self.screen_height,
                index
            )
            print(f"Saved screen dimensions ({self.screen_width}x{self.screen_height}) to settings")

    def process_data(self):
        """Start data processing in a separate thread"""
        # Disable UI elements during processing
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        
        # Clear the status log before starting new processing
        self.status_log.clear()

        # Create processing thread with current settings
        self.processing_thread = ProcessingThread(
            self.file_paths,
            self.output_dir,
            self.visualize_cb.isChecked(),
            self.extract_features_cb.isChecked(),
            self.generate_report_cb.isChecked(),
            self.selected_file_type,
            self.screen_width,
            self.screen_height
        )

        # Connect signals
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)

        # Start processing
        self.processing_thread.start()

    def update_progress(self, value):
        """Update progress bar with current progress"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """Update the status message and log with current processing status"""
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
        """Handle completion of data processing"""
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing complete!")

        # Re-enable UI elements
        self.process_btn.setEnabled(True)

        # Show results summary
        if "summary" in results:
            self._show_summary(results["summary"])
                
        # Update the features display if features were extracted
        if "features" in results and not results["features"].empty:
            self._update_feature_display(results)

        # Update visualization controls if visualizations were generated
        if "visualizations" in results and results["visualizations"]:
            self._update_visualization_controls(results)

        # Enable the report button if a report was generated
        if 'report_path' in results and os.path.exists(results['report_path']):
            self.report_path = results['report_path']
            self.report_btn.setEnabled(True)

    def _show_summary(self, summary):
        """Show processing summary in the status log"""
        # Add summary to the status log with highlighting
        self.status_log.setTextColor(Qt.blue if not self.theme_manager.is_dark_mode else Qt.cyan)
        self.status_log.append("\n--- PROCESSING SUMMARY ---")
        self.status_log.append(f"• Processed {summary['samples']} eye tracking samples")
        self.status_log.append(f"• Detected {summary['fixations']} fixations")
        self.status_log.append(f"• Detected {summary['saccades']} saccades")
        self.status_log.append(f"• Detected {summary['blinks']} blinks")
        self.status_log.append(f"• Processed {summary['frames']} video frames")
        self.status_log.append("--- END OF SUMMARY ---")
        self.status_log.setTextColor(Qt.black if not self.theme_manager.is_dark_mode else Qt.white)

    def _update_feature_display(self, results):
        """Update feature displays with processed data"""
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
        self.feature_table_manager.update_feature_tables(results["features"])

    def _update_visualization_controls(self, results):
        """Update visualization controls with processed data"""
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
            
        # Update screen dimensions in any active visualizations
        if hasattr(self, 'animated_scanpath'):
            self.animated_scanpath.screen_width = self.screen_width
            self.animated_scanpath.screen_height = self.screen_height

    def feature_movie_selected(self, index):
        """Handle movie selection in the features tab"""
        if index < 0 or not hasattr(self, 'movie_features'):
            return
            
        # Get the selected movie name
        movie_name = self.feature_movie_combo.currentText()
        
        # Update the features display with the selected movie's features
        if movie_name in self.movie_features:
            # Update feature tables
            self.feature_table_manager.update_feature_tables(self.movie_features[movie_name])
            
            # Update the header
            header_text = "Eye Movement Features for Autism Research" if movie_name == "All Data" else f"Eye Movement Features: {movie_name}"
            self.features_header.setText(header_text)

    def processing_error(self, error_msg):
        """Handle errors during data processing"""
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
        self.status_log.setTextColor(Qt.black if not self.theme_manager.is_dark_mode else Qt.white)
        
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

        # Get all plot files available for this movie
        available_visualizations = self._collect_visualizations(movie)

        # Add available visualizations to combo box
        if available_visualizations:
            # Update movie_visualizations
            self.movie_visualizations[movie] = available_visualizations
            
            # Clear and add static visualizations
            self.viz_type_combo.clear()
            self.viz_type_combo.addItems(sorted(available_visualizations.keys()))
            
            # Add animated scanpath visualization option
            self.viz_type_combo.addItem("Animated Scanpath")
            
            self.viz_type_combo.setEnabled(True)
            
            # Select the first visualization
            self.viz_type_combo.setCurrentIndex(0)
        else:
            self.viz_type_combo.setEnabled(False)
            self.image_label.setText(f"No visualizations found for movie: {movie}")

    def _collect_visualizations(self, movie):
        """Collect and organize visualizations for a movie"""
        available_visualizations = {}
        
        # First add visualizations from movie_visualizations dictionary
        if movie in self.movie_visualizations:
            for display_name, plot_path in self.movie_visualizations[movie].items():
                # Skip disabled plots
                if "Social vs Non-Social Balance" in display_name:
                    continue
                    
                if os.path.exists(plot_path):
                    available_visualizations[display_name] = plot_path

        # Then search through visualization_results
        for category, plot_paths in self.visualization_results[movie].items():
            for plot_path in plot_paths:
                if not os.path.exists(plot_path):
                    continue
                    
                # Extract the base filename
                basename = os.path.basename(plot_path)
                
                # Skip hidden files
                if basename.startswith('.'):
                    continue
                    
                # Skip paths we've already added
                if any(plot_path == existing_path for existing_path in available_visualizations.values()):
                    continue

                # Get display name for this plot
                display_name = VisualizationHelper.get_display_name_from_path(plot_path)
                if display_name:
                    available_visualizations[display_name] = plot_path

        return available_visualizations

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
            self._show_animated_scanpath(movie)
            return

        # For all other visualization types, switch to the image label widget
        self.viz_stack.setCurrentIndex(0)
        self.image_label.clear()

        if movie not in self.movie_visualizations or viz_type not in self.movie_visualizations[movie]:
            self.image_label.setText(f"Visualization not found: {viz_type}")
            return

        # Get the path to the visualization and display it
        plot_path = self.movie_visualizations[movie][viz_type]
        self._display_image(plot_path)

    def _show_animated_scanpath(self, movie):
        """Load and display the animated scanpath for the selected movie"""
        # Switch to the animated scanpath widget
        self.viz_stack.setCurrentIndex(1)
        
        try:
            # Find the data path for this movie
            data = VisualizationHelper.get_movie_data(movie, self.output_dir, self.visualization_results)
            if data is not None:
                # Extract real movie name from the data path
                data_path = data["data_path"]
                parts = os.path.basename(data_path).split('_unified_eye_metrics_')
                real_movie_name = parts[1].split('.')[0] if len(parts) > 1 and '.' in parts[1] else movie
                
                # Load the data into the animated scanpath widget
                self.animated_scanpath.load_data(
                    data["data"], 
                    None, 
                    real_movie_name, 
                    self.screen_width, 
                    self.screen_height
                )
            else:
                self.image_label.setText(f"No data available for movie: {movie}")
                self.viz_stack.setCurrentIndex(0)
        except Exception as e:
            self.image_label.setText(f"Error loading animated scanpath: {str(e)}")
            self.viz_stack.setCurrentIndex(0)

    def _display_image(self, image_path):
        """Display an image in the image label"""
        if not os.path.exists(image_path):
            self.image_label.setText(f"File not found: {image_path}")
            return
            
        try:
            # Display the image
            pixmap = QPixmap(image_path)
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
                self.image_label.setText(f"Failed to load image: {image_path}")
        except Exception as e:
            self.image_label.setText(f"Error displaying image: {str(e)}")

    def open_report(self):
        """Open the HTML report in the default web browser"""
        if hasattr(self, 'report_path') and os.path.exists(self.report_path):
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
        """Generate social attention plots based on ROI data"""
        # Get the current movie and plots directory
        movie = self.movie_combo.currentText()
        
        # Get movie data for the selected movie
        movie_data = VisualizationHelper.get_movie_data(
            movie, 
            self.output_dir, 
            self.visualization_results
        )
        
        # Get the plots directory
        plots_dir = VisualizationHelper.get_plots_directory(
            movie, 
            self.output_dir, 
            movie_data
        )
        print(f"Using plots directory: {plots_dir}")
        
        # Configure the plot generator
        self.plot_generator.roi_file_path = self.roi_file_path
        self.plot_generator.movie_combo = self.movie_combo
        self.plot_generator._get_movie_data = lambda m: VisualizationHelper.get_movie_data(
            m, self.output_dir, self.visualization_results
        )
        self.plot_generator.plots_dir = plots_dir
        
        # Pass report path and output directory information if available
        if hasattr(self, 'report_path'):
            self.plot_generator.report_path = self.report_path
        if hasattr(self, 'output_dir'):
            self.plot_generator.output_dir = self.output_dir
        
        # Generate the plots
        result = self.plot_generator.generate_social_attention_plots()
        
        # Handle the result
        if isinstance(result, dict) and result.get("success", False):
            self._handle_successful_plot_generation(result)
            return True
        elif isinstance(result, dict) and not result.get("success", False):
            self._handle_failed_plot_generation(result)
            return False
        elif isinstance(result, bool) and result:
            # Handle legacy boolean return
            self.movie_selected(self.movie_combo.currentIndex())
            return True
        
        return False

    def _handle_successful_plot_generation(self, result):
        """Handle successful generation of social attention plots"""
        movie = result.get("movie", self.movie_combo.currentText())
        
        # Update visualization results
        if movie not in self.visualization_results:
            self.visualization_results[movie] = {'basic': [], 'social': []}
            
        # Add the new plots to visualization_results
        for plot_path in result.get("plots", []):
            if os.path.exists(plot_path):
                if 'social' not in self.visualization_results[movie]:
                    self.visualization_results[movie]['social'] = []
                if plot_path not in self.visualization_results[movie]['social']:
                    self.visualization_results[movie]['social'].append(plot_path)
        
        # Force a refresh of the visualization dropdown
        if movie in self.movie_visualizations:
            del self.movie_visualizations[movie]
        
        # Reload the dropdown contents
        self.movie_selected(self.movie_combo.currentIndex())
        
        # Show success message
        message = "Social attention plots for {0} have been generated.\n\n".format(movie)
        if result.get("report_updated", False):
            message += "The plots have been added to the visualization dropdown and the HTML report has been updated."
        else:
            message += "The plots have been added to the visualization dropdown."
            
        QMessageBox.information(self, "Plot Generated", message)

    def _handle_failed_plot_generation(self, result):
        """Handle failed generation of social attention plots"""
        error_msg = result.get("error", "Unknown error")
        QMessageBox.critical(
            self,
            "Error Generating Plots",
            f"Failed to generate social attention plots: {error_msg}"
        )
    
    def showEvent(self, event):
        """Handle window show event - make sure the window is maximized on startup"""
        super().showEvent(event)
        # Maximize window when first shown
        self.showMaximized()


def run_gui(test_mode=False, source_file=None, destination_folder=None):
    """Run the GUI with optional test mode parameters.
    
    Args:
        test_mode: If True, run in test mode with predefined files
        source_file: Path to source file (.asc or .csv) for test mode
        destination_folder: Output folder for test mode
    
    Returns:
        The application exit code
    """
    app = QApplication(sys.argv)
    window = EyeMovementAnalysisGUI()
    
    # If in test mode, automatically set up the files
    if test_mode:
        print("Running in test mode")
        
        # Set file if provided - this needs to happen before we create and show the window
        source_file_path = None
        if source_file:
            # Try to convert to absolute path if needed
            file_path = os.path.abspath(source_file)
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
        if destination_folder:
            # Try to convert to absolute path if needed
            dest_path = os.path.abspath(destination_folder)
            
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
            # Process events to update UI
            app.processEvents()
            
            # If both file and output directory are specified, automatically process
            print("Auto-processing the specified file...")
            
            # Show the window first to make sure all UI elements are visible
            window.show()
            app.processEvents()  # Process events to render the window
            
            # Use QTimer to process data after the UI has settled
            QTimer.singleShot(200, window.process_data)
    else:
        # Only show the window here if not in test mode
        window.show()
    
    return app.exec_()