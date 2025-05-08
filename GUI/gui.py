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
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QCursor, QPalette

# Local application imports
from GUI.data.parser import (
    process_asc_file, process_multiple_files, 
    load_csv_file, load_multiple_csv_files
)
from eyelink_visualizer import MovieEyeTrackingVisualizer
from GUI.utils.documentation import (
    get_feature_explanations, get_visualization_explanations,
    get_formatted_feature_documentation, get_formatted_visualization_documentation
)
from animated_roi_scanpath import AnimatedROIScanpathWidget
from GUI.theme_manager import ThemeManager
from GUI.feature_table_manager import FeatureTableManager
from GUI.visualization.plot_generator import PlotGenerator
from GUI.data.processing_thread import ProcessingThread

# Global variable for plot progress tracking
current_plot_progress = "0/0"  # Will be updated during plot generation


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
        
        # Initialize plot generator
        self.plot_generator = PlotGenerator(
            self.screen_width, 
            self.screen_height, 
            self.visualization_results, 
            self.movie_visualizations
        )

        # Get feature and visualization explanations from the documentation module
        self.feature_explanations = get_feature_explanations()
        self.visualization_explanations = get_visualization_explanations()

        # Initialize theme manager
        self.theme_manager = ThemeManager(self)
        
        # Initialize feature table manager
        self.feature_table_manager = FeatureTableManager(self, self.theme_manager, self.feature_explanations)
        
        # Initialize feature_tables attribute
        self.feature_tables = self.feature_table_manager.feature_tables

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

        # Refresh visualization explanation text area
        if hasattr(self, 'viz_explanation'):
            if not self.theme_manager.is_dark_mode:
                self.viz_explanation.setStyleSheet("background-color: #f8f8f8; border: 1px solid #e0e0e0;")
            else:
                self.viz_explanation.setStyleSheet("")  # Default dark mode styling
            
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
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Apply theme-specific styles
        central_widget.setStyleSheet(self.theme_manager.get_theme_style())

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
        if self.theme_manager.is_dark_mode:
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

        # Create tables for different feature categories using the feature table manager
        self.feature_table_manager.create_feature_tables(features_layout)

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




    def update_feature_tables(self, features_df):
        """Update all feature tables with data from the features DataFrame"""
        if features_df is None or features_df.empty:
            return

        # Store the features data
        self.features_data = features_df
        
        # Update tables using the feature table manager
        self.feature_table_manager.update_feature_tables(features_df)
                
    
    
    

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
            self.status_label.setText("ASC Files: Original (converted) EyeLink data files. Processing may take longer "
                                      "but includes all raw data.")
        else:  # CSV Files
            self.status_label.setText("CSV Files: Pre-processed data files. Much faster loading for visualizations "
                                      "and analysis.")

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
            self.status_log.setTextColor(Qt.blue if not self.theme_manager.is_dark_mode else Qt.cyan)
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
            self.status_log.setTextColor(Qt.black if not self.theme_manager.is_dark_mode else Qt.white)
                
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
            self.feature_table_manager.update_feature_tables(results["features"])

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
            self.feature_table_manager.update_feature_tables(self.movie_features[movie_name])
            
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
                return {"data": data, "data_path": data_path, "data_dir": data_dir}
        
        return None
        
    def get_plots_directory(self, movie=None):
        """
        Get the correct plots directory for the given movie.
        This ensures all visualizations are saved to the same location.
        """
        # First try to get movie data to locate the plots directory
        if movie:
            movie_data = self._get_movie_data(movie)
            if movie_data and "data_dir" in movie_data:
                data_dir = movie_data["data_dir"]
                plots_dir = os.path.join(data_dir, 'plots')
                if os.path.exists(plots_dir):
                    return plots_dir
                    
        # If we have no movie data or can't find its plots directory,
        # use the output directory structure
        if hasattr(self, 'output_dir') and self.output_dir and os.path.exists(self.output_dir):
            data_dir = os.path.join(self.output_dir, 'data')
            if os.path.exists(data_dir):
                plots_dir = os.path.join(data_dir, 'plots')
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir, exist_ok=True)
                return plots_dir
            
            # If no data directory, use the main plots directory
            plots_dir = os.path.join(self.output_dir, 'plots')
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir, exist_ok=True)
            return plots_dir
            
        # Fallback to a default location
        default_plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(default_plots_dir, exist_ok=True)
        return default_plots_dir

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
        """Delegate to the PlotGenerator class and handle UI updates"""
        # Get the current movie
        movie = self.movie_combo.currentText()
        
        # Get the correct plots directory
        plots_dir = self.get_plots_directory(movie)
        print(f"Using plots directory: {plots_dir}")
        
        # Set necessary attributes on the plot generator
        self.plot_generator.roi_file_path = self.roi_file_path
        self.plot_generator.movie_combo = self.movie_combo
        self.plot_generator._get_movie_data = self._get_movie_data  # Method injection
        self.plot_generator.plots_dir = plots_dir
        
        # Pass report path and output directory information if available
        if hasattr(self, 'report_path'):
            self.plot_generator.report_path = self.report_path
        if hasattr(self, 'output_dir'):
            self.plot_generator.output_dir = self.output_dir
        
        # Call the generator
        result = self.plot_generator.generate_social_attention_plots()
        
        # Handle the result
        if isinstance(result, dict) and result.get("success", False):
            # Update the visualization dropdown to show the new plots
            movie = result.get("movie", self.movie_combo.currentText())
            
            # Ensure plots are properly added to visualization_results
            if movie not in self.visualization_results:
                self.visualization_results[movie] = {'basic': [], 'social': []}
                
            # Add the newly generated plots to visualization_results
            plots = result.get("plots", [])
            for plot_path in plots:
                if os.path.exists(plot_path):
                    if plot_path not in self.visualization_results[movie].get('social', []):
                        if 'social' not in self.visualization_results[movie]:
                            self.visualization_results[movie]['social'] = []
                        self.visualization_results[movie]['social'].append(plot_path)
            
            # Force a refresh of the visualization dropdown
            # First reset the movie_visualizations entry to ensure it's reloaded from disk
            if movie in self.movie_visualizations:
                del self.movie_visualizations[movie]
            
            # Then reload the dropdown contents
            self.movie_selected(self.movie_combo.currentIndex())
            
            # Show appropriate success message
            if result.get("report_updated", False):
                QMessageBox.information(
                    self,
                    "Plot Generated",
                    f"Social attention plots for {movie} have been generated.\n\n"
                    f"The plots have been added to the visualization dropdown and the HTML report has been updated."
                )
            else:
                QMessageBox.information(
                    self,
                    "Plot Generated",
                    f"Social attention plots for {movie} have been generated.\n\n"
                    f"The plots have been added to the visualization dropdown."
                )
            
            return True
        elif isinstance(result, dict) and not result.get("success", False):
            # Handle error case with details
            error_msg = result.get("error", "Unknown error")
            QMessageBox.critical(
                self,
                "Error Generating Plots",
                f"Failed to generate social attention plots: {error_msg}"
            )
            return False
        elif isinstance(result, bool):
            # Handle simple boolean return for backward compatibility
            if result:
                # Update the visualization dropdown
                self.movie_selected(self.movie_combo.currentIndex())
                return True
        
        return False


    
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
