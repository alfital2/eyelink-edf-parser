#!/usr/bin/env python3
"""
Eye Movement Analysis for Autism Classification - Application Entry Point

This script serves as the entry point for the eye tracking data analysis application.
It initializes the PyQt application and launches the main window.
"""

# Standard library imports
import sys
import os
import argparse

# Add parent directory to path, so we can import modules from the parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure matplotlib for thread safety BEFORE importing any matplotlib-related modules
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues

# PyQt5 imports
from PyQt5.QtWidgets import QApplication

# Import the main application window from the UI package
from ui.main_window import EyeMovementAnalysisGUI


def parse_args():
    """
    Parse command line arguments for the GUI.
    
    Returns:
        argparse.Namespace: The parsed command line arguments
    """
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


def main():
    """
    Main function to initialize and run the application
    """
    # Create the application
    app = QApplication(sys.argv)
    
    # Create the main window
    window = EyeMovementAnalysisGUI()
    
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
    
    # Start the application event loop
    sys.exit(app.exec_())


# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()