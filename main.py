"""
Main script for Eye Movement Analysis for Autism Classification
Author: Tal Alfi
Date: April 2025
"""

import os
import argparse
import time
from typing import List, Tuple
import glob
from datetime import datetime
import matplotlib.pyplot as plt

# Import our modules
from parser import process_asc_file, process_multiple_files, load_csv_file, load_multiple_csv_files
from eyelink_visualizer import MovieEyeTrackingVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Eye Movement Analysis for Autism Classification')

    # Input arguments
    parser.add_argument('--input', '-i', type=str, 
                        help='Path to ASC/CSV file or directory containing ASC/CSV files')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory for parsed data and visualizations')
    parser.add_argument('--screen_width', type=int, default=1280, help='Screen width in pixels')
    parser.add_argument('--screen_height', type=int, default=1024, help='Screen height in pixels')

    # Processing options
    parser.add_argument('--use_csv', action='store_true', 
                        help='Process CSV files instead of ASC files. Useful for faster loading of pre-processed data.')
    parser.add_argument('--unified_only', action='store_true', help='Only save unified eye metrics CSV')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--no_features', action='store_false', dest='extract_features', help='Skip feature extraction')
    parser.add_argument('--report', action='store_true', help='Generate HTML visualization report')

    return parser.parse_args()


def find_input_files(input_path: str, use_csv: bool = False) -> List[str]:
    """
    Find all input files (ASC or CSV) in the given directory or return a single file.
    
    Args:
        input_path: Path to a file or directory
        use_csv: If True, look for CSV files instead of ASC files
    
    Returns:
        List of file paths
    """
    file_ext = '*.csv' if use_csv else '*.asc'
    expected_ext = '.csv' if use_csv else '.asc'
    file_type = 'CSV' if use_csv else 'ASC'
    
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, file_ext)))
    elif os.path.isfile(input_path) and input_path.lower().endswith(expected_ext):
        return [input_path]
    else:
        raise ValueError(f"Input path is not a valid {file_type} file or directory: {input_path}")


def create_output_dirs(base_output_dir: str) -> Tuple[str, str, str]:
    """Create output directories for data, visualizations, and features."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_output_dir, timestamp)

    data_dir = os.path.join(output_dir, 'data')
    viz_dir = os.path.join(output_dir, 'plots')
    feature_dir = os.path.join(output_dir, 'features')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    return data_dir, viz_dir, feature_dir


def main():
    """Main execution function."""
    start_time = time.time()
    args = parse_args()
    
    # Determine file type to process
    file_type = "CSV" if args.use_csv else "ASC"

    # Find input files
    try:
        input_files = find_input_files(args.input, args.use_csv)
        print(f"Found {len(input_files)} {file_type} files.")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if not input_files:
        print(f"No {file_type} files found. Exiting.")
        return 1

    # Create output directories
    data_dir, viz_dir, feature_dir = create_output_dirs(args.output)
    print("Output directories created:")
    print(f"  - Data: {data_dir}")
    print(f"  - Visualizations: {viz_dir}")
    print(f"  - Features: {feature_dir}")

    # Process multiple files if needed
    if len(input_files) > 1:
        print(f"\nProcessing {len(input_files)} {file_type} files...")
        
        # Choose the appropriate processing function based on file type
        if args.use_csv:
            combined_features = load_multiple_csv_files(
                input_files,
                output_dir=data_dir
            )
        else:
            combined_features = process_multiple_files(
                input_files,
                output_dir=data_dir,
                unified_only=args.unified_only
            )

        # Save combined features
        if args.extract_features and not combined_features.empty:
            combined_features_path = os.path.join(feature_dir, 'combined_features.csv')
            combined_features.to_csv(combined_features_path, index=False)
            print(f"Combined features saved to {combined_features_path}")

            # Generate a summary report of extracted features
            summary_path = os.path.join(feature_dir, 'feature_summary.csv')
            feature_summary = combined_features.describe()
            feature_summary.to_csv(summary_path)
            print(f"Feature summary statistics saved to {summary_path}")

            # Generate a correlation heatmap of features
            plt.figure(figsize=(12, 10))
            corr_matrix = combined_features.select_dtypes(include=['number']).corr()
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            corr_path = os.path.join(feature_dir, 'feature_correlations.png')
            plt.savefig(corr_path, dpi=150, bbox_inches='tight')
            print(f"Feature correlation matrix saved to {corr_path}")
            plt.close()
    else:
        # Process a single file
        print(f"\nProcessing file: {input_files[0]}")
        
        # Choose the appropriate processing function based on file type
        if args.use_csv:
            result = load_csv_file(
                input_files[0],
                output_dir=data_dir,
                extract_features=args.extract_features
            )
        else:
            result = process_asc_file(
                input_files[0],
                output_dir=data_dir,
                extract_features=args.extract_features,
                unified_only=args.unified_only
            )

        print(f"Processing complete. Summary: {result['summary']}")

        # Save individual features
        if args.extract_features and 'features' in result and not result['features'].empty:
            # Get the base filename
            base_name = os.path.splitext(os.path.basename(input_files[0]))[0]
            features_path = os.path.join(feature_dir, f"{base_name}_features.csv")

            # Save features to CSV
            result['features'].to_csv(features_path, index=False)
            print(f"\nFeatures saved to {features_path}")

    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        try:
            # Get the base filename for participant ID
            participant_id = os.path.splitext(os.path.basename(input_files[0]))[0]
            
            # If it's a CSV file with unified_eye_metrics in the name, remove that part
            if args.use_csv:
                participant_id = participant_id.replace('_unified_eye_metrics', '')

            # Initialize the movie visualizer
            visualizer = MovieEyeTrackingVisualizer(
                base_dir=data_dir,
                screen_size=(args.screen_width, args.screen_height)
            )

            # Process all movies
            visualization_results = visualizer.process_all_movies(participant_id)

            # Generate HTML report if requested
            if args.report and visualization_results:
                report_dir = os.path.join(viz_dir, 'report')
                report_path = visualizer.generate_report(visualization_results, report_dir)
                print(f"Visualization report generated at: {report_path}")

        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging

    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")

    return 0


if __name__ == "__main__":
    exit(main())