"""
Main script for Eye Movement Analysis for Autism Classification
Author: Tal Alfi
Date: March 2025
"""

import os
import argparse
from typing import List, Tuple
import glob
from datetime import datetime
import matplotlib.pyplot as plt

# Import our modules
from parser import process_asc_file, process_multiple_files
from eyelink_visualizer import generate_visualizations, generate_multiple_visualizations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Eye Movement Analysis for Autism Classification')

    # Input arguments
    parser.add_argument('--input', '-i', type=str, help='Path to ASC file or directory containing ASC files')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory for parsed data and visualizations')
    parser.add_argument('--screen_width', type=int, default=1280, help='Screen width in pixels')
    parser.add_argument('--screen_height', type=int, default=1024, help='Screen height in pixels')

    # Processing options
    parser.add_argument('--unified_only', action='store_true', help='Only save unified eye metrics CSV')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--no_features', action='store_false', dest='extract_features', help='Skip feature extraction')

    return parser.parse_args()


def find_asc_files(input_path: str) -> List[str]:
    """Find all ASC files in the given directory or return a single file."""
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, '*.asc')))
    elif os.path.isfile(input_path) and input_path.lower().endswith('.asc'):
        return [input_path]
    else:
        raise ValueError(f"Input path is not a valid ASC file or directory: {input_path}")


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
    args = parse_args()

    # Find input files
    try:
        asc_files = find_asc_files(args.input)
        print(f"Found {len(asc_files)} ASC files.")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if not asc_files:
        print("No ASC files found. Exiting.")
        return 1

    # Create output directories
    data_dir, viz_dir, feature_dir = create_output_dirs(args.output)
    print(f"Output directories created:")
    print(f"  - Data: {data_dir}")
    print(f"  - Visualizations: {viz_dir}")
    print(f"  - Features: {feature_dir}")

    # Process multiple files if needed
    if len(asc_files) > 1:
        print(f"\nProcessing {len(asc_files)} files...")
        combined_features = process_multiple_files(
            asc_files,
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

            # Maybe generate a correlation heatmap of features
            plt.figure(figsize=(12, 10))
            corr_matrix = combined_features.select_dtypes(include=['number']).corr()
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            corr_path = os.path.join(feature_dir, 'feature_correlations.png')
            plt.savefig(corr_path, dpi=150, bbox_inches='tight')
            print(f"Feature correlation matrix saved to {corr_path}")
            plt.close()
    else:
        # Process a single file
        print(f"\nProcessing file: {asc_files[0]}")
        result = process_asc_file(
            asc_files[0],
            output_dir=data_dir,
            extract_features=args.extract_features,
            unified_only=args.unified_only
        )

        print(f"Processing complete. Summary: {result['summary']}")

        # Save individual features
        if args.extract_features and 'features' in result and not result['features'].empty:
            # Get the base filename
            base_name = os.path.splitext(os.path.basename(asc_files[0]))[0]
            features_path = os.path.join(feature_dir, f"{base_name}_features.csv")

            # Save features to CSV
            result['features'].to_csv(features_path, index=False)
            print(f"\nFeatures saved to {features_path}")

        # Generate visualizations if requested
        if args.visualize:
            print("\nGenerating visualizations...")
            try:
                # Set the participant ID in the dataframes
                if 'unified_eye_metrics' in result['dataframes']:
                    # Get the base filename for participant ID
                    participant_id = os.path.splitext(os.path.basename(asc_files[0]))[0]
                    # Add participant ID to the dataframes dictionary
                    result['dataframes']['participant_id'] = participant_id

                output_dir = generate_visualizations(
                    result['dataframes'],
                    output_dir=viz_dir,
                    screen_size=(args.screen_width, args.screen_height)
                )
                print(f"Visualizations saved to {output_dir}")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
                import traceback
                traceback.print_exc()  # Print the full traceback for debugging


if __name__ == "__main__":
    exit(main())
