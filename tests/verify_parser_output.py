#!/usr/bin/env python3
"""
Helper script to analyze and verify the output of the EyeLinkASCParser.

This script processes a sample ASC file and provides a detailed report
of the extracted data, allowing you to visually inspect the results and
verify that the parser is functioning correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import the parser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from parser import EyeLinkASCParser, process_asc_file
except ImportError:
    print("ERROR: Could not import the EyeLinkASCParser. Make sure parser.py is in the parent directory.")
    sys.exit(1)


def analyze_and_verify_parser(file_path):
    """Process the sample ASC file and generate verification reports."""
    # Path to the sample ASC file
    tests_dir = os.path.dirname(__file__)
    sample_asc_path = os.path.join(tests_dir, file_path)

    if not os.path.exists(sample_asc_path):
        print(f"ERROR: Sample ASC file not found at {sample_asc_path}")
        sys.exit(1)

    # Create output directory for reports
    output_dir = os.path.join(tests_dir, "verification_output")
    os.makedirs(output_dir, exist_ok=True)

    # Process the sample file
    print(f"Processing sample ASC file: {sample_asc_path}")
    result = process_asc_file(sample_asc_path, output_dir, extract_features=True)

    # Generate the verification report
    generate_report(result, output_dir)


def generate_report(result, output_dir):
    """Generate a comprehensive report of the parser's output."""
    print("\nGenerating verification report...")

    # Create a report file
    report_path = os.path.join(output_dir, "parser_verification_report.txt")

    with open(report_path, 'w') as f:
        # Write report header
        f.write("=" * 80 + "\n")
        f.write("EYELINK ASC PARSER VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        for key, value in result['summary'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Metadata
        f.write("METADATA\n")
        f.write("-" * 40 + "\n")
        for key, value in result['summary']['metadata'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Sample data statistics
        samples_df = result['dataframes']['samples']
        f.write("SAMPLE DATA STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {len(samples_df)}\n")
        f.write(f"Sample time range: {samples_df['timestamp'].min()} to {samples_df['timestamp'].max()}\n")
        f.write("\nSample columns summary statistics:\n")

        # Get statistics for relevant columns
        for col in ['x_left', 'y_left', 'pupil_left', 'x_right', 'y_right', 'pupil_right']:
            if col in samples_df.columns:
                stats = samples_df[col].describe()
                f.write(f"\n{col}:\n")
                f.write(f"  Mean: {stats['mean']:.2f}\n")
                f.write(f"  Min: {stats['min']:.2f}\n")
                f.write(f"  Max: {stats['max']:.2f}\n")
                f.write(f"  Std Dev: {stats['std']:.2f}\n")
        f.write("\n")

        # Eye events summary
        f.write("EYE EVENTS SUMMARY\n")
        f.write("-" * 40 + "\n")

        # Fixations
        for eye in ['left', 'right']:
            fixations_df = result['dataframes'].get(f'fixations_{eye}')
            if fixations_df is not None and not fixations_df.empty:
                f.write(f"\n{eye.capitalize()} eye fixations ({len(fixations_df)} events):\n")
                dur_stats = fixations_df['duration'].describe()
                f.write(f"  Duration Mean: {dur_stats['mean']:.2f} ms\n")
                f.write(f"  Duration Min: {dur_stats['min']:.2f} ms\n")
                f.write(f"  Duration Max: {dur_stats['max']:.2f} ms\n")
                if 'x' in fixations_df.columns and 'y' in fixations_df.columns:
                    f.write(f"  Average Position: ({fixations_df['x'].mean():.1f}, {fixations_df['y'].mean():.1f})\n")

        # Saccades
        for eye in ['left', 'right']:
            saccades_df = result['dataframes'].get(f'saccades_{eye}')
            if saccades_df is not None and not saccades_df.empty:
                f.write(f"\n{eye.capitalize()} eye saccades ({len(saccades_df)} events):\n")
                if 'duration' in saccades_df.columns:
                    dur_stats = saccades_df['duration'].describe()
                    f.write(f"  Duration Mean: {dur_stats['mean']:.2f} ms\n")
                    f.write(f"  Duration Min: {dur_stats['min']:.2f} ms\n")
                    f.write(f"  Duration Max: {dur_stats['max']:.2f} ms\n")
                if 'amplitude' in saccades_df.columns:
                    amp_stats = saccades_df['amplitude'].describe()
                    f.write(f"  Amplitude Mean: {amp_stats['mean']:.2f} degrees\n")
                    f.write(f"  Amplitude Min: {amp_stats['min']:.2f} degrees\n")
                    f.write(f"  Amplitude Max: {amp_stats['max']:.2f} degrees\n")

        # Blinks
        for eye in ['left', 'right']:
            blinks_df = result['dataframes'].get(f'blinks_{eye}')
            if blinks_df is not None and not blinks_df.empty:
                f.write(f"\n{eye.capitalize()} eye blinks ({len(blinks_df)} events):\n")
                if 'duration' in blinks_df.columns:
                    dur_stats = blinks_df['duration'].describe()
                    f.write(f"  Duration Mean: {dur_stats['mean']:.2f} ms\n")
                    f.write(f"  Duration Min: {dur_stats['min']:.2f} ms\n")
                    f.write(f"  Duration Max: {dur_stats['max']:.2f} ms\n")
        f.write("\n")

        # Messages summary
        messages_df = result['dataframes'].get('messages')
        if messages_df is not None and not messages_df.empty:
            f.write("MESSAGE MARKERS SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total messages: {len(messages_df)}\n")

            # Count message types
            message_types = {}
            for msg in messages_df['content']:
                # Extract message type (first word or prefix)
                if ' ' in msg:
                    msg_type = msg.split(' ')[0]
                else:
                    msg_type = msg

                message_types[msg_type] = message_types.get(msg_type, 0) + 1

            f.write("\nMessage types:\n")
            for msg_type, count in sorted(message_types.items(), key=lambda x: x[1], reverse=True):
                if count > 1:  # Only show types with multiple occurrences
                    f.write(f"  {msg_type}: {count} occurrences\n")

            # Show some example messages
            f.write("\nExample messages:\n")
            for i, msg in enumerate(messages_df.iloc[:5]['content']):
                f.write(f"  {i + 1}. {msg}\n")
            f.write("  ...\n")
            for i, msg in enumerate(messages_df.iloc[-5:]['content']):
                f.write(f"  {len(messages_df) - 4 + i}. {msg}\n")
        f.write("\n")

        # Frame markers
        frames_df = result['dataframes'].get('frames')
        if frames_df is not None and not frames_df.empty:
            f.write("VIDEO FRAME MARKERS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total frame markers: {len(frames_df)}\n")

            # List all frame numbers and timestamps
            f.write("\nFrame markers:\n")
            for i, row in frames_df.iterrows():
                f.write(f"  Frame #{row['frame']}: timestamp {row['timestamp']}\n")
        f.write("\n")

        # Extracted features
        features_df = result['features']
        if features_df is not None and not features_df.empty:
            f.write("EXTRACTED FEATURES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total features extracted: {len(features_df.columns) - 1}\n")  # -1 for participant_id

            # Group features by type
            feature_groups = {
                'Pupil': [col for col in features_df.columns if 'pupil_' in col],
                'Gaze': [col for col in features_df.columns if 'gaze_' in col],
                'Fixation': [col for col in features_df.columns if 'fixation_' in col],
                'Saccade': [col for col in features_df.columns if 'saccade_' in col],
                'Blink': [col for col in features_df.columns if 'blink_' in col],
                'Other': [col for col in features_df.columns if 'participant_id' not in col and
                          not any(x in col for x in ['pupil_', 'gaze_', 'fixation_', 'saccade_', 'blink_'])]
            }

            for group_name, feature_cols in feature_groups.items():
                if feature_cols:
                    f.write(f"\n{group_name} features ({len(feature_cols)}):\n")
                    for col in feature_cols:
                        value = features_df[col].iloc[0]
                        f.write(f"  {col}: {value}\n")

        # End of report
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Verification report generated: {report_path}")

    # Generate visualization plots
    generate_visualizations(result, output_dir)


def generate_visualizations(result, output_dir):
    """Generate plots to visualize the parser's output."""
    print("Generating visualization plots...")

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Gaze positions with events
    try:
        plot_gaze_path(result, os.path.join(plots_dir, "gaze_positions.png"))
    except Exception as e:
        print(f"WARNING: Failed to generate gaze positions plot: {str(e)}")

    # Plot 2: Pupil sizes
    try:
        plot_pupil_sizes(result, os.path.join(plots_dir, "pupil_sizes.png"))
    except Exception as e:
        print(f"WARNING: Failed to generate pupil sizes plot: {str(e)}")

    # Plot 3: Event durations
    try:
        plot_event_durations(result, os.path.join(plots_dir, "event_durations.png"))
    except Exception as e:
        print(f"WARNING: Failed to generate event durations plot: {str(e)}")

    print(f"Visualization plots saved to: {plots_dir}")


def plot_gaze_path(result, output_path):
    """Plot the gaze path with events marked."""
    samples_df = result['dataframes']['samples']
    unified_df = result['dataframes']['unified_eye_metrics']

    plt.figure(figsize=(10, 8))

    # Plot gaze positions for left eye
    plt.scatter(samples_df['x_left'], samples_df['y_left'], s=2, c='blue', alpha=0.3, label='Left eye')

    # Plot gaze positions for right eye
    plt.scatter(samples_df['x_right'], samples_df['y_right'], s=2, c='red', alpha=0.3, label='Right eye')

    # Highlight fixation positions for left eye
    if 'fixations_left' in result['dataframes']:
        fixations_left = result['dataframes']['fixations_left']
        if not fixations_left.empty and 'x' in fixations_left.columns and 'y' in fixations_left.columns:
            plt.scatter(fixations_left['x'], fixations_left['y'], s=30, marker='o',
                        edgecolor='blue', facecolor='none', linewidth=2, label='Left fixations')

    # Highlight fixation positions for right eye
    if 'fixations_right' in result['dataframes']:
        fixations_right = result['dataframes']['fixations_right']
        if not fixations_right.empty and 'x' in fixations_right.columns and 'y' in fixations_right.columns:
            plt.scatter(fixations_right['x'], fixations_right['y'], s=30, marker='o',
                        edgecolor='red', facecolor='none', linewidth=2, label='Right fixations')



    plt.title('Gaze Positions with Events')
    plt.xlabel('X position (pixels)')
    plt.ylabel('Y position (pixels)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Invert Y-axis (0,0 is top-left in screen coordinates)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pupil_sizes(result, output_path):
    """Plot the pupil sizes over time."""
    samples_df = result['dataframes']['samples']

    plt.figure(figsize=(12, 6))

    # Plot pupil sizes for both eyes
    plt.plot(samples_df['timestamp'], samples_df['pupil_left'], 'b-', alpha=0.7, label='Left eye')
    plt.plot(samples_df['timestamp'], samples_df['pupil_right'], 'r-', alpha=0.7, label='Right eye')

    # Mark blinks
    if 'blinks_left' in result['dataframes']:
        blinks_left = result['dataframes']['blinks_left']
        if not blinks_left.empty:
            for _, blink in blinks_left.iterrows():
                plt.axvspan(blink['start_time'], blink['end_time'], alpha=0.2, color='blue', label='_nolegend_')

    if 'blinks_right' in result['dataframes']:
        blinks_right = result['dataframes']['blinks_right']
        if not blinks_right.empty:
            for _, blink in blinks_right.iterrows():
                plt.axvspan(blink['start_time'], blink['end_time'], alpha=0.2, color='red', label='_nolegend_')

    # Mark frame markers
    if 'frames' in result['dataframes']:
        frames_df = result['dataframes']['frames']
        if not frames_df.empty:
            for _, frame in frames_df.iterrows():
                plt.axvline(x=frame['timestamp'], color='green', linestyle='--', alpha=0.5)
                plt.text(frame['timestamp'], plt.ylim()[1] * 0.95,
                         f"Frame #{frame['frame']}", rotation=90, va='top')

    plt.title('Pupil Size Over Time')
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('Pupil Size')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_event_durations(result, output_path):
    """Plot histograms of event durations."""
    events = []

    # Collect fixation durations
    for eye in ['left', 'right']:
        if f'fixations_{eye}' in result['dataframes']:
            fixations_df = result['dataframes'][f'fixations_{eye}']
            if not fixations_df.empty and 'duration' in fixations_df.columns:
                for duration in fixations_df['duration']:
                    events.append(('Fixation', eye, duration))

    # Collect saccade durations
    for eye in ['left', 'right']:
        if f'saccades_{eye}' in result['dataframes']:
            saccades_df = result['dataframes'][f'saccades_{eye}']
            if not saccades_df.empty and 'duration' in saccades_df.columns:
                for duration in saccades_df['duration']:
                    events.append(('Saccade', eye, duration))

    # Collect blink durations
    for eye in ['left', 'right']:
        if f'blinks_{eye}' in result['dataframes']:
            blinks_df = result['dataframes'][f'blinks_{eye}']
            if not blinks_df.empty and 'duration' in blinks_df.columns:
                for duration in blinks_df['duration']:
                    events.append(('Blink', eye, duration))

    # Create DataFrame for plotting
    events_df = pd.DataFrame(events, columns=['Event Type', 'Eye', 'Duration'])

    if not events_df.empty:
        # Reorganize data for grouped bar chart
        pivot_df = events_df.pivot_table(
            values='Duration',
            index='Event Type',
            columns='Eye',
            aggfunc=['mean', 'median', 'min', 'max', 'count']
        )

        # Plot the event statistics
        plt.figure(figsize=(10, 8))

        # Bar positions
        bar_width = 0.35
        r1 = np.arange(len(pivot_df.index))
        r2 = [x + bar_width for x in r1]

        # Create bars
        plt.bar(r1, pivot_df['mean']['left'], width=bar_width, label='Left Eye', color='blue', alpha=0.7)
        plt.bar(r2, pivot_df['mean']['right'], width=bar_width, label='Right Eye', color='red', alpha=0.7)

        # Labels and formatting
        plt.xlabel('Event Type')
        plt.ylabel('Mean Duration (ms)')
        plt.title('Mean Duration of Eye Events')
        plt.xticks([r + bar_width / 2 for r in range(len(pivot_df.index))], pivot_df.index)
        plt.legend()

        # Add count annotations
        for i, event_type in enumerate(pivot_df.index):
            left_count = pivot_df['count']['left'][event_type]
            right_count = pivot_df['count']['right'][event_type]

            plt.text(r1[i], pivot_df['mean']['left'][event_type] + 1, f'n={left_count}',
                     ha='center', va='bottom', color='blue')
            plt.text(r2[i], pivot_df['mean']['right'][event_type] + 1, f'n={right_count}',
                     ha='center', va='bottom', color='red')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    else:
        # Create a simple text plot if no events
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No event duration data available",
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    analyze_and_verify_parser("asc_files/smiley_test.asc")
    # analyze_and_verify_parser("asc_files/sample_test.asc")