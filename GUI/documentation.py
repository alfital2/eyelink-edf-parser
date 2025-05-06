"""
Documentation module for the Eye Movement Analysis for Autism Classification GUI.
Contains all explanations for features and visualizations.
"""


def get_feature_explanations():
    """Return a dictionary of explanations for each feature"""
    explanations = {
        # Pupil Size Features
        "pupil_left_mean": "Mean pupil size of the left eye. Research suggests individuals with ASD may show different pupil responses to social stimuli, potentially reflecting different autonomic nervous system activity.",
        "pupil_left_std": "Standard deviation of left pupil size, indicating variability. Greater pupil size variability has been observed in some studies of individuals with ASD.",
        "pupil_left_min": "Minimum pupil size recorded for the left eye during the session.",
        "pupil_left_max": "Maximum pupil size recorded for the left eye during the session.",
        "pupil_right_mean": "Mean pupil size of the right eye. Comparing left vs. right pupil metrics can also reveal potential asymmetries.",
        "pupil_right_std": "Standard deviation of right pupil size, indicating variability.",
        "pupil_right_min": "Minimum pupil size recorded for the right eye during the session.",
        "pupil_right_max": "Maximum pupil size recorded for the right eye during the session.",

        # Gaze Position Features
        "gaze_left_x_std": "Standard deviation of left eye horizontal position, reflecting how widely the eye moved horizontally. Individuals with ASD may show different scanning patterns with either more restricted or more erratic horizontal eye movements.",
        "gaze_left_y_std": "Standard deviation of left eye vertical position, reflecting how widely the eye moved vertically.",
        "gaze_left_dispersion": "Total area covered by left eye gaze, calculated as the product of horizontal and vertical range. Lower values might indicate more focused attention, while higher values might indicate more distributed attention.",
        "gaze_right_x_std": "Standard deviation of right eye horizontal position.",
        "gaze_right_y_std": "Standard deviation of right eye vertical position.",
        "gaze_right_dispersion": "Total area covered by right eye gaze.",

        # Fixation Features
        "fixation_left_count": "Total number of fixations made with the left eye. Individuals with ASD may show different fixation patterns when viewing social stimuli.",
        "fixation_left_duration_mean": "Average duration of fixations made with the left eye. Longer fixations might indicate deeper processing or difficulty disengaging attention.",
        "fixation_left_duration_std": "Variability in the duration of fixations made with the left eye.",
        "fixation_left_rate": "Number of fixations per second made with the left eye, calculated as fixation count divided by recording duration.",
        "fixation_right_count": "Total number of fixations made with the right eye.",
        "fixation_right_duration_mean": "Average duration of fixations made with the right eye.",
        "fixation_right_duration_std": "Variability in the duration of fixations made with the right eye.",
        "fixation_right_rate": "Number of fixations per second made with the right eye.",

        # Saccade Features
        "saccade_left_count": "Total number of saccades (rapid eye movements between fixations) made with the left eye. Saccade patterns may differ in ASD, potentially showing less predictable movement patterns.",
        "saccade_left_amplitude_mean": "Average distance covered by saccades made with the left eye, measured in visual degrees. Larger amplitudes might indicate more global scanning, while smaller ones might indicate more local focus.",
        "saccade_left_amplitude_std": "Variability in the distance covered by saccades made with the left eye.",
        "saccade_left_duration_mean": "Average duration of saccades made with the left eye.",
        "saccade_right_count": "Total number of saccades made with the right eye.",
        "saccade_right_amplitude_mean": "Average distance covered by saccades made with the right eye.",
        "saccade_right_amplitude_std": "Variability in the distance covered by saccades made with the right eye.",
        "saccade_right_duration_mean": "Average duration of saccades made with the right eye.",

        # Blink Features
        "blink_left_count": "Total number of blinks detected for the left eye. Blink rates may differ in ASD and could relate to attention patterns.",
        "blink_left_duration_mean": "Average duration of blinks for the left eye.",
        "blink_left_rate": "Number of blinks per second for the left eye.",
        "blink_right_count": "Total number of blinks detected for the right eye.",
        "blink_right_duration_mean": "Average duration of blinks for the right eye.",
        "blink_right_rate": "Number of blinks per second for the right eye.",

        # Head Movement Features
        "head_movement_mean": "Average magnitude of head movement during recording. Some studies suggest individuals with ASD may show different patterns of head movement during attention tasks.",
        "head_movement_std": "Variability in head movement magnitude.",
        "head_movement_max": "Maximum head movement magnitude recorded during the session.",
        "head_movement_frequency": "Frequency of head movement direction changes, potentially indicating restlessness or attention shifts.",

        # Basic Information
        "participant_id": "Unique identifier for the participant, extracted from the filename."
    }

    return explanations


def get_visualization_explanations():
    """Return a dictionary of explanations for each visualization type"""
    explanations = {
        "scanpath": """
        <h3>Scanpath Visualization</h3>
        <p>This visualization shows the trajectory of eye movements over time, tracing the path that eyes followed 
        while viewing the stimulus. Lines connect consecutive gaze points, and fixations are highlighted as larger points.</p>

        <p><strong>Research Relevance:</strong> Scanpaths can reveal distinct viewing strategies and attention patterns.
        In autism research, scanpath analysis often shows that individuals with ASD may:
        <ul>
            <li>Follow less predictable scanning patterns compared to neurotypical controls</li>
            <li>Show more idiosyncratic viewing strategies when viewing social scenes</li>
            <li>Exhibit different attention priorities, sometimes focusing less on socially relevant areas like faces</li>
        </ul></p>

        <p>Blue markers represent the left eye, while orange markers represent the right eye. 
        The larger green circles highlight fixation points where the eye remained relatively stable.</p>
        """,
        
        "animated_scanpath": """
        <h3>Animated Scanpath Visualization</h3>
        <p>This interactive visualization dynamically replays eye movements over time, allowing researchers to observe the 
        exact sequence and timing of visual attention as it unfolded during the viewing task.</p>
        
        <p><strong>Research Relevance:</strong> Animated scanpaths provide several advantages for autism research:
        <ul>
            <li>Temporal patterns that might be missed in static visualizations become apparent</li>
            <li>Timing of attention shifts can be precisely observed</li>
            <li>Fixation durations are experienced in real-time, providing intuitive understanding of attentional focus</li>
            <li>When combined with ROI data, reveals how attention moves between social and non-social elements</li>
        </ul></p>
        
        <p><strong>Controls:</strong> Use the play/pause button to control the animation. The speed slider adjusts playback rate.
        You can also use the frame slider to manually move to specific points in the recording. Toggle eye display options
        to focus on left, right, or both eyes.</p>
        """,
        
        "roi_scanpath": """
        <h3>ROI-Enhanced Scanpath Visualization</h3>
        <p>This visualization combines traditional scanpath data with pre-defined Regions of Interest (ROIs) to show how gaze 
        interacts with meaningful areas in the stimulus (such as faces, objects, or other relevant regions).</p>
        
        <p><strong>Research Relevance:</strong> ROI-enhanced visualizations are particularly valuable in autism research:
        <ul>
            <li>Clearly shows whether and how often social regions like faces and eyes are fixated</li>
            <li>Quantifies time spent looking at specific regions of social or clinical interest</li>
            <li>Reveals transitions between different regions (e.g., from faces to background objects)</li>
            <li>Enables comparison of attention patterns across different stimulus types or participant groups</li>
        </ul></p>
        
        <p>ROIs are displayed as colored, semi-transparent overlays. Current ROI information is displayed in real-time during 
        animation playback, including dwell time statistics for each region.</p>
        """,

        "heatmap": """
        <h3>Gaze Heatmap Visualization</h3>
        <p>This visualization shows the density of visual attention across the screen using a color gradient.
        Warmer colors (red, yellow) indicate areas that received more visual attention, while cooler colors
        (blue, green) or blank areas received less attention.</p>

        <p><strong>Research Relevance:</strong> Heatmaps are particularly valuable in autism research for:
        <ul>
            <li>Identifying differences in how social stimuli are processed (e.g., whether faces receive typical levels of attention)</li>
            <li>Revealing preference for non-social vs. social content in complex scenes</li>
            <li>Showing whether attention is distributed typically across features or focused on particular details</li>
        </ul></p>

        <p>Separate heatmaps for left and right eyes allow comparison of gaze patterns between eyes,
        which can reveal potential asymmetries or dominant eye effects.</p>
        """,

        "fixation_duration": """
        <h3>Fixation Duration Distribution</h3>
        <p>This histogram shows the distribution of fixation durations (how long the eye remained relatively 
        stable at each location) throughout the viewing period.</p>

        <p><strong>Research Relevance:</strong> Fixation duration analysis is important in autism research because:
        <ul>
            <li>Atypically long fixations may indicate difficulties with attentional disengagement or deeper processing</li>
            <li>Very short fixations might suggest reduced sustained attention or increased distractibility</li>
            <li>Different distributions between left and right eyes may indicate visual processing asymmetries</li>
        </ul></p>

        <p>The overlaid density curve (KDE) shows the overall distribution shape, while summary statistics 
        (mean, median, min, max) provide numerical insights into viewing behavior.</p>
        """,

        "saccade_amplitude": """
        <h3>Saccade Amplitude Distribution</h3>
        <p>This histogram shows the distribution of saccade amplitudes (the distance covered by rapid eye movements
        between fixations) measured in visual degrees.</p>

        <p><strong>Research Relevance:</strong> Saccade amplitude analysis is valuable in autism research because:
        <ul>
            <li>Smaller amplitudes might indicate more local, detail-oriented processing (common in some individuals with ASD)</li>
            <li>Larger amplitudes could reflect more global scanning patterns</li>
            <li>Highly variable amplitudes might suggest less structured viewing strategies</li>
        </ul></p>

        <p>The distribution shape can reveal whether viewing involved primarily short, medium, or long-distance eye movements,
        providing insights into visual search strategies and attentional scope.</p>
        """,

        "pupil_size": """
        <h3>Pupil Size Timeseries</h3>
        <p>This visualization shows how pupil size changed over time during stimulus viewing. Pupil size is influenced by
        both cognitive factors (cognitive load, emotional arousal) and physiological factors (lighting conditions).</p>
        
        <p><strong>Research Relevance:</strong> Pupil dynamics are increasingly studied in autism research:
        <ul>
           <li>Different pupillary responses to social vs. non-social stimuli have been observed in ASD</li>
           <li>Pupil responses may reflect differences in autonomic nervous system function</li>
           <li>Changes in pupil size can indicate shifts in cognitive processing or emotional responses</li>
        </ul></p>
        
        <p>The 'x' markers indicate the onset of blink events for each eye. Distinct patterns in pupil dilation or constriction 
        between blinks may reveal differences in cognitive processing or emotional reactivity that are characteristic of ASD. 
        The comparison between left and right eyes can also provide insights into potential asymmetries in autonomic responses.</p>
        """,

        "roi_attention_time": """
        <h3>ROI Attention Time Plot</h3>
        <p>This bar chart shows the total time spent looking at each defined Region of Interest (ROI) during stimulus viewing.
        Bar height represents the number of fixations in each region, while percentages inside bars show the proportion of total
        viewing time spent on each region.</p>

        <p><strong>Research Relevance:</strong> ROI attention time analysis is critical in autism research:
        <ul>
            <li>Quantifies precisely how attention was distributed across meaningful regions</li>
            <li>Directly compares social vs. non-social attention allocation (e.g., time spent on faces vs. objects)</li>
            <li>Can reveal subtle attention differences that may not be clinically observable</li>
            <li>Provides objective metrics for comparing attention patterns across participant groups</li>
        </ul></p>

        <p>Bars are ordered by fixation count (descending), making it easy to identify the most visually attended regions
        at a glance. The raw fixation counts and percentages provide multiple ways to interpret the data.</p>
        """,
        
        "roi_transition_matrix": """
        <h3>ROI Transition Matrix</h3>
        <p>This heatmap shows the frequency of gaze transitions between different Regions of Interest (ROIs).
        Brighter/warmer colors indicate more frequent transitions between regions, while darker/cooler colors 
        show less frequent transitions.</p>

        <p><strong>Research Relevance:</strong> Transition matrices provide unique insights for autism research:
        <ul>
            <li>Reveals the sequence and structure of attention shifting, not just where attention was focused</li>
            <li>Can identify unusual attention patterns (e.g., frequent transitions between unrelated regions)</li>
            <li>May show reduced transitions to/from social regions in individuals with ASD</li>
            <li>The overall transition pattern can indicate whether viewing was structured or more random</li>
        </ul></p>

        <p>The diagonal elements show "self-transitions" (continued attention within the same region),
        while off-diagonal elements show transitions between different regions. This visualization helps
        understand not just where participants looked, but how their attention flowed between regions.</p>
        """,
        
        "roi_fixation_sequence": """
        <h3>ROI Fixation Sequence</h3>
        <p>This timeline visualization shows the sequence of ROIs fixated throughout the viewing period.
        Each colored bar represents a different Region of Interest, with the x-axis showing the fixation 
        sequence number.</p>

        <p><strong>Research Relevance:</strong> Fixation sequence analysis is valuable in autism research:
        <ul>
            <li>Shows the temporal order of attention to different elements (e.g., whether faces are looked at first)</li>
            <li>Reveals attention patterns that may be characteristic of different viewing strategies</li>
            <li>Can identify perseverative attention patterns (getting "stuck" on certain regions)</li>
            <li>Shows whether participants follow typical scan patterns for specific types of stimuli</li>
        </ul></p>

        <p>The height of bars indicates fixation duration, so taller bars represent longer fixations.
        This allows researchers to see both the sequence of attended regions and the relative time spent on each.</p>
        """,
        
        "roi_first_fixation_latency": """
        <h3>ROI First Fixation Latency</h3>
        <p>This visualization shows how long it took for each Region of Interest to receive its first fixation.
        Shorter bars indicate regions that captured attention earlier in the viewing period.</p>

        <p><strong>Research Relevance:</strong> First fixation latency is particularly important in autism research:
        <ul>
            <li>Typically developing individuals often look at social elements (especially faces) first</li>
            <li>Individuals with ASD may show delayed first fixations to social regions</li>
            <li>The pattern of first fixations can reveal attention priorities and initial stimulus processing</li>
            <li>Comparison across ROIs shows which elements drew initial attention most effectively</li>
        </ul></p>

        <p>Regions are ordered by latency time (ascending), making it easy to identify which elements 
        captured attention earliest. Time is measured from the start of stimulus presentation.</p>
        """,
        
        "roi_fixation_duration_distribution": """
        <h3>ROI Fixation Duration Distribution</h3>
        <p>This violin plot shows the distribution of fixation durations for each Region of Interest.
        The width of each "violin" shows the density of fixations at different duration values.</p>

        <p><strong>Research Relevance:</strong> ROI-specific fixation duration analysis is valuable for autism research:
        <ul>
            <li>May reveal different processing styles for social vs. non-social content</li>
            <li>Longer fixations on specific regions might indicate deeper processing or difficulty disengaging</li>
            <li>Can show whether social regions receive typical patterns of sustained attention</li>
            <li>Different distribution shapes may indicate different cognitive processes for different region types</li>
        </ul></p>

        <p>Inside each violin plot, a box plot shows median and quartile values, providing statistical context.
        The overall shape reveals whether fixations on a region were consistent or highly variable in duration.</p>
        """,
        
        "roi_temporal_heatmap": """
        <h3>ROI Temporal Heatmap</h3>
        <p>This heatmap shows how attention to different Regions of Interest changed over time during stimulus viewing.
        The vertical axis shows different ROIs, while the horizontal axis represents time bins during viewing.</p>

        <p><strong>Research Relevance:</strong> Temporal attention patterns are crucial in autism research:
        <ul>
            <li>Shows whether attention patterns change throughout viewing (e.g., initial vs. sustained attention)</li>
            <li>Can reveal delayed attention shifts to social content in ASD</li>
            <li>May show different viewing strategies as stimulus processing progresses</li>
            <li>Helps distinguish between early and late attention differences that might have different cognitive bases</li>
        </ul></p>

        <p>Color intensity represents fixation density in each time bin, with warmer colors indicating more 
        attention. Special markers highlight first fixations to each region, showing the temporal sequence
        of initial attention to different elements.</p>
        """,

        "social_attention": """
        <h3>Social Attention Analysis</h3>
        <p>This visualization analyzes how visual attention was distributed between social elements (faces, eyes, people)
        and non-social elements in the stimulus.</p>

        <p><strong>Research Relevance:</strong> Social attention analysis is central to autism research:
        <ul>
            <li>Reduced attention to social stimuli (particularly faces and eyes) is one of the most consistently 
            observed patterns in individuals with ASD</li>
            <li>Quantifying the precise distribution of attention between social and non-social elements
            can potentially serve as a biomarker</li>
            <li>The timeline view reveals how social attention changes throughout viewing, which may reflect
            different patterns of engagement/disengagement</li>
        </ul></p>

        <p>Note: This visualization requires defined Areas of Interest (AOIs) that mark social regions in each frame.
        Without manual AOI data, the visualization creates simulated regions for demonstration purposes only.</p>
        """
    }

    return explanations


def get_formatted_feature_documentation():
    """Generate formatted HTML for the feature documentation tab"""
    feature_explanations = get_feature_explanations()

    html = """
    <h2>Eye Movement Features Guide for Autism Research</h2>
    <p>This guide explains the eye movement features extracted by the analysis tool and their potential relevance to autism research.</p>

    <h3>Background</h3>
    <p>Atypical gaze behavior is one of the earliest and most consistent observations in Autism Spectrum Disorder (ASD). 
    Eye-tracking technology offers a powerful, non-invasive window into how individuals visually engage with their environment.
    The features extracted by this tool are designed to capture aspects of visual attention that may differ between individuals
    with ASD and neurotypical controls.</p>

    <h3>Feature Categories</h3>
    """

    # Add each feature category with descriptions
    categories = [
        ("Pupil Size Features",
         "Measures related to pupil dilation, which can reflect cognitive processing, emotional responses, and autonomic nervous system activity."),
        ("Gaze Position Features",
         "Measures of how widely and variably gaze was distributed across the screen, reflecting scanning patterns."),
        ("Fixation Features",
         "Metrics related to eye fixations, which are periods when the eye is relatively stable and processing visual information."),
        ("Saccade Features",
         "Metrics related to saccades, which are rapid eye movements between fixations, reflecting how viewers scan between points of interest."),
        ("Blink Features", "Metrics related to eye blinks, which can reflect attention, fatigue, and cognitive load."),
        ("Head Movement Features",
         "Measures of head movement during eye tracking, which may indicate restlessness or compensation strategies.")
    ]

    for category, description in categories:
        html += f"<h4>{category}</h4><p>{description}</p><ul>"

        # Add features in this category
        for feature, explanation in feature_explanations.items():
            if category.lower().split()[0] in feature.lower():
                display_name = feature.replace('_', ' ').title()
                html += f"<li><strong>{display_name}:</strong> {explanation}</li>"

        html += "</ul>"

    html += """
    <h3>Research Context</h3>
    <p>When interpreting these features, it's important to consider:
    <ul>
        <li>The specific nature of the visual stimuli (social vs. non-social content)</li>
        <li>Individual differences within both ASD and control groups</li>
        <li>Age and developmental factors that may influence eye movement patterns</li>
        <li>The need for longitudinal assessment to capture potential changes over time</li>
    </ul></p>

    <p>These features are based on current eye-tracking research in ASD, which suggests that quantifiable differences
    in visual attention patterns may serve as potential biomarkers or aid in earlier identification of autism.</p>
    """

    return html


def get_formatted_visualization_documentation():
    """Generate formatted HTML for the visualization documentation tab"""
    visualization_explanations = get_visualization_explanations()

    html = """
    <h2>Eye Movement Visualizations Guide for Autism Research</h2>
    <p>This guide explains the visualizations generated by the analysis tool and their interpretation in the context of autism research.</p>

    <h3>Background</h3>
    <p>Visual attention patterns often differ between individuals with Autism Spectrum Disorder (ASD) and neurotypical controls,
    particularly when viewing social stimuli. These visualizations help researchers and clinicians analyze these differences
    through various graphical representations of eye movement data.</p>
    
    <h3>Visualization Categories</h3>
    <p>The visualizations in this tool are organized into several categories:</p>
    
    <h4>Basic Visualizations</h4>
    <ul>
        <li><strong>Scanpath:</strong> Static representation of gaze trajectory</li>
        <li><strong>Heatmap:</strong> Density visualization of visual attention</li>
        <li><strong>Fixation Duration Distribution:</strong> Statistical analysis of fixation times</li>
        <li><strong>Saccade Amplitude Distribution:</strong> Analysis of eye movement distances</li>
        <li><strong>Pupil Size Timeseries:</strong> Changes in pupil dilation over time</li>
    </ul>
    
    <h4>Animated Visualizations</h4>
    <ul>
        <li><strong>Animated Scanpath:</strong> Dynamic replay of eye movements with temporal information</li>
        <li><strong>ROI-Enhanced Scanpath:</strong> Animated eye movements with Region of Interest overlay</li>
    </ul>
    
    <h4>ROI-Based Social Attention Plots</h4>
    <ul>
        <li><strong>ROI Attention Time:</strong> Overall time spent on each Region of Interest</li>
        <li><strong>ROI Transition Matrix:</strong> Frequency of gaze shifts between different regions</li>
        <li><strong>ROI Fixation Sequence:</strong> Temporal order of fixations across regions</li>
        <li><strong>ROI First Fixation Latency:</strong> Time to first fixation for each region</li>
        <li><strong>ROI Fixation Duration Distribution:</strong> Statistical analysis of fixation times by region</li>
        <li><strong>ROI Temporal Heatmap:</strong> Changes in region attention over time</li>
    </ul>
    """

    # Add each visualization type with detailed explanations
    for viz_type, explanation in visualization_explanations.items():
        # Include the full HTML for the explanation
        html += explanation + "<hr>"

    html += """
    <h3>Working with ROI Data</h3>
    <p>Region of Interest (ROI) visualizations require predefined ROI data that specifies areas of particular interest in the stimulus.
    For social attention research, common ROIs include:</p>
    <ul>
        <li><strong>Social ROIs:</strong> Faces, eyes, mouth, hands, body parts of people in the scene</li>
        <li><strong>Object ROIs:</strong> Toys, furniture, background objects</li>
        <li><strong>Context ROIs:</strong> Room areas, background elements</li>
    </ul>
    
    <p>To use ROI-based visualizations:</p>
    <ol>
        <li>Load your eye tracking data</li>
        <li>Use the "Load ROI" button to select a JSON file containing ROI definitions</li>
        <li>Generate social attention plots with the ROI data</li>
        <li>For animated visualization with ROIs, enable the "Show ROIs" option in the Animated Scanpath tab</li>
    </ol>
    
    <h3>Using Visualizations in Clinical and Research Settings</h3>
    <p>These visualizations can be valuable in several contexts:
    <ul>
        <li><strong>Diagnostic Support:</strong> While not diagnostic on their own, these patterns may support clinical observations</li>
        <li><strong>Intervention Monitoring:</strong> Tracking changes in attention patterns over the course of interventions</li>
        <li><strong>Research:</strong> Investigating specific hypotheses about visual attention in ASD</li>
        <li><strong>Parent/Family Education:</strong> Helping families understand aspects of atypical visual attention</li>
    </ul></p>

    <p>When interpreting these visualizations, consider that eye movement patterns exist on a spectrum and 
    significant individual variation exists both within typical development and ASD. These tools are most valuable
    when integrated with other clinical and research information.</p>
    
    <h3>Exporting and Sharing Results</h3>
    <p>All visualizations can be saved and included in HTML reports for sharing with colleagues or research participants.
    The HTML report organizes all generated visualizations by movie/stimulus and type, creating an interactive presentation
    of your analysis results.</p>
    """

    return html