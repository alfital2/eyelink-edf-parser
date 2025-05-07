"""
Visualization Controller Module

This module contains the VisualizationController class which manages the interaction
between the data models and the visualization components in the GUI.
"""

from .base_controller import BaseController


class VisualizationController(BaseController):
    """
    Controller for managing data visualizations.
    
    This controller coordinates the generation and updating of various visualizations
    based on eye tracking data and analysis results, connecting the data models with
    the visualization components in the GUI.
    """
    
    def __init__(self, eye_tracking_data=None, roi_model=None, feature_model=None):
        """
        Initialize the visualization controller.
        
        Args:
            eye_tracking_data (EyeTrackingData, optional): The eye tracking data model
            roi_model (ROIModel, optional): The ROI data model
            feature_model (FeatureModel, optional): The feature data model
        """
        super().__init__()
        
        # Register models if provided
        if eye_tracking_data:
            self.register_model('eye_tracking_data', eye_tracking_data)
        
        if roi_model:
            self.register_model('roi', roi_model)
            
        if feature_model:
            self.register_model('feature', feature_model)
    
    def setup_connections(self):
        """
        Set up signal-slot connections between models and views.
        """
        # Get the models
        eye_tracking_data = self.get_model('eye_tracking_data')
        roi_model = self.get_model('roi')
        feature_model = self.get_model('feature')
        
        # Get the views
        plot_view = self.get_view('plot_view')
        gaze_view = self.get_view('gaze_view')
        roi_plot_view = self.get_view('roi_plot_view')
        status_bar = self.get_view('status_bar')
        
        # Connect eye tracking data signals to visualization updates
        if eye_tracking_data and plot_view:
            eye_tracking_data.data_processed.connect(
                lambda data: self.update_plots(data, plot_view)
            )
        
        if eye_tracking_data and gaze_view:
            eye_tracking_data.data_processed.connect(
                lambda data: self.update_gaze_plot(data, gaze_view)
            )
        
        # Connect ROI model signals to visualization updates
        if roi_model and roi_plot_view:
            roi_model.roi_analyzed.connect(
                lambda data: self.update_roi_visualization(data, roi_plot_view)
            )
        
        # Connect feature model signals to visualization updates if needed
        if feature_model and plot_view:
            feature_model.features_calculated.connect(
                lambda data: self.update_feature_plots(data, plot_view)
            )
        
        # Connect status updates to status bar if available
        if status_bar:
            if plot_view:
                # Setup connection for plot generation success/failure
                pass
    
    def update_plots(self, data, plot_view):
        """
        Update the plots based on processed eye tracking data.
        
        Args:
            data (dict): The processed eye tracking data
            plot_view: The view component that displays plots
            
        Returns:
            bool: True if plots were updated successfully, False otherwise
        """
        if hasattr(plot_view, 'update_plots'):
            return plot_view.update_plots(data)
        return False
    
    def update_gaze_plot(self, data, gaze_view):
        """
        Update the gaze plot based on processed eye tracking data.
        
        Args:
            data (dict): The processed eye tracking data
            gaze_view: The view component that displays gaze data
            
        Returns:
            bool: True if gaze plot was updated successfully, False otherwise
        """
        if hasattr(gaze_view, 'update_gaze_plot'):
            return gaze_view.update_gaze_plot(data)
        return False
    
    def update_roi_visualization(self, data, roi_plot_view):
        """
        Update the ROI visualization based on ROI analysis results.
        
        Args:
            data (dict): The ROI analysis results
            roi_plot_view: The view component that displays ROI visualizations
            
        Returns:
            bool: True if ROI visualization was updated successfully, False otherwise
        """
        if hasattr(roi_plot_view, 'update_roi_visualization'):
            return roi_plot_view.update_roi_visualization(data)
        return False
    
    def update_feature_plots(self, data, plot_view):
        """
        Update the feature plots based on calculated features.
        
        Args:
            data (pd.DataFrame): The calculated features
            plot_view: The view component that displays feature plots
            
        Returns:
            bool: True if feature plots were updated successfully, False otherwise
        """
        if hasattr(plot_view, 'update_feature_plots'):
            return plot_view.update_feature_plots(data)
        return False
    
    def generate_heatmap(self, fixation_data, image_path=None, width=None, height=None):
        """
        Generate a heatmap visualization of fixation data.
        
        Args:
            fixation_data (list): List of fixation data points
            image_path (str, optional): Path to background image for the heatmap
            width (int, optional): Width of the heatmap in pixels
            height (int, optional): Height of the heatmap in pixels
            
        Returns:
            object: The generated heatmap visualization or None if generation failed
        """
        plot_view = self.get_view('plot_view')
        
        if plot_view and hasattr(plot_view, 'generate_heatmap'):
            return plot_view.generate_heatmap(fixation_data, image_path, width, height)
        return None
    
    def generate_scanpath(self, fixation_data, saccade_data=None, image_path=None, width=None, height=None):
        """
        Generate a scanpath visualization of fixation and saccade data.
        
        Args:
            fixation_data (list): List of fixation data points
            saccade_data (list, optional): List of saccade data points
            image_path (str, optional): Path to background image for the scanpath
            width (int, optional): Width of the scanpath in pixels
            height (int, optional): Height of the scanpath in pixels
            
        Returns:
            object: The generated scanpath visualization or None if generation failed
        """
        plot_view = self.get_view('plot_view')
        
        if plot_view and hasattr(plot_view, 'generate_scanpath'):
            return plot_view.generate_scanpath(fixation_data, saccade_data, image_path, width, height)
        return None
    
    def export_visualization(self, viz_type, file_path, format='png', dpi=300):
        """
        Export a visualization to a file.
        
        Args:
            viz_type (str): Type of visualization to export ('heatmap', 'scanpath', etc.)
            file_path (str): Path to save the exported file
            format (str, optional): File format to use (png, pdf, svg, etc.)
            dpi (int, optional): Resolution for raster formats
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        plot_view = self.get_view('plot_view')
        
        if plot_view and hasattr(plot_view, 'export_visualization'):
            return plot_view.export_visualization(viz_type, file_path, format, dpi)
        return False
    
    def update_plot_settings(self, settings):
        """
        Update the settings for plot generation.
        
        Args:
            settings (dict): Dictionary of plot settings
            
        Returns:
            bool: True if settings were updated successfully, False otherwise
        """
        plot_view = self.get_view('plot_view')
        
        if plot_view and hasattr(plot_view, 'update_plot_settings'):
            return plot_view.update_plot_settings(settings)
        return False