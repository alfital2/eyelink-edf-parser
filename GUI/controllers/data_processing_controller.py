"""
Data Processing Controller Module

This module contains the DataProcessingController class which manages the interaction
between the eye tracking data models and the UI components that display that data.
"""

from .base_controller import BaseController


class DataProcessingController(BaseController):
    """
    Controller for managing eye tracking data processing.
    
    This controller coordinates the loading, processing, and analysis of
    eye tracking data, connecting the data models with the UI components
    that display and interact with the data.
    """
    
    def __init__(self, eye_tracking_data=None, roi_model=None):
        """
        Initialize the data processing controller.
        
        Args:
            eye_tracking_data (EyeTrackingData, optional): The eye tracking data model
            roi_model (ROIModel, optional): The ROI data model
        """
        super().__init__()
        
        # Register models if provided
        if eye_tracking_data:
            self.register_model('eye_tracking_data', eye_tracking_data)
        
        if roi_model:
            self.register_model('roi', roi_model)
    
    def setup_connections(self):
        """
        Set up signal-slot connections between models and views.
        """
        # Get the models
        eye_tracking_data = self.get_model('eye_tracking_data')
        roi_model = self.get_model('roi')
        
        # Get the views
        data_view = self.get_view('data_view')
        roi_view = self.get_view('roi_view')
        status_bar = self.get_view('status_bar')
        
        # Connect eye tracking data signals to view slots if available
        if eye_tracking_data and data_view:
            # Connect data_loaded signal to view update method
            eye_tracking_data.data_loaded.connect(data_view.update_data_display)
            # Connect data_processed signal to view update method
            eye_tracking_data.data_processed.connect(data_view.update_processed_data)
        
        # Connect ROI model signals to view slots if available
        if roi_model and roi_view:
            # Connect roi_loaded signal to view update method
            roi_model.roi_loaded.connect(roi_view.update_roi_display)
            # Connect roi_analyzed signal to view update method
            roi_model.roi_analyzed.connect(roi_view.update_roi_analysis)
        
        # Connect status updates to status bar if available
        if eye_tracking_data and status_bar:
            eye_tracking_data.data_loaded.connect(
                lambda data: status_bar.showMessage("Eye tracking data loaded successfully", 3000)
            )
            eye_tracking_data.data_processed.connect(
                lambda data: status_bar.showMessage("Eye tracking data processed successfully", 3000)
            )
        
        if roi_model and status_bar:
            roi_model.roi_loaded.connect(
                lambda data: status_bar.showMessage("ROI data loaded successfully", 3000)
            )
            roi_model.roi_analyzed.connect(
                lambda data: status_bar.showMessage("ROI analysis completed successfully", 3000)
            )
    
    def load_data(self, file_path):
        """
        Load eye tracking data from a file.
        
        Args:
            file_path (str): Path to the eye tracking data file
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        eye_tracking_data = self.get_model('eye_tracking_data')
        
        if eye_tracking_data:
            success = eye_tracking_data.load_data(file_path)
            
            if success:
                # If ROI model is available, analyze fixations in ROIs
                roi_model = self.get_model('roi')
                if roi_model and hasattr(roi_model, 'analyze_fixations'):
                    fixations = eye_tracking_data.get_fixations()
                    if fixations:
                        roi_model.analyze_fixations(fixations)
            
            return success
        
        return False
    
    def process_data(self, processing_params=None):
        """
        Process the loaded eye tracking data with optional parameters.
        
        Args:
            processing_params (dict, optional): Parameters for data processing
            
        Returns:
            bool: True if data was processed successfully, False otherwise
        """
        eye_tracking_data = self.get_model('eye_tracking_data')
        
        if eye_tracking_data:
            result = eye_tracking_data.process_data(processing_params)
            return result is not None
        
        return False
    
    def load_roi_data(self, file_path):
        """
        Load ROI data from a file.
        
        Args:
            file_path (str): Path to the ROI data file
            
        Returns:
            bool: True if ROI data was loaded successfully, False otherwise
        """
        roi_model = self.get_model('roi')
        
        if roi_model:
            success = roi_model.load_roi_data(file_path)
            
            if success:
                # If eye tracking data is available with fixations, analyze them with the new ROIs
                eye_tracking_data = self.get_model('eye_tracking_data')
                if eye_tracking_data and hasattr(eye_tracking_data, 'get_fixations'):
                    fixations = eye_tracking_data.get_fixations()
                    if fixations:
                        roi_model.analyze_fixations(fixations)
            
            return success
        
        return False
    
    def get_processing_results(self):
        """
        Get the results of the data processing.
        
        Returns:
            dict: The processed data results or None if no processing has been done
        """
        eye_tracking_data = self.get_model('eye_tracking_data')
        
        if eye_tracking_data and hasattr(eye_tracking_data, 'get_processed_data'):
            return eye_tracking_data.get_processed_data()
        
        return None
    
    def get_roi_analysis_results(self):
        """
        Get the results of the ROI analysis.
        
        Returns:
            dict: The ROI analysis results or None if no analysis has been done
        """
        roi_model = self.get_model('roi')
        
        if roi_model and hasattr(roi_model, 'get_analysis_results'):
            return roi_model.get_analysis_results()
        
        return None