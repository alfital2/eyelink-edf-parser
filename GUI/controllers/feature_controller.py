"""
Feature Controller Module

This module contains the FeatureController class which manages the interaction
between the feature model and the feature table views in the GUI.
"""

from .base_controller import BaseController


class FeatureController(BaseController):
    """
    Controller for managing eye movement feature extraction and display.
    
    This controller coordinates the calculation and display of eye movement features,
    connecting the feature model with the UI components that display the features.
    """
    
    def __init__(self, feature_model=None, eye_tracking_data=None):
        """
        Initialize the feature controller.
        
        Args:
            feature_model (FeatureModel, optional): The feature model
            eye_tracking_data (EyeTrackingData, optional): The eye tracking data model
        """
        super().__init__()
        
        # Register models if provided
        if feature_model:
            self.register_model('feature', feature_model)
        
        if eye_tracking_data:
            self.register_model('eye_tracking_data', eye_tracking_data)
    
    def setup_connections(self):
        """
        Set up signal-slot connections between models and views.
        """
        # Get the models
        feature_model = self.get_model('feature')
        eye_tracking_data = self.get_model('eye_tracking_data')
        
        # Get the views
        feature_table_view = self.get_view('feature_table_view')
        status_bar = self.get_view('status_bar')
        
        # Connect feature model signals to view slots if available
        if feature_model and feature_table_view:
            # Connect features_calculated signal to view update method
            feature_model.features_calculated.connect(
                lambda data: feature_table_view.update_feature_tables(data)
            )
            # Connect features_updated signal to view update method
            feature_model.features_updated.connect(
                lambda data: feature_table_view.update_feature_tables(data)
            )
        
        # Connect eye tracking data signals to trigger feature calculation
        if eye_tracking_data and feature_model:
            eye_tracking_data.data_processed.connect(
                lambda data: self.calculate_features(eye_tracking_data)
            )
        
        # Connect status updates to status bar if available
        if feature_model and status_bar:
            feature_model.features_calculated.connect(
                lambda data: status_bar.showMessage("Features calculated successfully", 3000)
            )
            feature_model.features_updated.connect(
                lambda data: status_bar.showMessage("Features updated successfully", 3000)
            )
    
    def calculate_features(self, eye_tracking_data=None):
        """
        Calculate features from eye tracking data.
        
        Args:
            eye_tracking_data (EyeTrackingData, optional): The eye tracking data model.
                If None, uses the model registered with the controller.
            
        Returns:
            pd.DataFrame: The calculated features or None if calculation failed
        """
        feature_model = self.get_model('feature')
        
        if not eye_tracking_data:
            eye_tracking_data = self.get_model('eye_tracking_data')
        
        if feature_model and eye_tracking_data:
            return feature_model.calculate_features(eye_tracking_data)
        
        return None
    
    def update_features(self, additional_features):
        """
        Update the features with additional features.
        
        Args:
            additional_features (dict): Dictionary of additional features to add
            
        Returns:
            pd.DataFrame: The updated features or None if update failed
        """
        feature_model = self.get_model('feature')
        
        if feature_model:
            return feature_model.update_features(additional_features)
        
        return None
    
    def get_features(self):
        """
        Get the current calculated features.
        
        Returns:
            pd.DataFrame: The features or None if no features have been calculated
        """
        feature_model = self.get_model('feature')
        
        if feature_model:
            return feature_model.get_features()
        
        return None
    
    def load_feature_categories(self, feature_categories):
        """
        Load feature categories structure from an external source.
        
        Args:
            feature_categories (list): Feature categories definition
            
        Returns:
            bool: True if categories were loaded successfully, False otherwise
        """
        feature_model = self.get_model('feature')
        
        if feature_model and hasattr(feature_model, 'load_feature_categories'):
            feature_model.load_feature_categories(feature_categories)
            return True
        
        return False
    
    def get_feature_categories(self):
        """
        Get the current feature categories structure.
        
        Returns:
            list: The feature categories structure or None if not available
        """
        feature_model = self.get_model('feature')
        
        if feature_model and hasattr(feature_model, 'get_feature_categories'):
            return feature_model.get_feature_categories()
        
        return None
    
    def get_feature_explanations(self):
        """
        Get the explanations for each feature.
        
        Returns:
            dict: Dictionary mapping feature keys to their explanations or None if not available
        """
        feature_model = self.get_model('feature')
        
        if feature_model and hasattr(feature_model, 'feature_explanations'):
            return feature_model.feature_explanations
        
        return None
    
    def export_features(self, file_path, format='csv'):
        """
        Export the calculated features to a file.
        
        Args:
            file_path (str): Path to save the exported file
            format (str, optional): File format to use (csv, xlsx, etc.)
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        feature_model = self.get_model('feature')
        
        if not feature_model:
            return False
        
        features_df = feature_model.get_features()
        
        if features_df is None or features_df.empty:
            return False
        
        try:
            if format.lower() == 'csv':
                features_df.to_csv(file_path, index=False)
            elif format.lower() == 'xlsx':
                features_df.to_excel(file_path, index=False)
            else:
                return False
            
            return True
        except Exception:
            return False