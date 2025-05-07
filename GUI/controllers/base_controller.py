"""
Base Controller Module

This module contains the BaseController class which provides common functionality
for all controllers in the Eye Movement Analysis application.
"""

class BaseController:
    """
    Base controller class that all controllers inherit from.
    
    This class provides common functionality and a standard interface
    for all controllers in the application.
    
    Attributes:
        models (dict): Dictionary of model instances accessible to the controller
        views (dict): Dictionary of view instances accessible to the controller
    """
    
    def __init__(self):
        """
        Initialize the controller with empty model and view dictionaries.
        """
        self.models = {}
        self.views = {}
    
    def register_model(self, name, model):
        """
        Register a model with the controller.
        
        Args:
            name (str): A name to identify the model
            model (object): The model instance to register
        """
        self.models[name] = model
    
    def register_view(self, name, view):
        """
        Register a view with the controller.
        
        Args:
            name (str): A name to identify the view
            view (object): The view instance to register
        """
        self.views[name] = view
    
    def get_model(self, name):
        """
        Get a registered model by name.
        
        Args:
            name (str): The name of the model to retrieve
            
        Returns:
            object: The requested model or None if not found
        """
        return self.models.get(name)
    
    def get_view(self, name):
        """
        Get a registered view by name.
        
        Args:
            name (str): The name of the view to retrieve
            
        Returns:
            object: The requested view or None if not found
        """
        return self.views.get(name)
    
    def setup_connections(self):
        """
        Set up signal-slot connections between models and views.
        
        This method should be implemented by subclasses to establish
        the appropriate connections between models and views.
        """
        pass