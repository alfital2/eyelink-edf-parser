from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QGroupBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QScrollArea, QSizePolicy, QToolTip)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
import pandas as pd

from feature_definitions import FEATURE_CATEGORIES


class FeatureTableManager:
    """Manages the creation and updates of feature tables in the GUI"""
    
    def __init__(self, parent, theme_manager, feature_explanations):
        """
        Initialize the table manager
        
        Args:
            parent: The parent widget that will display the tables
            theme_manager: The theme manager instance
            feature_explanations: Dictionary of feature explanations for tooltips
        """
        self.parent = parent
        self.theme_manager = theme_manager
        self.feature_explanations = feature_explanations
        self.feature_tables = {}
    
    def create_feature_tables(self, parent_layout):
        """Create organized tables for different categories of features in a grid layout"""
        # Create a scrollable widget for all tables
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        
        # Use a grid layout instead of vertical layout for better space utilization
        features_grid_layout = QGridLayout(scroll_content)
        features_grid_layout.setSpacing(15)  # Add more spacing between feature groups
        features_grid_layout.setContentsMargins(15, 15, 15, 15)  # Add margins around the grid

        # Fix the background color for light mode
        if not self.theme_manager.is_dark_mode:
            scroll_content.setStyleSheet("background-color: #f5f5f5;")

        # Define categories with their layouts
        categories = self._define_feature_categories()

        # Create a table for each category
        self.feature_tables = {}
        
        # Track all original feature keys for tooltip lookup
        all_feature_keys = {}
        
        # Process each category
        for category_info in categories:
            category_name = category_info[0]
            feature_data = category_info[1]
            row_pos = category_info[2]
            col_pos = category_info[3]
            
            # Create table for this category
            group_box, feature_keys = self._create_category_table(
                category_name, feature_data, row_pos, col_pos
            )
            
            # Store feature keys for tooltips
            all_feature_keys[category_name] = feature_keys
            
            # Add to grid layout at specified position
            features_grid_layout.addWidget(group_box, row_pos, col_pos)

        # Set column and row stretch factors to distribute space evenly
        for i in range(3):  # 3 columns
            features_grid_layout.setColumnStretch(i, 1)
        for i in range(3):  # 3 rows
            features_grid_layout.setRowStretch(i, 1)
        
        scroll_area.setWidget(scroll_content)
        parent_layout.addWidget(scroll_area)
        
        # Return the all_feature_keys dictionary for tooltip handling
        return all_feature_keys
    
    def _define_feature_categories(self):
        """Define the categories of features to display"""
        return FEATURE_CATEGORIES
    
    def _create_category_table(self, category_name, feature_data, row_pos, col_pos):
        """Create a table for a specific category of features"""
        # Create a group box for the category
        group_box = QGroupBox(category_name)
        group_layout = QVBoxLayout(group_box)
        group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Determine if this is a combined left/right table or a regular table
        is_combined = isinstance(feature_data[0], dict) and "left" in feature_data[0]
        
        # Create the appropriate table based on the data structure
        table, feature_keys = self._create_table_by_type(category_name, feature_data, is_combined)
        
        # Common table settings
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)  # Enable alternating row colors
        table.setSelectionMode(QTableWidget.SingleSelection)  # Allow selecting entire rows
        table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Additional styling for light mode
        if not self.theme_manager.is_dark_mode:
            table.setStyleSheet("QTableWidget { background-color: white; border: 1px solid #ddd; }")
        
        # Enable tooltips for the table
        table.setMouseTracking(True)
        table.cellEntered.connect(lambda row, col, t=table, cat=category_name: 
                                 self.show_feature_tooltip(row, col, t, feature_keys))
        
        # Store the table configuration
        self.feature_tables[category_name] = {
            "table": table,
            "features": feature_data,
            "is_combined": is_combined
        }
        
        # Add table to the group box
        group_layout.addWidget(table)
        
        return group_box, feature_keys
    
    def _create_table_by_type(self, category_name, feature_data, is_combined):
        """Create the appropriate table based on data structure"""
        feature_keys = []
        
        if is_combined:
            # Combined left/right table with 3 columns
            table = QTableWidget(0, 3)  # Metric, Left Eye, Right Eye
            table.setHorizontalHeaderLabels(["Metric", "Left Eye", "Right Eye"])
            
            # Equal column widths
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.Stretch)
            
            # Collect all the original feature keys for tooltips
            for item in feature_data:
                if "left" in item:
                    feature_keys.append(item["left"])
                if "right" in item:
                    feature_keys.append(item["right"])
            
        elif len(feature_data) > 0 and isinstance(feature_data[0], dict) and "key" in feature_data[0]:
            # Non-combined table with single value column
            table = QTableWidget(0, 2)  # Metric, Value
            table.setHorizontalHeaderLabels(["Metric", "Value"])
            
            # Equal column widths
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            
            # Collect feature keys for tooltips
            feature_keys = [item["key"] for item in feature_data]
            
        else:
            # Regular table (e.g., Basic Information)
            table = QTableWidget(0, 2)  # Feature, Value
            table.setHorizontalHeaderLabels(["Feature", "Value"])
            
            # Equal column widths
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            
            feature_keys = feature_data
        
        return table, feature_keys
    
    def update_feature_tables(self, features_df):
        """Update all feature tables with data from the features DataFrame"""
        if features_df is None or features_df.empty:
            return

        # For each category table, update the values
        for category_name, table_info in self.feature_tables.items():
            table = table_info["table"]
            feature_data = table_info["features"]
            is_combined = table_info.get("is_combined", False)

            # Clear the table
            table.setRowCount(0)

            # Handle different table types
            if is_combined:
                # This is a combined left/right eye table with 3 columns
                self._update_combined_table(table, feature_data, features_df)
            elif len(feature_data) > 0 and isinstance(feature_data[0], dict) and "key" in feature_data[0]:
                # This is a regular table with named metrics
                self._update_named_table(table, feature_data, features_df)
            else:
                # This is a simple table with direct feature keys
                self._update_simple_table(table, feature_data, features_df)
    
    def _update_combined_table(self, table, feature_data, features_df):
        """Update a table with left/right eye metrics in separate columns"""
        for i, item in enumerate(feature_data):
            # Skip if either left or right key is missing from the feature data
            if not all(key in features_df.columns for key in [item["left"], item["right"]]):
                continue
                
            row_position = table.rowCount()
            table.insertRow(row_position)
            
            # Create items
            name_item = QTableWidgetItem(item["name"])
            
            # Set tooltips with explanation if available
            tooltips = []
            if item["left"] in self.feature_explanations:
                tooltips.append(self.feature_explanations[item["left"]])
            if item["right"] in self.feature_explanations:
                tooltips.append(self.feature_explanations[item["right"]])
                
            if tooltips:
                name_item.setToolTip("\n\n".join(tooltips))
                
            # Get and format left eye value
            left_value = features_df[item["left"]].iloc[0]
            left_value_text = self._format_value(left_value)
            left_item = QTableWidgetItem(left_value_text)
            
            # Get and format right eye value
            right_value = features_df[item["right"]].iloc[0]
            right_value_text = self._format_value(right_value)
            right_item = QTableWidgetItem(right_value_text)
            
            # Add items to table
            table.setItem(row_position, 0, name_item)
            table.setItem(row_position, 1, left_item)
            table.setItem(row_position, 2, right_item)
    
    def _update_named_table(self, table, feature_data, features_df):
        """Update a table with named metrics (non-left/right)"""
        for i, item in enumerate(feature_data):
            if item["key"] not in features_df.columns:
                continue
                
            row_position = table.rowCount()
            table.insertRow(row_position)
            
            # Create items
            name_item = QTableWidgetItem(item["name"])
            
            # Set tooltip with explanation if available
            if item["key"] in self.feature_explanations:
                name_item.setToolTip(self.feature_explanations[item["key"]])
                
            # Get and format value
            value = features_df[item["key"]].iloc[0]
            value_text = self._format_value(value)
            value_item = QTableWidgetItem(value_text)
            
            # Add items to table
            table.setItem(row_position, 0, name_item)
            table.setItem(row_position, 1, value_item)
    
    def _update_simple_table(self, table, feature_keys, features_df):
        """Update a simple table with direct feature keys"""
        for i, feature_key in enumerate(feature_keys):
            if feature_key not in features_df.columns:
                continue
                
            row_position = table.rowCount()
            table.insertRow(row_position)
            
            # Format the feature name to be more readable
            display_name = feature_key.replace('_', ' ').title()
            
            # Create items
            name_item = QTableWidgetItem(display_name)
            
            # Set tooltip with explanation if available
            if feature_key in self.feature_explanations:
                name_item.setToolTip(self.feature_explanations[feature_key])
                
            # Get and format value
            value = features_df[feature_key].iloc[0]
            value_text = self._format_value(value)
            value_item = QTableWidgetItem(value_text)
            
            # Add items to table
            table.setItem(row_position, 0, name_item)
            table.setItem(row_position, 1, value_item)
    
    def _format_value(self, value):
        """Format a value for display in the table"""
        # Handle NaN values properly
        if pd.isna(value):
            return "N/A"
        elif isinstance(value, (int, float)):
            # Format number with appropriate precision
            try:
                if float(value).is_integer():
                    return str(int(value))
                else:
                    return f"{value:.4f}"
            except:
                # If conversion fails, use the value as is
                return str(value)
        else:
            return str(value)
    
    def show_feature_tooltip(self, row, col, table, features):
        """Show a tooltip with feature explanation when hovering over a feature name"""
        # Only show tooltips for feature names column (first column)
        if col == 0 and row < table.rowCount():
            # Get the tooltip directly from the table item if it exists
            cell_item = table.item(row, col)
            if cell_item and cell_item.toolTip():
                QToolTip.showText(QCursor.pos(), cell_item.toolTip())
            # If no tooltip in the item, try to find it from the feature list
            elif features and row < len(features):
                # Handle different feature list formats
                if isinstance(features, list):
                    if row < len(features):
                        # Get the feature key based on the type of item in the list
                        if isinstance(features[row], dict):
                            # If the features are dictionaries with keys
                            if "key" in features[row]:
                                feature_key = features[row]["key"]
                            elif "left" in features[row]:
                                # For left/right pairs, show tooltip for the left key
                                feature_key = features[row]["left"]
                            else:
                                return
                        else:
                            # Simple string keys
                            feature_key = features[row]
                        
                        # Show the tooltip if we have an explanation
                        if feature_key in self.feature_explanations:
                            explanation = self.feature_explanations[feature_key]
                            QToolTip.showText(QCursor.pos(), explanation)