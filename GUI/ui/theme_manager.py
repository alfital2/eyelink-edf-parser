"""
Theme Manager for UI

Provides theme management functionality with automatic light/dark mode detection
and appropriate stylesheets for the UI components.
"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import QObject, Qt


class ThemeManager(QObject):
    """
    A class to manage theme/appearance settings for Qt applications.
    Handles dark/light mode detection and provides appropriate stylesheets.
    """

    def __init__(self, parent=None):
        """Initialize the theme manager"""
        super().__init__(parent)

        # Check if dark mode is enabled by default
        self.is_dark_mode = self.is_dark_theme()

        # Install event filter to detect theme changes
        app = QApplication.instance()
        app.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Event filter to detect system theme changes"""
        if obj == QApplication.instance() and event.type() == event.ApplicationPaletteChange:
            # Theme has changed - update dark mode status
            new_dark_mode = self.is_dark_theme()
            if new_dark_mode != self.is_dark_mode:
                self.is_dark_mode = new_dark_mode
                # Notify parent if exists
                if self.parent():
                    if hasattr(self.parent(), 'refresh_theme'):
                        self.parent().refresh_theme()
        return super().eventFilter(obj, event)

    def is_dark_theme(self):
        """Detect if dark theme is active by checking background color"""
        app = QApplication.instance()
        palette = app.palette()
        background_color = palette.color(QPalette.Window)
        # If the background color is dark, assume dark theme
        return background_color.lightness() < 128

    def get_theme_style(self):
        """Get style sheet based on current theme"""
        if self.is_dark_mode:
            return """
            QGroupBox { 
                font-weight: bold; 
                font-size: 14px;
                color: #f0f0f0;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 16px;
                background-color: rgba(40, 40, 40, 150);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #333;
            }
            QTableWidget { 
                gridline-color: #555;
                background-color: rgba(60, 60, 60, 120);
                border: 1px solid #555;
                border-radius: 3px;
                alternate-background-color: rgba(70, 70, 70, 120);
            }
            QTableWidget::item {
                color: #f0f0f0;
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #2a82da;
            }
            QHeaderView::section {
                background-color: #444;
                color: #f0f0f0;
                border: 1px solid #555;
                padding: 4px;
                font-weight: bold;
            }
            QTextBrowser {
                background-color: rgba(60, 60, 60, 120);
                border: 1px solid #555;
                border-radius: 3px;
            }
            """
        else:
            return """
            QGroupBox { 
                font-weight: bold; 
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 16px;
                background-color: rgba(245, 245, 245, 150);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #f5f5f5;
            }
            QTableWidget { 
                gridline-color: #ccc;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 3px;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #0078d7;
                color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                padding: 4px;
                font-weight: bold;
            }
            QTextBrowser {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            """

    def get_text_color(self):
        """Get text color based on current theme"""
        return Qt.white if self.is_dark_mode else Qt.black

    def get_link_color(self):
        """Get hyperlink color based on current theme"""
        return "#58b0ff" if self.is_dark_mode else "#0078d7"

    def get_highlight_color(self):
        """Get highlight color based on current theme"""
        return Qt.cyan if self.is_dark_mode else Qt.blue

    def get_error_color(self):
        """Get error color based on current theme"""
        return Qt.red  # Red works for both light and dark modes