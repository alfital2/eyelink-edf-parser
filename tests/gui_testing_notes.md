# GUI Testing Notes

## Dialog Mocking in Automated Tests

When testing GUI components that use dialogs in PyQt, proper mocking is essential to avoid having real dialog windows appear during automated testing. This document outlines the approach used in this project to mock dialog windows.

### Best Practices for Mocking Dialog Windows

1. **Direct Module-Level Mocking**: Instead of using the `@patch` decorator which can sometimes be unreliable with PyQt, directly replace the methods at the module level during tests:

```python
# Save original method
original_getOpenFileNames = QFileDialog.getOpenFileNames

try:
    # Replace with mock
    QFileDialog.getOpenFileNames = MagicMock(return_value=([file_path], "Filter"))
    
    # Call the method that uses QFileDialog
    self.gui.select_files()
    
    # Test assertions
    self.assertEqual(self.gui.file_paths, [file_path])
finally:
    # Restore original method
    QFileDialog.getOpenFileNames = original_getOpenFileNames
```

2. **Always Restore Original Methods**: Use a try-finally block to ensure the original methods are restored even if the test fails.

3. **Import All Dialog Classes Explicitly**: Make sure to import all dialog-related classes directly:

```python
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
```

### Dialog Types to Mock

#### File Dialogs (QFileDialog)

These are the QFileDialog methods commonly used in the application:

- `QFileDialog.getOpenFileNames()` - Used for selecting multiple input files
- `QFileDialog.getOpenFileName()` - Used for selecting single files (e.g., ROI files)
- `QFileDialog.getExistingDirectory()` - Used for selecting output directories
- `QFileDialog.getSaveFileName()` - Used for saving files

#### Message Boxes (QMessageBox)

These QMessageBox methods should be mocked to avoid popup windows:

- `QMessageBox.information()` - Used for information messages
- `QMessageBox.warning()` - Used for warning messages
- `QMessageBox.critical()` - Used for error messages
- `QMessageBox.question()` - Used for confirmation dialogs

#### Other Dialog Types

Other dialog types that might need mocking:

- `QInputDialog` methods for text and number input
- `QColorDialog` for color selection
- `QFontDialog` for font selection
- `QErrorMessage` for error messages

### Example Implementations

#### Example 1: Mocking File Selection

```python
def test_file_selection(self):
    """Test file selection dialog."""
    # Directly mock QFileDialog.getOpenFileNames
    original_getOpenFileNames = QFileDialog.getOpenFileNames
    
    try:
        # Replace with mock returning predefined values
        QFileDialog.getOpenFileNames = MagicMock(return_value=([self.test_file], "Filter"))
        
        # Call method that triggers file dialog
        self.gui.select_files()
        
        # Assertions
        self.assertEqual(self.gui.file_paths, [self.test_file])
    finally:
        # Restore original method
        QFileDialog.getOpenFileNames = original_getOpenFileNames
```

#### Example 2: Mocking Message Boxes

```python
def test_error_message(self):
    """Test error message box."""
    # Save original method
    original_critical = QMessageBox.critical
    
    try:
        # Replace with mock
        QMessageBox.critical = MagicMock()
        
        # Call method that triggers error dialog
        self.gui.show_error("Test Error")
        
        # Verify dialog was shown with correct parameters
        QMessageBox.critical.assert_called_once()
        args = QMessageBox.critical.call_args[0]
        self.assertEqual(args[1], "Error")  # Title
        self.assertEqual(args[2], "Test Error")  # Message
    finally:
        # Restore original method
        QMessageBox.critical = original_critical
```

#### Example 3: Mocking External App Launching

```python
def test_open_external_app(self):
    """Test launching external application."""
    # Import and mock webbrowser
    import webbrowser
    original_open = webbrowser.open
    
    try:
        # Replace with mock
        webbrowser.open = MagicMock()
        
        # Call method that launches browser
        self.gui.open_report()
        
        # Verify browser was launched
        webbrowser.open.assert_called_once()
    finally:
        # Restore original method
        webbrowser.open = original_open
```

By following these practices, tests can be run non-interactively without dialog boxes appearing, enabling automated testing.