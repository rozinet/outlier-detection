"""
Building Sensor Outlier Detection System - Main Entry Point

This application provides a GUI for managing buildings, rooms, and sensors,
and running outlier detection on sensor data.

Architecture: MVC (Model-View-Controller)
  - Models: Data structures, configuration, detection logic
  - Views: UI components (tkinter)
  - Controllers: Business logic orchestration
"""

import sys
import os

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from controllers import AppController
from views import MainWindow


def main():
    """Main entry point for the application."""
    # Initialize application controller
    app_controller = AppController()

    # Create and run main window
    main_window = MainWindow(app_controller)
    main_window.run()


if __name__ == "__main__":
    main()
