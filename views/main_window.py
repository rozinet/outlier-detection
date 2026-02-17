"""Main application window."""

import tkinter as tk
from tkinter import ttk, messagebox
from .building_panel import BuildingPanel
from .settings_panel import SettingsPanel
from .results_panel import ResultsPanel


class MainWindow:
    """Main application window with tabbed interface."""

    def __init__(self, controller):
        """Initialize main window.

        Args:
            controller: AppController instance
        """
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Building Sensor Outlier Detection System")
        self.root.geometry("1200x800")

        # Configure styles
        self.setup_styles()

        # Create menu bar
        self.create_menu()

        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.building_panel = BuildingPanel(
            self.notebook,
            self.controller.get_building_controller()
        )
        self.notebook.add(self.building_panel.frame, text="Buildings & Sensors")

        self.settings_panel = SettingsPanel(
            self.notebook,
            self.controller.get_config()
        )
        self.notebook.add(self.settings_panel.frame, text="Settings")

        self.results_panel = ResultsPanel(
            self.notebook,
            self.controller.get_detection_controller(),
            self.controller.get_building_controller()
        )
        self.notebook.add(self.results_panel.frame, text="Detection Results")

        # Status bar
        self.create_status_bar(main_container)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))

    def create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save All", command=self.save_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_status_bar(self, parent):
        """Create status bar.

        Args:
            parent: Parent widget
        """
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)

    def set_status(self, message: str):
        """Update status bar message.

        Args:
            message: Status message
        """
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def save_all(self):
        """Save all application data."""
        try:
            self.controller.save_all()
            self.set_status("All data saved successfully")
            messagebox.showinfo("Save", "All data saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {e}")

    def show_about(self):
        """Show about dialog."""
        about_text = """Building Sensor Outlier Detection System

Version: 6.0

This application detects building pathologies from sensor data:
• Moisture Intrusion
• Condensation Risk
• Drying Failure
• Sensor Malfunction
• Rapid Moisture Change

© 2026 Rozinet"""

        messagebox.showinfo("About", about_text)

    def on_close(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to save before quitting?"):
            self.save_all()
        self.root.destroy()

    def run(self):
        """Start the application main loop."""
        self.root.mainloop()
