"""Settings configuration panel."""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json


class SettingsPanel:
    """Panel for configuring detection parameters."""

    def __init__(self, parent, config_manager):
        """Initialize settings panel.

        Args:
            parent: Parent widget
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.frame = ttk.Frame(parent)
        self.entries = {}

        self.create_layout()
        self.load_settings()

    def create_layout(self):
        """Create the panel layout."""
        # Header
        header = ttk.Label(self.frame, text="Detection Settings",
                          style='Heading.TLabel')
        header.pack(pady=10)

        # Buttons frame
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset to Defaults", command=self.reset_to_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export Config", command=self.export_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Import Config", command=self.import_config).pack(side=tk.LEFT, padx=5)

        # Create notebook for different setting categories
        notebook = ttk.Notebook(self.frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # General settings tab
        general_container, general_frame = self.create_scrollable_frame(notebook)
        notebook.add(general_container, text="General")
        self.create_general_settings(general_frame)

        # Moisture intrusion tab
        moisture_container, moisture_frame = self.create_scrollable_frame(notebook)
        notebook.add(moisture_container, text="Moisture Intrusion")
        self.create_moisture_settings(moisture_frame)

        # Condensation tab
        condensation_container, condensation_frame = self.create_scrollable_frame(notebook)
        notebook.add(condensation_container, text="Condensation")
        self.create_condensation_settings(condensation_frame)

        # Sensor malfunction tab
        sensor_container, sensor_frame = self.create_scrollable_frame(notebook)
        notebook.add(sensor_container, text="Sensor Malfunction")
        self.create_sensor_settings(sensor_frame)

        # Advanced tab
        advanced_container, advanced_frame = self.create_scrollable_frame(notebook)
        notebook.add(advanced_container, text="Advanced")
        self.create_advanced_settings(advanced_frame)

    def create_scrollable_frame(self, parent):
        """Create a scrollable frame.

        Args:
            parent: Parent widget

        Returns:
            Tuple of (container_frame, scrollable_frame)
        """
        # Create a container frame to hold canvas and scrollbar
        container = ttk.Frame(parent)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return container, scrollable_frame

    def add_setting_entry(self, parent, row, key, label, tooltip=""):
        """Add a setting entry field.

        Args:
            parent: Parent widget
            row: Grid row
            key: Configuration key
            label: Label text
            tooltip: Tooltip text
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=10, pady=5)

        entry = ttk.Entry(parent, width=20)
        entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self.entries[key] = entry

        if tooltip:
            tooltip_btn = ttk.Label(parent, text="?", foreground="blue")
            tooltip_btn.grid(row=row, column=2, padx=5)
            self.create_tooltip(tooltip_btn, tooltip)

    def create_tooltip(self, widget, text):
        """Create tooltip for widget.

        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text
        """
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="lightyellow",
                           relief=tk.SOLID, borderwidth=1, font=("Arial", 9))
            label.pack()
            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def create_general_settings(self, parent):
        """Create general settings section.

        Args:
            parent: Parent widget
        """
        self.add_setting_entry(parent, 0, "resample_interval", "Resample Interval:",
                              "Time interval for data resampling (e.g., 5min)")
        self.add_setting_entry(parent, 1, "min_data_hours", "Min Data Hours:",
                              "Minimum hours of data required for analysis")
        self.add_setting_entry(parent, 2, "median_filter_window", "Median Filter Window:",
                              "Window size for median filter (samples)")
        self.add_setting_entry(parent, 3, "smoothing_window", "Smoothing Window:",
                              "Window size for smoothing (samples)")
        self.add_setting_entry(parent, 4, "ewma_halflife_hours", "EWMA Half-life (hours):",
                              "Half-life for exponential weighted moving average")

    def create_moisture_settings(self, parent):
        """Create moisture intrusion settings section.

        Args:
            parent: Parent widget
        """
        self.add_setting_entry(parent, 0, "moisture_drop_threshold_24h", "24h Drop Threshold (%):",
                              "Moisture resistance drop threshold in 24 hours")
        self.add_setting_entry(parent, 1, "moisture_drop_threshold_7d", "7d Drop Threshold (%):",
                              "Moisture resistance drop threshold in 7 days")
        self.add_setting_entry(parent, 2, "cavity_rise_threshold_24h", "24h Cavity Rise (%):",
                              "Cavity humidity rise threshold in 24 hours")
        self.add_setting_entry(parent, 3, "moisture_intrusion_min_hours", "Min Duration (hours):",
                              "Minimum episode duration")
        self.add_setting_entry(parent, 4, "moisture_drop_min", "Min Drop (%):",
                              "Minimum drop to consider meaningful")
        self.add_setting_entry(parent, 5, "cavity_rise_min", "Min Cavity Rise (%):",
                              "Minimum cavity rise to consider meaningful")

    def create_condensation_settings(self, parent):
        """Create condensation settings section.

        Args:
            parent: Parent widget
        """
        self.add_setting_entry(parent, 0, "condensation_warning_pct", "Warning Threshold (%):",
                              "Cavity humidity % for WARNING level")
        self.add_setting_entry(parent, 1, "condensation_danger_pct", "Danger Threshold (%):",
                              "Cavity humidity % for DANGER level")
        self.add_setting_entry(parent, 2, "condensation_critical_pct", "Critical Threshold (%):",
                              "Cavity humidity % for CRITICAL level")
        self.add_setting_entry(parent, 3, "condensation_min_hours", "Min Duration (hours):",
                              "Minimum episode duration")
        self.add_setting_entry(parent, 4, "condensation_chronic_pct_warning", "Chronic % (WARNING):",
                              "% time above threshold for chronic WARNING")
        self.add_setting_entry(parent, 5, "condensation_chronic_pct_severe", "Chronic % (SEVERE):",
                              "% time above threshold for chronic DANGER/CRITICAL")
        self.add_setting_entry(parent, 6, "condensation_recurring_min_episodes", "Recurring Episodes:",
                              "Min episodes in 12mo for recurring pattern")
        self.add_setting_entry(parent, 7, "abs_humidity_warning_gkg", "Abs Humidity Warning (g/kg):",
                              "Absolute humidity WARNING threshold")

    def create_sensor_settings(self, parent):
        """Create sensor malfunction settings section.

        Args:
            parent: Parent widget
        """
        self.add_setting_entry(parent, 0, "flatline_window_hours", "Flatline Window (hours):",
                              "Window for flatline detection")
        self.add_setting_entry(parent, 1, "jump_threshold_temp", "Temp Jump Threshold (°C):",
                              "Temperature jump threshold")
        self.add_setting_entry(parent, 2, "jump_threshold_humidity", "Humidity Jump Threshold (%):",
                              "Humidity jump threshold")
        self.add_setting_entry(parent, 3, "jump_threshold_moisture", "Moisture Jump Threshold (%):",
                              "Moisture jump threshold")
        self.add_setting_entry(parent, 4, "jump_min_count", "Min Jump Count:",
                              "Minimum number of jumps to flag")
        self.add_setting_entry(parent, 5, "hampel_window", "Hampel Window:",
                              "Hampel filter window size")
        self.add_setting_entry(parent, 6, "hampel_threshold", "Hampel Threshold:",
                              "Hampel filter MAD multiplier")

    def create_advanced_settings(self, parent):
        """Create advanced settings section.

        Args:
            parent: Parent widget
        """
        self.add_setting_entry(parent, 0, "cusum_threshold", "CUSUM Threshold:",
                              "CUSUM alarm threshold")
        self.add_setting_entry(parent, 1, "cusum_drift", "CUSUM Drift:",
                              "CUSUM drift parameter")
        self.add_setting_entry(parent, 2, "drying_tau_warning_days", "Drying τ WARNING (days):",
                              "Exponential time constant for slow drying warning")
        self.add_setting_entry(parent, 3, "drying_tau_danger_days", "Drying τ DANGER (days):",
                              "Exponential time constant for very slow drying")
        self.add_setting_entry(parent, 4, "outlier_mad_zscore_threshold", "Outlier MAD Z-score:",
                              "MAD z-score threshold for outlier detection")
        self.add_setting_entry(parent, 5, "seasonal_baseline_window_days", "Seasonal Window (days):",
                              "Rolling window for seasonal baseline")

    def load_settings(self):
        """Load current settings into entry fields."""
        for key, entry in self.entries.items():
            value = self.config_manager.get(key)
            if value is not None:
                entry.delete(0, tk.END)
                entry.insert(0, str(value))

    def save_settings(self):
        """Save settings from entry fields."""
        try:
            updates = {}
            for key, entry in self.entries.items():
                value_str = entry.get().strip()
                if not value_str:
                    continue

                # Get original value to determine type
                original = self.config_manager.get(key)
                if original is not None:
                    if isinstance(original, bool):
                        value = value_str.lower() in ('true', '1', 'yes')
                    elif isinstance(original, int):
                        value = int(value_str)
                    elif isinstance(original, float):
                        value = float(value_str)
                    else:
                        value = value_str
                else:
                    # Try to infer type
                    try:
                        value = float(value_str)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        value = value_str

                updates[key] = value

            self.config_manager.update(updates)
            self.config_manager.save()
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        if messagebox.askyesno("Confirm", "Reset all settings to default values?"):
            self.config_manager.reset_to_defaults()
            self.load_settings()
            messagebox.showinfo("Success", "Settings reset to defaults")

    def export_config(self):
        """Export configuration to JSON file."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.config_manager.config, f, indent=2)
                messagebox.showinfo("Success", f"Configuration exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export configuration: {e}")

    def import_config(self):
        """Import configuration from JSON file."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    imported_config = json.load(f)
                self.config_manager.update(imported_config)
                self.load_settings()
                messagebox.showinfo("Success", "Configuration imported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import configuration: {e}")
