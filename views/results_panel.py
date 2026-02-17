"""Results display panel."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading


class ResultsPanel:
    """Panel for running detection and viewing results."""

    def __init__(self, parent, detection_controller, building_controller):
        """Initialize results panel.

        Args:
            parent: Parent widget
            detection_controller: DetectionController instance
            building_controller: BuildingController instance
        """
        self.detection_controller = detection_controller
        self.building_controller = building_controller
        self.frame = ttk.Frame(parent)
        self.current_results = None

        self.create_layout()

    def create_layout(self):
        """Create the panel layout."""
        # Header
        header = ttk.Label(self.frame, text="Detection Results",
                          style='Heading.TLabel')
        header.pack(pady=10)

        # Control frame
        control_frame = ttk.LabelFrame(self.frame, text="Run Detection", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Building selection
        ttk.Label(control_frame, text="Building:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.building_combo = ttk.Combobox(control_frame, state='readonly', width=40)
        self.building_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        self.refresh_buildings()

        # Data directory
        ttk.Label(control_frame, text="Data Directory:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        data_frame = ttk.Frame(control_frame)
        data_frame.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.data_dir_entry = ttk.Entry(data_frame, width=35)
        self.data_dir_entry.pack(side=tk.LEFT)
        ttk.Button(data_frame, text="Browse", command=self.browse_data_dir).pack(side=tk.LEFT, padx=5)

        # Run button
        self.run_btn = ttk.Button(control_frame, text="Run Detection", command=self.run_detection)
        self.run_btn.grid(row=2, column=1, sticky='w', padx=5, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                           maximum=100, length=300)
        self.progress_bar.grid(row=3, column=1, sticky='ew', padx=5, pady=5)

        self.progress_label = ttk.Label(control_frame, text="")
        self.progress_label.grid(row=4, column=1, sticky='w', padx=5)

        # Summary frame
        summary_frame = ttk.LabelFrame(self.frame, text="Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)

        self.summary_text = tk.Text(summary_frame, height=6, wrap=tk.WORD, state='disabled')
        self.summary_text.pack(fill=tk.X, padx=5, pady=5)

        # Results notebook
        results_notebook = ttk.Notebook(self.frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Problems tab
        problems_frame = ttk.Frame(results_notebook)
        results_notebook.add(problems_frame, text="Problems")
        self.create_problems_view(problems_frame)

        # Health scores tab
        health_frame = ttk.Frame(results_notebook)
        results_notebook.add(health_frame, text="Health Scores")
        self.create_health_view(health_frame)

        # Export button
        export_frame = ttk.Frame(self.frame)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(export_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)

    def create_problems_view(self, parent):
        """Create problems view.

        Args:
            parent: Parent widget
        """
        # Toolbar
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(toolbar, text="Filter by:").pack(side=tk.LEFT, padx=5)

        ttk.Label(toolbar, text="Type:").pack(side=tk.LEFT, padx=5)
        self.type_filter = ttk.Combobox(toolbar, state='readonly', width=20)
        self.type_filter['values'] = ['All', 'moisture_intrusion', 'condensation_risk',
                                      'drying_failure', 'sensor_malfunction', 'rapid_moisture_change']
        self.type_filter.current(0)
        self.type_filter.pack(side=tk.LEFT, padx=5)
        self.type_filter.bind('<<ComboboxSelected>>', self.filter_problems)

        ttk.Label(toolbar, text="Severity:").pack(side=tk.LEFT, padx=5)
        self.severity_filter = ttk.Combobox(toolbar, state='readonly', width=15)
        self.severity_filter['values'] = ['All', 'warning', 'danger', 'critical']
        self.severity_filter.current(0)
        self.severity_filter.pack(side=tk.LEFT, padx=5)
        self.severity_filter.bind('<<ComboboxSelected>>', self.filter_problems)

        # Treeview
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar_y = ttk.Scrollbar(tree_frame)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.problems_tree = ttk.Treeview(
            tree_frame,
            columns=('Device', 'Type', 'Severity', 'Start', 'End', 'Duration', 'Description'),
            show='headings',
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )

        self.problems_tree.heading('Device', text='Device ID')
        self.problems_tree.heading('Type', text='Problem Type')
        self.problems_tree.heading('Severity', text='Severity')
        self.problems_tree.heading('Start', text='Start Date')
        self.problems_tree.heading('End', text='End Date')
        self.problems_tree.heading('Duration', text='Duration (h)')
        self.problems_tree.heading('Description', text='Description')

        self.problems_tree.column('Device', width=100)
        self.problems_tree.column('Type', width=150)
        self.problems_tree.column('Severity', width=80)
        self.problems_tree.column('Start', width=100)
        self.problems_tree.column('End', width=100)
        self.problems_tree.column('Duration', width=80)
        self.problems_tree.column('Description', width=300)

        self.problems_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.config(command=self.problems_tree.yview)
        scrollbar_x.config(command=self.problems_tree.xview)

        # Configure severity colors
        self.problems_tree.tag_configure('warning', background='#fff3cd')
        self.problems_tree.tag_configure('danger', background='#f8d7da')
        self.problems_tree.tag_configure('critical', background='#dc3545', foreground='white')

    def create_health_view(self, parent):
        """Create health scores view.

        Args:
            parent: Parent widget
        """
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.health_tree = ttk.Treeview(
            tree_frame,
            columns=('Device', 'Score', 'Condensation', 'Moisture', 'Drying', 'Sensor', 'Rapid'),
            show='headings',
            yscrollcommand=scrollbar.set
        )

        self.health_tree.heading('Device', text='Device ID')
        self.health_tree.heading('Score', text='Health Score')
        self.health_tree.heading('Condensation', text='Condensation')
        self.health_tree.heading('Moisture', text='Moisture')
        self.health_tree.heading('Drying', text='Drying')
        self.health_tree.heading('Sensor', text='Sensor')
        self.health_tree.heading('Rapid', text='Rapid Change')

        self.health_tree.column('Device', width=150)
        self.health_tree.column('Score', width=100)
        self.health_tree.column('Condensation', width=100)
        self.health_tree.column('Moisture', width=100)
        self.health_tree.column('Drying', width=100)
        self.health_tree.column('Sensor', width=100)
        self.health_tree.column('Rapid', width=100)

        self.health_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.health_tree.yview)

        # Color code by health score
        self.health_tree.tag_configure('excellent', background='#d4edda')
        self.health_tree.tag_configure('good', background='#d1ecf1')
        self.health_tree.tag_configure('fair', background='#fff3cd')
        self.health_tree.tag_configure('poor', background='#f8d7da')

    def refresh_buildings(self):
        """Refresh buildings combobox."""
        buildings = self.building_controller.get_all_buildings()
        building_names = [b.name for b in buildings]
        self.building_combo['values'] = building_names
        if building_names:
            self.building_combo.current(0)

    def browse_data_dir(self):
        """Browse for data directory."""
        path = filedialog.askdirectory(title="Select Sensor Data Directory")
        if path:
            self.data_dir_entry.delete(0, tk.END)
            self.data_dir_entry.insert(0, path)

    def update_progress(self, message, progress):
        """Update progress bar and label.

        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
        """
        self.progress_var.set(progress * 100)
        self.progress_label.config(text=message)

    def run_detection(self):
        """Run detection in background thread."""
        building_name = self.building_combo.get()
        if not building_name:
            messagebox.showwarning("Warning", "Please select a building")
            return

        data_dir = self.data_dir_entry.get().strip()
        if not data_dir:
            messagebox.showwarning("Warning", "Please specify data directory")
            return

        # Get selected building
        buildings = self.building_controller.get_all_buildings()
        building = None
        for b in buildings:
            if b.name == building_name:
                building = b
                break

        if not building:
            messagebox.showerror("Error", "Building not found")
            return

        # Disable run button
        self.run_btn.config(state='disabled')
        self.progress_var.set(0)
        self.progress_label.config(text="Starting detection...")

        # Run in thread
        def run_thread():
            try:
                results = self.detection_controller.run_detection(
                    building, data_dir, progress_callback=self.update_progress
                )
                self.current_results = results
                self.frame.after(0, self.display_results)
            except Exception as e:
                self.frame.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {e}"))
            finally:
                self.frame.after(0, lambda: self.run_btn.config(state='normal'))

        thread = threading.Thread(target=run_thread, daemon=True)
        thread.start()

    def display_results(self):
        """Display detection results."""
        if not self.current_results:
            return

        # Display summary
        summary = self.detection_controller.get_results_summary()
        summary_text = f"""
Total Devices Analyzed: {summary['total_devices']}
Total Problems Detected: {summary['total_problems']}
Average Health Score: {summary['avg_health_score']:.1f}

Problems by Type:
{self._format_dict(summary['problems_by_type'])}

Problems by Severity:
{self._format_dict(summary['problems_by_severity'])}
        """.strip()

        self.summary_text.config(state='normal')
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert('1.0', summary_text)
        self.summary_text.config(state='disabled')

        # Display problems
        self.display_problems()

        # Display health scores
        self.display_health_scores()

        messagebox.showinfo("Complete", "Detection complete!")

    def _format_dict(self, d):
        """Format dictionary for display.

        Args:
            d: Dictionary

        Returns:
            Formatted string
        """
        if not d:
            return "  None"
        return "\n".join(f"  {k}: {v}" for k, v in d.items())

    def display_problems(self):
        """Display problems in treeview."""
        # Clear existing
        for item in self.problems_tree.get_children():
            self.problems_tree.delete(item)

        if not self.current_results:
            return

        # Add problems
        all_problems = self.current_results.get('problems', {})
        for device_id, problems in all_problems.items():
            for problem in problems:
                self.problems_tree.insert('', tk.END, values=(
                    device_id[:12],
                    problem.problem_type,
                    problem.severity.upper(),
                    problem.start.strftime('%Y-%m-%d'),
                    problem.end.strftime('%Y-%m-%d'),
                    f"{problem.duration_hours:.0f}",
                    problem.description
                ), tags=(problem.severity,))

    def filter_problems(self, event=None):
        """Filter problems by type and severity.

        Args:
            event: Event object
        """
        # Clear existing
        for item in self.problems_tree.get_children():
            self.problems_tree.delete(item)

        if not self.current_results:
            return

        type_filter = self.type_filter.get()
        severity_filter = self.severity_filter.get()

        # Add filtered problems
        all_problems = self.current_results.get('problems', {})
        for device_id, problems in all_problems.items():
            for problem in problems:
                # Apply filters
                if type_filter != 'All' and problem.problem_type != type_filter:
                    continue
                if severity_filter != 'All' and problem.severity != severity_filter:
                    continue

                self.problems_tree.insert('', tk.END, values=(
                    device_id[:12],
                    problem.problem_type,
                    problem.severity.upper(),
                    problem.start.strftime('%Y-%m-%d'),
                    problem.end.strftime('%Y-%m-%d'),
                    f"{problem.duration_hours:.0f}",
                    problem.description
                ), tags=(problem.severity,))

    def display_health_scores(self):
        """Display health scores in treeview."""
        # Clear existing
        for item in self.health_tree.get_children():
            self.health_tree.delete(item)

        if not self.current_results:
            return

        # Add health scores
        health_scores = self.current_results.get('health_scores', {})
        for device_id, score_data in health_scores.items():
            score = score_data.get('score', 0)
            breakdown = score_data.get('breakdown', {})

            # Determine color tag
            if score >= 90:
                tag = 'excellent'
            elif score >= 70:
                tag = 'good'
            elif score >= 50:
                tag = 'fair'
            else:
                tag = 'poor'

            self.health_tree.insert('', tk.END, values=(
                device_id[:12],
                f"{score:.1f}",
                breakdown.get('condensation_risk', 0),
                breakdown.get('moisture_intrusion', 0),
                breakdown.get('drying_failure', 0),
                breakdown.get('sensor_malfunction', 0),
                breakdown.get('rapid_moisture_change', 0)
            ), tags=(tag,))

    def export_results(self):
        """Export results to CSV."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Write problems
                    writer.writerow(['Device ID', 'Problem Type', 'Severity', 'Start', 'End', 'Duration (h)', 'Description'])

                    all_problems = self.current_results.get('problems', {})
                    for device_id, problems in all_problems.items():
                        for problem in problems:
                            writer.writerow([
                                device_id,
                                problem.problem_type,
                                problem.severity,
                                problem.start.strftime('%Y-%m-%d %H:%M:%S'),
                                problem.end.strftime('%Y-%m-%d %H:%M:%S'),
                                f"{problem.duration_hours:.1f}",
                                problem.description
                            ])

                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")
