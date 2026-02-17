"""Building management panel."""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog


class BuildingPanel:
    """Panel for managing buildings, rooms, and sensors."""

    def __init__(self, parent, building_controller):
        """Initialize building panel.

        Args:
            parent: Parent widget
            building_controller: BuildingController instance
        """
        self.building_controller = building_controller
        self.frame = ttk.Frame(parent)

        # Create main layout with three columns
        self.create_layout()
        self.refresh_buildings()

    def create_layout(self):
        """Create the panel layout."""
        # Header
        header = ttk.Label(self.frame, text="Building & Sensor Management",
                          style='Heading.TLabel')
        header.pack(pady=10)

        # Main content area with three columns
        content = ttk.Frame(self.frame)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Configure grid
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.columnconfigure(2, weight=2)

        # Column 1: Buildings
        self.create_buildings_column(content)

        # Column 2: Rooms
        self.create_rooms_column(content)

        # Column 3: Sensors
        self.create_sensors_column(content)

    def create_buildings_column(self, parent):
        """Create buildings column.

        Args:
            parent: Parent widget
        """
        col_frame = ttk.LabelFrame(parent, text="Buildings", padding=10)
        col_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Buttons
        btn_frame = ttk.Frame(col_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(btn_frame, text="New Building", command=self.new_building).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Edit", command=self.edit_building).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete", command=self.delete_building).pack(side=tk.LEFT, padx=2)

        # Listbox with scrollbar
        list_frame = ttk.Frame(col_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.buildings_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.buildings_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.buildings_list.yview)

        self.buildings_list.bind('<<ListboxSelect>>', self.on_building_select)

    def create_rooms_column(self, parent):
        """Create rooms column.

        Args:
            parent: Parent widget
        """
        col_frame = ttk.LabelFrame(parent, text="Rooms", padding=10)
        col_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        # Buttons
        btn_frame = ttk.Frame(col_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(btn_frame, text="New Room", command=self.new_room).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Edit", command=self.edit_room).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete", command=self.delete_room).pack(side=tk.LEFT, padx=2)

        # Listbox with scrollbar
        list_frame = ttk.Frame(col_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.rooms_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.rooms_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.rooms_list.yview)

        self.rooms_list.bind('<<ListboxSelect>>', self.on_room_select)

    def create_sensors_column(self, parent):
        """Create sensors column.

        Args:
            parent: Parent widget
        """
        col_frame = ttk.LabelFrame(parent, text="Sensors", padding=10)
        col_frame.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)

        # Buttons
        btn_frame = ttk.Frame(col_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(btn_frame, text="Add Sensor", command=self.add_sensor).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Edit", command=self.edit_sensor).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove", command=self.remove_sensor).pack(side=tk.LEFT, padx=2)

        # Treeview with scrollbar
        tree_frame = ttk.Frame(col_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sensors_tree = ttk.Treeview(tree_frame, columns=('ID', 'Name', 'Status'),
                                        show='headings', yscrollcommand=scrollbar.set)
        self.sensors_tree.heading('ID', text='Sensor ID')
        self.sensors_tree.heading('Name', text='Name')
        self.sensors_tree.heading('Status', text='Status')
        self.sensors_tree.column('ID', width=250)
        self.sensors_tree.column('Name', width=150)
        self.sensors_tree.column('Status', width=80)
        self.sensors_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.sensors_tree.yview)

    # Building operations
    def refresh_buildings(self):
        """Refresh buildings list."""
        self.buildings_list.delete(0, tk.END)
        buildings = self.building_controller.get_all_buildings()
        for building in buildings:
            self.buildings_list.insert(tk.END, building.name)
        self.rooms_list.delete(0, tk.END)
        self.sensors_tree.delete(*self.sensors_tree.get_children())

    def on_building_select(self, event):
        """Handle building selection.

        Args:
            event: Selection event
        """
        selection = self.buildings_list.curselection()
        if not selection:
            return

        idx = selection[0]
        buildings = self.building_controller.get_all_buildings()
        if idx < len(buildings):
            building = buildings[idx]
            self.refresh_rooms(building)

    def new_building(self):
        """Create new building."""
        dialog = BuildingDialog(self.frame, "New Building")
        if dialog.result:
            name = dialog.result.get('name')
            address = dialog.result.get('address', '')
            description = dialog.result.get('description', '')
            self.building_controller.create_building(name, address, description)
            self.refresh_buildings()

    def edit_building(self):
        """Edit selected building."""
        selection = self.buildings_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a building to edit")
            return

        idx = selection[0]
        buildings = self.building_controller.get_all_buildings()
        building = buildings[idx]

        dialog = BuildingDialog(self.frame, "Edit Building",
                               initial={'name': building.name,
                                      'address': building.address,
                                      'description': building.description})
        if dialog.result:
            self.building_controller.update_building(
                building.building_id,
                name=dialog.result.get('name'),
                address=dialog.result.get('address'),
                description=dialog.result.get('description')
            )
            self.refresh_buildings()

    def delete_building(self):
        """Delete selected building."""
        selection = self.buildings_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a building to delete")
            return

        idx = selection[0]
        buildings = self.building_controller.get_all_buildings()
        building = buildings[idx]

        if messagebox.askyesno("Confirm", f"Delete building '{building.name}'?\nThis will also delete all rooms and sensors."):
            self.building_controller.delete_building(building.building_id)
            self.refresh_buildings()

    # Room operations
    def refresh_rooms(self, building):
        """Refresh rooms list for a building.

        Args:
            building: Building object
        """
        self.current_building = building
        self.rooms_list.delete(0, tk.END)
        for room in building.rooms:
            display = f"{room.name}"
            if room.floor:
                display += f" (Floor {room.floor})"
            self.rooms_list.insert(tk.END, display)
        self.sensors_tree.delete(*self.sensors_tree.get_children())

    def on_room_select(self, event):
        """Handle room selection.

        Args:
            event: Selection event
        """
        selection = self.rooms_list.curselection()
        if not selection or not hasattr(self, 'current_building'):
            return

        idx = selection[0]
        if idx < len(self.current_building.rooms):
            room = self.current_building.rooms[idx]
            self.refresh_sensors(room)

    def new_room(self):
        """Create new room."""
        if not hasattr(self, 'current_building'):
            messagebox.showwarning("Warning", "Please select a building first")
            return

        dialog = RoomDialog(self.frame, "New Room")
        if dialog.result:
            self.building_controller.create_room(
                self.current_building.building_id,
                name=dialog.result.get('name'),
                floor=dialog.result.get('floor'),
                description=dialog.result.get('description', '')
            )
            self.refresh_rooms(self.current_building)

    def edit_room(self):
        """Edit selected room."""
        selection = self.rooms_list.curselection()
        if not selection or not hasattr(self, 'current_building'):
            messagebox.showwarning("Warning", "Please select a room to edit")
            return

        idx = selection[0]
        room = self.current_building.rooms[idx]

        dialog = RoomDialog(self.frame, "Edit Room",
                           initial={'name': room.name,
                                  'floor': room.floor or '',
                                  'description': room.description})
        if dialog.result:
            self.building_controller.update_room(
                self.current_building.building_id,
                room.room_id,
                name=dialog.result.get('name'),
                floor=dialog.result.get('floor'),
                description=dialog.result.get('description')
            )
            self.refresh_rooms(self.current_building)

    def delete_room(self):
        """Delete selected room."""
        selection = self.rooms_list.curselection()
        if not selection or not hasattr(self, 'current_building'):
            messagebox.showwarning("Warning", "Please select a room to delete")
            return

        idx = selection[0]
        room = self.current_building.rooms[idx]

        if messagebox.askyesno("Confirm", f"Delete room '{room.name}'?\nThis will also delete all sensors in this room."):
            self.building_controller.delete_room(self.current_building.building_id, room.room_id)
            self.refresh_rooms(self.current_building)

    # Sensor operations
    def refresh_sensors(self, room):
        """Refresh sensors list for a room.

        Args:
            room: Room object
        """
        self.current_room = room
        self.sensors_tree.delete(*self.sensors_tree.get_children())
        for sensor in room.sensors:
            status = "Active" if sensor.is_active else "Inactive"
            self.sensors_tree.insert('', tk.END, values=(sensor.sensor_id, sensor.name, status))

    def add_sensor(self):
        """Add sensor to room."""
        if not hasattr(self, 'current_room'):
            messagebox.showwarning("Warning", "Please select a room first")
            return

        dialog = SensorDialog(self.frame, "Add Sensor")
        if dialog.result:
            self.building_controller.create_sensor(
                self.current_building.building_id,
                self.current_room.room_id,
                sensor_id=dialog.result.get('sensor_id'),
                name=dialog.result.get('name'),
                data_path=dialog.result.get('data_path'),
                description=dialog.result.get('description', '')
            )
            self.refresh_sensors(self.current_room)

    def edit_sensor(self):
        """Edit selected sensor."""
        selection = self.sensors_tree.selection()
        if not selection or not hasattr(self, 'current_room'):
            messagebox.showwarning("Warning", "Please select a sensor to edit")
            return

        item = self.sensors_tree.item(selection[0])
        sensor_id = item['values'][0]
        sensor = self.current_room.get_sensor(sensor_id)

        dialog = SensorDialog(self.frame, "Edit Sensor",
                             initial={'sensor_id': sensor.sensor_id,
                                    'name': sensor.name,
                                    'data_path': sensor.data_path or '',
                                    'description': sensor.description,
                                    'is_active': sensor.is_active})
        if dialog.result:
            self.building_controller.update_sensor(
                self.current_building.building_id,
                self.current_room.room_id,
                sensor_id,
                name=dialog.result.get('name'),
                data_path=dialog.result.get('data_path'),
                description=dialog.result.get('description'),
                is_active=dialog.result.get('is_active')
            )
            self.refresh_sensors(self.current_room)

    def remove_sensor(self):
        """Remove selected sensor."""
        selection = self.sensors_tree.selection()
        if not selection or not hasattr(self, 'current_room'):
            messagebox.showwarning("Warning", "Please select a sensor to remove")
            return

        item = self.sensors_tree.item(selection[0])
        sensor_id = item['values'][0]

        if messagebox.askyesno("Confirm", f"Remove sensor '{sensor_id}'?"):
            self.building_controller.delete_sensor(
                self.current_building.building_id,
                self.current_room.room_id,
                sensor_id
            )
            self.refresh_sensors(self.current_room)


# Dialog classes
class BuildingDialog:
    """Dialog for creating/editing buildings."""

    def __init__(self, parent, title, initial=None):
        """Initialize dialog.

        Args:
            parent: Parent widget
            title: Dialog title
            initial: Initial values dictionary
        """
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Create form
        ttk.Label(self.dialog, text="Name:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.name_entry = ttk.Entry(self.dialog, width=40)
        self.name_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(self.dialog, text="Address:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.address_entry = ttk.Entry(self.dialog, width=40)
        self.address_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(self.dialog, text="Description:").grid(row=2, column=0, sticky='nw', padx=10, pady=5)
        self.desc_text = tk.Text(self.dialog, width=40, height=5)
        self.desc_text.grid(row=2, column=1, padx=10, pady=5)

        # Set initial values
        if initial:
            self.name_entry.insert(0, initial.get('name', ''))
            self.address_entry.insert(0, initial.get('address', ''))
            self.desc_text.insert('1.0', initial.get('description', ''))

        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        ttk.Button(btn_frame, text="OK", command=self.ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)

        self.dialog.wait_window()

    def ok(self):
        """Handle OK button."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Name is required")
            return

        self.result = {
            'name': name,
            'address': self.address_entry.get().strip(),
            'description': self.desc_text.get('1.0', tk.END).strip()
        }
        self.dialog.destroy()


class RoomDialog:
    """Dialog for creating/editing rooms."""

    def __init__(self, parent, title, initial=None):
        """Initialize dialog.

        Args:
            parent: Parent widget
            title: Dialog title
            initial: Initial values dictionary
        """
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x250")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Create form
        ttk.Label(self.dialog, text="Name:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.name_entry = ttk.Entry(self.dialog, width=40)
        self.name_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(self.dialog, text="Floor:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.floor_entry = ttk.Entry(self.dialog, width=40)
        self.floor_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(self.dialog, text="Description:").grid(row=2, column=0, sticky='nw', padx=10, pady=5)
        self.desc_text = tk.Text(self.dialog, width=40, height=4)
        self.desc_text.grid(row=2, column=1, padx=10, pady=5)

        # Set initial values
        if initial:
            self.name_entry.insert(0, initial.get('name', ''))
            self.floor_entry.insert(0, initial.get('floor', ''))
            self.desc_text.insert('1.0', initial.get('description', ''))

        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        ttk.Button(btn_frame, text="OK", command=self.ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)

        self.dialog.wait_window()

    def ok(self):
        """Handle OK button."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Name is required")
            return

        self.result = {
            'name': name,
            'floor': self.floor_entry.get().strip(),
            'description': self.desc_text.get('1.0', tk.END).strip()
        }
        self.dialog.destroy()


class SensorDialog:
    """Dialog for adding/editing sensors."""

    def __init__(self, parent, title, initial=None):
        """Initialize dialog.

        Args:
            parent: Parent widget
            title: Dialog title
            initial: Initial values dictionary
        """
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Create form
        ttk.Label(self.dialog, text="Sensor ID:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.id_entry = ttk.Entry(self.dialog, width=45)
        self.id_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(self.dialog, text="Name:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.name_entry = ttk.Entry(self.dialog, width=45)
        self.name_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(self.dialog, text="Data Path:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        path_frame = ttk.Frame(self.dialog)
        path_frame.grid(row=2, column=1, sticky='ew', padx=10, pady=5)
        self.path_entry = ttk.Entry(path_frame, width=35)
        self.path_entry.pack(side=tk.LEFT)
        ttk.Button(path_frame, text="Browse", command=self.browse_path).pack(side=tk.LEFT, padx=5)

        ttk.Label(self.dialog, text="Description:").grid(row=3, column=0, sticky='nw', padx=10, pady=5)
        self.desc_text = tk.Text(self.dialog, width=45, height=4)
        self.desc_text.grid(row=3, column=1, padx=10, pady=5)

        self.active_var = tk.BooleanVar(value=True)
        self.active_check = ttk.Checkbutton(self.dialog, text="Active", variable=self.active_var)
        self.active_check.grid(row=4, column=1, sticky='w', padx=10, pady=5)

        # Set initial values
        if initial:
            self.id_entry.insert(0, initial.get('sensor_id', ''))
            if initial.get('sensor_id'):  # Disable ID editing if editing existing sensor
                self.id_entry.config(state='readonly')
            self.name_entry.insert(0, initial.get('name', ''))
            self.path_entry.insert(0, initial.get('data_path', ''))
            self.desc_text.insert('1.0', initial.get('description', ''))
            self.active_var.set(initial.get('is_active', True))

        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)
        ttk.Button(btn_frame, text="OK", command=self.ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)

        self.dialog.wait_window()

    def browse_path(self):
        """Browse for data path."""
        path = filedialog.askdirectory(title="Select Data Directory")
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

    def ok(self):
        """Handle OK button."""
        sensor_id = self.id_entry.get().strip()
        name = self.name_entry.get().strip()

        if not sensor_id:
            messagebox.showwarning("Warning", "Sensor ID is required")
            return
        if not name:
            messagebox.showwarning("Warning", "Name is required")
            return

        self.result = {
            'sensor_id': sensor_id,
            'name': name,
            'data_path': self.path_entry.get().strip(),
            'description': self.desc_text.get('1.0', tk.END).strip(),
            'is_active': self.active_var.get()
        }
        self.dialog.destroy()
