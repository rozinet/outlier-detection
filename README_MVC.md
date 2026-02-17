# Building Sensor Outlier Detection System

A comprehensive GUI application for detecting building pathologies from sensor data.

## Features

- **Building Management**: Create and manage buildings with hierarchical room structure
- **Sensor Assignment**: Assign sensors to specific rooms for organized monitoring
- **Configurable Detection**: Adjust all detection parameters through the UI
- **Problem Detection**: Detects 5 types of building issues:
  - Moisture Intrusion
  - Condensation Risk
  - Drying Failure
  - Sensor Malfunction
  - Rapid Moisture Change
- **Health Scoring**: Composite health scores for each sensor
- **Results Export**: Export detection results to CSV

## Architecture

The application follows **MVC (Model-View-Controller)** architecture:

```
outlier-detection/
├── models/              # Data layer
│   ├── __init__.py
│   ├── building.py     # Building/Room/Sensor data structures
│   ├── config.py       # Configuration management
│   ├── problem.py      # Problem data structure
│   ├── data_loader.py  # Data loading utilities
│   ├── preprocessor.py # Signal preprocessing
│   └── detection_engine.py  # Detection algorithm wrapper
│
├── views/              # Presentation layer
│   ├── __init__.py
│   ├── main_window.py  # Main application window
│   ├── building_panel.py  # Building management UI
│   ├── settings_panel.py  # Configuration UI
│   └── results_panel.py   # Results display UI
│
├── controllers/        # Business logic layer
│   ├── __init__.py
│   ├── app_controller.py       # Main application controller
│   ├── building_controller.py  # Building management logic
│   └── detection_controller.py # Detection pipeline logic
│
├── main.py            # Application entry point
├── outlier.py         # Original detection algorithms (preserved)
├── config.json        # Configuration file (auto-generated)
└── buildings.json     # Building registry (auto-generated)
```

## Installation

### Requirements

- Python 3.10+
- Required packages:
  ```bash
  pip install numpy pandas scipy matplotlib
  ```

### Running the Application

```bash
cd outlier-detection
python main.py
```

## Usage Guide

### 1. Building Management

**Buildings & Sensors Tab:**

1. **Create a Building**:
   - Click "New Building"
   - Enter name, address, and description
   - Click OK

2. **Add Rooms**:
   - Select a building from the list
   - Click "New Room"
   - Enter room name, floor, and description
   - Click OK

3. **Assign Sensors**:
   - Select a room from the list
   - Click "Add Sensor"
   - Enter sensor ID (UUID from data files)
   - Provide a friendly name
   - Optionally set data path
   - Click OK

### 2. Configure Detection Settings

**Settings Tab:**

- Navigate through different category tabs:
  - **General**: Preprocessing parameters
  - **Moisture Intrusion**: Thresholds for water intrusion detection
  - **Condensation**: Humidity thresholds for condensation risk
  - **Sensor Malfunction**: Parameters for sensor fault detection
  - **Advanced**: CUSUM, drying curves, outlier detection

- Modify values as needed
- Click "Save Settings" to persist changes
- Use "Reset to Defaults" to restore factory settings
- Export/Import configuration as JSON files

### 3. Run Detection

**Detection Results Tab:**

1. Select a building from the dropdown
2. Browse to the sensor data directory (contains .json_line files)
3. Click "Run Detection"
4. Monitor progress bar
5. Review results:
   - **Summary**: Overview statistics
   - **Problems**: Detailed list of detected issues (filterable)
   - **Health Scores**: Device health scores with breakdown
6. Export results to CSV for further analysis

## Data Format

The application expects sensor data in `.json_line` format:

```
exported_data/
├── {sensor-uuid}_temperature_ambient_celsius.json_line
├── {sensor-uuid}_rel_humidity_ambient_pct.json_line
├── {sensor-uuid}_rel_humidity_cavity_pct.json_line
└── {sensor-uuid}_moisture_resistance_pct.json_line
```

Each `.json_line` file should contain:
```json
{
  "metric": {
    "__name__": "sensor_type",
    "installation_id": "building_id"
  },
  "timestamps": [1234567890000, ...],
  "values": [23.5, 24.1, ...]
}
```

## Configuration Files

### config.json
Stores all detection parameters. Auto-created on first run with default values.

### buildings.json
Stores building/room/sensor registry. Auto-created when you add your first building.

## Original Detection Logic

All detection algorithms from `outlier.py` are **preserved** and used by the application through the `DetectionEngine` wrapper. The MVC refactoring separates UI and data management while keeping the proven detection logic intact.

## Extending the Application

### Adding New Detection Algorithms

1. Implement detector function in `outlier.py`
2. Add function call in `models/detection_engine.py`
3. Update configuration in `models/config.py` if needed
4. Results will automatically appear in the UI

### Adding New UI Features

1. Create new panel class in `views/`
2. Add tab to `main_window.py` notebook
3. Implement controller logic if needed

## Troubleshooting

**Import Error: "Could not import detection functions"**
- Ensure `outlier.py` is in the same directory as `main.py`
- Check that all dependencies (numpy, pandas, scipy) are installed

**No data loaded**
- Verify data directory path is correct
- Check that sensor UUIDs in buildings.json match filenames
- Ensure .json_line files are properly formatted

**Detection fails**
- Check sensor has minimum required data (default: 48 hours)
- Verify all required sensor channels are present
- Review error messages in console

## Version

Version 6.0 - MVC Architecture

## License

© 2026 Senzomatic
