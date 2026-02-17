# Application Architecture

## MVC Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                      (Tkinter Windows)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    VIEW LAYER (views/)                       │
├─────────────────────────────────────────────────────────────┤
│  MainWindow          - Main application window               │
│  BuildingPanel       - Building/Room/Sensor management UI    │
│  SettingsPanel       - Configuration UI                      │
│  ResultsPanel        - Detection results display             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                CONTROLLER LAYER (controllers/)               │
├─────────────────────────────────────────────────────────────┤
│  AppController         - Main application coordinator        │
│  BuildingController    - Building CRUD operations            │
│  DetectionController   - Detection pipeline orchestration    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   MODEL LAYER (models/)                      │
├─────────────────────────────────────────────────────────────┤
│  ConfigManager       - Configuration management              │
│  Building/Room/      - Data structures & registry            │
│    Sensor                                                    │
│  Problem             - Detection result data structure       │
│  DataLoader          - Load sensor data from files           │
│  Preprocessor        - Signal preprocessing utilities        │
│  DetectionEngine     - Wrapper for detection algorithms      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ORIGINAL DETECTION LOGIC (outlier.py)           │
├─────────────────────────────────────────────────────────────┤
│  • Moisture Intrusion Detection                              │
│  • Condensation Risk Detection                               │
│  • Drying Failure Detection                                  │
│  • Sensor Malfunction Detection                              │
│  • Rapid Moisture Change Detection                           │
│  • Health Score Computation                                  │
│  • Installation Outlier Detection                            │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Building Management Flow
```
User Action (View)
  → BuildingController.create_building()
    → Building object created
      → BuildingRegistry.add_building()
        → buildings.json updated
          → View refreshed
```

### 2. Settings Configuration Flow
```
User modifies setting (View)
  → SettingsPanel.save_settings()
    → ConfigManager.update()
      → config.json updated
        → Settings available to DetectionEngine
```

### 3. Detection Pipeline Flow
```
User clicks "Run Detection" (View)
  → DetectionController.run_detection()
    → DataLoader.load_building_sensors()
      → Read .json_line files
    → Preprocessor.preprocess_device_data()
      → Multi-resolution signal processing
    → DetectionEngine.detect_all_problems()
      → Call outlier.py detectors
        → detect_moisture_intrusion()
        → detect_condensation_risk()
        → detect_drying_failure()
        → detect_sensor_malfunction()
        → detect_rapid_moisture_change()
      → Collect Problem objects
    → DetectionEngine.compute_device_health_score()
      → Calculate composite scores
    → Results returned to View
      → Display in ResultsPanel
```

## File Structure

```
outlier-detection/
│
├── main.py                      # Application entry point
│
├── models/                      # DATA LAYER
│   ├── __init__.py
│   ├── config.py               # ConfigManager class
│   ├── building.py             # Building/Room/Sensor/Registry classes
│   ├── problem.py              # Problem dataclass
│   ├── data_loader.py          # Data loading functions
│   ├── preprocessor.py         # Signal preprocessing functions
│   └── detection_engine.py     # DetectionEngine wrapper
│
├── views/                       # PRESENTATION LAYER
│   ├── __init__.py
│   ├── main_window.py          # MainWindow class
│   ├── building_panel.py       # BuildingPanel + dialogs
│   ├── settings_panel.py       # SettingsPanel class
│   └── results_panel.py        # ResultsPanel class
│
├── controllers/                 # BUSINESS LOGIC LAYER
│   ├── __init__.py
│   ├── app_controller.py       # AppController class
│   ├── building_controller.py  # BuildingController class
│   └── detection_controller.py # DetectionController class
│
├── outlier.py                   # ORIGINAL DETECTION ALGORITHMS
│                                 (Preserved, not modified)
│
├── config.json                  # Runtime configuration
├── buildings.json               # Building registry
│
└── Documentation
    ├── README_MVC.md            # Comprehensive documentation
    ├── QUICK_START.md           # Quick start guide
    └── ARCHITECTURE.md          # This file
```

## Key Design Principles

### 1. Separation of Concerns
- **Views**: Only handle UI rendering and user input
- **Controllers**: Orchestrate business logic, no UI code
- **Models**: Pure data and algorithms, no UI dependencies

### 2. Preservation of Original Logic
- All detection algorithms in `outlier.py` are **unchanged**
- `DetectionEngine` wraps original functions
- Ensures proven algorithms continue to work correctly

### 3. Data Persistence
- Buildings/Rooms/Sensors → `buildings.json`
- Configuration → `config.json`
- Both use JSON for human-readable storage

### 4. Extensibility
- Add new detectors: Modify `outlier.py` and `DetectionEngine`
- Add new UI features: Create new View/Controller
- Add new data types: Extend Model classes

### 5. User-Friendly Configuration
- All settings accessible via UI
- Import/Export configuration
- Reset to defaults option
- Tooltips explain each setting

## Component Responsibilities

### Models
- **ConfigManager**: Load/save/manage detection parameters
- **Building/Room/Sensor**: Hierarchical data structure
- **BuildingRegistry**: Persistent storage of building data
- **Problem**: Represents a detected issue
- **DataLoader**: Read sensor data from filesystem
- **Preprocessor**: Signal processing (filtering, smoothing)
- **DetectionEngine**: Interface to detection algorithms

### Views
- **MainWindow**: Application shell with tabs and menu
- **BuildingPanel**: Tree-based building hierarchy editor
- **SettingsPanel**: Categorized configuration forms
- **ResultsPanel**: Detection execution and result display

### Controllers
- **AppController**: Application lifecycle management
- **BuildingController**: CRUD operations for buildings
- **DetectionController**: Detection pipeline orchestration

## Technology Stack

- **UI Framework**: tkinter (Python standard library)
- **Data Processing**: numpy, pandas
- **Signal Processing**: scipy
- **Visualization**: matplotlib (for future charts)
- **Data Storage**: JSON files
- **Architecture**: MVC pattern

## Benefits of MVC Architecture

### For Users
- ✅ Easy-to-use graphical interface
- ✅ No command-line knowledge required
- ✅ Visual building/sensor management
- ✅ Real-time configuration changes
- ✅ Organized result viewing and filtering

### For Developers
- ✅ Clean separation of concerns
- ✅ Easy to test individual components
- ✅ Original detection logic preserved
- ✅ Simple to add new features
- ✅ Maintainable codebase

### For Maintenance
- ✅ Update UI without touching algorithms
- ✅ Update algorithms without touching UI
- ✅ Configuration changes don't require code changes
- ✅ Clear file organization
- ✅ Documented architecture
