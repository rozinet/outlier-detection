# Quick Start Guide

## Running the Application

### Step 1: Launch the UI

```bash
cd c:\Users\Max\Desktop\outlier\outlier-detection
python main.py
```

The application window will open with three main tabs:
- **Buildings & Sensors**: Manage your building hierarchy
- **Settings**: Configure detection parameters
- **Detection Results**: Run detection and view results

### Step 2: Create a Building

1. Go to the **"Buildings & Sensors"** tab
2. Click **"New Building"** button
3. Fill in:
   - **Name**: e.g., "Main Office Building"
   - **Address**: e.g., "123 Main St"
   - **Description**: (optional)
4. Click **OK**

### Step 3: Add Rooms

1. Select your building from the left panel
2. Click **"New Room"** button
3. Fill in:
   - **Name**: e.g., "Conference Room A"
   - **Floor**: e.g., "2"
   - **Description**: (optional)
4. Click **OK**
5. Repeat for all rooms in the building

### Step 4: Assign Sensors to Rooms

1. Select a room from the middle panel
2. Click **"Add Sensor"** button
3. Fill in:
   - **Sensor ID**: The UUID from your sensor data filename
     - Example: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
     - This is the first 36 characters of your `.json_line` filenames
   - **Name**: Friendly name, e.g., "Wall Sensor North"
   - **Data Path**: (optional) Custom path to data files
   - **Description**: (optional)
   - **Active**: Check to include in detection
4. Click **OK**
5. Repeat for all sensors in the room

### Step 5: Review Settings (Optional)

1. Go to the **"Settings"** tab
2. Browse through different categories:
   - General, Moisture Intrusion, Condensation, etc.
3. Adjust thresholds if needed (defaults are usually good)
4. Click **"Save Settings"** if you make changes

### Step 6: Run Detection

1. Go to the **"Detection Results"** tab
2. Select your building from the dropdown
3. Click **"Browse"** next to Data Directory
4. Navigate to your sensor data folder (contains `.json_line` files)
   - Default from original script: `G:\My Drive\Rozinet\RMind\Clients\Senzomatic\Data_sensors\exported_data_2026-01-22\exported_data`
5. Click **"Run Detection"**
6. Wait for detection to complete (progress bar shows status)

### Step 7: View Results

After detection completes:

**Summary Panel:**
- Total devices analyzed
- Total problems detected
- Average health score
- Breakdown by type and severity

**Problems Tab:**
- Filter by problem type or severity
- View detailed list of all detected issues
- Columns show: Device, Type, Severity, Dates, Duration, Description

**Health Scores Tab:**
- Overall health score per sensor (0-100)
- Breakdown by problem category
- Color-coded: Green (excellent) to Red (poor)

**Export:**
- Click **"Export Results"** to save as CSV

## Tips

### Sensor ID Format
- Sensor IDs are UUIDs in format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- Find them in your data filenames:
  ```
  a1b2c3d4-e5f6-7890-abcd-ef1234567890_temperature_ambient_celsius.json_line
  └─────────── Sensor ID ─────────────┘
  ```

### Data Organization
Your data directory should look like:
```
exported_data/
├── {uuid1}_temperature_ambient_celsius.json_line
├── {uuid1}_rel_humidity_ambient_pct.json_line
├── {uuid1}_rel_humidity_cavity_pct.json_line
├── {uuid1}_moisture_resistance_pct.json_line
├── {uuid2}_temperature_ambient_celsius.json_line
└── ...
```

### Saving Your Work
- Building/room/sensor data saves automatically to `buildings.json`
- Settings save when you click "Save Settings" to `config.json`
- Both files are in the same directory as `main.py`

### Keyboard Shortcuts
- File → Save All: Manually save all data
- File → Exit: Close application (prompts to save)

## Troubleshooting

**"Could not import detection functions"**
- Make sure `outlier.py` is in the same directory as `main.py`
- Install required packages: `pip install numpy pandas scipy matplotlib`

**No sensors loaded / "Loaded 0 devices"**
- Double-check sensor IDs match your data filenames exactly
- Verify data directory path is correct
- Ensure sensors are marked as "Active"

**Detection runs but finds no problems**
- This might be normal if sensors are healthy!
- Check that you have at least 48 hours of data per sensor
- Try lowering thresholds in Settings tab for testing

## Example Workflow

```
1. Launch: python main.py
2. Buildings Tab:
   - New Building: "Office HQ"
   - New Room: "Meeting Room 1", Floor "2"
   - Add Sensor: ID from filename, Name "North Wall Sensor"
3. Detection Tab:
   - Select "Office HQ"
   - Browse to: G:\...\exported_data
   - Run Detection
4. View results in Problems and Health Scores tabs
5. Export to CSV for reporting
```

## Next Steps

- Read [README_MVC.md](README_MVC.md) for detailed architecture information
- Explore Settings tab to customize detection sensitivity
- Run detection on different time periods by updating sensor data
- Export results for stakeholder reports
