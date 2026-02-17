"""Building, room, and sensor data structures."""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sensor:
    """Represents a sensor device."""
    sensor_id: str
    name: str
    room_id: Optional[str] = None
    data_path: Optional[str] = None
    description: str = ""
    is_active: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensor_id": self.sensor_id,
            "name": self.name,
            "room_id": self.room_id,
            "data_path": self.data_path,
            "description": self.description,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Sensor':
        """Create Sensor from dictionary."""
        return cls(**data)


@dataclass
class Room:
    """Represents a room in a building."""
    room_id: str
    name: str
    building_id: Optional[str] = None
    floor: Optional[str] = None
    description: str = ""
    sensors: list[Sensor] = field(default_factory=list)

    def add_sensor(self, sensor: Sensor) -> None:
        """Add a sensor to this room.

        Args:
            sensor: Sensor to add
        """
        sensor.room_id = self.room_id
        if sensor not in self.sensors:
            self.sensors.append(sensor)

    def remove_sensor(self, sensor_id: str) -> Optional[Sensor]:
        """Remove a sensor from this room.

        Args:
            sensor_id: ID of sensor to remove

        Returns:
            Removed sensor or None if not found
        """
        for i, sensor in enumerate(self.sensors):
            if sensor.sensor_id == sensor_id:
                removed = self.sensors.pop(i)
                removed.room_id = None
                return removed
        return None

    def get_sensor(self, sensor_id: str) -> Optional[Sensor]:
        """Get sensor by ID.

        Args:
            sensor_id: Sensor ID

        Returns:
            Sensor or None if not found
        """
        for sensor in self.sensors:
            if sensor.sensor_id == sensor_id:
                return sensor
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "room_id": self.room_id,
            "name": self.name,
            "building_id": self.building_id,
            "floor": self.floor,
            "description": self.description,
            "sensors": [s.to_dict() for s in self.sensors],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Room':
        """Create Room from dictionary."""
        sensors_data = data.pop("sensors", [])
        room = cls(**data)
        room.sensors = [Sensor.from_dict(s) for s in sensors_data]
        return room


@dataclass
class Building:
    """Represents a building with rooms and sensors."""
    building_id: str
    name: str
    address: str = ""
    description: str = ""
    rooms: list[Room] = field(default_factory=list)

    def add_room(self, room: Room) -> None:
        """Add a room to this building.

        Args:
            room: Room to add
        """
        room.building_id = self.building_id
        if room not in self.rooms:
            self.rooms.append(room)

    def remove_room(self, room_id: str) -> Optional[Room]:
        """Remove a room from this building.

        Args:
            room_id: ID of room to remove

        Returns:
            Removed room or None if not found
        """
        for i, room in enumerate(self.rooms):
            if room.room_id == room_id:
                removed = self.rooms.pop(i)
                removed.building_id = None
                return removed
        return None

    def get_room(self, room_id: str) -> Optional[Room]:
        """Get room by ID.

        Args:
            room_id: Room ID

        Returns:
            Room or None if not found
        """
        for room in self.rooms:
            if room.room_id == room_id:
                return room
        return None

    def get_all_sensors(self) -> list[Sensor]:
        """Get all sensors in this building.

        Returns:
            List of all sensors
        """
        sensors = []
        for room in self.rooms:
            sensors.extend(room.sensors)
        return sensors

    def find_sensor(self, sensor_id: str) -> Optional[tuple[Room, Sensor]]:
        """Find a sensor and its room.

        Args:
            sensor_id: Sensor ID

        Returns:
            Tuple of (room, sensor) or None if not found
        """
        for room in self.rooms:
            sensor = room.get_sensor(sensor_id)
            if sensor:
                return room, sensor
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "building_id": self.building_id,
            "name": self.name,
            "address": self.address,
            "description": self.description,
            "rooms": [r.to_dict() for r in self.rooms],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Building':
        """Create Building from dictionary."""
        rooms_data = data.pop("rooms", [])
        building = cls(**data)
        building.rooms = [Room.from_dict(r) for r in rooms_data]
        return building


class BuildingRegistry:
    """Registry for managing multiple buildings."""

    def __init__(self, registry_file: str = "buildings.json"):
        """Initialize building registry.

        Args:
            registry_file: Path to registry file
        """
        self.registry_file = registry_file
        self.buildings: dict[str, Building] = {}
        self.load()

    def add_building(self, building: Building) -> None:
        """Add a building to the registry.

        Args:
            building: Building to add
        """
        self.buildings[building.building_id] = building

    def remove_building(self, building_id: str) -> Optional[Building]:
        """Remove a building from the registry.

        Args:
            building_id: Building ID

        Returns:
            Removed building or None if not found
        """
        return self.buildings.pop(building_id, None)

    def get_building(self, building_id: str) -> Optional[Building]:
        """Get building by ID.

        Args:
            building_id: Building ID

        Returns:
            Building or None if not found
        """
        return self.buildings.get(building_id)

    def get_all_buildings(self) -> list[Building]:
        """Get all buildings.

        Returns:
            List of all buildings
        """
        return list(self.buildings.values())

    def load(self) -> None:
        """Load buildings from file."""
        import os
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.buildings = {
                        bid: Building.from_dict(bdata)
                        for bid, bdata in data.items()
                    }
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: Failed to load buildings from {self.registry_file}: {e}")

    def save(self) -> None:
        """Save buildings to file."""
        try:
            data = {
                bid: building.to_dict()
                for bid, building in self.buildings.items()
            }
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"Error: Failed to save buildings to {self.registry_file}: {e}")
