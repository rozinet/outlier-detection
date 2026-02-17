"""Controller for building management operations."""

import uuid
from typing import Optional
from models import Building, Room, Sensor, BuildingRegistry


class BuildingController:
    """Handles building, room, and sensor management operations."""

    def __init__(self, registry: BuildingRegistry):
        """Initialize building controller.

        Args:
            registry: BuildingRegistry instance
        """
        self.registry = registry

    def create_building(self, name: str, address: str = "", description: str = "") -> Building:
        """Create a new building.

        Args:
            name: Building name
            address: Building address
            description: Building description

        Returns:
            Created Building object
        """
        building_id = str(uuid.uuid4())
        building = Building(
            building_id=building_id,
            name=name,
            address=address,
            description=description
        )
        self.registry.add_building(building)
        self.registry.save()
        return building

    def delete_building(self, building_id: str) -> bool:
        """Delete a building.

        Args:
            building_id: Building ID

        Returns:
            True if deleted, False if not found
        """
        removed = self.registry.remove_building(building_id)
        if removed:
            self.registry.save()
            return True
        return False

    def update_building(self, building_id: str, name: Optional[str] = None,
                       address: Optional[str] = None, description: Optional[str] = None) -> Optional[Building]:
        """Update building information.

        Args:
            building_id: Building ID
            name: New name (optional)
            address: New address (optional)
            description: New description (optional)

        Returns:
            Updated Building or None if not found
        """
        building = self.registry.get_building(building_id)
        if not building:
            return None

        if name is not None:
            building.name = name
        if address is not None:
            building.address = address
        if description is not None:
            building.description = description

        self.registry.save()
        return building

    def create_room(self, building_id: str, name: str, floor: Optional[str] = None,
                   description: str = "") -> Optional[Room]:
        """Create a new room in a building.

        Args:
            building_id: Building ID
            name: Room name
            floor: Floor number/name
            description: Room description

        Returns:
            Created Room object or None if building not found
        """
        building = self.registry.get_building(building_id)
        if not building:
            return None

        room_id = str(uuid.uuid4())
        room = Room(
            room_id=room_id,
            name=name,
            building_id=building_id,
            floor=floor,
            description=description
        )
        building.add_room(room)
        self.registry.save()
        return room

    def delete_room(self, building_id: str, room_id: str) -> bool:
        """Delete a room from a building.

        Args:
            building_id: Building ID
            room_id: Room ID

        Returns:
            True if deleted, False if not found
        """
        building = self.registry.get_building(building_id)
        if not building:
            return False

        removed = building.remove_room(room_id)
        if removed:
            self.registry.save()
            return True
        return False

    def update_room(self, building_id: str, room_id: str, name: Optional[str] = None,
                   floor: Optional[str] = None, description: Optional[str] = None) -> Optional[Room]:
        """Update room information.

        Args:
            building_id: Building ID
            room_id: Room ID
            name: New name (optional)
            floor: New floor (optional)
            description: New description (optional)

        Returns:
            Updated Room or None if not found
        """
        building = self.registry.get_building(building_id)
        if not building:
            return None

        room = building.get_room(room_id)
        if not room:
            return None

        if name is not None:
            room.name = name
        if floor is not None:
            room.floor = floor
        if description is not None:
            room.description = description

        self.registry.save()
        return room

    def create_sensor(self, building_id: str, room_id: str, sensor_id: str, name: str,
                     data_path: Optional[str] = None, description: str = "") -> Optional[Sensor]:
        """Create/add a sensor to a room.

        Args:
            building_id: Building ID
            room_id: Room ID
            sensor_id: Sensor device ID
            name: Sensor name
            data_path: Path to sensor data (optional)
            description: Sensor description

        Returns:
            Created Sensor object or None if building/room not found
        """
        building = self.registry.get_building(building_id)
        if not building:
            return None

        room = building.get_room(room_id)
        if not room:
            return None

        sensor = Sensor(
            sensor_id=sensor_id,
            name=name,
            room_id=room_id,
            data_path=data_path,
            description=description
        )
        room.add_sensor(sensor)
        self.registry.save()
        return sensor

    def delete_sensor(self, building_id: str, room_id: str, sensor_id: str) -> bool:
        """Delete a sensor from a room.

        Args:
            building_id: Building ID
            room_id: Room ID
            sensor_id: Sensor ID

        Returns:
            True if deleted, False if not found
        """
        building = self.registry.get_building(building_id)
        if not building:
            return False

        room = building.get_room(room_id)
        if not room:
            return False

        removed = room.remove_sensor(sensor_id)
        if removed:
            self.registry.save()
            return True
        return False

    def update_sensor(self, building_id: str, room_id: str, sensor_id: str,
                     name: Optional[str] = None, data_path: Optional[str] = None,
                     description: Optional[str] = None, is_active: Optional[bool] = None) -> Optional[Sensor]:
        """Update sensor information.

        Args:
            building_id: Building ID
            room_id: Room ID
            sensor_id: Sensor ID
            name: New name (optional)
            data_path: New data path (optional)
            description: New description (optional)
            is_active: New active status (optional)

        Returns:
            Updated Sensor or None if not found
        """
        building = self.registry.get_building(building_id)
        if not building:
            return None

        room = building.get_room(room_id)
        if not room:
            return None

        sensor = room.get_sensor(sensor_id)
        if not sensor:
            return None

        if name is not None:
            sensor.name = name
        if data_path is not None:
            sensor.data_path = data_path
        if description is not None:
            sensor.description = description
        if is_active is not None:
            sensor.is_active = is_active

        self.registry.save()
        return sensor

    def get_all_buildings(self) -> list[Building]:
        """Get all buildings.

        Returns:
            List of Building objects
        """
        return self.registry.get_all_buildings()

    def get_building(self, building_id: str) -> Optional[Building]:
        """Get a building by ID.

        Args:
            building_id: Building ID

        Returns:
            Building object or None if not found
        """
        return self.registry.get_building(building_id)
