"""Main application controller."""

from models import ConfigManager, BuildingRegistry
from .building_controller import BuildingController
from .detection_controller import DetectionController


class AppController:
    """Main application controller coordinating all operations."""

    def __init__(self):
        """Initialize application controller."""
        self.config = ConfigManager()
        self.building_registry = BuildingRegistry()
        self.building_controller = BuildingController(self.building_registry)
        self.detection_controller = DetectionController(self.config)

    def get_config(self) -> ConfigManager:
        """Get configuration manager.

        Returns:
            ConfigManager instance
        """
        return self.config

    def get_building_controller(self) -> BuildingController:
        """Get building controller.

        Returns:
            BuildingController instance
        """
        return self.building_controller

    def get_detection_controller(self) -> DetectionController:
        """Get detection controller.

        Returns:
            DetectionController instance
        """
        return self.detection_controller

    def save_all(self):
        """Save all data (config and buildings)."""
        self.config.save()
        self.building_registry.save()
