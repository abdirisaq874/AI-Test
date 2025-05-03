"""
Configuration module for the Enhanced AI Creation Studio application
"""
import os

# Constants for app IDs
TEXT_TO_IMAGE_APP_ID = "c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network"
IMAGE_TO_3D_APP_ID = "f0b5f319156c4819b9827000b17e511a.node3.openfabric.network"

# Create output directories
VECTOR_DB_PATH = "output/vector_db"
MEMORY_DIR = "output/memory"
IMAGES_DIR = "output/images"
MODELS_DIR = "output/models"

for directory in [MEMORY_DIR, IMAGES_DIR, MODELS_DIR, VECTOR_DB_PATH]:
    os.makedirs(directory, exist_ok=True)

# Mock configuration for testing without actual config callback
class MockConfig:
    def __init__(self):
        self.app_ids = [
            TEXT_TO_IMAGE_APP_ID,  # Text to Image
            IMAGE_TO_3D_APP_ID     # Image to 3D
        ]

configurations = {'super-user': MockConfig()}