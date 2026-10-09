"""
Data preprocessing module for satellite images
Handles loading, normalization, and augmentation of satellite imagery
"""

import numpy as np
import cv2
from pathlib import Path
try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class SatelliteImageProcessor:
    """Process satellite images for deforestation detection"""
    
    def __init__(self, img_size=256):
        self.img_size = img_size
        
    def load_image(self, image_path):
        """
        Load satellite image from file
        Supports various formats including GeoTIFF (if rasterio is available)
        """
        # Try using rasterio for GeoTIFF files if available
        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(image_path) as src:
                    # Read RGB bands (assuming standard satellite imagery)
                    if src.count >= 3:
                        img = np.stack([src.read(i) for i in range(1, 4)], axis=-1)
                    else:
                        img = src.read(1)
                        img = np.stack([img] * 3, axis=-1)

                    # Normalize to 0-255
                    img = img.astype(np.float32)
                    for i in range(img.shape[2]):
                        band = img[:, :, i]
                        if band.max() > 0:
                            img[:, :, i] = (band - band.min()) / (band.max() - band.min()) * 255

                    img = img.astype(np.uint8)
                    return cv2.resize(img, (self.img_size, self.img_size))
            except:
                pass  # Fall through to OpenCV

        # Fallback to OpenCV for standard image formats
        img = cv2.imread(str(image_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, (self.img_size, self.img_size))

        return None
    
    def normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    def calculate_ndvi(self, image):
        """
        Calculate Normalized Difference Vegetation Index (NDVI)
        NDVI = (NIR - Red) / (NIR + Red)
        For RGB images, we'll use green channel as a proxy for vegetation
        """
        # Using green channel as vegetation indicator for RGB images
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 0].astype(np.float32)
        
        # Simple vegetation index using green and red
        ndvi = (green - red) / (green + red + 1e-8)
        return ndvi
    
    def detect_forest_coverage(self, image, threshold=0.3):
        """
        Estimate forest coverage based on vegetation index
        Returns percentage of area covered by forest
        """
        ndvi = self.calculate_ndvi(image)
        forest_mask = ndvi > threshold
        forest_coverage = np.sum(forest_mask) / forest_mask.size
        return forest_coverage
    
    def create_training_data_generator(self, train_dir, batch_size=16):
        """
        Create data generator for training
        Expects directory structure: train_dir/class_name/images
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        return train_generator
    
    def augment_image(self, image):
        """Apply random augmentations to image"""
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-20, 20)
            center = (self.img_size // 2, self.img_size // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (self.img_size, self.img_size))
        
        # Random flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)  # Horizontal flip
        
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)  # Vertical flip
        
        return image


def generate_sample_dataset(output_dir, num_samples=100):
    """
    Generate synthetic satellite images for testing
    Creates deforested and forested images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    forest_dir = output_dir / 'forest'
    deforested_dir = output_dir / 'deforested'
    forest_dir.mkdir(exist_ok=True)
    deforested_dir.mkdir(exist_ok=True)
    
    print(f"Generating {num_samples} sample images...")
    
    for i in range(num_samples):
        # Generate random base image
        img_size = 256
        image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        
        # Add green tones for forest
        if i < num_samples // 2:
            # Forested image - more green
            image[:, :, 1] = np.clip(image[:, :, 1] + 50, 0, 255)
            image[:, :, 0] = np.clip(image[:, :, 0] - 30, 0, 255)
            image[:, :, 2] = np.clip(image[:, :, 2] - 30, 0, 255)
            
            # Add some vegetation patterns
            for _ in range(20):
                x, y = np.random.randint(0, img_size, 2)
                radius = np.random.randint(10, 30)
                cv2.circle(image, (x, y), radius, (0, 200, 50), -1)
            
            cv2.imwrite(str(forest_dir / f'forest_{i:04d}.png'), image)
        else:
            # Deforested image - more brown/gray
            image[:, :, 0] = np.clip(image[:, :, 0] + 30, 0, 255)
            image[:, :, 1] = np.clip(image[:, :, 1] - 20, 0, 255)
            image[:, :, 2] = np.clip(image[:, :, 2] - 20, 0, 255)
            
            # Add barren patches
            for _ in range(15):
                x, y = np.random.randint(0, img_size, 2)
                radius = np.random.randint(15, 40)
                cv2.circle(image, (x, y), radius, (150, 100, 50), -1)
            
            cv2.imwrite(str(deforested_dir / f'deforested_{i-num_samples//2:04d}.png'), image)
    
    print(f"Sample dataset generated in {output_dir}")
    print(f"  - Forested images: {num_samples // 2}")
    print(f"  - Deforested images: {num_samples // 2}")


if __name__ == "__main__":
    # Test the processor
    processor = SatelliteImageProcessor(img_size=256)
    
    # Generate sample dataset
    generate_sample_dataset('data/sample_train', num_samples=100)
