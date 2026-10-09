"""
Unit tests for data preprocessing module
"""

import pytest
import numpy as np
from data_preprocessing import SatelliteImageProcessor


class TestSatelliteImageProcessor:
    """Test cases for SatelliteImageProcessor class"""

    def test_initialization(self):
        """Test processor initialization"""
        processor = SatelliteImageProcessor(img_size=256)
        assert processor.img_size == 256

    def test_initialization_default(self):
        """Test processor initialization with default size"""
        processor = SatelliteImageProcessor()
        assert processor.img_size == 256

    def test_normalize_image(self):
        """Test image normalization"""
        processor = SatelliteImageProcessor()
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        normalized = processor.normalize_image(image)
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0

    def test_calculate_ndvi(self):
        """Test NDVI calculation"""
        processor = SatelliteImageProcessor()
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        ndvi = processor.calculate_ndvi(image)
        assert ndvi.shape == (256, 256)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_detect_forest_coverage(self):
        """Test forest coverage detection"""
        processor = SatelliteImageProcessor()
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        coverage = processor.detect_forest_coverage(image)
        assert 0.0 <= coverage <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
