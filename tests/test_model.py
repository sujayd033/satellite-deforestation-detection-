"""
Unit tests for model module
"""

import pytest
import numpy as np
import tensorflow as tf
from model import DeforestationDetector


class TestDeforestationDetector:
    """Test cases for DeforestationDetector class"""

    def test_initialization(self):
        """Test detector initialization"""
        detector = DeforestationDetector(img_size=256, backbone='simple')
        assert detector.img_size == 256
        assert detector.backbone == 'simple'

    def test_build_simple_model(self):
        """Test building simple CNN model"""
        detector = DeforestationDetector(img_size=256, backbone='simple')
        model = detector.build_model(num_classes=2)
        assert model is not None
        assert model.input_shape == (None, 256, 256, 3)

    def test_compile_model(self):
        """Test model compilation"""
        detector = DeforestationDetector(img_size=256, backbone='simple')
        model = detector.build_model(num_classes=2)
        detector.compile_model(learning_rate=0.001)
        assert detector.model is not None
        assert detector.model.optimizer is not None

    def test_model_predict(self):
        """Test model prediction"""
        detector = DeforestationDetector(img_size=256, backbone='simple')
        model = detector.build_model(num_classes=2)
        detector.compile_model(learning_rate=0.001)
        
        # Create dummy input
        dummy_image = np.random.random((1, 256, 256, 3))
        prediction = detector.predict(dummy_image)
        assert 0.0 <= prediction <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
