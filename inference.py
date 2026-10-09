"""
Inference pipeline for deforestation detection
Handles loading models and making predictions on satellite images
"""

import numpy as np
import cv2
from pathlib import Path
import config
from data_preprocessing import SatelliteImageProcessor
from model import DeforestationDetector
import tensorflow as tf


class DeforestationPredictor:
    """Predictor class for deforestation detection"""
    
    def __init__(self, model_path=None, img_size=256):
        self.img_size = img_size
        self.processor = SatelliteImageProcessor(img_size=img_size)
        self.detector = DeforestationDetector(img_size=img_size)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("Warning: No model loaded. Please provide a valid model path.")
            self.model = None
    
    def load_model(self, model_path):
        """Load a trained model"""
        print(f"Loading model from {model_path}...")
        self.model = self.detector.load_model(model_path)
        print("Model loaded successfully!")
    
    def predict_single_image(self, image_path, threshold=0.5):
        """
        Predict deforestation for a single image
        Returns: (is_deforested, confidence, forest_coverage)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Load and preprocess image
        image = self.processor.load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Calculate forest coverage
        forest_coverage = self.processor.detect_forest_coverage(
            image, 
            threshold=config.FOREST_COVERAGE_THRESHOLD
        )
        
        # Normalize and predict
        normalized_image = self.processor.normalize_image(image)
        prediction = self.detector.predict(normalized_image)
        
        # Apply threshold
        is_deforested = prediction > threshold
        
        return {
            'is_deforested': bool(is_deforested),
            'confidence': float(prediction),
            'forest_coverage': float(forest_coverage),
            'image_path': str(image_path)
        }
    
    def predict_batch(self, image_paths, threshold=0.5):
        """
        Predict deforestation for multiple images
        Returns list of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path, threshold)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'is_deforested': None,
                    'confidence': None,
                    'forest_coverage': None,
                    'image_path': str(image_path),
                    'error': str(e)
                })
        return results
    
    def predict_directory(self, directory_path, threshold=0.5, pattern='*.png'):
        """
        Predict deforestation for all images in a directory
        """
        directory = Path(directory_path)
        image_paths = list(directory.glob(pattern))
        
        if not image_paths:
            print(f"No images found in {directory_path}")
            return []
        
        print(f"Found {len(image_paths)} images to process...")
        return self.predict_batch(image_paths, threshold)
    
    def compare_images(self, before_path, after_path, threshold=0.3):
        """
        Compare two images (before and after) to detect change
        Returns change magnitude and classification
        """
        # Load both images
        before_img = self.processor.load_image(before_path)
        after_img = self.processor.load_image(after_path)
        
        if before_img is None or after_img is None:
            raise ValueError("Could not load one or both images")
        
        # Calculate forest coverage for both
        before_coverage = self.processor.detect_forest_coverage(
            before_img, 
            threshold=config.FOREST_COVERAGE_THRESHOLD
        )
        after_coverage = self.processor.detect_forest_coverage(
            after_img, 
            threshold=config.FOREST_COVERAGE_THRESHOLD
        )
        
        # Calculate change
        coverage_change = before_coverage - after_coverage
        
        # Classify change
        if coverage_change > config.CHANGE_THRESHOLD:
            change_type = "deforestation"
        elif coverage_change < -config.CHANGE_THRESHOLD:
            change_type = "regrowth"
        else:
            change_type = "no significant change"
        
        return {
            'before_coverage': float(before_coverage),
            'after_coverage': float(after_coverage),
            'coverage_change': float(coverage_change),
            'change_type': change_type,
            'before_path': str(before_path),
            'after_path': str(after_path)
        }
    
    def predict_with_heatmap(self, image_path, threshold=0.5):
        """
        Predict deforestation and generate a heatmap visualization
        Returns prediction and heatmap image
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Load image
        image = self.processor.load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Create patches for heatmap
        patch_size = 64
        heatmap = np.zeros((self.img_size // patch_size, self.img_size // patch_size))
        
        normalized_image = self.processor.normalize_image(image)
        
        # Slide window prediction
        for i in range(0, self.img_size - patch_size, patch_size):
            for j in range(0, self.img_size - patch_size, patch_size):
                patch = normalized_image[i:i+patch_size, j:j+patch_size]
                if patch.shape != (patch_size, patch_size, 3):
                    continue
                
                patch_resized = cv2.resize(patch, (self.img_size, self.img_size))
                patch_resized = np.expand_dims(patch_resized, axis=0)
                prediction = self.detector.predict(patch_resized)
                
                heatmap[i//patch_size, j//patch_size] = prediction
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (self.img_size, self.img_size))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        
        # Overall prediction
        overall_prediction = self.predict_single_image(image_path, threshold)
        
        return {
            'prediction': overall_prediction,
            'heatmap': overlay,
            'heatmap_raw': heatmap_resized
        }


def quick_predict(image_path, model_path):
    """
    Quick prediction function for single image
    """
    predictor = DeforestationPredictor(model_path)
    result = predictor.predict_single_image(image_path)
    
    print("\n" + "=" * 50)
    print("Deforestation Detection Result")
    print("=" * 50)
    print(f"Image: {result['image_path']}")
    print(f"Deforested: {'YES' if result['is_deforested'] else 'NO'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Forest Coverage: {result['forest_coverage']:.2%}")
    print("=" * 50)
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <image_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    quick_predict(image_path, model_path)
