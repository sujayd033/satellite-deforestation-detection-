# API Documentation

## Core Modules

### SatelliteImageProcessor

Class for processing satellite images and calculating vegetation metrics.

#### Methods

##### `__init__(img_size=256)`
Initialize the processor with target image size.

**Parameters:**
- `img_size` (int): Target size for image resizing (default: 256)

##### `load_image(image_path)`
Load satellite image from file.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- `numpy.ndarray`: Loaded and resized image

##### `normalize_image(image)`
Normalize image to [0, 1] range.

**Parameters:**
- `image` (numpy.ndarray): Input image

**Returns:**
- `numpy.ndarray`: Normalized image

##### `calculate_ndvi(image)`
Calculate Normalized Difference Vegetation Index.

**Parameters:**
- `image` (numpy.ndarray): RGB image

**Returns:**
- `numpy.ndarray`: NDVI values

##### `detect_forest_coverage(image, threshold=0.3)`
Estimate forest coverage based on vegetation index.

**Parameters:**
- `image` (numpy.ndarray): RGB image
- `threshold` (float): Vegetation threshold (default: 0.3)

**Returns:**
- `float`: Forest coverage percentage

---

### DeforestationDetector

Deep learning model for deforestation detection.

#### Methods

##### `__init__(img_size=256, backbone='efficientnetb0')`
Initialize the detector.

**Parameters:**
- `img_size` (int): Input image size
- `backbone` (str): Model architecture ('resnet50', 'efficientnetb0', 'vgg16', 'simple')

##### `build_model(num_classes=2)`
Build the deforestation detection model.

**Parameters:**
- `num_classes` (int): Number of output classes (default: 2)

**Returns:**
- `tf.keras.Model`: Compiled model

##### `compile_model(learning_rate=0.001)`
Compile the model with optimizer and loss.

**Parameters:**
- `learning_rate` (float): Learning rate (default: 0.001)

##### `train(train_generator, val_generator, epochs, model_path)`
Train the model.

**Parameters:**
- `train_generator`: Training data generator
- `val_generator`: Validation data generator
- `epochs` (int): Number of training epochs
- `model_path` (str): Path to save model

**Returns:**
- `History`: Training history

##### `predict(image)`
Make prediction on a single image.

**Parameters:**
- `image` (numpy.ndarray): Input image

**Returns:**
- `float`: Prediction probability

---

### DeforestationPredictor

Predictor class for making predictions.

#### Methods

##### `__init__(model_path=None, img_size=256)`
Initialize the predictor.

**Parameters:**
- `model_path` (str): Path to trained model
- `img_size` (int): Input image size

##### `predict_single_image(image_path, threshold=0.5)`
Predict deforestation for a single image.

**Parameters:**
- `image_path` (str): Path to image
- `threshold` (float): Detection threshold (default: 0.5)

**Returns:**
- `dict`: Prediction results containing:
  - `is_deforested` (bool): Classification result
  - `confidence` (float): Prediction confidence
  - `forest_coverage` (float): Forest coverage percentage
  - `image_path` (str): Path to image

##### `predict_batch(image_paths, threshold=0.5)`
Predict deforestation for multiple images.

**Parameters:**
- `image_paths` (list): List of image paths
- `threshold` (float): Detection threshold (default: 0.5)

**Returns:**
- `list`: List of prediction dictionaries

##### `compare_images(before_path, after_path, threshold=0.3)`
Compare two images to detect change.

**Parameters:**
- `before_path` (str): Path to before image
- `after_path` (str): Path to after image
- `threshold` (float): Change threshold (default: 0.3)

**Returns:**
- `dict`: Comparison results containing:
  - `before_coverage` (float): Before forest coverage
  - `after_coverage` (float): After forest coverage
  - `coverage_change` (float): Coverage change
  - `change_type` (str): Type of change

---

### ResultVisualizer

Visualization tools for results.

#### Methods

##### `__init__(output_dir='results')`
Initialize the visualizer.

**Parameters:**
- `output_dir` (str): Directory to save visualizations

##### `plot_prediction(image, prediction, save_path=None)`
Plot image with prediction overlay.

**Parameters:**
- `image` (numpy.ndarray): Input image
- `prediction` (dict): Prediction results
- `save_path` (str): Path to save visualization

##### `plot_heatmap(image, heatmap, prediction, save_path=None)`
Plot image with deforestation heatmap.

**Parameters:**
- `image` (numpy.ndarray): Input image
- `heatmap` (numpy.ndarray): Heatmap values
- `prediction` (dict): Prediction results
- `save_path` (str): Path to save visualization

##### `plot_comparison(before_img, after_img, comparison_result, save_path=None)`
Plot before/after comparison.

**Parameters:**
- `before_img` (numpy.ndarray): Before image
- `after_img` (numpy.ndarray): After image
- `comparison_result` (dict): Comparison results
- `save_path` (str): Path to save visualization

##### `create_report(results, save_path=None)`
Create a summary report.

**Parameters:**
- `results` (list): List of prediction results
- `save_path` (str): Path to save report

**Returns:**
- `str`: Report text

---

## Configuration

### Configuration Parameters

```python
# Model Configuration
IMG_SIZE = 256              # Image size for processing
BATCH_SIZE = 16             # Training batch size
EPOCHS = 20                 # Number of training epochs
LEARNING_RATE = 0.001       # Learning rate

# Data Configuration
TRAIN_SPLIT = 0.8           # Training data split
VAL_SPLIT = 0.1             # Validation data split
TEST_SPLIT = 0.1            # Test data split

# Detection Thresholds
FOREST_COVERAGE_THRESHOLD = 0.3  # Minimum forest coverage
CHANGE_THRESHOLD = 0.2           # Minimum change to flag as deforestation
```

---

## CLI Interface

### Training Command

```bash
python main.py train [OPTIONS]
```

**Options:**
- `--epochs INT`: Number of training epochs
- `--batch-size INT`: Batch size for training
- `--backbone TEXT`: Model architecture
- `--plot-history`: Generate training history plots

### Prediction Command

```bash
python main.py predict [OPTIONS]
```

**Options:**
- `--model TEXT`: Path to trained model
- `--image TEXT`: Path to single image
- `--directory TEXT`: Path to directory of images
- `--pattern TEXT`: File pattern for directory search
- `--compare TEXT TEXT`: Compare two images
- `--threshold FLOAT`: Detection threshold
- `--visualize`: Generate visualization
- `--heatmap`: Generate heatmap visualization
