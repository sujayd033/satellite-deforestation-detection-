# Deforestation Detection from Satellite Images

A deep learning system for detecting deforestation in satellite imagery using convolutional neural networks (CNNs) with transfer learning.

## Features

- **Deep Learning Model**: CNN-based classification using transfer learning (ResNet50, EfficientNetB0, or VGG16)
- **Data Preprocessing**: Handles satellite imagery including GeoTIFF files
- **Vegetation Analysis**: Calculates NDVI (Normalized Difference Vegetation Index) and forest coverage
- **Change Detection**: Compare before/after images to detect deforestation over time
- **Batch Processing**: Process entire directories of satellite images
- **Visualization**: Generate heatmaps, comparison plots, and detailed reports
- **Sample Data Generation**: Automatically generates synthetic training data for testing

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── config.py                 # Configuration settings
├── data_preprocessing.py     # Image processing and data generation
├── model.py                  # Deep learning model architectures
├── train.py                  # Training script
├── inference.py              # Prediction and inference pipeline
├── visualization.py          # Visualization tools
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage

### Train the Model

Train a new deforestation detection model:

```bash
python main.py train --epochs 30 --batch-size 32
```

Options:
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 16)
- `--backbone`: Model architecture (resnet50, efficientnetb0, vgg16)
- `--plot-history`: Generate training history plots

The training script will automatically generate a sample dataset if none exists.

### Make Predictions

#### Single Image Prediction

```bash
python main.py predict --image path/to/image.png --visualize
```

#### Batch Prediction (Directory)

```bash
python main.py predict --directory path/to/images --visualize
```

#### Compare Two Images (Before/After)

```bash
python main.py predict --compare before.png after.png --visualize
```

#### Generate Heatmap Visualization

```bash
python main.py predict --image path/to/image.png --heatmap --visualize
```

### Options for Prediction

- `--model`: Path to trained model (default: models/deforestation_detector.h5)
- `--threshold`: Detection threshold (default: 0.5)
- `--pattern`: File pattern for directory search (default: *.png)
- `--visualize`: Generate visualization of results
- `--heatmap`: Generate heatmap visualization

## Configuration

Edit `config.py` to customize:

- `IMG_SIZE`: Image size for processing (default: 256)
- `BATCH_SIZE`: Training batch size (default: 16)
- `EPOCHS`: Number of training epochs (default: 20)
- `LEARNING_RATE`: Learning rate for optimizer (default: 0.001)
- `FOREST_COVERAGE_THRESHOLD`: Minimum forest coverage threshold (default: 0.3)
- `CHANGE_THRESHOLD`: Minimum change to flag as deforestation (default: 0.2)

## Model Architecture

The system uses transfer learning with pre-trained backbones:

1. **ResNet50**: Deep residual network with 50 layers
2. **EfficientNetB0**: Efficient convolutional neural network
3. **VGG16**: Classic VGG architecture with 16 layers

The model includes:
- Pre-trained backbone (frozen initially)
- Global average pooling
- Dense layers with dropout for regularization
- Binary classification output (sigmoid activation)

## Vegetation Analysis

The system calculates:
- **NDVI (Normalized Difference Vegetation Index)**: Measures vegetation health
- **Forest Coverage**: Percentage of area covered by forest
- **Change Detection**: Compares forest coverage between time periods

## Output

Results are saved in the `results/` directory:
- `prediction.png`: Single image prediction visualization
- `heatmap_prediction.png`: Heatmap showing deforestation probability
- `batch_predictions.png`: Grid of batch predictions
- `comparison.png`: Before/after comparison
- `detection_report.txt`: Detailed text report
- `training_history.png`: Training loss and accuracy plots

Trained models are saved in the `models/` directory.

## Example Workflow

1. **Train the model**:
```bash
python main.py train --epochs 30 --plot-history
```

2. **Predict on new images**:
```bash
python main.py predict --image satellite_image.png --visualize --heatmap
```

3. **Batch process a directory**:
```bash
python main.py predict --directory satellite_images/ --visualize
```

4. **Compare temporal changes**:
```bash
python main.py predict --compare 2020_image.png 2024_image.png --visualize
```

## Advanced Usage

### Using Custom Data

Organize your data as:
```
data/
└── custom_train/
    ├── forest/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── deforested/
        ├── image1.png
        ├── image2.png
        └── ...
```

Then modify `train.py` to use your custom data directory.

### Programmatic Usage

```python
from inference import DeforestationPredictor
from visualization import ResultVisualizer

# Load predictor
predictor = DeforestationPredictor(model_path='models/deforestation_detector.h5')

# Make prediction
result = predictor.predict_single_image('image.png')

# Visualize
visualizer = ResultVisualizer()
visualizer.plot_prediction(image, result, save_path='result.png')
```

## Technical Details

- **Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV, Rasterio (for GeoTIFF)
- **Visualization**: Matplotlib, Seaborn
- **Supported Formats**: PNG, JPG, GeoTIFF

## Limitations

- Synthetic sample data is for testing only - use real satellite imagery for production
- Model performance depends on quality and diversity of training data
- GeoTIFF support requires proper installation of rasterio and GDAL

## Future Improvements

- [ ] Support for multi-spectral satellite imagery
- [ ] Semantic segmentation for pixel-level deforestation maps
- [ ] Time-series analysis for deforestation trends
- [ ] Integration with satellite data APIs (Sentinel, Landsat)
- [ ] Web interface for easy predictions

## License

This project is provided as-is for educational and research purposes.
