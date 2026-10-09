# Getting Started Guide

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **pip**: Usually included with Python installation

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/sujayd033/satellite-deforestation-detection-.git
cd satellite-deforestation-detection-
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python main.py --help
```

You should see the CLI help message.

## Your First Prediction

### Step 1: Train the Model

The system includes sample data generation, so you can start training immediately:

```bash
python main.py train --epochs 5
```

This will:
- Generate 200 sample images (100 forested, 100 deforested)
- Train a CNN model
- Save the trained model to `models/deforestation_detector.h5`

### Step 2: Make a Prediction

```bash
python main.py predict --image data/sample_train/forest/forest_0000.png --visualize
```

This will:
- Load the trained model
- Predict deforestation status
- Generate a visualization
- Save results to `results/`

### Step 3: View Results

Check the `results/` directory for:
- `prediction.png`: Visualization of the prediction
- Detailed output in the terminal

## Understanding the Output

### Prediction Output

```
--------------------------------------------------
Image: data/sample_train/forest/forest_0000.png
Status: DEFORESTED
Confidence: 97.59%
Forest Coverage: 52.80%
--------------------------------------------------
```

- **Status**: Whether the image is classified as deforested or forested
- **Confidence**: Model's confidence in the prediction (0-100%)
- **Forest Coverage**: Percentage of area covered by forest

## Common Tasks

### Train with Custom Parameters

```bash
python main.py train --epochs 30 --batch-size 32 --backbone simple --plot-history
```

### Batch Process Images

```bash
python main.py predict --directory data/sample_train/forest --visualize
```

### Compare Two Images

```bash
python main.py predict --compare before.png after.png --visualize
```

### Generate Heatmap

```bash
python main.py predict --image image.png --heatmap --visualize
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Make sure you've activated your virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: TensorFlow Import Error

**Solution:** Ensure you have the correct Python version (3.8+) and TensorFlow version:
```bash
python --version
pip show tensorflow
```

### Issue: Out of Memory During Training

**Solution:** Reduce the batch size in `config.py`:
```python
BATCH_SIZE = 8  # Reduce from 16
```

### Issue: Model Not Found

**Solution:** Train the model first:
```bash
python main.py train --epochs 5
```

## Next Steps

1. **Use Your Own Data**: Replace the sample data with real satellite imagery
2. **Experiment with Models**: Try different backbones (resnet50, efficientnetb0, vgg16)
3. **Adjust Parameters**: Tune hyperparameters in `config.py`
4. **Explore Visualizations**: Try different visualization options
5. **Read the Documentation**: Check `docs/API.md` for detailed API documentation

## Getting Help

- **Documentation**: Check the [README.md](../README.md) and [docs/](./) folder
- **Issues**: Report bugs on [GitHub Issues](https://github.com/sujayd033/satellite-deforestation-detection-/issues)
- **Email**: Contact sujaydharmavar@gmail.com

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/guides/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

Happy coding! 🌲
