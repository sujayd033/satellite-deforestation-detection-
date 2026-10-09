"""
Configuration settings for deforestation detection system
"""

# Model configuration
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

# Data configuration
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model architecture
NUM_CLASSES = 2  # 0: No deforestation, 1: Deforestation
BACKBONE = 'simple'  # Options: 'resnet50', 'efficientnetb0', 'vgg16', 'simple'

# Paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Deforestation detection thresholds
FOREST_COVERAGE_THRESHOLD = 0.3  # Minimum forest coverage to be considered forested
CHANGE_THRESHOLD = 0.2  # Minimum change to be flagged as deforestation
