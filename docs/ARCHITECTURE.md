# System Architecture

## Overview

The Deforestation Detection System is built using a modular architecture that separates concerns into distinct components:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (CLI)                     │
│                          main.py                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Training   │ │  Inference  │ │ Visualization│
    │   Module    │ │   Module    │ │   Module    │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Core Modules   │
                  │                 │
                  │  • config.py    │
                  │  • model.py     │
                  │  • data_        │
                  │    preprocessing│
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Data Storage   │
                  │                 │
                  │  • data/        │
                  │  • models/      │
                  │  • results/     │
                  └─────────────────┘
```

## Component Details

### 1. Configuration Layer (`config.py`)
- Centralized configuration management
- Model hyperparameters
- Data processing settings
- Detection thresholds

### 2. Data Processing Layer (`data_preprocessing.py`)
- Image loading and preprocessing
- NDVI calculation
- Forest coverage estimation
- Data augmentation
- Sample data generation

### 3. Model Layer (`model.py`)
- CNN architecture definitions
- Transfer learning implementations
- Model compilation and training
- Fine-tuning strategies

### 4. Training Layer (`train.py`)
- Training pipeline orchestration
- Data generator setup
- Model checkpointing
- Early stopping and learning rate scheduling

### 5. Inference Layer (`inference.py`)
- Single image prediction
- Batch processing
- Change detection
- Heatmap generation

### 6. Visualization Layer (`visualization.py`)
- Prediction visualization
- Heatmap generation
- Comparison plots
- Training history plots
- Report generation

### 7. Interface Layer (`main.py`)
- CLI argument parsing
- Command routing
- User interaction

## Data Flow

### Training Flow
```
Raw Images → Preprocessing → Augmentation → Model Training → Validation → Model Save
```

### Inference Flow
```
Input Image → Preprocessing → Model Inference → Post-processing → Visualization
```

### Batch Processing Flow
```
Directory → Image List → Batch Processing → Results Aggregation → Report Generation
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Interface | Python argparse |
| ML Framework | TensorFlow/Keras |
| Image Processing | OpenCV, Rasterio |
| Visualization | Matplotlib, Seaborn |
| Data Handling | NumPy, Pandas |

## Design Patterns

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Dependency Injection**: Models and processors are injected rather than hardcoded
3. **Factory Pattern**: Different model architectures can be selected at runtime
4. **Strategy Pattern**: Different preprocessing strategies can be applied

## Extension Points

The system is designed to be extensible at several points:

1. **New Model Architectures**: Add to `model.py`
2. **New Preprocessing Methods**: Add to `data_preprocessing.py`
3. **New Visualization Types**: Add to `visualization.py`
4. **New Data Sources**: Extend image loading in `data_preprocessing.py`
5. **New CLI Commands**: Add to `main.py`
