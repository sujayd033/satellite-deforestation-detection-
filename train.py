"""
Training script for deforestation detection model
"""

import os
from pathlib import Path
import config
from data_preprocessing import SatelliteImageProcessor, generate_sample_dataset
from model import DeforestationDetector
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_data_generators(data_dir):
    """
    Prepare training, validation, and test data generators
    """
    # Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def train_model():
    """
    Main training function
    """
    print("=" * 60)
    print("Deforestation Detection Model Training")
    print("=" * 60)
    
    # Create directories
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    Path(config.MODELS_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    
    # Generate sample dataset if it doesn't exist
    data_dir = Path(config.DATA_DIR) / 'sample_train'
    if not data_dir.exists():
        print("\nGenerating sample dataset...")
        generate_sample_dataset(str(data_dir), num_samples=200)
    
    # Prepare data generators
    print("\nPreparing data generators...")
    train_generator, val_generator = prepare_data_generators(data_dir)
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
    # Build model
    print("\nBuilding model...")
    detector = DeforestationDetector(
        img_size=config.IMG_SIZE,
        backbone=config.BACKBONE
    )
    model = detector.build_model(num_classes=config.NUM_CLASSES)
    detector.compile_model(learning_rate=config.LEARNING_RATE)
    
    print(f"\nModel architecture: {config.BACKBONE}")
    model.summary()
    
    # Train model
    print("\nStarting training...")
    model_path = os.path.join(config.MODELS_DIR, 'deforestation_detector.h5')
    
    history = detector.train(
        train_generator,
        val_generator,
        epochs=config.EPOCHS,
        model_path=model_path
    )
    
    # Fine-tune the model
    print("\nFine-tuning model...")
    detector.fine_tune(unfreeze_layers=10)
    
    history_fine = detector.train(
        train_generator,
        val_generator,
        epochs=10,
        model_path=model_path
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = detector.evaluate(val_generator)
    print("\nValidation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nModel saved to: {model_path}")
    print("Training complete!")
    
    return history, results


if __name__ == "__main__":
    train_model()
