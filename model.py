"""
Deep learning model for deforestation detection
Uses CNN architecture with transfer learning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config


class DeforestationDetector:
    """CNN model for detecting deforestation in satellite images"""
    
    def __init__(self, img_size=256, backbone='efficientnetb0'):
        self.img_size = img_size
        self.backbone = backbone
        self.model = None
        
    def build_model(self, num_classes=2):
        """
        Build the deforestation detection model
        Uses transfer learning with pre-trained backbone
        """
        # Input layer
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Choose backbone
        if self.backbone == 'resnet50':
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        elif self.backbone == 'efficientnetb0':
            base_model = applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        elif self.backbone == 'vgg16':
            base_model = applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        else:
            # Simple CNN if no backbone specified
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D()(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
            self.model = models.Model(inputs=inputs, outputs=outputs)
            return self.model
        
        # Freeze the backbone initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=base_model.input, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
    
    def fine_tune(self, unfreeze_layers=10):
        """
        Fine-tune the model by unfreezing some backbone layers
        """
        # Unfreeze the last N layers of the backbone
        if self.backbone in ['resnet50', 'efficientnetb0', 'vgg16']:
            self.model.trainable = True
            for layer in self.model.layers[:-unfreeze_layers]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
    
    def get_callbacks(self, model_path):
        """Get training callbacks"""
        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, train_generator, val_generator, epochs, model_path):
        """Train the model"""
        callbacks = self.get_callbacks(model_path)
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_generator):
        """Evaluate the model on test data"""
        results = self.model.evaluate(test_generator, verbose=1)
        return dict(zip(self.model.metrics_names, results))
    
    def predict(self, image):
        """
        Make prediction on a single image
        Returns probability of deforestation
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)[0][0]
        return prediction
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        return self.model


class ChangeDetectionModel:
    """
    Model for detecting changes between two satellite images
    Compares before and after images to detect deforestation
    """
    
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.model = None
        
    def build_siamese_model(self):
        """
        Build a Siamese network for change detection
        Takes two images (before and after) and outputs change probability
        """
        # Shared feature extractor
        def build_feature_extractor():
            inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D()(x)
            x = layers.GlobalAveragePooling2D()(x)
            return models.Model(inputs, x)
        
        feature_extractor = build_feature_extractor()
        
        # Two input branches
        input_before = layers.Input(shape=(self.img_size, self.img_size, 3))
        input_after = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Extract features
        features_before = feature_extractor(input_before)
        features_after = feature_extractor(input_after)
        
        # Calculate difference
        diff = layers.Subtract()([features_after, features_before])
        abs_diff = layers.Lambda(lambda x: tf.abs(x))(diff)
        
        # Classification
        x = layers.Dense(128, activation='relu')(abs_diff)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(
            inputs=[input_before, input_after],
            outputs=outputs
        )
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model


if __name__ == "__main__":
    # Test model creation
    detector = DeforestationDetector(img_size=config.IMG_SIZE, backbone=config.BACKBONE)
    model = detector.build_model()
    detector.compile_model(learning_rate=config.LEARNING_RATE)
    model.summary()
