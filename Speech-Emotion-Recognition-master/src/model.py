import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class SpeechEmotionModel:
    """
    RNN/LSTM model for speech emotion recognition.
    """
    
    def __init__(self, input_shape, num_classes, lstm_units=128, dropout_rate=0.3):
        """
        Initialize the model.
        
        Args:
            input_shape: Shape of input features (timesteps, features)
            num_classes: Number of emotion classes
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_lstm_model(self):
        """
        Build an LSTM-based model.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First LSTM layer
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.BatchNormalization(),
            
            # Third LSTM layer
            layers.LSTM(
                self.lstm_units // 4,
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_bidirectional_lstm_model(self):
        """
        Build a bidirectional LSTM model for better context understanding.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    self.lstm_units,
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                )
            ),
            layers.BatchNormalization(),
            
            # Second bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    self.lstm_units // 2,
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                )
            ),
            layers.BatchNormalization(),
            
            # Third bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(
                    self.lstm_units // 4,
                    return_sequences=False,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                )
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_gru_model(self):
        """
        Build a GRU-based model (alternative to LSTM).
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First GRU layer
            layers.GRU(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.BatchNormalization(),
            
            # Second GRU layer
            layers.GRU(
                self.lstm_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.BatchNormalization(),
            
            # Third GRU layer
            layers.GRU(
                self.lstm_units // 4,
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_cnn_lstm_model(self):
        """
        Build a hybrid CNN-LSTM model for feature extraction and sequence modeling.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Reshape for CNN (add channel dimension)
            layers.Reshape((self.input_shape[0], self.input_shape[1], 1)),
            
            # CNN layers for feature extraction
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            # Reshape back for LSTM
            layers.Reshape((-1, 64)),
            
            # LSTM layers
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.BatchNormalization(),
            
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model must be built before compiling")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully")
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
        
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Test loss and accuracy
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features
        
        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def save_model(self, model_path='models/speech_emotion_model.h5'):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/speech_emotion_model.h5'):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return self.model


def create_model(input_shape, num_classes, model_type='lstm', **kwargs):
    """
    Factory function to create different model types.
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of emotion classes
        model_type: Type of model ('lstm', 'bidirectional_lstm', 'gru', 'cnn_lstm')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Compiled SpeechEmotionModel instance
    """
    model = SpeechEmotionModel(input_shape, num_classes, **kwargs)
    
    if model_type == 'lstm':
        model.build_lstm_model()
    elif model_type == 'bidirectional_lstm':
        model.build_bidirectional_lstm_model()
    elif model_type == 'gru':
        model.build_gru_model()
    elif model_type == 'cnn_lstm':
        model.build_cnn_lstm_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.compile_model()
    
    return model
