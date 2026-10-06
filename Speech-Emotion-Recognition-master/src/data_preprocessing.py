import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_combined_features, load_audio_data


class DataPreprocessor:
    """
    Handles data loading, preprocessing, and splitting for speech emotion recognition.
    """
    
    def __init__(self, data_dir, emotion_labels, max_len=100, test_size=0.2, random_state=42):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Directory containing emotion-labeled audio files
            emotion_labels: Dictionary mapping emotion names to integer labels
            max_len: Maximum length for feature sequences
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.emotion_labels = emotion_labels
        self.max_len = max_len
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess(self):
        """
        Load audio files and extract features.
        
        Returns:
            X: Feature array
            y: Label array
        """
        print("Loading and preprocessing audio data...")
        
        X, y = load_audio_data(
            self.data_dir, 
            self.emotion_labels, 
            max_len=self.max_len
        )
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
        print(f"Shape: {X.shape}")
        
        return X, y
    
    def split_data(self, X, y):
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Feature array
            y: Label array
        
        Returns:
            X_train, X_val, X_test: Split feature arrays
            y_train, y_val, y_test: Split label arrays
        """
        # First split: train + val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_train_val
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(self, X_train, X_val, X_test):
        """
        Normalize features using mean and std from training set.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
        
        Returns:
            Normalized feature arrays
        """
        # Calculate mean and std from training set
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1
        
        # Normalize
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std
        
        # Save normalization parameters
        np.save('models/mean.npy', mean)
        np.save('models/std.npy', std)
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def encode_labels(self, y_train, y_val, y_test):
        """
        Encode labels using LabelEncoder.
        
        Args:
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
        
        Returns:
            Encoded label arrays
        """
        # Fit on training data
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def prepare_dataset(self):
        """
        Complete pipeline to prepare the dataset.
        
        Returns:
            Dictionary containing all split and preprocessed data
        """
        # Load and extract features
        X, y = self.load_and_preprocess()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Normalize features
        X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
        
        # Encode labels
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'num_classes': len(self.emotion_labels)
        }


def create_ravdess_emotion_labels():
    """
    Create emotion labels for RAVDESS dataset.
    
    Returns:
        Dictionary mapping emotion names to labels
    """
    return {
        'neutral': 0,
        'calm': 1,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fearful': 5,
        'disgust': 6,
        'surprised': 7
    }


def create_custom_emotion_labels():
    """
    Create emotion labels for custom dataset.
    
    Returns:
        Dictionary mapping emotion names to labels
    """
    return {
        'happy': 0,
        'sad': 1,
        'angry': 2,
        'neutral': 3,
        'fear': 4
    }
