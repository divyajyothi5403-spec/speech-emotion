"""
Configuration file for Speech Emotion Recognition project.
"""

# Audio processing parameters
SAMPLE_RATE = 22050
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256
MAX_LEN = 100

# Model parameters
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

# Dataset parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Emotion labels for custom dataset
CUSTOM_EMOTIONS = {
    'happy': 0,
    'sad': 1,
    'angry': 2,
    'neutral': 3,
    'fear': 4
}

# Emotion labels for RAVDESS dataset
RAVDESS_EMOTIONS = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}

# Paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
MODEL_PATH = 'models/speech_emotion_model.h5'
CONFIG_PATH = 'models/model_config.pkl'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'

# Recording parameters
RECORDING_DURATION = 3  # seconds
RECORDING_CHANNELS = 1  # mono
RECORDING_CHUNK_SIZE = 1024
