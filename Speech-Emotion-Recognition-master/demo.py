import wave
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import sys

try:
    import pyaudio
except ImportError:
    pyaudio = None

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.feature_extraction import extract_combined_features


class MicrophoneRecorder:
    """
    Record audio from microphone.
    """
    
    def __init__(self, sample_rate=22050, channels=1, chunk_size=1024):
        """
        Initialize the microphone recorder.
        
        Args:
            sample_rate: Sampling rate for recording
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_size: Size of audio chunks
        """
        if pyaudio is None:
            raise RuntimeError(
                'PyAudio is required for microphone recording. '
                'Install it locally with: pip install pyaudio'
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        
    def record(self, duration=3, output_file='temp_recording.wav'):
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            output_file: Path to save the recorded audio
        
        Returns:
            Path to the recorded audio file
        """
        print(f"Recording for {duration} seconds...")
        print("Speak now...")
        
        # Open audio stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        
        # Record audio
        for i in range(int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        print("Recording finished!")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        
        # Save recording to file
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return output_file
    
    def close(self):
        """Close the audio interface."""
        self.audio.terminate()


class EmotionPredictor:
    """
    Predict emotions from audio using trained model.
    """
    
    def __init__(self, model_path='models/speech_emotion_model.h5', 
                 config_path='models/model_config.pkl',
                 label_encoder_path='models/label_encoder.pkl'):
        """
        Initialize the emotion predictor.
        
        Args:
            model_path: Path to trained model
            config_path: Path to model configuration
            label_encoder_path: Path to label encoder
        """
        # Load model configuration
        if os.path.exists(config_path):
            self.config = joblib.load(config_path)
            self.emotion_labels = self.config['emotion_labels']
            self.max_len = self.config['max_len']
        else:
            raise FileNotFoundError(f"Model configuration not found at {config_path}")
        
        # Load label encoder
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
        else:
            raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")
        
        # Load trained model
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load normalization parameters
        if os.path.exists('models/mean.npy') and os.path.exists('models/std.npy'):
            self.mean = np.load('models/mean.npy')
            self.std = np.load('models/std.npy')
        else:
            self.mean = None
            self.std = None
            print("Warning: Normalization parameters not found. Predictions may be inaccurate.")
    
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file for prediction.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Preprocessed feature array
        """
        # Extract features
        features = extract_combined_features(audio_path, max_len=self.max_len)
        
        if features is None:
            raise ValueError("Failed to extract features from audio")
        
        # Normalize features if parameters are available
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        
        return features
    
    def predict(self, audio_path):
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary containing predicted emotion and probabilities
        """
        # Preprocess audio
        features = self.preprocess_audio(audio_path)
        
        # Make prediction
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Get predicted class
        predicted_class = np.argmax(predictions)
        predicted_emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Create emotion probability dictionary
        emotion_probs = {}
        for i, prob in enumerate(predictions):
            emotion = self.label_encoder.inverse_transform([i])[0]
            emotion_probs[emotion] = float(prob)
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': float(predictions[predicted_class]),
            'probabilities': emotion_probs
        }
    
    def predict_from_microphone(self, duration=3):
        """
        Record audio from microphone and predict emotion.
        
        Args:
            duration: Recording duration in seconds
        
        Returns:
            Prediction results
        """
        # Record audio
        recorder = MicrophoneRecorder()
        temp_file = recorder.record(duration=duration)
        recorder.close()
        
        # Predict emotion
        try:
            result = self.predict(temp_file)
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return result
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise


def display_prediction(result):
    """
    Display prediction results in a formatted way.
    
    Args:
        result: Prediction result dictionary
    """
    print("\n" + "=" * 50)
    print("EMOTION PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Emotion: {result['predicted_emotion'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll Probabilities:")
    print("-" * 50)
    
    # Sort emotions by probability
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    for emotion, prob in sorted_probs:
        bar_length = int(prob * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        print(f"{emotion:12} | {bar} | {prob:.2%}")
    
    print("=" * 50 + "\n")


def main():
    """
    Main function for the demo application.
    """
    print("=" * 50)
    print("Speech Emotion Recognition Demo")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('models/speech_emotion_model.h5'):
        print("Error: Trained model not found!")
        print("Please train the model first by running: python train.py")
        print("Make sure you have audio data in the 'data' directory.")
        return
    
    try:
        # Initialize predictor
        print("\nLoading model...")
        predictor = EmotionPredictor()
        
        # Get recording duration from user
        try:
            duration = float(input("\nEnter recording duration in seconds (default=3): ") or "3")
        except ValueError:
            duration = 3
        
        # Record and predict
        print("\nStarting microphone recording...")
        result = predictor.predict_from_microphone(duration=duration)
        
        # Display results
        display_prediction(result)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have:")
        print("1. A trained model in the 'models' directory")
        print("2. A working microphone")
        print("3. PyAudio installed (pip install pyaudio)")


if __name__ == '__main__':
    main()
