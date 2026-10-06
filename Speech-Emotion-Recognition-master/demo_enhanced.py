import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.feature_extraction import extract_combined_features
from demo import MicrophoneRecorder, EmotionPredictor, display_prediction


class FileUploadPredictor:
    """
    Predict emotions from uploaded audio files.
    """
    
    def __init__(self, predictor):
        """
        Initialize with an existing EmotionPredictor.
        
        Args:
            predictor: EmotionPredictor instance
        """
        self.predictor = predictor
    
    def predict_from_file(self, file_path):
        """
        Predict emotion from an audio file.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Prediction results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file is audio
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Invalid audio file. Supported formats: {valid_extensions}")
        
        print(f"\nProcessing file: {file_path}")
        result = self.predictor.predict(file_path)
        return result
    
    def batch_predict(self, directory):
        """
        Predict emotions for all audio files in a directory.
        
        Args:
            directory: Path to directory containing audio files
        
        Returns:
            List of prediction results
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        results = []
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        print(f"\nProcessing all audio files in: {directory}")
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                file_path = os.path.join(directory, filename)
                try:
                    result = self.predict_from_file(file_path)
                    result['filename'] = filename
                    results.append(result)
                    print(f"✓ Processed: {filename}")
                except Exception as e:
                    print(f"✗ Error processing {filename}: {e}")
        
        return results


def interactive_menu():
    """
    Interactive menu for choosing between file upload and live voice.
    """
    print("=" * 60)
    print("SPEECH EMOTION RECOGNITION - ENHANCED DEMO")
    print("=" * 60)
    print("\nChoose an option:")
    print("1. Live Voice Recording (Microphone)")
    print("2. Upload Audio File")
    print("3. Batch Process Directory")
    print("4. Exit")
    
    return input("\nEnter your choice (1-4): ")


def main():
    """
    Main function for the enhanced demo application.
    """
    print("=" * 60)
    print("Speech Emotion Recognition - Enhanced Demo")
    print("=" * 60)
    
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
        file_predictor = FileUploadPredictor(predictor)
        
        while True:
            choice = interactive_menu()
            
            if choice == '1':
                # Live voice recording
                try:
                    duration_input = input("\nEnter recording duration in seconds (default=3): ")
                    duration = float(duration_input) if duration_input else 3
                    
                    print("\n🎤 Starting microphone recording...")
                    result = predictor.predict_from_microphone(duration=duration)
                    display_prediction(result)
                    
                except Exception as e:
                    print(f"\n❌ Error during recording: {e}")
                    print("Make sure you have a working microphone.")
            
            elif choice == '2':
                # File upload
                file_path = input("\nEnter the path to your audio file: ").strip('"')
                
                try:
                    result = file_predictor.predict_from_file(file_path)
                    display_prediction(result)
                    
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    print("Make sure the file exists and is a valid audio format.")
            
            elif choice == '3':
                # Batch processing
                directory = input("\nEnter the path to directory containing audio files: ").strip('"')
                
                try:
                    results = file_predictor.batch_predict(directory)
                    
                    if results:
                        print(f"\n{'='*60}")
                        print(f"BATCH PROCESSING RESULTS - {len(results)} files processed")
                        print(f"{'='*60}")
                        
                        for i, result in enumerate(results, 1):
                            print(f"\n{i}. {result['filename']}")
                            print(f"   Predicted: {result['predicted_emotion'].upper()}")
                            print(f"   Confidence: {result['confidence']:.2%}")
                    else:
                        print("\nNo valid audio files found in the directory.")
                        
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            
            elif choice == '4':
                print("\nExiting... Goodbye!")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter a number between 1-4.")
            
            # Ask if user wants to continue
            if choice != '4':
                continue_choice = input("\nDo you want to continue? (y/n): ").lower()
                if continue_choice != 'y':
                    print("\nExiting... Goodbye!")
                    break
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have:")
        print("1. A trained model in the 'models' directory")
        print("2. Required dependencies installed")


if __name__ == '__main__':
    main()
