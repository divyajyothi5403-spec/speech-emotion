import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.feature_extraction import extract_combined_features
from demo import MicrophoneRecorder, EmotionPredictor, display_prediction


class AdvancedFileProcessor:
    """
    Advanced file processor with multiple file support and history.
    """
    
    def __init__(self, predictor):
        """
        Initialize with an existing EmotionPredictor.
        
        Args:
            predictor: EmotionPredictor instance
        """
        self.predictor = predictor
        self.history_file = 'models/prediction_history.json'
        self.history = self.load_history()
    
    def load_history(self):
        """Load prediction history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return defaultdict(list)
        return defaultdict(list)
    
    def save_history(self):
        """Save prediction history to file."""
        os.makedirs('models', exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_to_history(self, source, filename, result):
        """Add prediction to history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'predicted_emotion': result['predicted_emotion'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        }
        self.history[source].append(entry)
        self.save_history()
    
    def process_multiple_files(self, file_paths):
        """
        Process multiple audio files.
        
        Args:
            file_paths: List of file paths to process
        
        Returns:
            List of prediction results
        """
        results = []
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        print(f"\nProcessing {len(file_paths)} files...")
        
        for i, file_path in enumerate(file_paths, 1):
            if not os.path.exists(file_path):
                print(f"✗ File not found: {file_path}")
                continue
            
            if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
                print(f"✗ Invalid format: {file_path}")
                continue
            
            try:
                print(f"[{i}/{len(file_paths)}] Processing: {os.path.basename(file_path)}")
                result = self.predictor.predict(file_path)
                result['filename'] = os.path.basename(file_path)
                result['file_path'] = file_path
                results.append(result)
                self.add_to_history('file_upload', os.path.basename(file_path), result)
                print(f"✓ Predicted: {result['predicted_emotion']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")
        
        return results
    
    def display_summary(self, results):
        """Display summary of multiple predictions."""
        if not results:
            print("No results to display.")
            return
        
        # Count emotions
        emotion_counts = defaultdict(int)
        total_confidence = 0
        
        for result in results:
            emotion_counts[result['predicted_emotion']] += 1
            total_confidence += result['confidence']
        
        print(f"\n{'='*60}")
        print(f"SUMMARY - {len(results)} files processed")
        print(f"{'='*60}")
        print(f"Average Confidence: {total_confidence/len(results):.2%}")
        print("\nEmotion Distribution:")
        
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            bar = '█' * int(percentage / 3)
            print(f"  {emotion:12} | {bar} | {count} ({percentage:.1f}%)")
    
    def get_history(self, source=None, limit=10):
        """Get prediction history."""
        if source:
            return self.history.get(source, [])[-limit:]
        else:
            all_entries = []
            for source_entries in self.history.values():
                all_entries.extend(source_entries)
            return sorted(all_entries, key=lambda x: x['timestamp'], reverse=True)[:limit]


class AudioVisualizer:
    """
    Audio visualization utilities.
    """
    
    @staticmethod
    def display_waveform(audio_path, save_path='temp_waveform.png'):
        """
        Display and save audio waveform.
        
        Args:
            audio_path: Path to audio file
            save_path: Path to save the waveform image
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr)
            plt.title('Audio Waveform')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            return save_path
        except Exception as e:
            print(f"Error creating waveform: {e}")
            return None
    
    @staticmethod
    def display_spectrogram(audio_path, save_path='temp_spectrogram.png'):
        """
        Display and save audio spectrogram.
        
        Args:
            audio_path: Path to audio file
            save_path: Path to save the spectrogram image
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            plt.figure(figsize=(12, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            return save_path
        except Exception as e:
            print(f"Error creating spectrogram: {e}")
            return None
    
    @staticmethod
    def display_mfcc(audio_path, save_path='temp_mfcc.png'):
        """
        Display and save MFCC features.
        
        Args:
            audio_path: Path to audio file
            save_path: Path to save the MFCC image
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(mfcc, sr=sr, x_axis='time')
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            return save_path
        except Exception as e:
            print(f"Error creating MFCC visualization: {e}")
            return None


def advanced_interactive_menu():
    """
    Advanced interactive menu with more options.
    """
    print("=" * 70)
    print("🎤 SPEECH EMOTION RECOGNITION - ADVANCED DEMO")
    print("=" * 70)
    print("\nChoose an option:")
    print("1. 🎙️  Live Voice Recording (Microphone)")
    print("2. 📁 Upload Single Audio File")
    print("3. 📂 Upload Multiple Audio Files")
    print("4. 📊 Process Entire Directory")
    print("5. 📈 View Prediction History")
    print("6. 🎨 Audio Visualization")
    print("7. 🆚 Compare Recordings")
    print("8. ❌ Exit")
    
    return input("\nEnter your choice (1-8): ")


def main():
    """
    Main function for the advanced demo application.
    """
    print("=" * 70)
    print("Speech Emotion Recognition - Advanced Demo")
    print("=" * 70)
    
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
        file_processor = AdvancedFileProcessor(predictor)
        visualizer = AudioVisualizer()
        
        while True:
            choice = advanced_interactive_menu()
            
            if choice == '1':
                # Live voice recording
                try:
                    duration_input = input("\nEnter recording duration in seconds (default=3): ")
                    duration = float(duration_input) if duration_input else 3
                    
                    print("\n🎙️ Starting microphone recording...")
                    result = predictor.predict_from_microphone(duration=duration)
                    display_prediction(result)
                    file_processor.add_to_history('live_recording', f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav', result)
                    
                except Exception as e:
                    print(f"\n❌ Error during recording: {e}")
            
            elif choice == '2':
                # Single file upload
                file_path = input("\nEnter the path to your audio file: ").strip('"')
                
                try:
                    result = file_processor.process_multiple_files([file_path])
                    if result:
                        display_prediction(result[0])
                        
                        # Ask for visualization
                        viz_choice = input("\nWould you like to see audio visualizations? (y/n): ").lower()
                        if viz_choice == 'y':
                            print("Generating visualizations...")
                            visualizer.display_waveform(file_path)
                            visualizer.display_spectrogram(file_path)
                            print("✓ Visualizations saved as temp files")
                    
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            
            elif choice == '3':
                # Multiple file upload
                print("\nEnter multiple file paths (one per line). Enter 'done' when finished:")
                file_paths = []
                while True:
                    file_path = input(f"File {len(file_paths) + 1}: ").strip('"')
                    if file_path.lower() == 'done':
                        break
                    if file_path:
                        file_paths.append(file_path)
                
                if file_paths:
                    results = file_processor.process_multiple_files(file_paths)
                    file_processor.display_summary(results)
                else:
                    print("No files provided.")
            
            elif choice == '4':
                # Directory processing
                directory = input("\nEnter the path to directory containing audio files: ").strip('"')
                
                try:
                    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
                    file_paths = []
                    
                    for filename in os.listdir(directory):
                        if any(filename.lower().endswith(ext) for ext in valid_extensions):
                            file_paths.append(os.path.join(directory, filename))
                    
                    if file_paths:
                        results = file_processor.process_multiple_files(file_paths)
                        file_processor.display_summary(results)
                    else:
                        print("No valid audio files found in the directory.")
                        
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            
            elif choice == '5':
                # View history
                print("\n📈 Prediction History")
                print("=" * 70)
                
                source_filter = input("Filter by source (file_upload/live_recording/all, default=all): ").lower()
                
                if source_filter in ['file_upload', 'live_recording']:
                    history = file_processor.get_history(source=source_filter, limit=20)
                else:
                    history = file_processor.get_history(limit=20)
                
                if history:
                    for i, entry in enumerate(history, 1):
                        print(f"\n{i}. {entry['filename']}")
                        print(f"   Time: {entry['timestamp']}")
                        print(f"   Predicted: {entry['predicted_emotion']} ({entry['confidence']:.2%})")
                else:
                    print("No history found.")
            
            elif choice == '6':
                # Audio visualization
                file_path = input("\nEnter the path to audio file for visualization: ").strip('"')
                
                if os.path.exists(file_path):
                    print("Generating visualizations...")
                    
                    print("1. Waveform...")
                    visualizer.display_waveform(file_path, 'waveform.png')
                    print("   ✓ Saved as waveform.png")
                    
                    print("2. Spectrogram...")
                    visualizer.display_spectrogram(file_path, 'spectrogram.png')
                    print("   ✓ Saved as spectrogram.png")
                    
                    print("3. MFCC...")
                    visualizer.display_mfcc(file_path, 'mfcc.png')
                    print("   ✓ Saved as mfcc.png")
                    
                    print("\n✓ All visualizations saved successfully!")
                else:
                    print("File not found.")
            
            elif choice == '7':
                # Compare recordings
                print("\n🆚 Compare Recordings")
                print("Enter paths to 2 audio files to compare:")
                
                file1 = input("File 1: ").strip('"')
                file2 = input("File 2: ").strip('"')
                
                if os.path.exists(file1) and os.path.exists(file2):
                    try:
                        print("\nAnalyzing files...")
                        result1 = predictor.predict(file1)
                        result2 = predictor.predict(file2)
                        
                        print(f"\n{'='*70}")
                        print("COMPARISON RESULTS")
                        print(f"{'='*70}")
                        
                        print(f"\nFile 1: {os.path.basename(file1)}")
                        print(f"  Predicted: {result1['predicted_emotion']} ({result1['confidence']:.2%})")
                        
                        print(f"\nFile 2: {os.path.basename(file2)}")
                        print(f"  Predicted: {result2['predicted_emotion']} ({result2['confidence']:.2%})")
                        
                        if result1['predicted_emotion'] == result2['predicted_emotion']:
                            print(f"\n✓ Both files have the same predicted emotion: {result1['predicted_emotion']}")
                        else:
                            print(f"\n✗ Different emotions detected")
                            print(f"  File 1: {result1['predicted_emotion']}")
                            print(f"  File 2: {result2['predicted_emotion']}")
                        
                        # Compare confidence
                        conf_diff = abs(result1['confidence'] - result2['confidence'])
                        print(f"\nConfidence difference: {conf_diff:.2%}")
                        
                    except Exception as e:
                        print(f"Error during comparison: {e}")
                else:
                    print("One or both files not found.")
            
            elif choice == '8':
                print("\nExiting... Goodbye!")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter a number between 1-8.")
            
            # Ask if user wants to continue
            if choice != '8':
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
