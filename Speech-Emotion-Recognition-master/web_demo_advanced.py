import streamlit as st
import os
import sys
import numpy as np
import tempfile
import librosa
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from collections import defaultdict

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.feature_extraction import extract_combined_features

# Page configuration
st.set_page_config(
    page_title="Advanced Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f8ff;
        margin: 20px 0;
        border: 2px solid #1f77b4;
    }
    .emotion-bar {
        height: 30px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .history-item {
        padding: 15px;
        border-radius: 8px;
        background-color: #f9f9f9;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class AdvancedStreamlitPredictor:
    """
    Advanced emotion predictor for Streamlit with history and visualization.
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.label_encoder = None
        self.config = None
        self.mean = None
        self.std = None
        self.history_file = 'models/prediction_history.json'
        self.history = self.load_history()
        self.load_model()
    
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
    
    def load_model(self):
        """Load the trained model and configurations."""
        try:
            # Load model configuration
            if os.path.exists('models/model_config.pkl'):
                self.config = joblib.load('models/model_config.pkl')
                self.emotion_labels = self.config['emotion_labels']
                self.max_len = self.config['max_len']
            else:
                st.error("Model configuration not found. Please train the model first.")
                return False
            
            # Load label encoder
            if os.path.exists('models/label_encoder.pkl'):
                self.label_encoder = joblib.load('models/label_encoder.pkl')
            else:
                st.error("Label encoder not found. Please train the model first.")
                return False
            
            # Load trained model
            if os.path.exists('models/speech_emotion_model.h5'):
                self.model = keras.models.load_model('models/speech_emotion_model.h5')
            else:
                st.error("Trained model not found. Please train the model first.")
                return False
            
            # Load normalization parameters
            if os.path.exists('models/mean.npy') and os.path.exists('models/std.npy'):
                self.mean = np.load('models/mean.npy')
                self.std = np.load('models/std.npy')
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file for prediction."""
        try:
            features = extract_combined_features(audio_path, max_len=self.max_len)
            
            if features is None:
                raise ValueError("Failed to extract features from audio")
            
            # Normalize features if parameters are available
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            
            # Add batch dimension
            features = np.expand_dims(features, axis=0)
            
            return features
            
        except Exception as e:
            st.error(f"Error preprocessing audio: {e}")
            return None
    
    def predict(self, audio_path):
        """Predict emotion from audio file."""
        if self.model is None:
            st.error("Model not loaded. Please train the model first.")
            return None
        
        # Preprocess audio
        features = self.preprocess_audio(audio_path)
        if features is None:
            return None
        
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
    
    def create_waveform(self, audio_path):
        """Create waveform visualization."""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title('Audio Waveform')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            plt.tight_layout()
            
            return fig
        except Exception as e:
            st.error(f"Error creating waveform: {e}")
            return None
    
    def create_spectrogram(self, audio_path):
        """Create spectrogram visualization."""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
            ax.set_title('Spectrogram')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.tight_layout()
            
            return fig
        except Exception as e:
            st.error(f"Error creating spectrogram: {e}")
            return None


def display_results(result, show_visualizations=False, audio_path=None, predictor=None):
    """Display prediction results with visualizations."""
    if result is None:
        return
    
    # Main result
    st.markdown(f"""
    <div class="result-box">
        <h2 style="text-align: center; color: #1f77b4;">
            Predicted Emotion: {result['predicted_emotion'].upper()}
        </h2>
        <p style="text-align: center; font-size: 1.2rem;">
            Confidence: {result['confidence']:.2%}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability bars
    st.subheader("Emotion Probabilities")
    
    # Sort emotions by probability
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    for emotion, prob in sorted_probs:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 10px 0;">
            <div style="width: 100px; font-weight: bold;">{emotion}</div>
            <div style="flex: 1; background-color: #e0e0e0; border-radius: 5px; height: 30px; margin: 0 10px;">
                <div style="width: {prob * 100}%; background-color: #1f77b4; height: 100%; border-radius: 5px;"></div>
            </div>
            <div style="width: 60px; text-align: right;">{prob:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create bar chart
    st.subheader("Probability Distribution")
    emotions = [item[0] for item in sorted_probs]
    probabilities = [item[1] for item in sorted_probs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(emotions, probabilities, color='steelblue')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Probability Distribution')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Audio visualizations
    if show_visualizations and audio_path and predictor:
        st.subheader("Audio Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Waveform")
            waveform_fig = predictor.create_waveform(audio_path)
            if waveform_fig:
                st.pyplot(waveform_fig)
        
        with col2:
            st.write("Spectrogram")
            spectrogram_fig = predictor.create_spectrogram(audio_path)
            if spectrogram_fig:
                st.pyplot(spectrogram_fig)


def display_history(history):
    """Display prediction history."""
    if not history:
        st.info("No prediction history found.")
        return
    
    st.subheader("Prediction History")
    
    # Flatten history from all sources
    all_entries = []
    for source_entries in history.values():
        all_entries.extend(source_entries)
    
    # Sort by timestamp (most recent first)
    all_entries = sorted(all_entries, key=lambda x: x['timestamp'], reverse=True)
    
    # Display recent entries
    for entry in all_entries[:20]:  # Show last 20 entries
        timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        
        st.markdown(f"""
        <div class="history-item">
            <strong>📁 {entry['filename']}</strong><br/>
            <small>🕒 {timestamp}</small><br/>
            <strong>Emotion:</strong> {entry['predicted_emotion'].upper()} 
            ({entry['confidence']:.2%})
        </div>
        """, unsafe_allow_html=True)


def compare_results(result1, result2, name1="File 1", name2="File 2"):
    """Compare two prediction results."""
    st.subheader("Comparison Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="result-box">
            <h3>{name1}</h3>
            <h2 style="color: #1f77b4;">{result1['predicted_emotion'].upper()}</h2>
            <p>Confidence: {result1['confidence']:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-box">
            <h3>{name2}</h3>
            <h2 style="color: #1f77b4;">{result2['predicted_emotion'].upper()}</h2>
            <p>Confidence: {result2['confidence']:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison summary
    if result1['predicted_emotion'] == result2['predicted_emotion']:
        st.success(f"✓ Both files have the same predicted emotion: {result1['predicted_emotion']}")
    else:
        st.warning(f"✗ Different emotions detected")
        st.info(f"File 1: {result1['predicted_emotion']} | File 2: {result2['predicted_emotion']}")
    
    # Confidence comparison
    conf_diff = abs(result1['confidence'] - result2['confidence'])
    st.info(f"Confidence difference: {conf_diff:.2%}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Advanced Speech Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using RNN/LSTM Deep Learning with Enhanced Features</p>', unsafe_allow_html=True)
    
    # Initialize predictor
    @st.cache_resource
    def get_predictor():
        return AdvancedStreamlitPredictor()
    
    predictor = get_predictor()
    
    if predictor.model is None:
        st.error("⚠️ Model not found. Please train the model first by running: python train.py")
        st.info("Make sure you have audio data in the 'data' directory and a trained model in the 'models' directory.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose Feature:",
        ["📁 File Upload", "🎙️ Live Recording", "📂 Multiple Files", "🆚 Compare Files", "📈 History"]
    )
    
    if page == "📁 File Upload":
        st.header("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
        )
        
        show_viz = st.checkbox("Show audio visualizations", value=True)
        
        if uploaded_file is not None:
            st.info(f"Uploaded file: {uploaded_file.name}")
            st.audio(uploaded_file)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            try:
                if st.button("🔍 Predict Emotion", key="single_predict"):
                    with st.spinner("Analyzing audio..."):
                        result = predictor.predict(tmp_path)
                        display_results(result, show_visualizations=show_viz, audio_path=tmp_path, predictor=predictor)
                        predictor.add_to_history('file_upload', uploaded_file.name, result)
            
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    elif page == "🎙️ Live Recording":
        st.header("Record Live Audio")
        
        duration = st.slider("Recording duration (seconds)", 1, 10, 3)
        show_viz = st.checkbox("Show audio visualizations", value=True)
        
        st.info("Click the button below and start speaking when prompted.")
        
        if st.button("🎙️ Start Recording", key="record_button"):
            try:
                from demo import MicrophoneRecorder
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_path = tmp_file.name
                
                with st.spinner(f"Recording for {duration} seconds... Speak now!"):
                    recorder = MicrophoneRecorder()
                    recorder.record(duration=duration, output_file=tmp_path)
                    recorder.close()
                
                st.success("Recording completed!")
                st.audio(tmp_path)
                
                with st.spinner("Analyzing audio..."):
                    result = predictor.predict(tmp_path)
                    display_results(result, show_visualizations=show_viz, audio_path=tmp_path, predictor=predictor)
                    predictor.add_to_history('live_recording', f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav', result)
            
            except Exception as e:
                st.error(f"Error during recording: {e}")
                st.info("Make sure you have a working microphone and PyAudio installed.")
            
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    elif page == "📂 Multiple Files":
        st.header("Upload Multiple Files")
        
        uploaded_files = st.file_uploader(
            "Choose multiple audio files",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            accept_multiple_files=True,
            help="Select multiple files for batch processing"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files")
            
            if st.button("🔍 Process All Files", key="batch_predict"):
                results = []
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            result = predictor.predict(tmp_path)
                            result['filename'] = uploaded_file.name
                            results.append(result)
                            predictor.add_to_history('batch_upload', uploaded_file.name, result)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display summary
                st.success(f"✓ Processed {len(results)} files")
                
                # Summary statistics
                emotion_counts = {}
                total_confidence = 0
                
                for result in results:
                    emotion = result['predicted_emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    total_confidence += result['confidence']
                
                st.subheader("Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Files", len(results))
                    st.metric("Average Confidence", f"{total_confidence/len(results):.2%}")
                
                with col2:
                    st.metric("Unique Emotions", len(emotion_counts))
                    st.metric("Most Common", max(emotion_counts, key=emotion_counts.get))
                
                # Emotion distribution
                st.subheader("Emotion Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                emotions = list(emotion_counts.keys())
                counts = list(emotion_counts.values())
                ax.bar(emotions, counts, color='steelblue')
                ax.set_xlabel('Emotions')
                ax.set_ylabel('Count')
                ax.set_title('Emotion Distribution Across Files')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Detailed results
                st.subheader("Detailed Results")
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result['filename']}"):
                        st.write(f"Predicted: {result['predicted_emotion']} ({result['confidence']:.2%})")
    
    elif page == "🆚 Compare Files":
        st.header("Compare Two Audio Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            file1 = st.file_uploader("First file", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], key="file1")
        
        with col2:
            file2 = st.file_uploader("Second file", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], key="file2")
        
        if file1 and file2:
            st.audio(file1)
            st.audio(file2)
            
            if st.button("🆚 Compare", key="compare_button"):
                tmp_paths = []
                try:
                    # Save files temporarily
                    for i, uploaded_file in enumerate([file1, file2]):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_paths.append(tmp_file.name)
                    
                    with st.spinner("Analyzing files..."):
                        result1 = predictor.predict(tmp_paths[0])
                        result2 = predictor.predict(tmp_paths[1])
                        
                        compare_results(result1, result2, file1.name, file2.name)
                        
                        predictor.add_to_history('comparison', file1.name, result1)
                        predictor.add_to_history('comparison', file2.name, result2)
                
                finally:
                    for tmp_path in tmp_paths:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
    
    elif page == "📈 History":
        st.header("Prediction History")
        
        display_history(predictor.history)
        
        if st.button("🗑️ Clear History"):
            predictor.history = defaultdict(list)
            predictor.save_history()
            st.success("History cleared!")
            st.rerun()
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About & Features"):
        st.markdown("""
        ### Advanced Features:
        
        - **📁 File Upload**: Process individual audio files with visualizations
        - **🎙️ Live Recording**: Record and analyze audio in real-time
        - **📂 Multiple Files**: Batch process multiple audio files
        - **🆚 Compare Files**: Compare emotions between two audio files
        - **📈 History**: Track your prediction history over time
        
        ### Audio Visualizations:
        - Waveform display
        - Spectrogram analysis
        - Probability distribution charts
        
        ### How it works:
        1. Audio is converted to MFCC and spectral features
        2. Features are processed by a trained neural network
        3. The model predicts the most likely emotion with confidence scores
        """)


if __name__ == '__main__':
    main()
