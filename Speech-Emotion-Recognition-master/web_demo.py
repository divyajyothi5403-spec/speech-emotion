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

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.feature_extraction import extract_combined_features


# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f8ff;
        margin: 20px 0;
    }
    .emotion-bar {
        height: 30px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitEmotionPredictor:
    """
    Emotion predictor for Streamlit web interface.
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.label_encoder = None
        self.config = None
        self.mean = None
        self.std = None
        self.load_model()
    
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


def display_results(result):
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


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Speech Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using RNN/LSTM Deep Learning</p>', unsafe_allow_html=True)
    
    # Initialize predictor
    @st.cache_resource
    def get_predictor():
        return StreamlitEmotionPredictor()
    
    predictor = get_predictor()
    
    if predictor.model is None:
        st.error("⚠️ Model not found. Please train the model first by running: python train.py")
        st.info("Make sure you have audio data in the 'data' directory and a trained model in the 'models' directory.")
        return
    
    # Input method selection
    st.header("Choose Input Method")
    input_method = st.radio(
        "Select how you want to provide audio:",
        ["📁 Upload Audio File", "🎤 Record Live Audio"]
    )
    
    if input_method == "📁 Upload Audio File":
        st.subheader("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"Uploaded file: {uploaded_file.name}")
            st.audio(uploaded_file)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            try:
                # Add predict button
                if st.button("🔍 Predict Emotion", key="upload_predict"):
                    with st.spinner("Analyzing audio..."):
                        result = predictor.predict(tmp_path)
                        display_results(result)
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    else:  # Live audio recording
        st.subheader("Record Live Audio")
        
        # Recording duration
        duration = st.slider("Recording duration (seconds)", 1, 10, 3)
        
        st.info("Click the button below and start speaking when prompted.")
        
        # Record button
        if st.button("🎙️ Start Recording", key="record_button"):
            try:
                # Import recording functionality
                from demo import MicrophoneRecorder
                
                # Create temporary file for recording
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_path = tmp_file.name
                
                # Record audio
                with st.spinner(f"Recording for {duration} seconds... Speak now!"):
                    recorder = MicrophoneRecorder()
                    recorder.record(duration=duration, output_file=tmp_path)
                    recorder.close()
                
                st.success("Recording completed!")
                st.audio(tmp_path)
                
                # Predict
                with st.spinner("Analyzing audio..."):
                    result = predictor.predict(tmp_path)
                    display_results(result)
            
            except Exception as e:
                st.error(f"Error during recording: {e}")
                st.info("Make sure you have a working microphone and PyAudio installed.")
            
            finally:
                # Clean up temporary file
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    This application uses deep learning (RNN/LSTM) to recognize emotions from speech audio.
    
    **Supported Emotions:**
    - Happy
    - Sad  
    - Angry
    - Neutral
    - Fear (if included in training data)
    
    **How it works:**
    1. Audio is converted to MFCC and spectral features
    2. Features are processed by a trained neural network
    3. The model predicts the most likely emotion
    
    **Model Architecture:**
    - Input: MFCC + spectral features
    - Model: LSTM/Bidirectional LSTM/GRU/CNN-LSTM
    - Output: Emotion classification with confidence scores
    """)


if __name__ == '__main__':
    main()
