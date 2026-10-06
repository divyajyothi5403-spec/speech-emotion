import librosa
import numpy as np
import soundfile as sf
import os


def extract_mfcc(audio_path, n_mfcc=40, n_fft=512, hop_length=256):
    """
    Extract MFCC features from audio file.
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients to extract
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
    
    Returns:
        mfcc_features: MFCC features array
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # Transpose to have time steps as first dimension
        mfcc_features = mfcc.T
        
        return mfcc_features
    except Exception as e:
        print(f"Error extracting MFCC from {audio_path}: {e}")
        return None


def extract_spectral_features(audio_path):
    """
    Extract spectral features from audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        spectral_features: Dictionary containing various spectral features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y)[0]
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Mel-frequency cepstral coefficients
        features['mfcc'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        return features
    except Exception as e:
        print(f"Error extracting spectral features from {audio_path}: {e}")
        return None


def extract_combined_features(audio_path, n_mfcc=40, max_len=100):
    """
    Extract combined MFCC and spectral features with fixed length.
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        max_len: Maximum length for feature sequence (padding/truncation)
    
    Returns:
        combined_features: Combined feature array with fixed length
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Combine features
        combined = np.vstack([
            mfcc,
            spectral_centroid,
            spectral_rolloff,
            spectral_bandwidth,
            zero_crossing_rate
        ])
        
        # Transpose to have time steps as first dimension
        combined = combined.T
        
        # Pad or truncate to fixed length
        if combined.shape[0] < max_len:
            # Pad with zeros
            padding = np.zeros((max_len - combined.shape[0], combined.shape[1]))
            combined = np.vstack([combined, padding])
        elif combined.shape[0] > max_len:
            # Truncate
            combined = combined[:max_len, :]
        
        return combined
    except Exception as e:
        print(f"Error extracting combined features from {audio_path}: {e}")
        return None


def load_audio_data(data_dir, emotion_labels, max_len=100):
    """
    Load and extract features from audio files in a directory structure.
    
    Args:
        data_dir: Root directory containing emotion subdirectories
        emotion_labels: Dictionary mapping emotion names to labels
        max_len: Maximum length for feature sequences
    
    Returns:
        X: Feature array
        y: Label array
    """
    X = []
    y = []
    
    for emotion, label in emotion_labels.items():
        emotion_dir = os.path.join(data_dir, emotion)
        
        if not os.path.exists(emotion_dir):
            print(f"Directory not found: {emotion_dir}")
            continue
        
        for audio_file in os.listdir(emotion_dir):
            if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
                audio_path = os.path.join(emotion_dir, audio_file)
                
                # Extract features
                features = extract_combined_features(audio_path, max_len=max_len)
                
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    return np.array(X), np.array(y)


def augment_audio(y, sr):
    """
    Apply data augmentation to audio.
    
    Args:
        y: Audio time series
        sr: Sampling rate
    
    Returns:
        augmented_audio: List of augmented audio samples
    """
    augmented = []
    
    # Original
    augmented.append(y)
    
    # Noise injection
    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)
    
    # Time stretching
    augmented.append(librosa.effects.time_stretch(y, rate=1.2))
    augmented.append(librosa.effects.time_stretch(y, rate=0.8))
    
    # Pitch shifting
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    
    return augmented
