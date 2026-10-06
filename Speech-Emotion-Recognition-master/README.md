# Speech Emotion Recognition using RNN/LSTM

A deep learning project for recognizing emotions from speech audio using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The system extracts MFCC and spectral features from audio and classifies emotions such as happy, sad, angry, neutral, etc.

## Features

- **Audio Feature Extraction**: MFCC and spectral features using librosa
- **Multiple Model Architectures**: LSTM, Bidirectional LSTM, GRU, and CNN-LSTM hybrid models
- **Multiple Input Methods**: 
  - Live microphone recording
  - Single audio file upload (WAV, MP3, FLAC, OGG, M4A)
  - Multiple file upload and batch processing
  - Directory processing
- **Advanced Features**:
  - Audio visualization (waveform, spectrogram, MFCC)
  - Prediction history tracking
  - File comparison functionality
  - Real-time confidence scores
- **Web Interface**: Browser-based application with Streamlit
- **Interactive Demo**: Enhanced command-line interface with menu system
- **Data Preprocessing**: Automatic normalization, augmentation, and train/val/test splitting
- **Flexible Dataset Support**: Works with custom datasets and RAVDESS format

## Project Structure

```
.
├── data/                      # Directory for audio datasets
│   ├── happy/                # Emotion-labeled audio files
│   ├── sad/
│   ├── angry/
│   └── neutral/
├── models/                    # Saved models and configurations
│   ├── speech_emotion_model.h5
│   ├── model_config.pkl
│   ├── label_encoder.pkl
│   ├── mean.npy
│   └── std.npy
├── src/                       # Source code
│   ├── __init__.py
│   ├── feature_extraction.py   # Audio feature extraction
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   └── model.py                # Model architectures
├── train.py                   # Training script
├── demo.py                    # Simple microphone demo
├── demo_enhanced.py           # Enhanced demo with file upload & live voice
├── demo_advanced.py           # Advanced demo with visualization & history
├── web_demo.py                # Basic Streamlit web interface
├── web_demo_advanced.py       # Advanced Streamlit web interface
├── config.py                  # Configuration file
├── requirements.txt           # Python dependencies
├── USAGE_GUIDE.md             # Detailed usage instructions
└── README.md                  # This file
```

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

The root `requirements.txt` contains the lightweight dependencies needed by the Vercel web deployment. For local training, Streamlit demos, and model prediction, install the full dependency set instead:

```bash
pip install -r requirements-local.txt
```

**Note**: PyAudio is optional and is only required for live microphone recording. It is not required for the web upload interface or Vercel deployment. It may require additional system dependencies:
- **Windows**: Usually works with pip install
- **Linux**: `sudo apt-get install python3-pyaudio portaudio19-dev`
- **macOS**: `brew install portaudio` then `pip install pyaudio`

## Dataset Preparation

### Option 1: Custom Dataset

Organize your audio files in the following structure:

```
data/
├── happy/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── sad/
│   ├── sample1.wav
│   └── ...
├── angry/
│   └── ...
├── neutral/
│   └── ...
└── ... (other emotions)
```

Supported audio formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

### Option 2: RAVDESS Dataset

Download the RAVDESS dataset and organize it by emotion labels. The project supports 8 emotions:
- neutral, calm, happy, sad, angry, fearful, disgusted, surprised

## Training

Train the model with your dataset:

```bash
python train.py --data_dir data --model_type lstm --epochs 50
```

### Training Options

- `--data_dir`: Directory containing audio data (default: `data`)
- `--model_type`: Model architecture - `lstm`, `bidirectional_lstm`, `gru`, `cnn_lstm` (default: `lstm`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--lstm_units`: Number of LSTM units (default: 128)
- `--dropout_rate`: Dropout rate (default: 0.3)
- `--max_len`: Maximum sequence length (default: 100)
- `--dataset`: Dataset type - `ravdess` or `custom` (default: `custom`)

### Example Commands

```bash
# Train with LSTM model
python train.py --data_dir data --model_type lstm --epochs 50

# Train with Bidirectional LSTM for better accuracy
python train.py --data_dir data --model_type bidirectional_lstm --epochs 100

# Train with CNN-LSTM hybrid
python train.py --data_dir data --model_type cnn_lstm --epochs 75

# Train with RAVDESS dataset
python train.py --data_dir ravdess_data --dataset ravdess --model_type lstm
```

## Real-time Demo

The project offers multiple ways to use the emotion recognition system:

### 1. Advanced Command-Line Demo (Recommended)

Full-featured interactive menu with advanced options:

```bash
python demo_advanced.py
```

Features:
- **🎙️ Live Voice Recording**: Record from microphone with custom duration
- **📁 Single File Upload**: Process any audio file on your computer
- **📂 Multiple File Upload**: Process multiple files at once
- **📊 Directory Processing**: Analyze entire directories of audio files
- **📈 Prediction History**: View and track past predictions
- **🎨 Audio Visualization**: Waveform, spectrogram, and MFCC displays
- **🆚 File Comparison**: Compare emotions between two audio files
- **Detailed Results**: Confidence scores and probability breakdowns

### 2. Enhanced Command-Line Demo

Interactive menu with basic options:

```bash
python demo_enhanced.py
```

Features:
- **Live Voice Recording**: Record from microphone with custom duration
- **File Upload**: Process any audio file on your computer
- **Batch Processing**: Analyze entire directories of audio files
- **Detailed Results**: Confidence scores and probability breakdowns

### 3. Advanced Web Interface (Streamlit)

Browser-based interface with advanced features:

```bash
streamlit run web_demo_advanced.py
```

Features:
- **📁 File Upload**: Single and multiple file upload
- **🎙️ Live Recording**: Built-in microphone recording
- **📂 Batch Processing**: Process multiple files at once
- **🆚 File Comparison**: Compare two audio files side-by-side
- **📈 History Tracking**: View prediction history
- **🎨 Visualizations**: Waveform, spectrogram, and probability charts
- **User-Friendly**: No command-line knowledge needed

### 4. Basic Web Interface (Streamlit)

Simple browser-based interface:

```bash
streamlit run web_demo.py
```

Features:
- **Drag & Drop File Upload**: Easy audio file upload
- **Live Recording**: Built-in microphone recording
- **Visual Results**: Beautiful charts and probability bars

### 5. Simple Demo

Basic microphone-based prediction:

```bash
python demo.py
```

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

## Model Architectures

### 1. LSTM
Standard LSTM network with 3 layers for sequence modeling.

### 2. Bidirectional LSTM
Processes sequences in both directions for better context understanding.

### 3. GRU
Gated Recurrent Units - lighter alternative to LSTM with similar performance.

### 4. CNN-LSTM Hybrid
Combines CNN for feature extraction with LSTM for sequence modeling.

## Feature Extraction

The system extracts the following features from audio:

- **MFCC (Mel-Frequency Cepstral Coefficients)**: Primary feature representation
- **Spectral Centroid**: Brightness of the sound
- **Spectral Rolloff**: Frequency below which a certain percentage of energy lies
- **Spectral Bandwidth**: Width of the spectrum
- **Zero Crossing Rate**: Rate at which signal changes sign

## Performance Tips

1. **Dataset Quality**: Ensure clear, emotion-labeled audio recordings
2. **Data Quantity**: More samples per emotion improve accuracy
3. **Audio Quality**: Use consistent recording conditions
4. **Model Selection**: Bidirectional LSTM often gives best results
5. **Training Time**: Increase epochs if validation loss is still decreasing

## Troubleshooting

### PyAudio Installation Issues
If PyAudio fails to install:
- Windows: Try `pip install pipwin` then `pipwin install pyaudio`
- Linux: Install PortAudio development headers first
- macOS: Install PortAudio via Homebrew

### Model Loading Errors
Ensure you have:
1. Trained the model first (`python train.py`)
2. All model files in the `models/` directory
3. Correct data directory structure

### Microphone Access
- Ensure your microphone is connected and working
- Check system microphone permissions
- Try recording duration of 2-5 seconds for best results

## Advanced Usage

### Custom Feature Extraction
Modify `src/feature_extraction.py` to add custom features or adjust extraction parameters.

### Model Architecture Customization
Edit `src/model.py` to create custom architectures or modify existing ones.

### Batch Prediction
Use the `EmotionPredictor` class in your own scripts to batch process audio files.

## Requirements

- Python 3.7+
- TensorFlow 2.13.0+
- librosa 0.10.0+
- numpy, pandas, scikit-learn
- pyaudio (for microphone input)
- matplotlib, seaborn (for visualization)
- streamlit (for web interface)

## License

This project is for educational and research purposes.

## Acknowledgments

- RAVDESS Dataset for providing emotional speech data
- librosa for audio processing capabilities
- TensorFlow/Keras for deep learning framework

## Future Improvements

- [ ] Support for more emotion categories
- [ ] Real-time streaming emotion detection
- [x] Web-based interface (Streamlit)
- [ ] Multi-language support
- [ ] Transfer learning from pre-trained models
- [ ] Attention mechanisms for better performance
- [ ] Mobile application support

## Contact

For questions or issues, please open an issue in the project repository.
