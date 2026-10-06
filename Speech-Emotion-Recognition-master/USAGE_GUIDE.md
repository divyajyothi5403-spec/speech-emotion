# Usage Guide - Speech Emotion Recognition

This guide explains how to use the different features of the Speech Emotion Recognition project.

## Features Overview

The project now includes three ways to use the emotion recognition system:

1. **Enhanced Command-Line Demo** (`demo_enhanced.py`) - Interactive menu with file upload and live voice
2. **Web Interface** (`web_demo.py`) - Browser-based interface with file upload and recording
3. **Original Demo** (`demo.py`) - Simple microphone-based prediction

## 1. Enhanced Command-Line Demo

### Running the Enhanced Demo

```bash
python demo_enhanced.py
```

### Features

- **Option 1: Live Voice Recording**
  - Records audio from your microphone
  - Customizable recording duration
  - Instant emotion prediction with confidence scores

- **Option 2: Upload Audio File**
  - Process any audio file on your computer
  - Supports: WAV, MP3, FLAC, OGG, M4A
  - Detailed probability breakdown

- **Option 3: Batch Processing**
  - Process entire directories of audio files
  - Get predictions for multiple files at once
  - Summary of all results

### Example Usage

```
============================================================
SPEECH EMOTION RECOGNITION - ENHANCED DEMO
============================================================

Choose an option:
1. Live Voice Recording (Microphone)
2. Upload Audio File
3. Batch Process Directory
4. Exit

Enter your choice (1-4): 1

Enter recording duration in seconds (default=3): 5

🎤 Starting microphone recording...
Recording for 5 seconds...
Speak now...
Recording finished!

============================================================
EMOTION PREDICTION RESULTS
============================================================
Predicted Emotion: HAPPY
Confidence: 87.5%

All Probabilities:
--------------------------------------------------
happy        | ████████████████████████████░░░░ | 87.5%
neutral      | ████████░░░░░░░░░░░░░░░░░░░░░░░░ | 12.3%
sad          | ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ | 0.2%
============================================================
```

## 2. Web Interface (Streamlit)

### Running the Web Demo

```bash
streamlit run web_demo.py
```

This will launch a web browser with the interactive interface.

### Features

- **📁 Upload Audio File**
  - Drag and drop or click to upload
  - Preview audio before prediction
  - Visual probability charts
  - Supports multiple audio formats

- **🎤 Record Live Audio**
  - Built-in microphone recording
  - Adjustable recording duration (1-10 seconds)
  - Real-time prediction
  - Visual results display

### Web Interface Benefits

- User-friendly graphical interface
- No command-line knowledge needed
- Beautiful visualizations
- Works in any modern browser
- Can be shared as a web application

### Example Web Interface

The web interface provides:
- Large, clear emotion prediction display
- Color-coded probability bars
- Interactive charts showing emotion distribution
- File information and audio preview
- Responsive design for different screen sizes

## 3. Original Demo (Simple)

### Running the Original Demo

```bash
python demo.py
```

### Features

- Simple microphone recording
- Basic emotion prediction
- Text-based output

## Prerequisites

### Before running any demo, ensure:

1. **Model is trained**: Run `python train.py` first
2. **Dependencies installed**: Run `pip install -r requirements.txt`
3. **Audio data available**: Place emotion-labeled audio files in `data/` directory
4. **Microphone working** (for live recording features)

### File Upload Requirements

For file upload and batch processing:
- Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`
- Files should contain clear speech
- Longer recordings may take more time to process

## Troubleshooting

### Common Issues

**"Model not found" error**
- Solution: Train the model first using `python train.py`
- Ensure the `models/` directory contains the trained model

**"Microphone not working" error**
- Solution: Check your microphone permissions
- Ensure PyAudio is installed: `pip install pyaudio`
- Test your microphone in system settings

**"Error processing audio file" error**
- Solution: Ensure the file is a valid audio format
- Check that the file is not corrupted
- Try converting to WAV format

**PyAudio installation issues**
- Windows: Try `pip install pipwin` then `pipwin install pyaudio`
- Linux: `sudo apt-get install python3-pyaudio portaudio19-dev`
- macOS: `brew install portaudio` then `pip install pyaudio`

## Advanced Usage

### Custom Recording Duration

**Command-line demo:**
```bash
python demo_enhanced.py
# Then enter custom duration when prompted
```

**Web interface:**
- Use the slider to adjust duration (1-10 seconds)

### Batch Processing

**Process entire directories:**
```bash
python demo_enhanced.py
# Choose option 3 and enter directory path
```

### Integration with Other Applications

The `EmotionPredictor` class can be imported and used in your own Python scripts:

```python
from demo import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor()

# Predict from file
result = predictor.predict('path/to/audio.wav')

# Predict from microphone
result = predictor.predict_from_microphone(duration=3)
```

## Performance Tips

1. **Audio Quality**: Use clear, noise-free recordings for better accuracy
2. **Recording Length**: 2-5 seconds is optimal for emotion detection
3. **File Format**: WAV files generally process faster than compressed formats
4. **Model Selection**: Bidirectional LSTM typically gives best accuracy

## Data Privacy

- All audio processing is done locally on your machine
- No audio data is sent to external servers
- Temporary files are automatically deleted after processing
- Web interface runs locally on your computer

## Next Steps

1. Train the model with your own dataset
2. Experiment with different model architectures
3. Integrate emotion recognition into your applications
4. Customize the web interface for your needs

## Support

For issues or questions:
- Check the main README.md for project overview
- Review training documentation in README.md
- Ensure all dependencies are properly installed
