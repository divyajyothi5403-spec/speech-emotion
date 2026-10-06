# Advanced Features Guide

This document describes the advanced features added to the Speech Emotion Recognition project.

## 🚀 New Features Overview

### 1. Advanced Command-Line Demo (`demo_advanced.py`)

A full-featured command-line interface with professional capabilities:

#### 🎙️ Live Voice Recording
- Record audio from microphone with customizable duration (1-10 seconds)
- Real-time emotion prediction
- Automatic history tracking

#### 📁 File Upload Options
- **Single File Upload**: Process individual audio files
- **Multiple File Upload**: Process several files at once
- **Directory Processing**: Analyze entire folders of audio files
- Supports: WAV, MP3, FLAC, OGG, M4A formats

#### 🎨 Audio Visualization
- **Waveform Display**: Visual representation of audio amplitude over time
- **Spectrogram**: Frequency spectrum visualization
- **MFCC Display**: Mel-frequency cepstral coefficients visualization
- Saves visualizations as PNG files

#### 📈 Prediction History
- Tracks all predictions with timestamps
- Stores results in JSON format
- Filter by source (file upload/live recording)
- View recent predictions

#### 🆚 File Comparison
- Compare emotions between two audio files
- Side-by-side confidence comparison
- Emotion matching analysis
- Confidence difference calculation

### 2. Advanced Web Interface (`web_demo_advanced.py`)

A professional web application with advanced features:

#### 📁 File Upload Tab
- Single file upload with preview
- Audio visualization toggle
- Real-time prediction with charts
- Automatic history saving

#### 🎙️ Live Recording Tab
- Adjustable recording duration (1-10 seconds)
- Real-time audio preview
- Instant emotion prediction
- Optional visualizations

#### 📂 Multiple Files Tab
- Batch file upload
- Progress tracking
- Summary statistics:
  - Total files processed
  - Average confidence
  - Unique emotions detected
  - Most common emotion
- Emotion distribution chart
- Detailed results for each file

#### 🆚 Compare Files Tab
- Side-by-side file upload
- Direct comparison interface
- Visual confidence comparison
- Emotion matching analysis

#### 📈 History Tab
- View all past predictions
- Filterable by source
- Timestamp tracking
- Clear history option
- Persistent storage

### 3. Enhanced Features

#### Audio Visualizations
All visualizations use librosa and matplotlib:

**Waveform:**
```python
# Shows amplitude over time
librosa.display.waveshow(y, sr=sr)
```

**Spectrogram:**
```python
# Shows frequency content over time
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
```

**MFCC:**
```python
# Shows mel-frequency cepstral coefficients
librosa.display.specshow(mfcc, sr=sr, x_axis='time')
```

#### Prediction History System
- Stores predictions in `models/prediction_history.json`
- Tracks:
  - Timestamp
  - Filename
  - Predicted emotion
  - Confidence score
  - Full probability distribution
- Persistent across sessions

#### Batch Processing
- Process multiple files efficiently
- Progress tracking
- Summary statistics
- Emotion distribution analysis
- Detailed per-file results

## 🎯 Usage Examples

### Advanced Command-Line Demo

```bash
python demo_advanced.py
```

**Example Workflow:**

1. **Live Recording:**
```
Enter your choice (1-8): 1
Enter recording duration in seconds (default=3): 5
🎙️ Starting microphone recording...
Recording for 5 seconds...
Speak now...
Recording finished!
```

2. **Multiple File Upload:**
```
Enter your choice (1-8): 3
Enter multiple file paths (one per line). Enter 'done' when finished:
File 1: "C:\audio\happy1.wav"
File 2: "C:\audio\sad1.wav"
File 3: done

Processing 2 files...
[1/2] Processing: happy1.wav
✓ Predicted: happy (87.5%)
[2/2] Processing: sad1.wav
✓ Predicted: sad (92.3%)

SUMMARY - 2 files processed
Average Confidence: 89.9%
Emotion Distribution:
  happy        | ████████████ | 1 (50.0%)
  sad          | ████████████ | 1 (50.0%)
```

3. **Audio Visualization:**
```
Enter your choice (1-8): 6
Enter the path to audio file for visualization: "audio.wav"
Generating visualizations...
1. Waveform...
   ✓ Saved as waveform.png
2. Spectrogram...
   ✓ Saved as spectrogram.png
3. MFCC...
   ✓ Saved as mfcc.png
```

### Advanced Web Interface

```bash
streamlit run web_demo_advanced.py
```

**Features accessible through sidebar navigation:**

1. **File Upload**: Upload single files with visualization options
2. **Live Recording**: Record audio with adjustable duration
3. **Multiple Files**: Batch upload with progress tracking
4. **Compare Files**: Side-by-side comparison interface
5. **History**: View and manage prediction history

## 🔧 Technical Implementation

### History System
```python
class AdvancedFileProcessor:
    def __init__(self, predictor):
        self.history_file = 'models/prediction_history.json'
        self.history = self.load_history()
    
    def add_to_history(self, source, filename, result):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'predicted_emotion': result['predicted_emotion'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        }
        self.history[source].append(entry)
        self.save_history()
```

### Audio Visualization
```python
class AudioVisualizer:
    @staticmethod
    def display_waveform(audio_path, save_path):
        y, sr = librosa.load(audio_path, sr=None)
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title('Audio Waveform')
        plt.savefig(save_path)
        plt.close()
```

### Batch Processing
```python
def process_multiple_files(self, file_paths):
    results = []
    for i, file_path in enumerate(file_paths, 1):
        print(f"[{i}/{len(file_paths)}] Processing: {file_path}")
        result = self.predictor.predict(file_path)
        results.append(result)
        self.add_to_history('batch_upload', file_path, result)
    return results
```

## 📊 Comparison Features

### Command-Line Comparison
```
Enter your choice (1-8): 7
Enter paths to 2 audio files to compare:
File 1: "audio1.wav"
File 2: "audio2.wav"

COMPARISON RESULTS
File 1: audio1.wav
  Predicted: HAPPY (87.5%)

File 2: audio2.wav
  Predicted: SAD (92.3%)

✗ Different emotions detected
  File 1: happy
  File 2: sad

Confidence difference: 4.8%
```

### Web Interface Comparison
- Side-by-side upload interface
- Visual comparison cards
- Automatic emotion matching analysis
- Confidence difference display

## 🎨 Visualization Benefits

### Waveform
- Shows audio amplitude over time
- Helps identify speech patterns
- Useful for quality assessment

### Spectrogram
- Shows frequency content over time
- Reveals spectral characteristics
- Helps understand acoustic features

### MFCC
- Shows mel-frequency cepstral coefficients
- Represents the spectral envelope
- Used as input features for the model

## 💾 Data Management

### History Storage
- Location: `models/prediction_history.json`
- Format: JSON
- Structure:
```json
{
  "file_upload": [
    {
      "timestamp": "2026-09-11T14:30:00",
      "filename": "audio.wav",
      "predicted_emotion": "happy",
      "confidence": 0.875,
      "probabilities": {
        "happy": 0.875,
        "sad": 0.125,
        ...
      }
    }
  ],
  "live_recording": [...],
  "batch_upload": [...],
  "comparison": [...]
}
```

### Temporary Files
- Waveform: `waveform.png`
- Spectrogram: `spectrogram.png`
- MFCC: `mfcc.png`
- Recording: `temp_recording.wav`
- All automatically cleaned up

## 🚀 Performance Optimizations

### Batch Processing
- Progress tracking
- Efficient memory usage
- Error handling per file
- Summary statistics generation

### Web Interface
- Cached model loading
- Efficient file handling
- Responsive design
- Real-time updates

## 🎯 Use Cases

### Research & Analysis
- Batch process datasets
- Compare different recordings
- Track prediction patterns
- Visual analysis of audio

### Development & Testing
- Test model on multiple files
- Compare different audio sources
- Monitor prediction consistency
- Debug audio processing

### Production Use
- Process user uploads
- Real-time emotion detection
- Historical tracking
- Quality assurance

## 🔮 Future Enhancements

Potential additions:
- Real-time streaming visualization
- Export results to CSV/Excel
- Advanced statistical analysis
- Custom visualization themes
- Audio quality metrics
- Emotion trend analysis
- Multi-language support

## 📝 Notes

- All visualizations use librosa and matplotlib
- History is persistent across sessions
- Temporary files are automatically cleaned
- Web interface requires Streamlit
- Advanced features require matplotlib for visualizations

## 🛠️ Troubleshooting

### Visualization Issues
- Ensure matplotlib is installed
- Check file permissions for saving images
- Verify audio file format compatibility

### History Issues
- Check `models/` directory permissions
- Verify JSON file integrity
- Ensure sufficient disk space

### Performance Issues
- Reduce batch size for large directories
- Disable visualizations for faster processing
- Use simpler audio formats (WAV recommended)
