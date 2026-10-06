# Speech Emotion Recognition Website

A modern, interactive website demonstrating Speech Emotion Recognition using RNN/LSTM deep learning.

## Features

- **Modern Dark Theme**: Beautiful dark background with glowing blue/purple accents
- **Interactive Process Visualization**: Three-step process demonstration
- **Live Audio Recording**: Real-time microphone input with audio visualization
- **Emotion Prediction**: Real-time emotion detection with confidence scores
- **Responsive Design**: Works on desktop and mobile devices
- **LSTM/RNN Visualization**: Interactive diagrams showing neural network architecture
- **Audio Waveforms**: Real-time audio visualization during recording

## Structure

```
website/
├── index.html          # Main HTML structure
├── styles.css          # Styling with dark theme and glowing effects
├── script.js           # Interactive JavaScript for audio visualization
├── app.py             # Flask backend for prediction integration
└── README.md          # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install flask flask-cors
```

The website also requires the main project dependencies:
```bash
pip install -r ../requirements.txt
```

### 2. Train the Model (Optional)

For real predictions, train the model first:
```bash
cd ..
python train.py --data_dir data --model_type lstm --epochs 50
```

If you don't train the model, the website will run in **demo mode** with simulated predictions.

## Running the Website

### Option 1: With Flask Backend (Recommended)

```bash
cd website
python app.py
```

Then open your browser to: `http://localhost:5000`

### Option 2: Static Files Only

Open `index.html` directly in your browser. The website will run in demo mode with simulated predictions.

## Features Breakdown

### 1. Hero Section
- Eye-catching title with gradient text
- Floating microphone animation with sound waves
- Call-to-action buttons
- Modern dark theme with glowing effects

### 2. Process Section
Three interactive steps showing how the system works:

**Step 1: Microphone Input & Feature Extraction**
- Audio bars visualization
- Feature extraction icons (MFCC, Chroma, Mel Spectrogram)

**Step 2: RNN/LSTM Model**
- Interactive LSTM cell diagram
- Input, Forget, and Output gates
- Hover effects on gates

**Step 3: Emotion Prediction**
- Animated emotion chart
- Emotion icons with probability bars
- Real-time confidence display

### 3. Live Demo Section
- Start/Stop recording buttons
- Real-time audio visualization on canvas
- Emotion prediction display with large icon
- Probability bars for all emotions
- Recording status indicator

### 4. About Section
- Technology cards with icons
- Hover effects and animations
- Information about deep learning, real-time processing, etc.

### 5. Footer
- Navigation links
- Technology information
- Copyright notice

## API Endpoints

### `/api/predict` (POST)
Predict emotion from uploaded audio file.

**Request:**
- `audio`: Audio file (WAV, MP3, etc.)

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.85,
  "probabilities": {
    "happy": 0.85,
    "sad": 0.05,
    "angry": 0.03,
    "neutral": 0.07
  },
  "demo_mode": false
}
```

### `/api/health` (GET)
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "demo_mode": false,
  "emotions": ["happy", "sad", "angry", "neutral"]
}
```

### `/api/model-info` (GET)
Get model information.

**Response:**
```json
{
  "demo_mode": false,
  "model_type": "lstm",
  "num_classes": 4,
  "emotions": ["happy", "sad", "angry", "neutral"],
  "input_shape": [100, 44]
}
```

## Demo Mode

If the trained model is not available, the website automatically runs in **demo mode**:
- Simulated emotion predictions
- Random probability distributions
- Full UI functionality for demonstration
- Clear indication that predictions are simulated

## Customization

### Colors
Edit the CSS variables in `styles.css`:
```css
:root {
    --primary-color: #00d4ff;
    --secondary-color: #9b59b6;
    --accent-color: #ff6b6b;
    --bg-dark: #0a0a1a;
    --bg-darker: #050510;
    --bg-card: #1a1a2e;
}
```

### Emotions
Add more emotions in `script.js`:
```javascript
const emotionIcons = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    neutral: '😐',
    // Add more emotions here
};
```

### LSTM Visualization
Customize the LSTM diagram in `index.html`:
```html
<div class="lstm-gates">
    <div class="gate input-gate">Input Gate</div>
    <div class="gate forget-gate">Forget Gate</div>
    <div class="gate output-gate">Output Gate</div>
</div>
```

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- Opera: ✅ Full support

**Note:** Microphone access requires HTTPS or localhost.

## Troubleshooting

### Microphone Not Working
- Ensure you're on HTTPS or localhost
- Check browser microphone permissions
- Try a different browser

### Backend Not Starting
- Check if Flask is installed: `pip install flask flask-cors`
- Check if port 5000 is available
- Try a different port: `python app.py --port 5001`

### Model Not Loading
- Ensure the model is trained: `python train.py`
- Check if `models/speech_emotion_model.h5` exists
- The website will automatically fall back to demo mode

## Performance

- **Static Assets**: CSS and JavaScript are minified for fast loading
- **Audio Processing**: Optimized for real-time performance
- **Visualization**: Canvas-based rendering for smooth animations
- **Responsive**: Optimized for mobile and desktop

## Security

- Input validation on all endpoints
- Temporary file cleanup after processing
- CORS enabled for development
- No sensitive data storage

## Future Enhancements

- [ ] WebAssembly for client-side audio processing
- [ ] WebSocket for real-time streaming
- [ ] Multi-language support
- [ ] Export prediction history
- [ ] Advanced analytics dashboard
- [ ] Mobile app version

## Credits

Built with:
- **HTML5/CSS3** - Modern web technologies
- **JavaScript (ES6+)** - Interactive features
- **Flask** - Python backend
- **Canvas API** - Audio visualization
- **Web Audio API** - Microphone access

## License

This project is for educational and research purposes.

## Support

For issues or questions:
1. Check the main project README.md
2. Review troubleshooting section
3. Ensure all dependencies are installed
