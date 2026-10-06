"""
Flask backend for Speech Emotion Recognition website
Integrates the web interface with the existing Python prediction system
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import tempfile

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import prediction modules
try:
    from demo import EmotionPredictor
    PREDICTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import prediction modules: {e}")
    print("Website will run in demo mode with simulated predictions.")
    PREDICTION_AVAILABLE = False
    EmotionPredictor = None

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize predictor
predictor = None

DEMO_EMOTIONS = ['happy', 'sad', 'angry', 'neutral']

def demo_prediction():
    """Return a simulated prediction when no trained model is deployed."""
    import random

    predicted_emotion = random.choice(DEMO_EMOTIONS)
    probabilities = {emotion: random.random() for emotion in DEMO_EMOTIONS}
    total = sum(probabilities.values())
    probabilities = {key: value / total for key, value in probabilities.items()}
    return {
        'emotion': predicted_emotion,
        'confidence': probabilities[predicted_emotion],
        'probabilities': probabilities,
        'demo_mode': True
    }

def get_predictor():
    """Get or initialize the emotion predictor."""
    global predictor
    if predictor is None:
        try:
            if PREDICTION_AVAILABLE and EmotionPredictor is not None:
                predictor = EmotionPredictor()
            else:
                print("EmotionPredictor class not available or prediction disabled")
                return None
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            return None
    return predictor

@app.route('/')
def index():
    """Serve the main website."""
    return send_from_directory(current_dir, 'index.html')

@app.route('/styles.css')
def styles():
    """Serve the CSS file."""
    return send_from_directory(current_dir, 'styles.css')

@app.route('/script.js')
def script():
    """Serve the JavaScript file."""
    return send_from_directory(current_dir, 'script.js')

@app.route('/api/predict', methods=['POST'])
def predict_emotion():
    """
    Predict emotion from uploaded audio file.
    
    Expected POST data:
    - audio: Audio file (WAV, MP3, etc.)
    
    Returns:
    - JSON response with prediction results
    """
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Keep the original container extension so audio decoders can identify it.
        file_extension = os.path.splitext(audio_file.filename)[1].lower() or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Check if prediction is available
            if not PREDICTION_AVAILABLE:
                return jsonify(demo_prediction())
            
            # Get predictor
            pred = get_predictor()
            if pred is None:
                return jsonify(demo_prediction())
            
            # Make prediction
            result = pred.predict(tmp_path)
            
            if result is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
            # Format response
            response = {
                'emotion': result['predicted_emotion'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'demo_mode': False
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if not PREDICTION_AVAILABLE:
        return jsonify({
            'status': 'healthy',
            'model_loaded': False,
            'demo_mode': True,
            'emotions': DEMO_EMOTIONS
        })
    
    pred = get_predictor()
    if pred is None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': False,
            'demo_mode': True,
            'emotions': DEMO_EMOTIONS,
            'message': 'Running in demo mode because no trained model is deployed'
        })
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'demo_mode': False,
        'emotions': list(pred.emotion_labels.keys()) if hasattr(pred, 'emotion_labels') else []
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information."""
    if not PREDICTION_AVAILABLE:
        return jsonify({
            'demo_mode': True,
            'emotions': DEMO_EMOTIONS,
            'message': 'Running in demo mode with simulated predictions'
        })
    
    pred = get_predictor()
    if pred is None:
        return jsonify({
            'demo_mode': True,
            'emotions': DEMO_EMOTIONS,
            'message': 'Running in demo mode because no trained model is deployed'
        })
    
    try:
        info = {
            'demo_mode': False,
            'model_type': pred.config.get('model_type', 'unknown') if hasattr(pred, 'config') else 'unknown',
            'num_classes': pred.config.get('num_classes', 0) if hasattr(pred, 'config') else 0,
            'emotions': list(pred.emotion_labels.keys()) if hasattr(pred, 'emotion_labels') else [],
            'input_shape': pred.config.get('input_shape', []) if hasattr(pred, 'config') else []
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Speech Emotion Recognition Web Server...")
    print("Open your browser to: http://localhost:5000")
    
    # Check if model exists
    if not os.path.exists('models/speech_emotion_model.h5'):
        print("WARNING: Trained model not found!")
        print("Please train the model first by running: python train.py")
        print("The website will run in demo mode with simulated predictions.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)