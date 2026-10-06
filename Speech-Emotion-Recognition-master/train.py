import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_preprocessing import DataPreprocessor, create_ravdess_emotion_labels, create_custom_emotion_labels
from src.model import create_model


def plot_training_history(history, save_path='models/training_history.png'):
    """
    Plot and save training history.
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing audio data')
    parser.add_argument('--model_type', type=str, default='lstm', 
                        choices=['lstm', 'bidirectional_lstm', 'gru', 'cnn_lstm'],
                        help='Type of model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lstm_units', type=int, default=128, help='Number of LSTM units')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--dataset', type=str, default='custom', 
                        choices=['ravdess', 'custom'],
                        help='Type of dataset')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Set up emotion labels based on dataset
    if args.dataset == 'ravdess':
        emotion_labels = create_ravdess_emotion_labels()
    else:
        emotion_labels = create_custom_emotion_labels()
    
    print("=" * 50)
    print("Speech Emotion Recognition Training")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Emotion labels: {emotion_labels}")
    print("=" * 50)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        emotion_labels=emotion_labels,
        max_len=args.max_len
    )
    
    # Prepare dataset
    print("\nPreparing dataset...")
    data = preprocessor.prepare_dataset()
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    num_classes = data['num_classes']
    
    print(f"\nInput shape: {X_train.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"\nBuilding {args.model_type} model...")
    model = create_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=num_classes,
        model_type=args.model_type,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate
    )
    
    # Train model
    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model('models/speech_emotion_model.h5')
    
    # Save training configuration
    config = {
        'model_type': args.model_type,
        'input_shape': (X_train.shape[1], X_train.shape[2]),
        'num_classes': num_classes,
        'emotion_labels': emotion_labels,
        'max_len': args.max_len,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }
    
    joblib.dump(config, 'models/model_config.pkl')
    print("Model configuration saved to models/model_config.pkl")
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
