import os
import sys
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import argparse
from arabic_r_disorder_classifier import CNNLSTMAttention, extract_features as extract_features_cnn

# Define constants
SAMPLE_RATE = 16000
CLASSES = ['Deletion', 'Distortion', 'Normal', 'Substitution_gh', 'Substitution_l']
NUM_CLASSES = len(CLASSES)
MAX_LENGTH = 250  # Maximum sequence length for transformer

def load_models():
    """Load all trained models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CNN-LSTM model
    cnn_input_dim = 169  # This should match the feature dimension from extract_features_cnn
    cnn_hidden_dim = 128
    cnn_model = CNNLSTMAttention(cnn_input_dim, cnn_hidden_dim, NUM_CLASSES)
    if os.path.exists('final_cnn_lstm_model.pth'):
        cnn_model.load_state_dict(torch.load('final_cnn_lstm_model.pth', map_location=device))
        cnn_model = cnn_model.to(device)
        cnn_model.eval()
        print("CNN-LSTM model loaded successfully")
    else:
        print("CNN-LSTM model not found")
        cnn_model = None
    

    return cnn_model, device

def predict_file(file_path, cnn_model,  device):
    """Make predictions on a single audio file using all available models"""
    results = {}
    
    # Extract features
    cnn_features = extract_features_cnn(file_path)
    
    # Convert to tensors
    cnn_features_tensor = torch.FloatTensor(cnn_features).unsqueeze(0).to(device)
    attention_mask = torch.ones(1, MAX_LENGTH).to(device)
    
    # Predict with CNN-LSTM model
    if cnn_model is not None:
        with torch.no_grad():
            cnn_output, _ = cnn_model(cnn_features_tensor)
            cnn_probs = torch.nn.functional.softmax(cnn_output, dim=1)
            cnn_pred = torch.argmax(cnn_probs, dim=1).item()
            results['CNN-LSTM'] = {
                'prediction': CLASSES[cnn_pred],
                'confidence': cnn_probs[0][cnn_pred].item(),
                'all_probs': cnn_probs[0].cpu().numpy()
            }

    
    return results

def process_directory(directory, cnn_model, device):
    """Process all WAV files in a directory"""
    results = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                try:
                    file_results = predict_file(file_path, cnn_model, device)
                    results[file_path] = file_results
                    
                    # Print results
                    print(f"Results for {file_path}:")
                    for model_name, model_results in file_results.items():
                        print(f"  {model_name}: {model_results['prediction']} (confidence: {model_results['confidence']:.4f})")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict Arabic [ر] pronunciation disorders')
    parser.add_argument('input', help='Path to audio file or directory')
   
    
    args = parser.parse_args()
    
    # Load models
    cnn_model, device = load_models()
    
    # Select models to use based on user choice
    if cnn_model is None:
            print("CNN-LSTM model not available")
            return

    # Process input
    if os.path.isfile(args.input):
        if not args.input.endswith('.wav'):
            print(f"Error: {args.input} is not a WAV file")
            return
        
        results = predict_file(args.input, cnn_model, device)
        
        # Print results
        print(f"\nResults for {args.input}:")
        for model_name, model_results in results.items():
            print(f"\n{model_name} model:")
            print(f"  Prediction: {model_results['prediction']}")
            print(f"  Confidence: {model_results['confidence']:.4f}")
            print("  Class probabilities:")
            for i, (cls, prob) in enumerate(zip(CLASSES, model_results['all_probs'])):
                print(f"    {cls}: {prob:.4f}")
    
    elif os.path.isdir(args.input):
        results = process_directory(args.input, cnn_model,  device)
        
        # Print summary
        print("\nSummary:")
        total_files = len(results)
        print(f"Processed {total_files} files")
        
        # Count predictions by model
        if args.model == 'all' or args.model == 'ensemble':
            ensemble_counts = {}
            for file_results in results.values():
                if 'Ensemble' in file_results:
                    pred = file_results['Ensemble']['prediction']
                    ensemble_counts[pred] = ensemble_counts.get(pred, 0) + 1
            
            print("\nEnsemble model predictions:")
            for cls in CLASSES:
                count = ensemble_counts.get(cls, 0)
                percentage = (count / total_files) * 100 if total_files > 0 else 0
                print(f"  {cls}: {count} files ({percentage:.1f}%)")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main() 