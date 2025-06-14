# Arabic R-Disorder Classifier

## Project Overview
This project implements a deep learning-based classifier for detecting and classifying pronunciation disorders related to the Arabic letter "ر" (r). The system can identify different types of pronunciation disorders including:

- **Deletion**: When the "ر" sound is completely omitted
- **Distortion**: When the "ر" sound is pronounced incorrectly
- **Normal**: Correct pronunciation of "ر"
- **Substitution_gh**: When "ر" is substituted with a "غ"-like sound
- **Substitution_l**: When "ر" is substituted with an "ل"-like sound

## Technology Stack
- **Python 3.x**
- **PyTorch & TorchAudio**: For building and training the deep learning model
- **NumPy & Pandas**: For data manipulation
- **Matplotlib & Seaborn**: For visualization
- **Scikit-learn**: For evaluation metrics
- **SoundFile**: For audio processing

## Model Architecture
The project uses a hybrid CNN-LSTM-Attention architecture:

1. **Feature Extraction**: 
   - MFCC (Mel-Frequency Cepstral Coefficients)
   - Mel Spectrograms
   - Feature normalization and resizing

2. **Model Components**:
   - Convolutional layers for feature learning
   - Bidirectional LSTM for sequence modeling
   - Attention mechanism to focus on important parts of the audio
   - Fully connected layers for classification

3. **Training Techniques**:
   - Data augmentation (noise injection, time shifting, frequency masking)
   - Early stopping to prevent overfitting
   - Learning rate scheduling
   - Gradient clipping
   - Label smoothing
   - Dropout regularization

## Project Structure
```
.
├── Train/                  # Training data directory
│   ├── Deletion/           # Contains deletion disorder samples
│   ├── Distortion/         # Contains distortion disorder samples
│   ├── Normal/             # Contains normal pronunciation samples
│   ├── Substitution_gh/    # Contains غ substitution samples
│   └── Substitution_l/     # Contains ل substitution samples
├── Test/                   # Testing data directory (same structure as Train/)
├── arabic_r_disorder_classifier.py    # Main training script
├── inference.py            # Script for inference on new audio files
├── best_model.pth          # Saved model weights
└── requirements.txt        # Python dependencies
```

Each class directory contains:
- `wav/`: Directory containing WAV audio files
- `MFCC/`: Directory containing pre-computed MFCC features
- `list.txt`: List of audio files

## How It Works
1. **Data Preparation**:
   - Audio files are organized by disorder category
   - Features are extracted using MFCC and Mel spectrogram techniques
   - Data is split into training, validation, and test sets

2. **Training Process**:
   - The model is trained using cross-entropy loss
   - Early stopping monitors validation loss to prevent overfitting
   - The best model is saved during training

3. **Evaluation**:
   - Model performance is evaluated using accuracy, confusion matrix, and classification report
   - Visualizations are generated to show training progress and model performance

4. **Inference**:
   - New audio samples can be classified using the trained model
   - Both single files and directories can be processed

## How to Use

### Installation
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Training
To train the model:
```
python arabic_r_disorder_classifier.py
```

This will:
- Load the training data
- Train the model
- Save the best model to `best_model.pth`
- Generate performance visualizations

### Inference
To classify new audio files:
```
python inference.py path/to/audio.wav
```

For batch processing:
```
python inference.py path/to/directory
```

## Performance
The model achieves high accuracy in distinguishing between different types of pronunciation disorders, with particular strength in identifying normal pronunciation versus disorders.

## Techniques Used to Combat Overfitting
- Feature normalization
- Data augmentation
- Dropout regularization
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Label smoothing
- Reduced model complexity

## Future Improvements
- Implement more advanced audio augmentation techniques
- Explore transformer-based architectures
- Add support for real-time classification
- Develop a user-friendly interface for non-technical users

## Requirements
See `requirements.txt` for the full list of dependencies. 