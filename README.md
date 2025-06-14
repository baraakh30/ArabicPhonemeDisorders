# Arabic [ر] Pronunciation Disorder Classification

This project aims to identify pronunciation disorders of the Arabic phoneme [ر] (corresponding to [r] in IPA) at the beginning of words. The system classifies the disorder into five categories:

1. Normal (correct pronunciation)
2. Deletion (omitting the sound)
3. Distortion (incorrect articulation)
4. Substitution with [غ] (gh)
5. Substitution with [ل] (l)

## Dataset Structure

The dataset is organized as follows:
```
├── Train/
│   ├── Normal/
│   │   └── wav/
│   ├── Deletion/
│   │   └── wav/
│   ├── Distortion/
│   │   └── wav/
│   ├── Substitution_gh/
│   │   └── wav/
│   └── Substitution_l/
│       └── wav/
└── Test/
    ├── Normal/
    │   └── wav/
    ├── Deletion/
    │   └── wav/
    ├── Distortion/
    │   └── wav/
    ├── Substitution_gh/
    │   └── wav/
    └── Substitution_l/
        └── wav/
```

## Model Architecture

The system uses a hybrid CNN-LSTM architecture with attention mechanism:

1. **Feature Extraction**: Multiple acoustic features are extracted from audio files:
   - MFCC (Mel-Frequency Cepstral Coefficients)
   - Mel Spectrogram
   - Chroma Features
   - Spectral Contrast
   - Tonnetz
   - Zero Crossing Rate
   - Spectral Centroid, Bandwidth, and Rolloff

2. **Deep Learning Model**:
   - CNN layers for feature extraction
   - Bidirectional LSTM for temporal modeling
   - Attention mechanism to focus on important time steps
   - Fully connected layers for classification

## Requirements

To install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate the model:

```bash
python arabic_r_disorder_classifier.py
```

This will:
1. Load and preprocess the training and test data
2. Train the model for 50 epochs
3. Save the best model based on validation accuracy
4. Generate a classification report and confusion matrix
5. Save visualizations of the training history and results

## Results

The model performance is evaluated using:
- Accuracy
- Precision, recall, and F1-score for each class
- Confusion matrix

Results are saved as:
- `best_model.pth`: The trained model with the best validation accuracy
- `confusion_matrix.png`: Visualization of the model's predictions
- `training_history.png`: Training and validation loss/accuracy curves 