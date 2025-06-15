import os
import sys
from time import time
import psutil
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 16000
CLASSES = [
    'Normal',
    'Distortion',
    'Deletion',
    'Substitution_gh',
    'Substitution_l'
]

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using traditional ML methods only.")

def extract_features(file_path, n_mfcc=13, n_fft=1024, hop_length=256, n_mels=64):
    """Extract multiple features from audio file with better parameter tuning"""
    try:
        # Load audio file with torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            resampler = T.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ensure waveform is 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Pre-emphasize (high-pass filter)
        waveform_emphasized = torch.cat([waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1)
        
        # Pad or truncate to fixed length (1 second)
        target_length = SAMPLE_RATE
        if waveform_emphasized.shape[1] < target_length:
            padding = target_length - waveform_emphasized.shape[1]
            waveform_fixed = torch.nn.functional.pad(waveform_emphasized, (0, padding))
        else:
            waveform_fixed = waveform_emphasized[:, :target_length]
            
        # Extract MFCC with fewer coefficients
        mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
        mfccs = mfcc_transform(waveform_fixed)
        
        # Extract Mel spectrogram   
        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mel_spectrogram = mel_transform(waveform_fixed)
        mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
        
        # Convert to numpy arrays
        mfccs_np = mfccs.squeeze(0).cpu().numpy()
        mel_np = mel_spectrogram.squeeze(0).cpu().numpy()
        
        # Better feature size management
        target_width = 63  # Better aligned with audio processing
        
        # Resize features using interpolation for better quality
        def resize_feature(feature, target_width):
            if feature.shape[1] == target_width:
                return feature
            elif feature.shape[1] > target_width:
                # Downsample using averaging
                step = feature.shape[1] / target_width
                indices = np.arange(0, feature.shape[1], step).astype(int)[:target_width]
                return feature[:, indices]
            else:
                # Zero-pad
                padding = target_width - feature.shape[1]
                return np.pad(feature, ((0, 0), (0, padding)), mode='constant')
        
        mfccs_np = resize_feature(mfccs_np, target_width)
        mel_np = resize_feature(mel_np, target_width)
        
        # Only use most relevant features to reduce overfitting
        # Combine features 
        features = np.vstack([mfccs_np, mel_np[:32, :]])  # Use only first 32 mel bands
        
        # Better normalization - per-feature standardization
        features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        # Return appropriate dummy features
        dummy_features = np.zeros((45, 63))  # 13 mfcc + 32 mel features
        return dummy_features

# Data augmentation with more realistic transformations
class AudioAugmentation(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, features):
        if np.random.random() > self.p:
            return features
            
        # More conservative augmentation to prevent overfitting
        aug_type = np.random.choice(['noise', 'time_shift', 'freq_mask', 'none'], p=[0.3, 0.3, 0.3, 0.1])
        
        if aug_type == 'noise':
            # added noise level
            noise = torch.randn_like(features) * 0.02 
            return features + noise
        elif aug_type == 'time_shift':
            # Time shifting instead of masking
            shift = np.random.randint(-5, 6)
            if shift > 0:
                features = torch.cat([features[:, shift:], torch.zeros_like(features[:, :shift])], dim=1)
            elif shift < 0:
                features = torch.cat([torch.zeros_like(features[:, :abs(shift)]), features[:, :shift]], dim=1)
            return features
        elif aug_type == 'freq_mask':
            # More conservative frequency masking
            freq_dim = features.shape[0]
            mask_len = max(1, int(freq_dim * 0.1)) 
            mask_start = np.random.randint(0, max(1, freq_dim - mask_len))
            features[mask_start:mask_start+mask_len, :] = 0
            return features
        else:
            return features

# Dataset class (same as original but updated for new feature dimensions)
class ArabicRDisorderDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None, cache_features=True):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.cache_features = cache_features
        self.feature_cache = {}
        
        # Get all audio files and their labels
        self.files = []
        self.labels = []
        
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(data_dir, class_name, 'wav')
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.wav'):
                        self.files.append(os.path.join(class_dir, file_name))
                        self.labels.append(class_idx)
        
        # Convert labels to numpy array of type int64 (long)
        self.labels = np.array(self.labels, dtype=np.int64)
        len_files = len(self.files)
        if len_files == 0:
            print(f"No files found in {data_dir} for {mode} set")
            sys.exit(1)
        print(f"Loaded {len(self.files)} files for {mode} set")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Check if features are already cached
        if self.cache_features and file_path in self.feature_cache:
            features = self.feature_cache[file_path]
        else:
            # Extract features
            features = extract_features(file_path)
            
            # Cache features if needed
            if self.cache_features and features is not None:
                self.feature_cache[file_path] = features
        
        # Ensure features is not None
        if features is None:
            features = np.zeros((45, 63))  # Updated dimensions
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        # Apply transforms if any
        if self.transform:
            features = self.transform(features)
        
        return features, label

class ArabicRDisorderClassifier:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.features = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.class_folders = CLASSES

    def extract_audio_features(self, audio_path):
        """Wrapper for the new extract_features function"""
        features = extract_features(audio_path)
        if features is not None:
            # Flatten the features for traditional ML models
            return features.flatten()
        return None

    def load_dataset_pytorch(self, train_path, test_path=None):
        """Load dataset using PyTorch Dataset classes"""
        print("Loading dataset with PyTorch...")
        
        # Create datasets
        train_transform = AudioAugmentation(p=0.3) if train_path else None
        train_dataset = ArabicRDisorderDataset(train_path, mode='train', transform=train_transform)
        
        test_dataset = None
        if test_path and os.path.exists(test_path):
            test_dataset = ArabicRDisorderDataset(test_path, mode='test', transform=None)
        
        # Extract features and labels for traditional ML models
        train_features = []
        train_labels = []
        
        for i in range(len(train_dataset)):
            features, label = train_dataset[i]
            # Remove augmentation for feature extraction
            train_dataset.transform = None
            clean_features, _ = train_dataset[i]
            train_dataset.transform = train_transform
            
            train_features.append(clean_features.numpy().flatten())
            train_labels.append(label)
        
        test_features = []
        test_labels = []
        
        if test_dataset:
            for i in range(len(test_dataset)):
                features, label = test_dataset[i]
                test_features.append(features.numpy().flatten())
                test_labels.append(label)
        
        # Convert to numpy arrays
        self.X_train = np.array(train_features)
        self.y_train = np.array(train_labels)
        
        if test_features:
            self.X_test = np.array(test_features)
            self.y_test = np.array(test_labels)
        else:
            # Split training data if no separate test set
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
            )
        
        # Encode labels
        all_labels = np.concatenate([self.y_train, self.y_test])
        self.label_encoder.fit(all_labels)
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Dataset loaded: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
        print(f"Feature dimension: {self.X_train.shape[1]}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train_encoded, self.y_test_encoded

    def load_dataset(self, train_path, test_path=None):
        """Original dataset loading method for backwards compatibility"""
        return self.load_dataset_pytorch(train_path, test_path)

    def train_traditional_models(self):
        print("Training traditional ML models...")
        
        # Random Forest
        start_time = time()
        memory_before = get_memory_usage()
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train_encoded)
        memory_after = get_memory_usage()
        self.models['Random Forest'] = rf_model
        print(f"Random Forest trained in {time() - start_time:.2f} seconds")
        print(f"Random Forest memory usage: {memory_after - memory_before:.2f} MB (Total: {memory_after:.2f} MB)")
        
        # SVM
        start_time = time()
        memory_before = get_memory_usage()
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_model.fit(self.X_train_scaled, self.y_train_encoded)
        memory_after = get_memory_usage()
        self.models['SVM'] = svm_model
        print(f"SVM trained in {time() - start_time:.2f} seconds")
        print(f"SVM memory usage: {memory_after - memory_before:.2f} MB (Total: {memory_after:.2f} MB)")
        print("Traditional models trained successfully!")

    def train_deep_learning_model(self):
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping deep learning model.")
            return
            
        print("Training deep learning model...")
        
        start_time = time()
        memory_before = get_memory_usage()
        
        # Convert labels to categorical
        y_train_categorical = to_categorical(self.y_train_encoded)
        y_test_categorical = to_categorical(self.y_test_encoded)
        
        # Create model
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            self.X_train_scaled, y_train_categorical,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        memory_after = get_memory_usage()
        training_time = time() - start_time
        
        self.models['Deep Learning'] = model
        self.dl_history = history
        print(f"Deep learning model trained in {training_time:.2f} seconds")
        print(f"Deep learning model memory usage: {memory_after - memory_before:.2f} MB (Total: {memory_after:.2f} MB)")
        print("Deep learning model trained successfully!")    
    def evaluate_models(self):
        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} Results:")
            print("-" * 30)
            
            # Track memory usage during evaluation
            memory_before_eval = get_memory_usage()
            
            if model_name == 'Deep Learning' and TENSORFLOW_AVAILABLE:
                y_pred_proba = model.predict(self.X_test_scaled, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(self.X_test_scaled)
            
            memory_after_eval = get_memory_usage()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test_encoded, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test_encoded, y_pred, average='weighted'
            )
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Evaluation memory usage: {memory_after_eval - memory_before_eval:.2f} MB (Current total: {memory_after_eval:.2f} MB)")
              # Detailed classification report
            print("\nDetailed Classification Report:")
            target_names = CLASSES
            print(classification_report(self.y_test_encoded, y_pred, target_names=target_names))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test_encoded, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
        
        return results

    def analyze_feature_importance(self):
        if 'Random Forest' not in self.models:
            print("Random Forest model not trained. Cannot analyze feature importance.")
            return
        
        rf_model = self.models['Random Forest']
        feature_importance = rf_model.feature_importances_
        
        plt.figure(figsize=(12, 8))
        indices = np.argsort(feature_importance)[::-1][:20]
        plt.bar(range(20), feature_importance[indices])
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
        
        print("Top 10 most important features:")
        for i in range(10):
            print(f"Feature {indices[i]}: {feature_importance[indices[i]]:.4f}")

    def predict_single_audio(self, audio_path, model_name='Random Forest'):
        if model_name not in self.models:
            print(f"Model {model_name} not trained.")
            return None
        
        # Extract features using new method
        features = extract_features(audio_path)
        if features is None or features.size == 0:
            print("Failed to extract features from audio file.")
            return None
        
        # Flatten and scale features
        features_flat = features.flatten().reshape(1, -1)
        features_scaled = self.scaler.transform(features_flat)
        
        model = self.models[model_name]
        
        if model_name == 'Deep Learning' and TENSORFLOW_AVAILABLE:
            prediction_proba = model.predict(features_scaled, verbose=0)
            prediction = np.argmax(prediction_proba, axis=1)[0]
            confidence = np.max(prediction_proba)
        else:
            prediction = model.predict(features_scaled)[0]
            if hasattr(model, 'predict_proba'):
                confidence = np.max(model.predict_proba(features_scaled))
            else:
                confidence = None
        
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_index': prediction
        }

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("Arabic /r/ Phoneme Disorder Classification System")
    print("=" * 50)
    
    # Initialize classifier
    classifier = ArabicRDisorderClassifier()
    
    # Define paths
    train_path = "Train"
    test_path = "Test"
    
    # Load and process dataset
    print("Loading and processing dataset...")
    X_train, X_test, y_train, y_test = classifier.load_dataset(train_path, test_path)
    
    # Train models
    print("Training models...")
    classifier.train_traditional_models()
    start_time = time()
    classifier.train_deep_learning_model()
    print(f"Deep learning model training completed in {time() - start_time:.2f} seconds")
    # Evaluate models
    print("Evaluating models...")
    results = classifier.evaluate_models()
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    classifier.analyze_feature_importance()
    
    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()