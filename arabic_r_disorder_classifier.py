import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
import time
import psutil
import gc
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
SAMPLE_RATE = 16000
CLASSES = ['Deletion', 'Distortion', 'Normal', 'Substitution_gh', 'Substitution_l']
NUM_CLASSES = len(CLASSES)

# More robust feature extraction with better normalization
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
        
        #  Better normalization - per-feature standardization
        features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        # Return appropriate dummy features
        dummy_features = np.zeros((45, 63))  # 13 mfcc + 32 mel features
        return dummy_features

#  data augmentation with more realistic transformations
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

# Simplified attention mechanism to reduce overfitting
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        # Simplified attention with fewer parameters
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.Tanh(),
            nn.Dropout(0.3),  # Added dropout
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        batch_size, time_steps, features = x.size()
        
        x_reshaped = x.reshape(-1, features)
        attention_weights = self.attention(x_reshaped)
        attention_weights = attention_weights.reshape(batch_size, time_steps, 1)
        
        attended = x * attention_weights
        context = torch.sum(attended, dim=1)
        
        return context, attention_weights

class CNNLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.6):
        super(CNNLSTMAttention, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.4)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(64, hidden_dim//2, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, dropout=0 if num_layers == 1 else dropout)
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim) 
        
        #Simplified output layer with more dropout
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim//2, num_classes)
        
    def forward(self, x):
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM layers
        x, _ = self.lstm(x)
        
        # Attention
        x, attention_weights = self.attention(x)
        
        # Output layer
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x, attention_weights

#  early stopping with more conservative parameters
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=False):  
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'best_model.pth')
        self.val_loss_min = val_loss

# Training function with  regularization
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    model.train()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    early_stopping = EarlyStopping(patience=10, verbose=True, min_delta=0.001)
    
    # Initialize time and memory tracking
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Track GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        initial_gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB
    else:
        initial_gpu_memory = 0
        initial_gpu_memory_reserved = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            
            loss.backward()
            # More aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        current_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        memory_used = current_memory - initial_memory
        
        # Track GPU memory if available
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            current_gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            current_gpu_memory = 0
            current_gpu_memory_reserved = 0
            peak_gpu_memory = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Gap: {abs(epoch_acc - val_acc):.4f}")  # Monitor overfitting gap
        print(f"Epoch Time: {epoch_time:.2f} seconds")
        print(f"Current Memory Usage: {current_memory:.2f} MB (Increase: {memory_used:.2f} MB)")
        if torch.cuda.is_available():
            print(f"GPU Memory: {current_gpu_memory:.2f} MB allocated, {current_gpu_memory_reserved:.2f} MB reserved")
            print(f"Peak GPU Memory: {peak_gpu_memory:.2f} MB")
        
        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print("Loaded best model from early stopping")
    
    # Calculate total training time
    total_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    total_memory_increase = final_memory - initial_memory
    
    # Final GPU memory stats
    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        final_gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        gpu_memory_increase = final_gpu_memory - initial_gpu_memory
    else:
        final_gpu_memory = 0
        final_gpu_memory_reserved = 0
        peak_gpu_memory = 0
        gpu_memory_increase = 0
    
    # Log training metrics
    training_metrics = {
        "total_training_time": total_time,
        "epochs_completed": epoch + 1,
        "average_epoch_time": total_time / (epoch + 1),
        "initial_memory_usage_mb": initial_memory,
        "final_memory_usage_mb": final_memory,
        "memory_increase_mb": total_memory_increase,
        "initial_gpu_memory_mb": initial_gpu_memory,
        "final_gpu_memory_mb": final_gpu_memory,
        "gpu_memory_increase_mb": gpu_memory_increase,
        "peak_gpu_memory_mb": peak_gpu_memory,
        "gpu_memory_reserved_mb": final_gpu_memory_reserved,
        "final_train_accuracy": train_accs[-1],
        "final_val_accuracy": val_accs[-1]
    }
    
    return train_losses, train_accs, val_losses, val_accs, training_metrics

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def main():
    print(f"Using device: {device}")
    
    # Print GPU information if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Data directories
    train_dir = 'Train'
    test_dir = 'Test'
    
    train_transform = AudioAugmentation(p=0.4) 
    train_dataset = ArabicRDisorderDataset(train_dir, mode='train', transform=train_transform)
    test_dataset = ArabicRDisorderDataset(test_dir, mode='test')

    batch_size = 16 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Get input dimensions
    sample_features, _ = train_dataset[0]
    input_dim = sample_features.shape[0]
    
    hidden_dim = 64  
    model = CNNLSTMAttention(input_dim, hidden_dim, NUM_CLASSES, dropout=0.6)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3) 
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    num_epochs = 100
    
    # Train the model
    train_losses, train_accs, val_losses, val_accs, training_metrics = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs)
    
    # Final evaluation
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASSES))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Plot training history with overfitting indicators
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train', linewidth=2)
    plt.plot(val_accs, label='Validation', linewidth=2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # overfitting gap plot
    plt.subplot(1, 3, 3)
    gap = [abs(t - v) for t, v in zip(train_accs, val_accs)]
    plt.plot(gap, label='Train-Val Gap', linewidth=2, color='red')
    plt.title('Overfitting Gap')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final overfitting gap: {abs(train_accs[-1] - val_accs[-1]):.4f}")
    
    # Save training metrics to a text file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'arabic_r_disorder_model_metrics_{timestamp}.txt', 'w') as f:
        f.write("=== Arabic R-Disorder Classifier Training Metrics ===\n")
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== Hardware Information ===\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            f.write(f"Total GPU Memory: {gpu_memory:.1f} GB\n\n")
        else:
            f.write("GPU: Not available\n\n")
        
        f.write("=== Training Time ===\n")
        f.write(f"Total Training Time: {training_metrics['total_training_time']:.2f} seconds ({training_metrics['total_training_time']/60:.2f} minutes)\n")
        f.write(f"Epochs Completed: {training_metrics['epochs_completed']}\n")
        f.write(f"Average Time per Epoch: {training_metrics['average_epoch_time']:.2f} seconds\n\n")
        
        f.write("=== Memory Usage ===\n")
        f.write(f"Initial Memory Usage: {training_metrics['initial_memory_usage_mb']:.2f} MB\n")
        f.write(f"Final Memory Usage: {training_metrics['final_memory_usage_mb']:.2f} MB\n")
        f.write(f"Memory Increase: {training_metrics['memory_increase_mb']:.2f} MB\n\n")
        
        f.write("=== GPU Memory Usage ===\n")
        f.write(f"Initial GPU Memory Usage: {training_metrics['initial_gpu_memory_mb']:.2f} MB\n")
        f.write(f"Final GPU Memory Usage: {training_metrics['final_gpu_memory_mb']:.2f} MB\n")
        f.write(f"GPU Memory Increase: {training_metrics['gpu_memory_increase_mb']:.2f} MB\n")
        f.write(f"Peak GPU Memory Usage: {training_metrics['peak_gpu_memory_mb']:.2f} MB\n")
        f.write(f"GPU Memory Reserved: {training_metrics['gpu_memory_reserved_mb']:.2f} MB\n\n")
        
        f.write("=== Model Performance ===\n")
        f.write(f"Final Training Accuracy: {training_metrics['final_train_accuracy']:.4f}\n")
        f.write(f"Final Validation Accuracy: {training_metrics['final_val_accuracy']:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Overfitting Gap: {abs(train_accs[-1] - val_accs[-1]):.4f}\n\n")
        
        f.write("=== Model Architecture ===\n")
        f.write(f"Input Dimension: {input_dim}\n")
        f.write(f"Hidden Dimension: {hidden_dim}\n")
        f.write(f"Number of Classes: {NUM_CLASSES}\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n\n")
        
        f.write("=== Classification Report ===\n")
        f.write(classification_report(test_labels, test_preds, target_names=CLASSES))
    
    print(f"Training metrics saved to 'arabic_r_disorder_model_metrics_{timestamp}.txt'")

if __name__ == "__main__":
    main()