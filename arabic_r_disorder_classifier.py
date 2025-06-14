import os
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

# Feature extraction functions using torchaudio
def extract_features(file_path, n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128):
    """Extract multiple features from audio file using torchaudio for faster processing"""
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
        # Approximate librosa's preemphasis with a simple filter
        waveform_emphasized = torch.cat([waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1)
        
        # Pad or truncate to fixed length (1 second)
        target_length = SAMPLE_RATE
        if waveform_emphasized.shape[1] < target_length:
            # Pad
            padding = target_length - waveform_emphasized.shape[1]
            waveform_fixed = torch.nn.functional.pad(waveform_emphasized, (0, padding))
        else:
            # Truncate
            waveform_fixed = waveform_emphasized[:, :target_length]
            
        # Extract MFCC - simplified parameters to avoid dimension issues
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
        mel_spectrogram = torch.log(mel_spectrogram + 1e-9)  # Log-Mel spectrogram
        
        # Convert to numpy arrays
        mfccs_np = mfccs.squeeze(0).cpu().numpy()
        mel_np = mel_spectrogram.squeeze(0).cpu().numpy()
        
        # Ensure all features have the same second dimension
        target_width = 32  # Fixed width for all features
        
        # Resize features if needed
        if mfccs_np.shape[1] != target_width:
            mfccs_resized = np.zeros((mfccs_np.shape[0], target_width))
            width = min(mfccs_np.shape[1], target_width)
            mfccs_resized[:, :width] = mfccs_np[:, :width]
            mfccs_np = mfccs_resized
            
        if mel_np.shape[1] != target_width:
            mel_resized = np.zeros((mel_np.shape[0], target_width))
            width = min(mel_np.shape[1], target_width)
            mel_resized[:, :width] = mel_np[:, :width]
            mel_np = mel_resized
        
        # Create a fixed-size zero crossing rate feature
        zcr_np = np.zeros((1, target_width))
        
        # Combine features
        features = np.vstack([mfccs_np, mel_np, zcr_np])
        
        # Normalize features
        features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        # Return a dummy feature array instead of None to avoid TypeError
        dummy_features = np.zeros((169, 32))  # Match expected dimensions (40 mfcc + 128 mel + 1 zcr)
        return dummy_features

# Dataset class
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
            # Use a dummy feature array if extraction failed
            features = np.zeros((169, 32))  # Match expected dimensions
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        # Apply transforms if any
        if self.transform:
            features = self.transform(features)
        
        return features, label

# Attention mechanism
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (batch, time, features)
        batch_size, time_steps, features = x.size()
        
        # Reshape for attention calculation
        x_reshaped = x.reshape(-1, features)  # (batch*time, features)
        
        # Calculate attention weights
        attention_weights = self.attention(x_reshaped)  # (batch*time, 1)
        attention_weights = attention_weights.reshape(batch_size, time_steps, 1)  # (batch, time, 1)
        
        # Apply attention weights
        attended = x * attention_weights  # (batch, time, features)
        
        # Sum over time dimension
        context = torch.sum(attended, dim=1)  # (batch, features)
        
        return context, attention_weights

# CNN-LSTM model with attention
class CNNLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(CNNLSTMAttention, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layers
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, features, time)
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)
        
        # LSTM layers
        x, _ = self.lstm(x)
        
        # Attention
        x, attention_weights = self.attention(x)
        
        # Output layer
        x = self.fc(x)
        
        return x, attention_weights

# Training function for all epochs
def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
    model.train()
    train_losses = []
    train_accs = []
    
    # Start training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Process all batches in this epoch
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
            # Convert labels to long type to fix potential "nll_loss_forward_reduce_cuda_kernel_2d_index" error
            labels = labels.long()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        # Save model at regular intervals
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f'cnn_lstm_model_epoch_{epoch+1}.pth')
            print(f"Model checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_cnn_lstm_model.pth')
    print("Final model saved")
    
    return train_losses, train_accs

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
            
            # Forward pass
            outputs, _ = model(inputs)
            # Convert labels to long type to fix potential "nll_loss_forward_reduce_cuda_kernel_2d_index" error
            labels = labels.long()
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Main function
def main():
    # Print device information once at the beginning
    print(f"Using device: {device}")
    
    # Data directories
    train_dir = 'Train'
    test_dir = 'Test'
    
    # Create datasets
    train_dataset = ArabicRDisorderDataset(train_dir, mode='train')
    test_dataset = ArabicRDisorderDataset(test_dir, mode='test')
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Get input dimensions from the first sample
    sample_features, _ = train_dataset[0]
    input_dim = sample_features.shape[0]  # Number of features
    
    # Create model
    hidden_dim = 128
    model = CNNLSTMAttention(input_dim, hidden_dim, NUM_CLASSES)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training parameters
    num_epochs = 20
    
    # Train the model (all epochs at once)
    train_losses, train_accs = train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)
    
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
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('cnn_lstm_training_history.png')
    
    print(f"Final test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main() 