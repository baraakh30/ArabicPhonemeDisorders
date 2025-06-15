import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import os
import numpy as np
from pathlib import Path
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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
random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Class mapping
class_names = ['Deletion', 'Distortion', 'Normal', 'Substitution_gh', 'Substitution_l']
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_sample_rate=16000, max_length=3.0):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.max_length = max_length
        self.max_samples = int(target_sample_rate * max_length)
        
        self.audio_files = []
        self.labels = []
        
        # Load all audio files
        for class_name in class_names:
            class_dir = self.root_dir / class_name / 'wav'
            if class_dir.exists():
                for audio_file in class_dir.glob('*.wav'):
                    self.audio_files.append(audio_file)
                    self.labels.append(class_to_idx[class_name])
        
        print(f"Loaded {len(self.audio_files)} audio files from {root_dir}")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure waveform is 2D (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = T.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
            
            # Pad or truncate to fixed length
            if waveform.shape[1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            else:
                padding = self.max_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            
            # Ensure we have exactly the right shape: (max_samples,)
            waveform = waveform.squeeze(0)
            
            # Apply transforms
            if self.transform:
                waveform = self.transform(waveform)
            
            # Final check - ensure 1D tensor of correct length
            if waveform.dim() != 1 or waveform.shape[0] != self.max_samples:
                waveform = waveform.flatten()[:self.max_samples]
                if waveform.shape[0] < self.max_samples:
                    padding = self.max_samples - waveform.shape[0]
                    waveform = F.pad(waveform, (0, padding))
            
            return waveform, label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence with correct shape if loading fails
            return torch.zeros(self.max_samples), label

class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def __call__(self, waveform):
        # Ensure input is 1D
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        
        # Random choice of augmentation
        aug_type = random.choice(['noise', 'time_shift', 'pitch_shift', 'volume', 'none'])
        
        if aug_type == 'noise':
            # Add white noise
            noise_factor = random.uniform(0.001, 0.005)
            noise = torch.randn_like(waveform) * noise_factor
            waveform = waveform + noise
            
        elif aug_type == 'time_shift':
            # Random time shift
            shift_samples = random.randint(-1600, 1600)  # ±0.1 seconds
            waveform = torch.roll(waveform, shift_samples)
            
        elif aug_type == 'pitch_shift':
            # Speed perturbation (affects both pitch and tempo)
            speed_factor = random.uniform(0.9, 1.1)
            if speed_factor != 1.0:
                # Create new indices for resampling
                original_length = len(waveform)
                new_indices = torch.arange(0, original_length, speed_factor)
                new_indices = new_indices[new_indices < original_length].long()
                
                if len(new_indices) > 0:
                    waveform = waveform[new_indices]
                    
                    # Pad or truncate to original length
                    if len(waveform) < original_length:
                        padding = original_length - len(waveform)
                        waveform = F.pad(waveform, (0, padding))
                    else:
                        waveform = waveform[:original_length]
        
        elif aug_type == 'volume':
            # Volume augmentation
            volume_factor = random.uniform(0.8, 1.2)
            waveform = waveform * volume_factor
        
        # Ensure output is 1D and correct length
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
            
        return waveform

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, 64, 11, padding=5)
        
    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))
        out4 = F.relu(self.conv4(x))
        return torch.cat([out1, out2, out3, out4], dim=1)

class AdvancedAudioClassifier(nn.Module):
    def __init__(self, num_classes=5, sample_rate=16000):
        super(AdvancedAudioClassifier, self).__init__()
        self.sample_rate = sample_rate
        
        # Spectral feature extraction
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            f_min=0,
            f_max=8000
        )
        
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': 1024,
                'hop_length': 256,
                'n_mels': 128,
                'f_min': 0,
                'f_max': 8000
            }
        )
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(141)  # 128 mel + 13 mfcc
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 1024, stride=2),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(1024, 8, batch_first=True)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract spectral features
        mel_spec = self.mel_spectrogram(x)
        mel_spec = torch.log(mel_spec + 1e-8)  # Log mel spectrogram
        
        mfcc_features = self.mfcc(x)
        
        # Combine features
        features = torch.cat([mel_spec, mfcc_features], dim=1)
        
        # Multi-scale feature extraction
        features = self.multi_scale(features)
        
        # Residual blocks
        features = self.res_blocks(features)
        
        # Attention mechanism
        features_transposed = features.transpose(1, 2)  # (batch, seq, features)
        attended_features, _ = self.attention(features_transposed, features_transposed, features_transposed)
        features = attended_features.transpose(1, 2)  # Back to (batch, features, seq)
        
        # Global pooling
        avg_pool = self.global_avg_pool(features).squeeze(-1)
        max_pool = self.global_max_pool(features).squeeze(-1)
        
        # Combine pooled features
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
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
        # Training phase
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * train_correct / train_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        scheduler.step()
        
        # Calculate epoch time and memory usage
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
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print(f'  Epoch Time: {epoch_time:.2f} seconds')
        print(f'  Current Memory Usage: {current_memory:.2f} MB (Increase: {memory_used:.2f} MB)')
        if torch.cuda.is_available():
            print(f'  GPU Memory: {current_gpu_memory:.2f} MB allocated, {current_gpu_memory_reserved:.2f} MB reserved')
            print(f'  Peak GPU Memory: {peak_gpu_memory:.2f} MB')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_2.pth')
            print(f'  New best model saved! Validation Accuracy: {val_acc:.2f}%')
        
        print('-' * 60)
    
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
        "final_train_accuracy": train_accuracies[-1],
        "final_val_accuracy": val_accuracies[-1],
        "best_val_accuracy": best_val_acc
    }
    
    return train_losses, val_losses, train_accuracies, val_accuracies, training_metrics

def evaluate_model(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Classification report
    print('\nClassification Report:')
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_model_2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_acc, all_predictions, all_targets

def main():
    # Data paths
    train_dir = 'Train'
    test_dir = 'Test'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Data augmentation for training
    train_transform = AudioAugmentation()
    
    # Datasets
    train_dataset = AudioDataset(train_dir, transform=train_transform)
    test_dataset = AudioDataset(test_dir, transform=None)
    
    # Data loaders with error handling
    batch_size = 16  # Moderate batch size for 16GB GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)  # Set num_workers=0 for debugging
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)   # Set num_workers=0 for debugging
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")
    
    # Model
    model = AdvancedAudioClassifier(num_classes=len(class_names))
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies, training_metrics = train_model(model, train_loader, test_loader, 
                                              num_epochs=100, learning_rate=0.001)
    
    # Load best model
    model.load_state_dict(torch.load('best_model_2.pth'))
    
    # Final evaluation
    print("\nFinal evaluation on test set:")
    test_acc, predictions, targets = evaluate_model(model, test_loader)
    
    # Plot training history with overfitting indicators
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train', linewidth=2)
    plt.plot(val_accuracies, label='Validation', linewidth=2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.savefig('training_history_model_2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print("Model saved as 'best_model_2.pth'")
    
    # Save training metrics to a text file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'model2_metrics_{timestamp}.txt', 'w') as f:
        f.write("=== Advanced Audio Classifier (Model2) Training Metrics ===\n")
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
        f.write(f"Final Training Accuracy: {training_metrics['final_train_accuracy']:.2f}%\n")
        f.write(f"Final Validation Accuracy: {training_metrics['final_val_accuracy']:.2f}%\n")
        f.write(f"Best Validation Accuracy: {training_metrics['best_val_accuracy']:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n\n")
        
        f.write("=== Model Architecture ===\n")
        f.write(f"Model Type: AdvancedAudioClassifier\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        
        f.write("=== Classification Report ===\n")
        f.write(classification_report(targets, predictions, target_names=class_names))
    
    print(f"Training metrics saved to 'arabic_r_disorder_classifier_2_metrics_{timestamp}.txt'")

if __name__ == "__main__":
    main()