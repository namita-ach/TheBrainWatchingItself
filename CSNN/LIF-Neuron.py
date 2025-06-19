import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()

# Define a custom dataset class
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))  # Ensure class order consistency
        self.image_paths = []
        self.labels = []
        self.data_shape = None

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith('.npy'):
                        self.image_paths.append(os.path.join(class_path, file))
                        self.labels.append(class_idx)

        # Detect data shape from first file
        if len(self.image_paths) > 0:
            try:
                sample_data = np.load(self.image_paths[0])
                self.data_shape = sample_data.shape
                print(f"Detected data shape: {self.data_shape}")
            except Exception as e:
                print(f"Error loading sample data: {e}")
                raise

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            spike_train = np.load(self.image_paths[idx])  # Load encoded spike train

            # Reshape if necessary
            if len(spike_train.shape) == 1:
                spike_train = spike_train.reshape(self.target_shape)

            # Ensure correct dimension order (time, H, W)
            if len(spike_train.shape) == 3:
                # If shape is (H, W, T), permute to (T, H, W)
                if spike_train.shape[2] < spike_train.shape[0] and spike_train.shape[2] < spike_train.shape[1]:
                    spike_train = np.transpose(spike_train, (2, 0, 1))
                # If shape is (T, H, W), keep as is
                elif spike_train.shape[0] < spike_train.shape[1] and spike_train.shape[0] < spike_train.shape[2]:
                    pass  # Already in correct format
                else:
                    # Default: last dimension is time
                    spike_train = np.transpose(spike_train, (2, 0, 1))

            spike_train = torch.tensor(spike_train, dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return spike_train, label

        except Exception as e:
            print(f"Error loading file {self.image_paths[idx]}: {e}")
            
# Define the SNN model with LIF neurons
class CSNN(nn.Module):
    def __init__(self, num_classes=4, input_channels=100, spatial_size=128, dropout_rate=0.3):
        super(CSNN, self).__init__()
        self.input_channels = input_channels
        self.spatial_size = spatial_size

        # First convolution block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9)
        self.pool1 = nn.AvgPool2d(2)  # Reduce spatial dimensions

        # Second convolution block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=0.9)
        self.pool2 = nn.AvgPool2d(2)  # Further reduce dimensions

        # Calculate flattened size after convolutions and pooling
        # After two 2x2 pooling layers, spatial dimensions reduced by factor of 4
        flattened_size = 64 * (spatial_size // 4) * (spatial_size // 4)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(flattened_size, 128)
        self.lif_fc1 = snn.Leaky(beta=0.9)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

        print(f"Model initialized with:")
        print(f"  Input channels: {input_channels}")
        print(f"  Spatial size: {spatial_size}")
        print(f"  Flattened size: {flattened_size}")

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_fc1 = self.lif_fc1.init_leaky()

        # First convolutional block
        spk1, mem1 = self.lif1(self.conv1(x), mem1)
        spk1 = self.pool1(spk1)  # Apply pooling

        # Second convolutional block
        spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
        spk2 = self.pool2(spk2)  # Apply pooling

        # Fully connected layers
        spk2 = spk2.flatten(start_dim=1)
        spk_fc1, mem_fc1 = self.lif_fc1(self.fc1(spk2), mem_fc1)
        spk_fc1 = self.dropout(spk_fc1)  # Apply dropout
        out = self.fc2(spk_fc1)

        return out

# Enhanced evaluation function with comprehensive metrics
def comprehensive_evaluation(model, test_loader, device, class_names=None):
    #Comprehensive evaluation including precision, recall, F1-score, sensitivity, specificity, confusion matrix, and AUC-ROC
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for spike_trains, labels in test_loader:
            spike_trains, labels = spike_trains.to(device), labels.to(device)
            outputs = model(spike_trains)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)

    # Get unique classes
    n_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(n_classes)]

    print("="*80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*80)

    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Macro and micro averages
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    print("PER-CLASS METRICS:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall (Sensitivity): {recall_per_class[i]:.4f}")
        print(f"  F1-Score: {f1_per_class[i]:.4f}")

    print("\nMACRO AVERAGES:")
    print("-" * 30)
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall: {recall_macro:.4f}")
    print(f"F1-Score: {f1_macro:.4f}")

    print("\nMICRO AVERAGES:")
    print("-" * 30)
    print(f"Precision: {precision_micro:.4f}")
    print(f"Recall: {recall_micro:.4f}")
    print(f"F1-Score: {f1_micro:.4f}")

    # Sensitivity and Specificity for each class (one-vs-rest)
    print("\nSENSITIVITY AND SPECIFICITY (One-vs-Rest):")
    print("-" * 50)

    sensitivity_specificity_data = []
    for i, class_name in enumerate(class_names):
        # Binary classification for class i vs rest
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        # True positives, false positives, true negatives, false negatives
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        # Sensitivity (Recall) and Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"{class_name}:")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")

        sensitivity_specificity_data.append({
            'Class': class_name,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })

    # Confusion Matrix
    print("\nCONFUSION MATRIX:")
    print("-" * 20)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # AUC-ROC Scores
    print("\nAUC-ROC SCORES:")
    print("-" * 15)

    # Binarize labels for multiclass ROC
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    # Per-class AUC-ROC
    auc_scores = []
    for i, class_name in enumerate(class_names):
        if n_classes == 2 and i == 1:  # For binary classification, avoid duplicate
            break

        if n_classes > 2:
            auc_score = roc_auc_score(y_true_binarized[:, i], y_proba[:, i])
        else:
            auc_score = roc_auc_score(y_true, y_proba[:, 1])

        auc_scores.append(auc_score)
        print(f"{class_name} AUC-ROC: {auc_score:.4f}")

    # Macro and Micro AUC-ROC for multiclass
    if n_classes > 2:
        macro_auc = roc_auc_score(y_true_binarized, y_proba, average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true_binarized, y_proba, average='micro', multi_class='ovr')
        print(f"\nMacro AUC-ROC: {macro_auc:.4f}")
        print(f"Micro AUC-ROC: {micro_auc:.4f}")
    else:
        macro_auc = auc_scores[0]
        micro_auc = auc_scores[0]

    # Plot ROC curves
    plt.figure(figsize=(12, 8))

    if n_classes == 2:
        # Binary classification ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_names[1]} (AUC = {auc_scores[0]:.4f})')
    else:
        # Multiclass ROC curves
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc_scores[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Prepare comprehensive metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'auc_per_class': auc_scores,
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'confusion_matrix': cm.tolist(),
        'sensitivity_specificity': sensitivity_specificity_data
    }

    return metrics

# Measure inference time, CPU & RAM usage
def measure_performance(model, test_loader, device):
    process = psutil.Process()
    cpu_percent = psutil.cpu_percent(interval=1)  # Measure CPU usage over 1 second
    ram_usage = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    # Get a single sample for inference time measurement
    test_sample, _ = next(iter(test_loader))
    test_sample = test_sample[:1].to(device)  # Take a single image

    # Measure inference time
    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            start_time = time.time()
            _ = model(test_sample)
            times.append(time.time() - start_time)

    avg_inference_time = sum(times) / len(times)
    print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
    print(f"CPU Usage: {cpu_percent}%")
    print(f"RAM Usage: {ram_usage:.2f} MB")

    return avg_inference_time, cpu_percent, ram_usage

# Count parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Load dataset
train_dataset = BrainTumorDataset("Encoding/Temporal-Encoding/Training")
test_dataset = BrainTumorDataset("Encoding/Temporal-Encoding/Testing")
# train_dataset = BrainTumorDataset("Encoding/Rate-Encoding/Training")
# test_dataset = BrainTumorDataset("Encoding/Rate-Encoding/Testing")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Get class names from dataset
class_names = train_dataset.classes

# Determine model parameters from data
if hasattr(train_dataset, 'target_shape'):
    if len(train_dataset.target_shape) == 3:
        sample_data, _ = train_dataset[0]
        time_steps, height, width = sample_data.shape
        input_channels = time_steps
        spatial_size = height
    else:
        input_channels = 100
        spatial_size = 128
else:
    input_channels = 100
    spatial_size = 128
print(f"Model will use: {input_channels} input channels, {spatial_size}x{spatial_size} spatial size")

# Define training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Classes: {class_names}")

model = CSNN(num_classes=len(class_names),
             input_channels=input_channels,
             spatial_size=spatial_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation tracking
num_epochs = 30
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for spike_trains, labels in train_loader:
        spike_trains, labels = spike_trains.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(spike_trains)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = 100 * correct_train / total_train
    train_loss /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for spike_trains, labels in test_loader:
            spike_trains, labels = spike_trains.to(device), labels.to(device)
            outputs = model(spike_trains)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = 100 * correct_val / total_val
    val_loss /= len(test_loader)

    # Store metrics for plotting
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

print("Training complete!")

# Save model
save_path = "rate-csnn_lif_model_rate.pth"
torch.save(model.state_dict(), save_path)
print(f"SNN model saved as '{save_path}'")

# Comprehensive evaluation
print("STARTING COMPREHENSIVE EVALUATION")

evaluation_metrics = comprehensive_evaluation(model, test_loader, device, class_names)

# Measure performance
print("PERFORMANCE METRICS")
avg_time, cpu_usage, ram_usage = measure_performance(model, test_loader, device)
params = count_parameters(model)
print(f"Total Parameters: {params:,}")

# Prepare comprehensive metrics for CSV
detailed_metrics = {
    "Metric": [],
    "Value": []
}

# Add basic metrics
detailed_metrics["Metric"].extend([
    "Overall Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
    "Precision (Micro)", "Recall (Micro)", "F1-Score (Micro)",
    "AUC-ROC (Macro)", "AUC-ROC (Micro)",
    "Inference Time (s)", "CPU Usage (%)", "RAM Usage (MB)", "Total Parameters"
])

detailed_metrics["Value"].extend([
    f"{evaluation_metrics['accuracy']:.4f}",
    f"{evaluation_metrics['precision_macro']:.4f}",
    f"{evaluation_metrics['recall_macro']:.4f}",
    f"{evaluation_metrics['f1_macro']:.4f}",
    f"{evaluation_metrics['precision_micro']:.4f}",
    f"{evaluation_metrics['recall_micro']:.4f}",
    f"{evaluation_metrics['f1_micro']:.4f}",
    f"{evaluation_metrics['macro_auc']:.4f}",
    f"{evaluation_metrics['micro_auc']:.4f}",
    f"{avg_time:.6f}",
    f"{cpu_usage:.2f}",
    f"{ram_usage:.2f}",
    f"{params}"
])

# Add per-class metrics
for i, class_name in enumerate(class_names):
    detailed_metrics["Metric"].extend([
        f"{class_name} - Precision",
        f"{class_name} - Recall",
        f"{class_name} - F1-Score",
        f"{class_name} - AUC-ROC"
    ])

    detailed_metrics["Value"].extend([
        f"{evaluation_metrics['precision_per_class'][i]:.4f}",
        f"{evaluation_metrics['recall_per_class'][i]:.4f}",
        f"{evaluation_metrics['f1_per_class'][i]:.4f}",
        f"{evaluation_metrics['auc_per_class'][i]:.4f}" if i < len(evaluation_metrics['auc_per_class']) else "N/A"
    ])

# Add sensitivity and specificity
for item in evaluation_metrics['sensitivity_specificity']:
    detailed_metrics["Metric"].extend([
        f"{item['Class']} - Sensitivity",
        f"{item['Class']} - Specificity"
    ])

    detailed_metrics["Value"].extend([
        f"{item['Sensitivity']:.4f}",
        f"{item['Specificity']:.4f}"
    ])

# print all the metrics
print("\nCOMPREHENSIVE METRICS:")
for metric, value in zip(detailed_metrics["Metric"], detailed_metrics["Value"]):
    print(f"{metric}: {value}")
