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

torch.cuda.empty_cache()

# Define a custom dataset class
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))  # Ensure class order consistency
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.npy'):
                    self.image_paths.append(os.path.join(class_path, file))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        spike_train = np.load(self.image_paths[idx])  # Load encoded spike train
        spike_train = torch.tensor(spike_train, dtype=torch.float32).permute(2, 0, 1)  # (time, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spike_train, label

# Define the SNN model with LIF neurons
class CSNN(nn.Module):
    def __init__(self, num_classes=4, time_window=100, dropout_rate=0.3):
        super(CSNN, self).__init__()
        self.time_window = time_window
        
        # First convolution block
        self.conv1 = nn.Conv2d(100, 32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9)
        self.pool1 = nn.AvgPool2d(2)  # Reduce spatial dimensions
        
        # Second convolution block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=0.9)
        self.pool2 = nn.AvgPool2d(2)  # Further reduce dimensions
        
        # Fully connected layers with dropout
        # After two 2x2 pooling layers, dimensions reduced by factor of 4
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # 128x128 -> 64x64 -> 32x32
        self.lif_fc1 = snn.Leaky(beta=0.9)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
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
train_dataset = BrainTumorDataset("Encoding/Rate-Encoding/Training")
test_dataset = BrainTumorDataset("Encoding/Rate-Encoding/Testing")

# train_dataset = BrainTumorDataset("Encoding/Temporal-Encoding/Training")
# test_dataset = BrainTumorDataset("Encoding/Temporal-Encoding/Testing")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation tracking
num_epochs = 30
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
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

print("Training complete!")

# Save model
save_path = "/home/pes1ug22am100/Documents/Research and Experimentation/neuralNeurosis/Neural-Models/csnn_lif_model_rate.pth"
torch.save(model.state_dict(), save_path)
print(f"SNN model saved as '{save_path}'")

# Measure performance
avg_time, cpu_usage, ram_usage = measure_performance(model, test_loader, device)
params = count_parameters(model)

# Save metrics to CSV
metrics = {
    "Inference Time (s)": [avg_time],
    "CPU Usage (%)": [cpu_usage],
    "RAM Usage (MB)": [ram_usage],
    "Total Parameters": [params]
}

metrics_df = pd.DataFrame(metrics)
csv_path = "csnn_metrics.csv"
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics saved to '{csv_path}'")