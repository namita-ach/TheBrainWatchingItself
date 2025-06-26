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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()

# Seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, target_shape=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))  # Ensure class order consistency
        self.image_paths = []
        self.labels = []
        self.data_shape = None
        self.target_shape = target_shape

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

            if self.target_shape and len(spike_train.shape) == 1:
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
                    spike_train = np.transpose(spike_train, (2, 0, 1))

            spike_train = torch.tensor(spike_train, dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return spike_train, label

        except Exception as e:
            print(f"Error loading file {self.image_paths[idx]}: {e}")
            return None, None

# SNN model with LIF neurons
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

def prepare_data(encoding_type, batch_size=8):
    if encoding_type.lower() == 'temporal':
        train_dir = "Encoding/Temporal-Encoding/Training"
        test_dir = "Encoding/Temporal-Encoding/Testing"
    elif encoding_type.lower() == 'rate':
        train_dir = "Encoding/Rate-Encoding/Training"
        test_dir = "Encoding/Rate-Encoding/Testing"
    else:
        raise ValueError("Encoding type must be 'temporal' or 'rate'")
    
    train_dataset = BrainTumorDataset(train_dir)
    test_dataset = BrainTumorDataset(test_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.classes

def build_csnn(num_classes=4, input_channels=100, spatial_size=128, dropout_rate=0.3):
    return CSNN(num_classes=num_classes, 
                input_channels=input_channels, 
                spatial_size=spatial_size, 
                dropout_rate=dropout_rate)

def compile_and_train_csnn(model, train_loader, test_loader, device, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for spike_trains, labels in train_loader:
            spike_trains, labels = spike_trains.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spike_trains)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation accuracy
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for spike_trains, labels in test_loader:
                spike_trains, labels = spike_trains.to(device), labels.to(device)
                outputs = model(spike_trains)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return {'train_loss': train_losses, 'val_accuracy': val_accuracies}

def evaluate_single_run_csnn(model, test_loader, device, class_names): #Evaluate single CSNN run
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
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    
    # Macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    # AUC calculation
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    auc_scores = []
    
    if len(class_names) == 2:
        # Binary classification
        auc_score = roc_auc_score(y_true, y_proba[:, 1])
        auc_scores = [auc_score]
        mean_auc = auc_score
    else:
        # Multiclass classification
        for i in range(len(class_names)):
            try:
                auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                auc_scores.append(auc_score)
            except ValueError:
                auc_scores.append(0.0)  # Handle cases where class doesn't exist in test set
        mean_auc = np.mean(auc_scores)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'auc_macro': mean_auc,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'auc_per_class': auc_scores
    }

def multiple_runs_experiment_csnn(encoding_type, n_runs=5, epochs=30, batch_size=8): #Run multiple training experiments for CSNN with specified encoding
    print(f"Starting {n_runs} training runs for CSNN with {encoding_type} encoding...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    all_results = []
    
    for run in range(n_runs):
        print(f"\nCSNN {encoding_type} - Run {run + 1}/{n_runs}")
        
        # Set different seed for each run
        set_seeds(42 + run)
        
        # Prepare fresh data
        train_loader, test_loader, class_names = prepare_data(encoding_type, batch_size)
        
        # Determine model parameters from data
        sample_data, _ = next(iter(train_loader))
        if len(sample_data.shape) == 5:  # Batch, Time, Channels, H, W
            time_steps, height, width = sample_data.shape[2], sample_data.shape[3], sample_data.shape[4]
            input_channels = sample_data.shape[1]
            spatial_size = height
        elif len(sample_data.shape) == 4:  # Batch, Channels, H, W
            input_channels = sample_data.shape[1]
            spatial_size = sample_data.shape[2]
        else:
            input_channels = 100
            spatial_size = 128
        
        print(f"Model parameters: {input_channels} channels, {spatial_size}x{spatial_size} spatial")
        
        # Build and train model
        model = build_csnn(num_classes=len(class_names), 
                          input_channels=input_channels, 
                          spatial_size=spatial_size).to(device)
        
        history = compile_and_train_csnn(model, train_loader, test_loader, device, epochs)
        
        # Evaluate
        results = evaluate_single_run_csnn(model, test_loader, device, class_names)
        results['run'] = run + 1
        results['encoding'] = encoding_type
        results['input_channels'] = input_channels
        results['spatial_size'] = spatial_size
        all_results.append(results)
        
        print(f"Run {run + 1} - Accuracy: {results['accuracy']:.4f}, F1: {results['f1_macro']:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return all_results, class_names

def compare_multiple_encodings(encoding_types=['temporal', 'rate'], n_runs=5, epochs=30):
    all_encoding_results = {}
    
    for encoding_type in encoding_types:
            print(f"TRAINING CSNN WITH {encoding_type.upper()} ENCODING")
        
    results, class_names = multiple_runs_experiment_csnn(encoding_type, n_runs, epochs)
    all_encoding_results[encoding_type] = results
    
    return all_encoding_results, class_names

def calculate_statistics_csnn(all_results, class_names):
    # Extract metrics across all runs
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']
    stats_summary = {}
    
    for metric in metrics:
        values = [result[metric] for result in all_results]
        stats_summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    # Per-class statistics
    per_class_stats = {}
    for i, class_name in enumerate(class_names):
        per_class_stats[class_name] = {}
        
        for metric_type in ['precision', 'recall', 'f1', 'auc']:
            key = f'{metric_type}_per_class'
            if key in all_results[0]:
                values = [result[key][i] if i < len(result[key]) else 0.0 for result in all_results]
                per_class_stats[class_name][metric_type] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
    
    return stats_summary, per_class_stats

def perform_statistical_tests_csnn(all_results, encoding_type, baseline_accuracy=None): # Perform statistical significance tests for CSNN
    accuracies = [result['accuracy'] for result in all_results]
    
    print(f"\nStatistical Analysis for CSNN with {encoding_type} encoding:")
    
    # Basic statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    n = len(accuracies)
    
    # Calculate 95% confidence interval
    ci_95 = stats.t.interval(0.95, n-1, loc=mean_acc, scale=std_acc/np.sqrt(n))
    
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Standard Deviation: {std_acc:.4f}")
    print(f"95% Confidence Interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    
    if baseline_accuracy is not None:
        t_stat, p_value = stats.ttest_1samp(accuracies, baseline_accuracy)
        print(f"\nOne-sample t-test against baseline accuracy {baseline_accuracy:.4f}:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("*** Significant improvement over baseline (p < 0.05) ***")
        else:
            print("No significant improvement over baseline (p >= 0.05)")
        return t_stat, p_value, ci_95
    
    return None, None, ci_95

def create_results_table_csnn(stats_summary, per_class_stats, class_names, encoding_type):
    # Overall metrics table
    overall_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'AUC-ROC (Macro)'],
        'Mean': [stats_summary[k]['mean'] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']],
        'Std Dev': [stats_summary[k]['std'] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']],
        'Min': [stats_summary[k]['min'] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']],
        'Max': [stats_summary[k]['max'] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']]
    })
    overall_df['Encoding'] = encoding_type
    
    # Per-class metrics table
    per_class_rows = []
    for class_name in class_names:
        for metric in ['precision', 'recall', 'f1', 'auc']:
            if metric in per_class_stats[class_name]:
                per_class_rows.append({
                    'Class': class_name,
                    'Metric': metric.capitalize(),
                    'Mean': per_class_stats[class_name][metric]['mean'],
                    'Std Dev': per_class_stats[class_name][metric]['std'],
                    'Encoding': encoding_type
                })
    
    per_class_df = pd.DataFrame(per_class_rows)
    
    return overall_df, per_class_df

def create_comparison_table_csnn(all_encoding_results):
    comparison_rows = []
    
    for encoding_type, results in all_encoding_results.items():
        # Calculate statistics for this encoding
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']
        
        for metric in metrics:
            values = [result[metric] for result in results]
            comparison_rows.append({
                'Encoding': encoding_type.capitalize(),
                'Metric': metric.replace('_', ' ').title(),
                'Mean': np.mean(values),
                'Std Dev': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values)
            })
    
    return pd.DataFrame(comparison_rows)

def perform_encoding_comparison_tests(all_encoding_results):
    encoding_types = list(all_encoding_results.keys())
    
    if len(encoding_types) < 2:
        print("Need at least 2 encoding types for comparison.")
        return
    
    print("STATISTICAL COMPARISON BETWEEN ENCODINGS")
    
    # Compare each pair of encodings
    for i in range(len(encoding_types)):
        for j in range(i+1, len(encoding_types)):
            encoding1, encoding2 = encoding_types[i], encoding_types[j]
            
            accuracies1 = [result['accuracy'] for result in all_encoding_results[encoding1]]
            accuracies2 = [result['accuracy'] for result in all_encoding_results[encoding2]]
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(accuracies1, accuracies2)
            
            print(f"\n{encoding1.capitalize()} vs {encoding2.capitalize()}:")
            print(f"Mean accuracy - {encoding1}: {np.mean(accuracies1):.4f}")
            print(f"Mean accuracy - {encoding2}: {np.mean(accuracies2):.4f}")
            print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                better_encoding = encoding1 if np.mean(accuracies1) > np.mean(accuracies2) else encoding2
                print(f"*** {better_encoding.capitalize()} is significantly better (p < 0.05) ***")
            else:
                print("No significant difference (p >= 0.05)")

if __name__ == "__main__":
    # Define encodings to test
    encodings_to_test = ['temporal', 'rate']
    n_runs = 5  # Can increase for better statistics
    epochs = 30
    
    # Compare multiple encodings
    all_encoding_results, class_names = compare_multiple_encodings(encodings_to_test, n_runs, epochs)
    
    # Generate comprehensive analysis
    print("COMPREHENSIVE CSNN STATISTICAL ANALYSIS")
    
    # Individual encoding analysis
    for encoding_type, results in all_encoding_results.items():
        print(f"\n{encoding_type.upper()} ENCODING ANALYSIS:")

        
        # Calculate statistics
        stats_summary, per_class_stats = calculate_statistics_csnn(results, class_names)
        
        # Create results tables
        overall_df, per_class_df = create_results_table_csnn(stats_summary, per_class_stats, class_names, encoding_type)
        
        print(f"\nOverall Performance Metrics (across {n_runs} runs):")
        print(overall_df.round(4))
        
        # Statistical tests
        perform_statistical_tests_csnn(results, encoding_type)
    
    # Encoding comparison
    print("ENCODING COMPARISON TABLE")
    
    comparison_table = create_comparison_table_csnn(all_encoding_results)
    print("\nComparison of All Encodings:")
    print(comparison_table)

    
    print("CSNN ANALYSIS COMPLETE")
