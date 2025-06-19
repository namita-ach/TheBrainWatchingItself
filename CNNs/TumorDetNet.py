import os
import time
import psutil
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Data Preparation
data_dir = "Brain-Tumor-Classification-DataSet"
img_size = (224, 224)
batch_size = 32

def prepare_data(train_dir, test_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, 
        class_mode="categorical", shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, 
        class_mode="categorical", shuffle=False
    )
    return train_gen, test_gen

train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")
train_gen, test_gen = prepare_data(train_dir, test_dir, img_size, batch_size)
num_classes = len(train_gen.class_indices)
class_names = list(train_gen.class_indices.keys())

# Build TumorDetNet Model
def build_tumordetnet(input_shape=(224, 224, 3), num_classes=4):
    model = models.Sequential([
        # Convolutional block 1
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv block 2
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv block 3
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv block 4
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Fully Connected Layer
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def calculate_sensitivity_specificity(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = []
    specificity = []
    
    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivity.append(sens)
        specificity.append(spec)
    
    return sensitivity, specificity

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    return cm

def plot_roc_curves(y_true_bin, y_pred_proba, class_names, model_name):
    plt.figure(figsize=(12, 8))
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc

def comprehensive_evaluation(model, test_gen, class_names, model_name):
    print(f"\n{model_name} Comprehensive Evaluation")
    
    # Get predictions
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes
    
    # Binarize labels for ROC calculation
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Basic accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1-score per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    # Macro and Micro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
    )
    
    # Sensitivity and Specificity
    sensitivity, specificity = calculate_sensitivity_specificity(y_true, y_pred, class_names)
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    
    # Plot ROC curves and get AUC values
    roc_auc = plot_roc_curves(y_true_bin, y_pred_proba, class_names, model_name)
    
    # Create detailed metrics DataFrame
    detailed_metrics = []
    for i, class_name in enumerate(class_names):
        detailed_metrics.append({
            'Class': class_name,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-Score': f1[i],
            'Sensitivity': sensitivity[i],
            'Specificity': specificity[i],
            'AUC-ROC': roc_auc[i],
            'Support': support[i]
        })
    
    # Add macro/micro averages
    detailed_metrics.append({
        'Class': 'Macro Average',
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1-Score': f1_macro,
        'Sensitivity': np.mean(sensitivity),
        'Specificity': np.mean(specificity),
        'AUC-ROC': np.mean(list(roc_auc.values())),
        'Support': np.sum(support)
    })
    
    detailed_metrics.append({
        'Class': 'Micro Average',
        'Precision': precision_micro,
        'Recall': recall_micro,
        'F1-Score': f1_micro,
        'Sensitivity': accuracy,
        'Specificity': np.nan,
        'AUC-ROC': np.nan,
        'Support': np.sum(support)
    })
    
    detailed_df = pd.DataFrame(detailed_metrics)
    
    # Display results
    print("\nPer-Class Metrics:")
    print(detailed_df.round(4))
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return detailed_df, cm, roc_auc

# Measure inference time, CPU & RAM usage
def measure_performance(model, test_gen):
    process = psutil.Process()
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_usage = process.memory_info().rss / (1024 * 1024)
    
    test_sample, _ = next(test_gen)
    test_sample = test_sample[:1]
    
    times = []
    for _ in range(100):
        start_time = time.time()
        _ = model.predict(test_sample, verbose=0)
        times.append(time.time() - start_time)
    
    avg_inference_time = sum(times) / len(times)
    print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
    print(f"CPU Usage: {cpu_percent}%")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    
    return avg_inference_time, cpu_percent, ram_usage

def compile_and_train(model, train_gen, test_gen, epochs=10):
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_gen, validation_data=test_gen, epochs=epochs)
    return history

# Main execution
print("Training TumorDetNet...")
tumordetnet_model = build_tumordetnet(input_shape=(224, 224, 3), num_classes=4)
history = compile_and_train(tumordetnet_model, train_gen, test_gen, epochs=100)

# Save the model
model_path = "tumordetnet.h5"
tumordetnet_model.save(model_path)
print(f"TumorDetNet model saved as '{model_path}'")

# Comprehensive evaluation
detailed_metrics, confusion_mat, auc_scores = comprehensive_evaluation(
    tumordetnet_model, test_gen, class_names, "TumorDetNet"
)

# Measure performance
avg_time, cpu_usage, ram_usage = measure_performance(tumordetnet_model, test_gen)
params = tumordetnet_model.count_params()

# Combine all metrics
performance_metrics = {
    "Model": ["TumorDetNet"],
    "Inference Time (s)": [avg_time],
    "CPU Usage (%)": [cpu_usage],
    "RAM Usage (MB)": [ram_usage],
    "Total Parameters": [params],
    "Overall Accuracy": [detailed_metrics[detailed_metrics['Class'] == 'Micro Average']['Recall'].values[0]],
    "Macro Avg Precision": [detailed_metrics[detailed_metrics['Class'] == 'Macro Average']['Precision'].values[0]],
    "Macro Avg Recall": [detailed_metrics[detailed_metrics['Class'] == 'Macro Average']['Recall'].values[0]],
    "Macro Avg F1-Score": [detailed_metrics[detailed_metrics['Class'] == 'Macro Average']['F1-Score'].values[0]],
    "Mean AUC-ROC": [detailed_metrics[detailed_metrics['Class'] == 'Macro Average']['AUC-ROC'].values[0]]
}

# Save all metrics
performance_df = pd.DataFrame(performance_metrics)
detailed_metrics.to_csv("tumordetnet_detailed_metrics.csv", index=False)
performance_df.to_csv("tumordetnet_performance_summary.csv", index=False)

print(f"Detailed metrics saved to 'tumordetnet_detailed_metrics.csv'")
print(f"Performance summary saved to 'tumordetnet_performance_summary.csv'")

# Save confusion matrix as CSV
cm_df = pd.DataFrame(confusion_mat, columns=class_names, index=class_names)
cm_df.to_csv("tumordetnet_confusion_matrix.csv")
print(f"Confusion matrix saved to 'tumordetnet_confusion_matrix.csv'")
