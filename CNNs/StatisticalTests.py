import os
import time
import psutil
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Keep existing functions the same
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

def build_alexnet(input_shape=(224, 224, 3), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', 
                     input_shape=input_shape, kernel_regularizer=l2(0.0005)),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(256, (5, 5), activation='relu', padding="same", 
                     kernel_regularizer=l2(0.0005)),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(384, (3, 3), activation='relu', padding="same", 
                     kernel_regularizer=l2(0.0005)),
        layers.Conv2D(384, (3, 3), activation='relu', padding="same", 
                     kernel_regularizer=l2(0.0005)),
        layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
                     kernel_regularizer=l2(0.0005)),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_densenet(input_shape=(224, 224, 3), num_classes=4):
    base_model = applications.DenseNet201(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = True
    for layer in base_model.layers[:350]:  # Freeze first 350 layers
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

def build_efficientnet(input_shape=(224, 224, 3), num_classes=4):
    base_model = applications.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # Freeze first 100 layers
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

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

def compile_and_train(model, train_gen, test_gen, epochs=10):
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_gen, validation_data=test_gen, epochs=epochs, verbose=0)
    return history

def evaluate_single_run(model, test_gen, class_names): # for a single model
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes
    
    accuracy = np.mean(y_pred == y_true)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    # Macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    # AUC calculation
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    auc_scores = []
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        auc_scores.append(auc(fpr, tpr))
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

def multiple_runs_experiment(model_name, n_runs=5, epochs=30): #  multiple training runs for a specific model and get statistics    
    # Data setup
    data_dir = "Brain-Tumor-Classification-DataSet"
    img_size = (224, 224)
    batch_size = 32
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")
    
    model_builders = {
        'AlexNet': build_alexnet,
        'DenseNet': build_densenet,
        'EfficientNet': build_efficientnet,
        'TumorDetNet': build_tumordetnet
    }
    
    if model_name not in model_builders:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_builders.keys())}")
    
    all_results = []
    
    print(f"Starting {n_runs} training runs for {model_name}...")
    
    for run in range(n_runs):
        print(f"\n{model_name} - Run {run + 1}/{n_runs}")
        
        # Set different seed for each run
        set_seeds(42 + run)
        
        # Prepare fresh data generators
        train_gen, test_gen = prepare_data(train_dir, test_dir, img_size, batch_size)
        class_names = list(train_gen.class_indices.keys())
        
        # Build and train model
        model = model_builders[model_name](input_shape=(224, 224, 3), num_classes=len(class_names))
        history = compile_and_train(model, train_gen, test_gen, epochs=epochs)
        
        # Evaluate
        results = evaluate_single_run(model, test_gen, class_names)
        results['run'] = run + 1
        results['model'] = model_name
        all_results.append(results)
        
        print(f"Run {run + 1} - Accuracy: {results['accuracy']:.4f}, F1: {results['f1_macro']:.4f}")
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    return all_results, class_names

def compare_multiple_models(model_names, n_runs=5, epochs=30):
    all_model_results = {}
    
    for model_name in model_names:

        print(f"TRAINING {model_name.upper()}")

        
        results, class_names = multiple_runs_experiment(model_name, n_runs, epochs)
        all_model_results[model_name] = results
    
    return all_model_results, class_names

def calculate_statistics(all_results, class_names): #Calculate mean, std, and confidence intervals from multiple runs
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
                values = [result[key][i] for result in all_results]
                per_class_stats[class_name][metric_type] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
    
    return stats_summary, per_class_stats

def perform_statistical_tests(all_results, model_name, baseline_accuracy=None): # statistical significance tests
    accuracies = [result['accuracy'] for result in all_results]
    
    print(f"\nStatistical Analysis for {model_name}:")
    
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

def create_results_table(stats_summary, per_class_stats, class_names):    
    # Overall metrics table
    overall_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'AUC-ROC (Macro)'],
        'Mean': [stats_summary[k]['mean'] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']],
        'Std Dev': [stats_summary[k]['std'] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']],
        'Min': [stats_summary[k]['min'] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']],
        'Max': [stats_summary[k]['max'] for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']]
    })
    
    # Per-class metrics table
    per_class_rows = []
    for class_name in class_names:
        for metric in ['precision', 'recall', 'f1', 'auc']:
            if metric in per_class_stats[class_name]:
                per_class_rows.append({
                    'Class': class_name,
                    'Metric': metric.capitalize(),
                    'Mean': per_class_stats[class_name][metric]['mean'],
                    'Std Dev': per_class_stats[class_name][metric]['std']
                })
    
    per_class_df = pd.DataFrame(per_class_rows)
    
    return overall_df, per_class_df

# Main execution for statistical analysis
if __name__ == "__main__":
    # Define models to test
    models_to_test = ['AlexNet', 'DenseNet', 'EfficientNet', 'TumorDetNet']
    n_runs = 5  # We can increase this for better statistics
    epochs = 30
    
    # Compare multiple models
    all_model_results, class_names = compare_multiple_models(models_to_test, n_runs, epochs)
    
    # Generate comprehensive analysis
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    
    # Individual model analysis
    for model_name, results in all_model_results.items():
        print(f"\n{model_name.upper()} ANALYSIS:")
        
        # Calculate statistics
        stats_summary, per_class_stats = calculate_statistics(results, class_names)
        
        # Create results tables
        overall_df, per_class_df = create_results_table(stats_summary, per_class_stats, class_names)
        
        print(f"\nOverall Performance Metrics (across {n_runs} runs):")
        print(overall_df.round(4))
        
        # Statistical tests
        perform_statistical_tests(results, model_name)

        
    print("ANALYSIS COMPLETE")
