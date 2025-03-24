import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd

# Data Preparation
data_dir = "Brain-Tumor-Classification-DataSet"
img_size = (224, 224)
batch_size = 32

def prepare_data(train_dir, test_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=True
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    return train_gen, test_gen

train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")
train_gen, test_gen = prepare_data(train_dir, test_dir, img_size, batch_size)
num_classes = len(train_gen.class_indices)

# Build efficientnet Model
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

        # Output dude
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Measure inference time, CPU & RAM usage
def measure_performance(model, test_gen):
    process = psutil.Process()
    cpu_percent = psutil.cpu_percent(interval=1)  # Measure CPU usage over 1 second
    ram_usage = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    test_sample, _ = next(test_gen)
    test_sample = test_sample[:1]  # Take a single image

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

print("Training tumordetnet...")
tumordetnet_model = build_tumordetnet(input_shape=(224, 224, 3), num_classes=4)
history = compile_and_train(tumordetnet_model, train_gen, test_gen, epochs=100)
# Save the model
model_path = "tumordetnet.h5"

tumordetnet_model.save(model_path)
print(f"tumordetnet model saved as '{model_path}'")

# Measure performance
avg_time, cpu_usage, ram_usage = measure_performance(tumordetnet_model, test_gen)
params = tumordetnet_model.count_params()

# Save metrics to CSV
metrics = {
    "Inference Time (s)": [avg_time],
    "CPU Usage (%)": [cpu_usage],
    "RAM Usage (MB)": [ram_usage],
    "Total Parameters": [params]
}

metrics_df = pd.DataFrame(metrics)
csv_path = "tumordetnet_metrics.csv"
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics saved to '{csv_path}'")