import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def encode_rate_coding(image_path, time_window=100, max_spikes=20):
    print(f"Encoding image: {image_path}")
    img = image.load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    
    spike_train = np.zeros((img_array.shape[0], img_array.shape[1], time_window))
    
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            num_spikes = int(img_array[i, j] * max_spikes)
            spike_train[i, j, :num_spikes] = 1
    
    return img_array, spike_train

def visualize_encoding(img_array, spike_train):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original Image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original MRI Image')
    
    # Encoded Spike Train (Summed Over Time)
    spike_visual = np.sum(spike_train, axis=-1)
    axes[1].imshow(spike_visual, cmap='hot')
    axes[1].set_title('Rate Coding Encoded Image')
    
    plt.colorbar(axes[1].imshow(spike_visual, cmap='hot'))
    plt.show()

def process_directory(input_dir, output_dir, time_window=100, max_spikes=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_count = 0

    for split in ['Training', 'Testing']:
        split_path = os.path.join(input_dir, split)
        output_split_path = os.path.join(output_dir, split)

        if not os.path.exists(split_path):
            print(f"Skipping {split_path}, does not exist.")  # Debugging print
            continue

        os.makedirs(output_split_path, exist_ok=True)

        for tumor_class in os.listdir(split_path):
            class_path = os.path.join(split_path, tumor_class)
            output_class_path = os.path.join(output_split_path, tumor_class)

            if not os.path.isdir(class_path):
                continue

            os.makedirs(output_class_path, exist_ok=True)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue  # Skip non-image files
                
                print(f"Processing: {img_path}")  # Debugging print
                
                img_array, spike_train = encode_rate_coding(img_path, time_window, max_spikes)
                
                # Save as .npy
                save_path = os.path.join(output_class_path, os.path.splitext(img_name)[0] + '.npy')
                np.save(save_path, spike_train)
                print(f"Saved: {save_path}")  # Debugging print
                
                img_count += 1
                
                # Every 50 images, visualize before and after encoding
                if img_count % 50 == 0:
                    visualize_encoding(img_array, spike_train)
                    print(f"Processed {img_count} images so far...")

    print("Processing complete!")


input_directory = "Brain-Tumor-Classification-DataSet"
output_directory = "Encoding/Rate-Encoding"
process_directory(input_directory, output_directory)
