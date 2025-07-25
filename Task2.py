import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")

# Define class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- 1. Data Loading and Preprocessing ---
print("Loading and preprocessing CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encode the labels
train_labels_one_hot = to_categorical(train_labels, num_classes=10)
test_labels_one_hot = to_categorical(test_labels, num_classes=10)

print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels_one_hot.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels_one_hot.shape}")

# --- Visualize some sample images ---
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index [0]
    plt.xlabel(class_names[train_labels[i][0]])
plt.suptitle('Sample CIFAR-10 Images')
plt.show()

# --- 2. Model Architecture (Convolutional Neural Network) ---
print("Building the CNN model...")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization()) # Added for better training stability
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25)) # Added for regularization

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# --- 3. Model Compilation ---
print("Compiling the model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. Model Training ---
print("Training the model...")
history = model.fit(train_images, train_labels_one_hot, epochs=20,
                    validation_data=(test_images, test_labels_one_hot),
                    batch_size=64)

# --- 5. Model Evaluation ---
print("\nEvaluating the model...")
test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot, verbose=2)
print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")

# --- 6. Visualizations of Results ---

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Visualize some predictions
print("\nVisualizing some predictions...")
predictions = model.predict(test_images)

plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    
    true_label = class_names[test_labels[i][0]]
    predicted_label = class_names[np.argmax(predictions[i])]
    
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
plt.suptitle('Sample Predictions (Green: Correct, Red: Incorrect)', fontsize=16)
plt.show()

# Visualize misclassified images
print("\nVisualizing some misclassified images...")
misclassified_indices = np.where(np.argmax(predictions, axis=1) != test_labels.flatten())[0]

plt.figure(figsize=(15, 15))
# Show up to 25 misclassified images
num_to_show = min(len(misclassified_indices), 25)
for i in range(num_to_show):
    idx = misclassified_indices[i]
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[idx])
    
    true_label = class_names[test_labels[idx][0]]
    predicted_label = class_names[np.argmax(predictions[idx])]
    
    plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}", color='red')
plt.suptitle(f'Misclassified Images (Showing {num_to_show} out of {len(misclassified_indices)})', fontsize=16)
plt.show()

# --- Optional: Visualize filters from the first convolutional layer ---
print("\nVisualizing filters from the first convolutional layer (might be abstract)...")
# Get the first convolutional layer
first_conv_layer = model.layers[0]

# Get the weights (filters) from this layer
# The weights are usually in the format (kernel_height, kernel_width, input_channels, output_channels)
filters, biases = first_conv_layer.get_weights()

# Normalize filter values for visualization
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Plot the first few filters (e.g., 16 filters)
n_filters = min(filters.shape[3], 16) # Display up to 16 filters
rows = int(np.sqrt(n_filters))
cols = int(np.ceil(n_filters / rows))

plt.figure(figsize=(cols * 2, rows * 2))
for i in range(n_filters):
    ax = plt.subplot(rows, cols, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    # Display the filter. Since CIFAR-10 images are RGB (3 channels),
    # the filters will also have 3 channels.
    # We display the first channel or average across channels for grayscale visualization.
    # For RGB filters, we can display it directly.
    plt.imshow(filters[:, :, :, i]) # Display the i-th filter (all 3 input channels)
plt.suptitle('Filters from the First Convolutional Layer', fontsize=16)
plt.show()