import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import matplotlib.pyplot as plt
import os
from itertools import chain

# F1ScoreCallback
class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        predictions = np.argmax(self.model.predict(X_val), axis=1)
        true_labels = np.argmax(y_val, axis=1)
        f1 = f1_score(true_labels, predictions, average='weighted')
        self.f1_scores.append(f1)
        print(f"Epoch {epoch + 1}: F1-Score: {f1:.4f}")

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.96

# Data Collection
main_folder = "C:\\Users\\mahdi\\OneDrive\\Desktop\\AISContest_Data"

# List to store all file paths
final_file_list = []

# Process folders from 0 to 4
for label in range(5):
    folder_path = os.path.join(main_folder, str(label))
    
    if not os.path.exists(folder_path):  # Check if folder exists
        print(f"Warning: Folder {folder_path} does not exist.")
        continue
    
    print(f"Processing folder: {folder_path}")
    
    # Collect all .npy files in the folder
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npy")]
    final_file_list.append(file_list)
    print(f"Found {len(file_list)} files in folder {label}.")

# Flatten the list of lists into one list
file_list = list(chain.from_iterable(final_file_list))

# Load data and labels
data_list = []
labels_list = []

for file_path in file_list:
    label = int(os.path.basename(os.path.dirname(file_path)))  # Extract label from folder name
    data = np.load(file_path)  # Load data from .npy file
    data_list.append(data)
    labels_list.append(label)

# Convert data and labels to NumPy arrays
data_array = np.stack(data_list, axis=0)
labels_array = np.array(labels_list)

# Add a channel dimension to the data
data_array = data_array[..., np.newaxis]

print(f"Data shape: {data_array.shape}")
print(f"Labels shape: {labels_array.shape}")

# Standardize features
scaler = StandardScaler()
data_array_flat = data_array.reshape(-1, data_array.shape[2])
data_array_flat = scaler.fit_transform(data_array_flat)
data_array = data_array_flat.reshape(-1, 51, data_array.shape[2], 1)

# Handle class imbalance with SMOTE
X_flat = data_array.reshape(data_array.shape[0], -1)
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_flat, labels_array)
data_array = X_resampled.reshape(-1, 51, data_array.shape[2], 1)

# Convert labels to one-hot encoding
labels_array = to_categorical(y_resampled, num_classes=5)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, random_state=42)

# Callbacks
f1_callback = F1ScoreCallback(validation_data=(X_test, y_test))
callbacks = [
    LearningRateScheduler(scheduler),
    f1_callback
]

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(51, 59, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Reshape((11, -1)),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(5, activation='softmax')
])


# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model and save training history
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# Evaluate model
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Compute F1-Score
f1 = f1_score(true_classes, predicted_classes, average='weighted')
print(f"Final F1-Score: {f1:.4f}")

# Plot F1-Score over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(f1_callback.f1_scores) + 1), f1_callback.f1_scores, marker='o', label='F1-Score')
plt.title('F1-Score Over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

# Display all epochs details
for epoch, acc, val_acc, loss, val_loss in zip(
        range(1, len(history.history['accuracy']) + 1),
        history.history['accuracy'],
        history.history['val_accuracy'],
        history.history['loss'],
        history.history['val_loss']):
    print(f"Epoch {epoch}: Accuracy: {acc:.4f}, Validation Accuracy: {val_acc:.4f}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")