import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GRU, LSTM, Reshape
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import matplotlib.pyplot as plt
import os

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

# بارگذاری و پیش‌پردازش داده‌ها به روش جدید
main_folder = r"C:\Users\hp\Desktop\AISContest_Data\AISContest_Data"

data_list = []
labels_list = []

# پردازش هر پوشه (۰ تا ۴)
for label in range(5):
    folder_path = os.path.join(main_folder, str(label))  # مسیر پوشه
    file_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npy")])
    
    # بارگذاری داده‌های هر فایل و افزودن برچسب
    for file in file_list:
        data = np.load(file)  # بارگذاری داده از فایل npy
        data_list.append(data)  # اضافه کردن داده به لیست
        labels_list.append(label)  # اضافه کردن برچسب به لیست

# تبدیل داده‌ها به آرایه NumPy
data_array = np.stack(data_list, axis=0)  # داده‌ها به آرایه سه‌بعدی
labels_array = np.array(labels_list)  # برچسب‌ها به آرایه یک‌بعدی

# افزودن کانال به داده‌ها
data_array = data_array[..., np.newaxis]  # اضافه کردن بعد چهارم: (نمونه‌ها، 59، 51، 1)

print(f"ابعاد داده: {data_array.shape}")
print(f"ابعاد برچسب‌ها: {labels_array.shape}")

# استانداردسازی ویژگی‌ها
scaler = StandardScaler()
data_array_flat = data_array.reshape(-1, data_array.shape[2])
data_array_flat = scaler.fit_transform(data_array_flat)
data_array = data_array_flat.reshape(-1, 51, data_array.shape[2], 1)

# مقابله با عدم تعادل کلاس‌ها با SMOTE
X_flat = data_array.reshape(data_array.shape[0], -1)
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_flat, labels_array)
data_array = X_resampled.reshape(-1, 51, data_array.shape[2], 1)

# تبدیل برچسب‌ها به one-hot encoding
labels_array = to_categorical(y_resampled, num_classes=5)

# تقسیم داده‌ها به مجموعه‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, random_state=42)

# Callbacks
f1_callback = F1ScoreCallback(validation_data=(X_test, y_test))
callbacks = [
    LearningRateScheduler(scheduler),
    f1_callback
]

# معماری مدل
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

# کامپایل مدل
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# آموزش مدل و ذخیره تاریخچه آموزش
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# ارزیابی مدل
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)


# محاسبه F1-Score
f1 = f1_score(true_classes, predicted_classes, average='weighted')
print(f"Final F1-Score: {f1:.4f}")

# رسم F1-Score در طول epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(f1_callback.f1_scores) + 1), f1_callback.f1_scores, marker='o', label='F1-Score')
plt.title('F1-Score Over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

# بررسی تمام Epochs
for epoch, acc, val_acc, loss, val_loss in zip(
        range(1, len(history.history['accuracy']) + 1),
        history.history['accuracy'],
        history.history['val_accuracy'],
        history.history['loss'],
        history.history['val_loss']):
    print(f"Epoch {epoch}: Accuracy: {acc:.4f}, Validation Accuracy: {val_acc:.4f}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
