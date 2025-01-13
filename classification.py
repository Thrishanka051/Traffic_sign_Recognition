import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Define path to Train folder
TRAIN_PATH = "/kaggle/input/gtsrb-german-traffic-sign/Train"

# Load Train images and labels
def load_train_data(train_path):
    data = []
    labels = []
    
    for folder in os.listdir(train_path):
        folder_path = os.path.join(train_path, folder)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                try:
                    img = cv2.imread(os.path.join(folder_path, img_name))
                    img = cv2.resize(img, (32, 32))
                    data.append(img)
                    labels.append(int(folder))  # Folder name is the class label
                except Exception as e:
                    print(f"Error reading {img_name}: {e}")
    
    return np.array(data), np.array(labels)

# Load training data
X_train, y_train = load_train_data(TRAIN_PATH)

# Normalize train data
X_train = X_train / 255.0

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)

# Define path to Test.csv
TEST_CSV_PATH = "/kaggle/input/gtsrb-german-traffic-sign/Test.csv"
TEST_IMAGES_PATH = "/kaggle/input/gtsrb-german-traffic-sign/Test"

# Load Test.csv
test_data = pd.read_csv(TEST_CSV_PATH)

# Correct image paths
def correct_path(row):
    image_number = row['Path'].split('/')[-1]  # Extract only the image number
    corrected_path = os.path.join(TEST_IMAGES_PATH, image_number)  # Correct directory path
    return corrected_path

test_data['CorrectedPath'] = test_data.apply(correct_path, axis=1)

# Load test images and labels
def load_test_data(test_data):
    data = []
    labels = []
    
    for index, row in test_data.iterrows():
        img_path = row['CorrectedPath']
        label = row['ClassId']  # Assuming ClassId column contains labels
        
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return np.array(data), np.array(labels)

# Load Test data
X_test, y_test = load_test_data(test_data)

# Normalize test data
X_test = X_test / 255.0

# One-hot encode test labels
y_test = to_categorical(y_test, num_classes=num_classes)

# Define the CNN model
model = Sequential([
    Input(shape=(32, 32, 3)),  # Explicitly define input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Apply data augmentation to the training data
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20,
                    validation_data=(X_test, y_test))

# Save the trained model
model.save("traffic_sign_model.h5")
print("Model saved successfully!")

# Evaluate on corrected Test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Function to predict a single image
def predict_sign(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return class_index

# Test with a custom image
image_path = "/kaggle/input/gtsrb-german-traffic-sign/Test/00000.png"
predicted_class = predict_sign(image_path)
print(f"Predicted Class: {predicted_class}")
