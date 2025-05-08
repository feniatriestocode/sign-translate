import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load the preprocessed data
X = np.load("../workspace/X.npy")  # Landmark features
y = np.load("../workspace/y.npy")  # One-hot encoded labels (if you did one-hot encoding)

# Assume X contains your features (e.g., MediaPipe landmarks)
# Normalize the entire dataset
scaler = MinMaxScaler()
# Split the normalized data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, '../workspace/minmax_scaler.pkl')
print("After saving, min_ seen by scaler:", scaler.min_) 

early_stop = EarlyStopping(
    monitor='val_loss',   # You can also monitor 'val_accuracy'
    patience=5,           # Number of epochs to wait for improvement
    restore_best_weights=True  # Roll back to the best weights seen
)


# 3. Build the model (a simple feed-forward neural network)
model = keras.Sequential([
    keras.layers.Input(shape=(63,)),  # 63 features (hand landmarks x,y,z)
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(y.shape[1], activation='softmax')  # Output layer: one unit per class
])

# 4. Compile the model
model.compile(
    optimizer='adam',  # Optimizer
    loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
    metrics=['accuracy']  # Metric to track during training
)

# 5. Train the model
history = model.fit(
    X_train, y_train,  # Training data
    epochs=100,  # Number of epochs to train
    validation_data=(X_val, y_val),  # Validation data to track accuracy
    batch_size=32,  # Batch size for training
    callbacks=[early_stop]
)

# Optionally save the trained model
model.save("../models/asl_model_norm.h5")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print("Training complete. Model saved as 'asl_model.h5'")

