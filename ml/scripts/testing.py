import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import mediapipe as mp
import joblib

# Load the previously fitted scaler
scaler = joblib.load('../workspace/minmax_scaler.pkl')
# Load model
model = tf.keras.models.load_model('../models/asl_model_norm.h5')

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# Your image-label map (update with real labels)
image_labels = {
    'A_test.jpg': 0,
    'B_test.jpg': 1,
    'C_test.jpg': 2,
    'D_test.jpg': 3,
    'E_test.jpg': 4,
    'F_test.jpg': 5,
    'G_test.jpg': 6,
    'H_test.jpg': 7,
    'I_test.jpg': 8,
    'J_test.jpg': 9, 
    'K_test.jpg': 10,
    'L_test.jpg': 11,
    'M_test.jpg': 12,
    'N_test.jpg': 13,
    'O_test.jpg': 14,
    'P_test.jpg': 15,
    'Q_test.jpg': 16,
    'R_test.jpg': 17,
    'S_test.jpg': 18,
    'T_test.jpg': 19,
    'U_test.jpg': 20,
    'V_test.jpg': 21,
    'W_test.jpg': 22,
    'X_test.jpg': 23,
    'Y_test.jpg': 24,
    'Z_test.jpg': 25,
    'aug_A_test.jpg': 0,
    'aug_B_test.jpg': 1,
    'aug_C_test.jpg': 2,
    'aug_D_test.jpg': 3,
    'aug_E_test.jpg': 4,
    'aug_F_test.jpg': 5,
    'aug_G_test.jpg': 6,
    'aug_H_test.jpg': 7,
    'aug_I_test.jpg': 8,
    'aug_J_test.jpg': 9, 
    'aug_K_test.jpg': 10,
    'aug_L_test.jpg': 11,
    'aug_M_test.jpg': 12,
    'aug_N_test.jpg': 13,
    'aug_O_test.jpg': 14,
    'aug_P_test.jpg': 15,
    'aug_Q_test.jpg': 16,
    'aug_R_test.jpg': 17,
    'aug_S_test.jpg': 18,
    'aug_T_test.jpg': 19,
    'aug_U_test.jpg': 20,
    'aug_V_test.jpg': 21,
    'aug_W_test.jpg': 22,
    'aug_X_test.jpg': 23,
    'aug_Y_test.jpg': 24,
    'aug_Z_test.jpg': 25,
    'aug2_A_test.jpg': 0,
    'aug2_B_test.jpg': 1,
    'aug2_C_test.jpg': 2,
    'aug2_D_test.jpg': 3,
    'aug2_E_test.jpg': 4,
    'aug2_F_test.jpg': 5,
    'aug2_G_test.jpg': 6,
    'aug2_H_test.jpg': 7,
    'aug2_I_test.jpg': 8,
    'aug2_J_test.jpg': 9, 
    'aug2_K_test.jpg': 10,
    'aug2_L_test.jpg': 11,
    'aug2_M_test.jpg': 12,
    'aug2_N_test.jpg': 13,
    'aug2_O_test.jpg': 14,
    'aug2_P_test.jpg': 15,
    'aug2_Q_test.jpg': 16,
    'aug2_R_test.jpg': 17,
    'aug2_S_test.jpg': 18,
    'aug2_T_test.jpg': 19,
    'aug2_U_test.jpg': 20,
    'aug2_V_test.jpg': 21,
    'aug2_W_test.jpg': 22,
    'aug2_X_test.jpg': 23,
    'aug2_Y_test.jpg': 24,
    'aug2_Z_test.jpg': 25,
    'aug2_aug_A_test.jpg':0,
    'aug2_aug_B_test.jpg':1,
    'aug2_aug_C_test.jpg':2,
    'aug2_aug_D_test.jpg':3,
    'aug2_aug_E_test.jpg':4,
    'aug2_aug_F_test.jpg':5,
    'aug2_aug_G_test.jpg':6,
    'aug2_aug_H_test.jpg':7,
    'aug2_aug_I_test.jpg':8,
    'aug2_aug_J_test.jpg':9,  
    'aug2_aug_K_test.jpg':10, 
    'aug2_aug_L_test.jpg':11, 
    'aug2_aug_M_test.jpg':12, 
    'aug2_aug_N_test.jpg':13, 
    'aug2_aug_O_test.jpg':14, 
    'aug2_aug_P_test.jpg':15, 
    'aug2_aug_Q_test.jpg':16, 
    'aug2_aug_R_test.jpg':17,
    'aug2_aug_S_test.jpg':18, 
    'aug2_aug_T_test.jpg':19, 
    'aug2_aug_U_test.jpg':20, 
    'aug2_aug_V_test.jpg':21, 
    'aug2_aug_W_test.jpg':22, 
    'aug2_aug_X_test.jpg':23, 
    'aug2_aug_Y_test.jpg': 24,
    'aug2_aug_Z_test.jpg':25, 
    'nothing_test.jpg': 27,
    'aug_nothing_test.jpg':27,
    'aug2_nothing_test.jpg': 27,
    'aug2_aug_nothing_test.jpg': 27,
    'aug2_aug_space_test.jpg': 28,
    'aug2_space_test.jpg':28,
    'aug_space_test.jpg': 28,
    'space_test.jpg': 28
}

# Folder with test images (original or augmented)
image_folder = '../data/asl_alphabet_test'  # Update with your test folder

X_test = []
y_test = []

def extract_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    return None

# Process each image
for filename, label in image_labels.items():
    path = os.path.join(image_folder, filename)
    img = cv2.imread(path)
    if img is None:
        continue

    landmarks = extract_landmarks(img)
    if landmarks is not None:
        X_test.append(landmarks)
        y_test.append(label)

# Final arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Apply the same scaler to test data
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("✅ Test Accuracy:", acc)
print("✅ Test F1 Score:", f1)
print("✅ Test Precision:", precision)
print("✅ Test Recall:", recall)

