import cv2
import numpy as np
import mediapipe as mp
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

print("Path to dataset files:", os.path.abspath("../data/asl_alphabet_train"))


DATASET_DIR = os.path.relpath("../data/asl_alphabet_train")
OUTPUT_X = "../workspace/X.npy"
OUTPUT_Y = "../workspace/y.npy"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

X = []
y = []

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
    return None

for label_folder in sorted(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label_folder)
    if not os.path.isdir(label_path):
        continue  # Skip non-directories

    print(f"Processing label '{label_folder}'...")

    for img_file in tqdm(os.listdir(label_path)):
        img_path = os.path.join(label_path, img_file)
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue  # Failed to read
            landmarks = extract_landmarks(image)
            if landmarks:
                X.append(landmarks)
                y.append(label_folder)
        except Exception as e:
            print(f"Error with image {img_path}: {e}")

# Convert to numpy arrays
X = np.array(X)

# Label encoding (integer encoding)
le = LabelEncoder()
y_int = le.fit_transform(y)  # Converts labels like 'A', 'B' to 0, 1, ...
print(f"Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
# Optionally, use one-hot encoding
y_onehot = to_categorical(y_int)  # Converts to one-hot vectors

# Save processed data
np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y_onehot) 

print(f"Saved {len(X)} samples to '{OUTPUT_X}' and '{OUTPUT_Y}'")