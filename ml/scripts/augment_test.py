import cv2
import os
import numpy as np
import random

# Paths
input_folder = '../data/asl_alphabet_test'           # Folder with original test images
output_folder =  '../data/asl_alphabet_test'    # Where to save augmented images
os.makedirs(output_folder, exist_ok=True)

def augment_image(img):
    """Apply random augmentations to the input image."""
    # Random rotation
    angle = random.uniform(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # Random brightness
    brightness = random.uniform(0.4, 2.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    return img

# Process and save
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue  # skip unreadable files

        aug_img = augment_image(img)

        new_filename = 'aug2_' + filename
        save_path = os.path.join(output_folder, new_filename)
        cv2.imwrite(save_path, aug_img)

print("✅ Augmented images saved to:", output_folder)
