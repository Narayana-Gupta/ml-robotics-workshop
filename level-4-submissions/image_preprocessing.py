import os
import zipfile
import cv2
import numpy as np
from pathlib import Path

ZIP_FILE = 'images.zip'
EXTRACT_DIR = 'resized_images'

Path(EXTRACT_DIR).mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

OUTPUT_DIR = 'processed_images'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for file in os.listdir(EXTRACT_DIR):
    img_file_path = os.path.join(EXTRACT_DIR, file)
    
    if os.path.isfile(img_file_path):
        image = cv2.imread(img_file_path)
        
        if image is None:
            print(f"Failed to load image: {file}")
            continue
        
        resized_image = cv2.resize(image, (128, 128))
        normalized_image = resized_image / 255.0
        blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
        output_image = (blurred_image * 255).astype(np.uint8)
        output_path = os.path.join(OUTPUT_DIR, file)
        cv2.imwrite(output_path, output_image)
        print(f"Processed image saved: {output_path}")

print("Image processing completed!")
