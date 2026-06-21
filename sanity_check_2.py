import cv2
import os
from ultralytics import YOLO

# --- DIRECTORIES ---
images_dir = "/export/data/lriethm/CV_Project/datasets/real_origami/test/images/"
labels_dir = "/export/data/lriethm/CV_Project/datasets/real_origami/test/labels/"
model_path = "/export/data/lriethm/CV_Project/yolo_runs/sim2real_v1-2/weights/best.pt"

# 1. Automatically find the first image file in the directory
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
all_files = os.listdir(images_dir)
image_files = [f for f in all_files if f.endswith(valid_extensions)]

if not image_files:
    raise FileNotFoundError(f"No image files found in {images_dir}")

# Grab the first available image
image_name = image_files[0]
image_path = os.path.join(images_dir, image_name)

# Find corresponding label file (assuming same base name with .txt)
base_name = os.path.splitext(image_name)[0]
label_path = os.path.join(labels_dir, f"{base_name}.txt")

print(f"Loading Model: {model_path}")
model = YOLO(model_path)

print(f"Automatically selected image: {image_name}")
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"OpenCV could not read the file at {image_path}. Check file integrity.")

h, w, _ = img.shape

# 2. Draw PREDICTIONS (Red Boxes)
print("Running prediction...")
results = model(img, verbose=False)
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3) # RED
    cv2.putText(img, "Pred", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 3. Draw GROUND TRUTH (Green Boxes)
print("Loading ground truth...")
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            class_id, x_c, y_c, bw, bh = map(float, parts)
            # Convert YOLO normalized coordinates back to standard pixel coordinates
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3) # GREEN
            cv2.putText(img, "Truth", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
else:
    print(f"Warning: No ground truth label file found at {label_path}")

# Save the resulting image
output_file = "sanity_check_output.jpg"
cv2.imwrite(output_file, img)
print(f"Done! Saved comparison to: {os.path.abspath(output_file)}")