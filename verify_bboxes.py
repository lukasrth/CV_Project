import cv2
import os
import glob

# ==========================================
# 1. PATHS (Update if needed)
# ==========================================
OUT_DIR = "/export/data/lriethm/CV_Project/synthetic_output_2/"
DEBUG_DIR = "/export/data/lriethm/CV_Project/sanity_checks_2/"

# Create a folder for the debug images
os.makedirs(DEBUG_DIR, exist_ok=True)

# Grab just the first 10 PNG images to check
image_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.png")))[:10]

print(f"Checking {len(image_files)} images...")

for img_path in image_files:
    # Find the corresponding .txt label file
    txt_path = img_path.replace(".png", ".txt")
    
    if not os.path.exists(txt_path):
        print(f"Warning: No label file found for {img_path}")
        continue

    # Load the image using OpenCV
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape

    # Read the YOLO label
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            # Convert YOLO normalized coordinates (0.0 to 1.0) back to raw pixels
            x_center = int(x_center_norm * img_width)
            y_center = int(y_center_norm * img_height)
            box_width = int(width_norm * img_width)
            box_height = int(height_norm * img_height)

            # Calculate top-left and bottom-right corners for OpenCV
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            x_max = int(x_center + (box_width / 2))
            y_max = int(y_center + (box_height / 2))

            # Draw the bounding box (Color: Green, Thickness: 2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw a small red dot at the exact center to verify alignment
            cv2.circle(img, (x_center, y_center), 3, (0, 0, 255), -1)

    # Save the debug image
    filename = os.path.basename(img_path)
    save_path = os.path.join(DEBUG_DIR, filename)
    cv2.imwrite(save_path, img)

print(f"Done! Check the '{DEBUG_DIR}' folder to see your drawn boxes.")