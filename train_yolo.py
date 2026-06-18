from ultralytics import YOLO
import os

# Create a directory on your data drive to save the heavy model weights
RUNS_DIR = "/export/data/lriethm/CV_Project/yolo_runs/"
os.makedirs(RUNS_DIR, exist_ok=True)

print("Loading YOLOv8-Nano model...")
# Load the pre-trained lightweight model
model = YOLO('yolov8n.pt')

print("Starting training...")
# Start training!
results = model.train(
    data='dataset.yaml',   # Path to your config file
    epochs=50,             # Number of training loops
    imgsz=640,             # Image size (YOLO standard)
    batch=16,              # How many images to process at once
    project=RUNS_DIR,      # Where to save the output weights and charts
    name='sim2real_v1'     # Name of this specific training run
)

print("Training complete! Model weights saved to:", RUNS_DIR)