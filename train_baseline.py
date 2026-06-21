from ultralytics import YOLO
import os

RUNS_DIR = "/export/data/lriethm/CV_Project/yolo_runs/"

print("Loading a fresh YOLOv8-Nano model...")
# We load the base pre-trained model (untrained on your origami)
model = YOLO('yolov8n.pt')

print("Starting BASELINE training on REAL data...")
results = model.train(
    data="/export/data/lriethm/CV_Project/datasets/real_origami/data.yaml",   
    epochs=1000,             
    imgsz=640,             
    batch=16,              
    project=RUNS_DIR,      
    name='real_baseline_v1' # Saves to a new, separate folder!
)

print("Training complete! Now evaluating baseline on test set...")
# Automatically evaluate the baseline model on the test split
metrics = model.val(split="test")