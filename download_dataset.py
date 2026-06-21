import os
from roboflow import Roboflow

try:
    print("1. Initializing Roboflow API...")
    rf = Roboflow(api_key="eC5LMqP2r4dYSBkc3Gjf")
    
    print("2. Connecting to workspace and project...")
    project = rf.workspace("alina-meng-m-gmail-com").project("real_origami_bbox")
    version = project.version(1)
    
    target_dir = "/export/data/lriethm/CV_Project/datasets/real_origami"
    print(f"3. Attempting download to: {target_dir}")
    
    # Run download
    dataset = version.download("yolov8", location=target_dir)
    
    print("\n--- SUCCESS! ---")
    print(f"Reported download location: {dataset.location}")
    print(f"Contents of that directory: {os.listdir(dataset.location)}")

except Exception as e:
    print("\n--- CRITICAL ERROR ---")
    print(f"The download failed with the following error:\n{str(e)}")