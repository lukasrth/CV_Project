# Sim2Real: Synthetic Data Generation & YOLOv8 Training Pipeline

This repository contains a complete "Sim2Real" (Simulation to Reality) computer vision pipeline. It automates the generation of synthetic bounding-box training data using 3D `.glb` scans and headless Blender, and subsequently trains a YOLOv8 object detection model—all without requiring manual image annotation.

## Pipeline Overview

1. **3D Asset Ingestion:** Loads custom 3D scans (`.glb`) into a headless Blender instance.
2. **Synthetic Photography:** Automates a virtual camera to orbit the object, randomizing lighting, backgrounds (COCO), object scale, and viewing angles.
3. **Automated Annotation:** Mathematically projects the 3D mesh boundaries onto the 2D camera plane to generate pixel-perfect YOLO bounding box labels.
4. **Sanity Checking:** Validates the math by drawing bounding boxes onto debug images using OpenCV.
5. **Model Training:** Trains a pre-trained YOLOv8-Nano model exclusively on the synthetic dataset to prepare it for real-world inference.

---

## Environment Setup & Installation

This project is designed to run on a remote Linux server with GPU access. To conserve limited `/home` directory space, heavy assets and software are installed on the `/data` drive.

### 1. Project Directory Structure
We split the lightweight code from the heavy data:
* **Code Workspace:** `/export/home/your_username/Computer_Vision/Project/`
* **Data Workspace:** `/export/data/your_username/CV_Project/`

### 2. Headless Blender Installation (No `sudo` required)
Blender is used as our rendering engine. We use a portable Linux build to bypass admin privilege requirements.

```bash
cd /export/data/your_username/CV_Project/software
wget https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz
tar -xf blender-4.1.0-linux-x64.tar.xz
```

### 3. Python Environment (`uv`)
Blender uses its own internal Python environment for rendering. However, for OpenCV sanity checks and YOLO training, we create a blazing-fast virtual environment using `uv`.

```bash
cd /export/home/your_username/Computer_Vision/Project
uv venv .venv
source .venv/bin/activate
uv pip install ultralytics opencv-python
```

---

## Phase 1: Data Generation Engine

The core of this project is `generate_dataset.py`, a script that utilizes the Blender Python API (`bpy`).

To generate the dataset, run Blender headlessly in the background:
```bash
/export/data/your_username/CV_Project/software/blender-4.1.0-linux-x64/blender --background --python generate_dataset.py
```

### Known Quirk: The "Too Small" 3D Scale Issue
When importing `.glb` files from various scanning apps (e.g., RealityScan, Luma AI, Khronos samples), the internal scale of the objects varies wildly. A common issue is the object rendering as a tiny speck or blowing up to cover the entire camera lens.

**How we solved it:**
We decoupled the camera distance from the object's physical scale. 
1. **Camera Orbit:** The camera orbits the object at a steady, fixed distance radius (e.g., `15.0` to `20.0` units).
2. **Root Scaling:** We randomly physicalize the scale of the object (`scale_factor = random.uniform(0.3, 1.5)`). *Crucially*, we only apply this scale to the **root objects** (`if obj.parent is None:`). Scaling child nodes in a `.glb` hierarchy will cause the math to double up and the 3D mesh to explode or vanish entirely.
3. **Lens Shift:** We use `shift_x` and `shift_y` on the camera sensor to push the object into the corners of the image without distorting the 3D perspective.

---

## Phase 2: Sanity Checks

**Never trust your synthetic generation pipeline blindly.** Before training, always verify that the mathematical projection from 3D space to 2D YOLO coordinates (`[class] [x_center] [y_center] [width] [height]`) is accurate.

Run the verification script using your `uv` environment:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python verify_bboxes.py
```
This script reads the first 10 generated `.png` and `.txt` files, converts the normalized coordinates back to pixels, and draws a green bounding box over the image. Check the `/sanity_checks/` output folder to ensure the boxes perfectly hug the object.

---

## Phase 3: YOLOv8 Training

Once the dataset (images + labels) is generated, training is handled by the Ultralytics library. 

1. Ensure your `dataset.yaml` points to the absolute paths of your synthetic output directory.
2. Run the training script, specifying your target GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train_yolo.py
```

The script will download the lightweight `yolov8n.pt` weights and fine-tune them on your synthetic dataset. Training artifacts, loss graphs, and the final `best.pt` weights will be saved to your `/data` drive to prevent storage overflow.

---

## Troubleshooting & FAQ

* **Massive blocks of `libEGL warning` in the terminal:** Ignore these. This is Blender attempting to open a GUI window on a headless server, failing, and falling back to background rendering perfectly.
* **Blender fails with `Error: Please select a file`:** Your `GLB_PATH` in the Python script is pointing to a file that doesn't exist. Check your spelling and ensure you are using absolute paths.
* **`verify_bboxes.py` says "Checking 0 images":** Ensure the verification script is looking for `.png` files, as Blender's background renderer forces PNG exports by default.