# Sim2Real: Synthetic Data Generation & YOLOv8 Training Pipeline

This repository contains a complete "Sim2Real" (Simulation to Reality) computer vision pipeline. It automates the generation of synthetic bounding-box training data using 3D `.glb` scans and headless Blender, and subsequently trains a YOLOv8 object detection model. 

Crucially, this project serves as a case study in **overcoming the Sim2Real domain gap and the Clever Hans effect**, featuring built-in scripts for geometry sanitization, domain randomization, and adversarial testing.

## Pipeline Overview

1. **3D Asset Ingestion & Sanitization:** Loads custom 3D scans (`.glb`) into a headless Blender instance and surgically scrubs baked vertex colors and artifactual tracking markers to prevent shortcut learning.
2. **Synthetic Photography & Domain Randomization:** Automates a virtual camera to orbit the object, randomizing lighting, backgrounds (COCO), object scale, and dynamically shifting material shades to force structural learning.
3. **Automated Annotation:** Mathematically projects the 3D mesh boundaries onto the 2D camera plane to generate pixel-perfect YOLO bounding box labels.
4. **Adversarial & Sanity Checking:** Validates bounding box math and includes "data poisoning" scripts to test for model overfitting.
5. **Model Training:** Trains a YOLOv8 model exclusively on the synthetic dataset (with a 9:1 Train/Val split).
6. **Real-World Evaluation:** Zero-shot evaluation of the synthetic model on a manually annotated real-world dataset.

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
wget [https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz](https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz)
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

### Key Feature: Geometry Surgery & Texture Stripping
Many `.glb` files from 3D scanning apps (like Polycam or Luma) embed tracking markers or artifactual data directly into the mesh's vertex colors. If left intact, the neural network will suffer from the **Clever Hans effect**, memorizing the marker instead of the object's geometry.
Our script automatically mitigates this by:
1. Stripping all existing materials.
2. Iterating through and deleting all `color_attributes` (Vertex Colors).
3. Applying a `DynamicPaper` material that randomly shifts its grayscale value on every frame to prevent color-based shortcut learning.

### Known Quirk: The "Too Small" 3D Scale Issue
Because internal scaling varies wildly across `.glb` files, we decouple camera distance from the object's physical scale. 
* We orbit the camera at a fixed radius (`15.0` to `45.0` units).
* We randomly scale the **root objects** (`scale_factor = random.uniform(3.0, 15.0)`). Scaling child nodes in a `.glb` hierarchy will cause the math to double up and the mesh to explode.

---

## Phase 2: Sanity Checks & Adversarial Testing

**Never trust your synthetic generation pipeline blindly.**

### 1. The Visual Sanity Check
Verify that the mathematical projection from 3D space to 2D YOLO coordinates (`class x_center y_center width height`) is accurate. 
```bash
CUDA_VISIBLE_DEVICES=0 uv run python sanity_check_2.py
```
This script auto-detects images, converts YOLO coordinates back to pixels, and draws green (Ground Truth) and red (Prediction) bounding boxes over the image so you can visually confirm coordinate alignment.

### 2. Data Poisoning (The Shortcut Test)
If your model achieves 99% synthetic mAP but 1% real-world mAP, it may have learned an artifact. You can use our adversarial script to inject a tiny red tracking dot into the center of your real-world test images:
```bash
uv run python poison_test_data.py
```
If evaluating the model on this "poisoned" dataset causes the mAP to suddenly spike, you have proven the model is overfitting to a specific visual artifact rather than learning the object morphology.

---

## Phase 3: YOLOv8 Training

Once the dataset is generated, training is handled by the Ultralytics library. Ensure your `dataset.yaml` points to the absolute paths of your synthetic output directory and utilizes a strict **9:1 Train/Validation Split** to monitor for synthetic overfitting.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train_yolo.py
```

The script will download the lightweight `yolov8n.pt` weights and fine-tune them. Training artifacts and the final `best.pt` weights will be saved to your `/data` drive to prevent storage overflow.

---

## Phase 4: Real-World Evaluation

To test true generalization, we evaluate the synthetic model against a manually annotated real-world dataset. 
* **The Dataset:** Hosted on Roboflow ([Link to Dataset](https://universe.roboflow.com/alina-meng-m-gmail-com/real_origami_bbox/dataset/1)).
* **Evaluation Strategy:** We use a "Settings-Based Split", ensuring the environments in the test dataset have absolutely no overlap with the synthetic training backgrounds to prevent data leakage.

Run the zero-shot evaluation script:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python eval_synthetic.py
```

---

## Troubleshooting & FAQ

* **Model scores 99% in training but 0% on real data:** You have a severe Sim2Real domain gap. Check for shortcut learning (like the Clever Hans red dot), verify your coordinates are normalized (between 0.0 and 1.0, not absolute pixels), and ensure your lighting/shadows aren't causing the object to look like a flat "cut-out".
* **Massive blocks of `libEGL warning` in the terminal:** Ignore these. This is Blender attempting to open a GUI window on a headless server, failing, and falling back to background rendering perfectly.
* **Blender fails with `Error: Please select a file`:** Your `GLB_PATH` in the Python script is pointing to a file that doesn't exist. Check your spelling and ensure you are using absolute paths.
