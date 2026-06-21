import bpy
import bpy_extras
import mathutils
import math
import random
import os
import glob

# ==========================================
# 1. CONFIGURATION & PATHS (Update These)
# ==========================================
GLB_PATH = "/export/data/lriethm/CV_Project/assets/3DModel_origami.glb"
BG_DIR = "/export/data/lriethm/CV_Project/datasets/val2017/" # COCO Backgrounds
OUT_DIR = "/export/data/lriethm/CV_Project/synthetic_output_2/"
NUM_IMAGES = 2 # Change to 5000+ for the final run
CLASS_ID = 0 # YOLO class ID for your object

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_scene():
    """Deletes the default cube, light, and camera to start fresh."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def get_2d_bounding_box(scene, camera_object, target_object):
    """
    Projects the 3D object vertices onto the 2D camera plane.
    Returns YOLO format: (x_center, y_center, width, height)
    """
    # Ensure the object has mesh data
    if target_object.type != 'MESH':
        return None

    # Get the object's global matrix to convert local vertices to world space
    matrix = target_object.matrix_world
    frame = [matrix @ v.co for v in target_object.data.vertices]

    # Convert world coordinates to 2D camera coordinates (values between 0.0 and 1.0)
    camera_coords = [bpy_extras.object_utils.world_to_camera_view(scene, camera_object, v) for v in frame]

    # Find min and max coordinates
    x_coords = [c.x for c in camera_coords]
    y_coords = [c.y for c in camera_coords]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Clamp values to the image boundaries (0.0 to 1.0)
    x_min, x_max = max(0.0, x_min), min(1.0, x_max)
    y_min, y_max = max(0.0, y_min), min(1.0, y_max)

    # If the object is completely off-screen, return None
    if x_min == x_max or y_min == y_max:
        return None

    # Calculate YOLO format
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2.0)
    
    # Blender's Y axis is bottom-up. YOLO expects top-down. We must invert Y.
    y_center = 1.0 - (y_min + (height / 2.0))

    return (x_center, y_center, width, height)

def setup_compositor_for_background():
    """Sets up the node tree to composite the 3D render over a 2D image."""
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()

    # Create nodes
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    bg_image_node = tree.nodes.new(type="CompositorNodeImage")
    scale_node = tree.nodes.new(type="CompositorNodeScale")
    alpha_over = tree.nodes.new(type="CompositorNodeAlphaOver")
    composite_out = tree.nodes.new(type="CompositorNodeComposite")

    # Set background to scale to render size (Fit to Camera)
    scale_node.space = 'RENDER_SIZE'

    # Link nodes
    tree.links.new(bg_image_node.outputs["Image"], scale_node.inputs["Image"])
    tree.links.new(scale_node.outputs["Image"], alpha_over.inputs[1]) # Background
    tree.links.new(render_layers.outputs["Image"], alpha_over.inputs[2]) # Foreground (3D Object)
    tree.links.new(alpha_over.outputs["Image"], composite_out.inputs["Image"])
    
    # Enable transparent rendering for the 3D scene background
    bpy.context.scene.render.film_transparent = True
    
    return bg_image_node

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
clean_scene()

# A. Setup Camera
bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(math.radians(90), 0, 0))
cam = bpy.context.object
bpy.context.scene.camera = cam

# B. Import the .glb Object
import os
if not os.path.exists(GLB_PATH):
    raise FileNotFoundError(f"CRITICAL ERROR: Could not find the 3D model at {GLB_PATH}")

bpy.ops.import_scene.gltf(filepath=GLB_PATH)

imported_objects = list(bpy.context.selected_objects)

# Find the main root object for bounding box calculations
target_obj = imported_objects[0]

# FORCE Blender to accept standard XYZ degrees instead of Quaternions!
target_obj.rotation_mode = 'XYZ'

# ==========================================
print("Stripping baked textures to destroy the red dot shortcut...")
target_obj.data.materials.clear() # This deletes the texture with the red dot

# Create a clean, blank material to replace it
paper_mat = bpy.data.materials.new(name="DynamicPaper")
paper_mat.use_nodes = True
target_obj.data.materials.append(paper_mat)
# ==========================================




# C. Setup Lighting
bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
sun = bpy.context.object

# D. Setup Compositor for Backgrounds
bg_node = setup_compositor_for_background()
background_files = glob.glob(os.path.join(BG_DIR, "*.jpg"))

# E. THE RENDERING LOOP
for i in range(NUM_IMAGES):
    #print(f"Rendering image {i+1}/{NUM_IMAGES}...")

    # ============================================
    # 1. ORBIT THE CAMERA (Replaces Object Rotation/Scale)
    # ============================================
    scale_factor = random.uniform(3.0, 15.0)
    
    # Loop through our permanent list, not the active selection!
    for obj in imported_objects:
        # Only scale the "Root" objects (objects with no parents). 
        # If we scale the children too, the math doubles up and explodes the mesh.
        if obj.parent is None:
            obj.scale = (scale_factor, scale_factor, scale_factor)
    distance = random.uniform(15.0, 45.0) 

    # Pick random spherical angles for 360-degree coverage
    theta = random.uniform(0, 2 * math.pi) # Longitude (Around the sides)
    phi = random.uniform(0.2, math.pi - 0.2) # Latitude (Top to bottom)

    # Move the camera to that spherical coordinate
    cam.location.x = distance * math.sin(phi) * math.cos(theta)
    cam.location.y = distance * math.sin(phi) * math.sin(theta)
    cam.location.z = distance * math.cos(phi)

    # Point the camera AT the object, but add a slight random offset 
    # so the helmet isn't always perfectly dead-center in the final image.
    target_x = random.uniform(-0.6, 0.6)
    target_y = random.uniform(-0.6, 0.6)
    target_z = random.uniform(-0.6, 0.6)
    target_vector = mathutils.Vector((target_x, target_y, target_z))
    
    # Calculate the direction from the camera to the jittered target
    direction = target_vector - cam.location
    
    # Use Blender's built-in math to aim the camera perfectly
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    cam.rotation_mode = 'XYZ' # Switch back to Euler to be safe
    # ============================================

    cam.data.shift_x = random.uniform(-0.17, 0.17)
    cam.data.shift_y = random.uniform(-0.17, 0.17)

    # 2. Randomize Lighting
    sun.rotation_euler = (
        random.uniform(0, math.pi),
        random.uniform(0, math.pi),
        0
    )
    sun.data.energy = random.uniform(1.0, 5.0)

    shade = random.uniform(0.4, 0.9) # Pick a random grayscale value
    # Apply to the Principled BSDF node (Red, Green, Blue, Alpha)
    paper_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (shade, shade, shade, 1.0)

    # 3. Assign Random Background
    bg_img_path = random.choice(background_files)
    img = bpy.data.images.load(bg_img_path)
    bg_node.image = img

    # 4. Calculate Bounding Box
    bpy.context.view_layer.update() # Force Blender to update matrices
    bbox = get_2d_bounding_box(bpy.context.scene, cam, target_obj)

    if bbox is None:
        #print("Object off screen, skipping...")
        continue # Skip rendering if the object isn't visible

    x_c, y_c, w, h = bbox

    # 5. Render & Save
    file_prefix = f"synthetic_{i:05d}"
    
    # Save Image (.png is default in background rendering)
    bpy.context.scene.render.filepath = os.path.join(OUT_DIR, f"{file_prefix}.png")
    bpy.ops.render.render(write_still=True)
    
    # Save YOLO Label (.txt)
    txt_path = os.path.join(OUT_DIR, f"{file_prefix}.txt")
    with open(txt_path, "w") as f:
        f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    # Cleanup image memory so Blender doesn't crash after 1000 loops
    bpy.data.images.remove(img)

print("Dataset generation complete!")