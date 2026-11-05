import os
import sys
import json
import logging

# Set up logging for the terminal to keep it clean
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==============================================================================
# --- 1. DEPENDENCY CHECK (nothing is installed automatically) ---
# ==============================================================================
REQUIRED_MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "PIL": "pillow",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime-gpu",
    "insightface": "insightface",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "gradio": "gradio",
}


def check_dependencies():
    """Stops with a clear message if a package is missing, and warns if the GPU can't be used."""
    import importlib.util

    missing = [pip_name for module_name, pip_name in REQUIRED_MODULES.items()
               if importlib.util.find_spec(module_name) is None]
    if missing:
        print("\nSome Python packages are missing: " + ", ".join(missing))
        print("Install them from this folder with:\n")
        print("    pip install -r requirements.txt\n")
        print("For the NVIDIA GPU version of PyTorch, see 'Install' in README.md.")
        if sys.stdin and sys.stdin.isatty():
            input("Press Enter to close...")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        logging.warning("PyTorch can't use an NVIDIA GPU (CPU-only build or no CUDA driver). "
                        "The photo labels will be slow. See 'Install' in README.md for the GPU version.")
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        logging.warning("ONNX Runtime can't use the GPU, so face detection runs on the CPU (slow). "
                        "See 'Troubleshooting' in README.md.")


# Use only the models already saved on this PC. This must be set before transformers is imported,
# because the Hugging Face libraries read it when they load.
os.environ["HF_HUB_OFFLINE"] = "1"

check_dependencies()

# ==============================================================================
# --- 2. MAIN APPLICATION LOGIC ---
# ==============================================================================
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import gradio as gr  # noqa: E402
import shutil  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import pipeline  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402

logging.info("Initializing AI Models...")

# Load SigLIP 2 (Offline Mode, see HF_HUB_OFFLINE above)
try:
    logging.info("Loading SigLIP 2...")
    image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip2-so400m-patch14-384")
except Exception as e:
    logging.warning(f"Failed to load offline SigLIP 2 model: {e}")
    image_classifier = None

# Load InsightFace (GPU)
try:
    logging.info("Loading InsightFace on NVIDIA GPU...")
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))  # 0 = the first GPU
except Exception as e:
    logging.error(f"Error loading InsightFace: {e}")
    face_app = None

candidate_labels = [
    "A photo of exactly one person",
    "A photo of two or more people",
    "A photo of a pet or animal",
    "A photo of nature or scenery",
    "A photo of an object, building, or other subject"
]

def sanitize_folder_name(name):
    """Removes invalid characters for folder creation."""
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip()

def sidecar_path(image_path):
    """IMG_1.jpg -> IMG_1.jpg.json, so IMG_1.jpg and IMG_1.png never share one sidecar."""
    return image_path + ".json"


def unique_destination(destination_dir, filename):
    """A name in destination_dir that neither the photo nor its sidecar already uses.
    IMG_1.jpg becomes IMG_1 (1).jpg, IMG_1 (2).jpg, ... so nothing is ever overwritten."""
    stem, ext = os.path.splitext(filename)
    candidate, number = filename, 1
    while (os.path.exists(os.path.join(destination_dir, candidate))
           or os.path.exists(sidecar_path(os.path.join(destination_dir, candidate)))):
        candidate = f"{stem} ({number}){ext}"
        number += 1
    return os.path.join(destination_dir, candidate)


def choose_folder(metadata, sort_mode):
    """The folder name for one photo, or None when the photo stays where it is."""
    if sort_mode == "Sort by Broad Category (Nature, Pets, etc.)":
        return metadata.get("auto_category", "Unsorted")
    people = metadata.get("people_detected", [])
    if len(people) == 1:
        return people[0]
    if len(people) > 1:
        return "Group Photos"
    if sort_mode == "Sort by Specific People (Person ID)":
        return "No People Detected"
    if sort_mode == "Faces Only (Ignore everything else)":
        return None
    return "Unsorted"


def process_photo_folder(directory_path, sort_mode, strictness_slider, preview_only=True):
    if not os.path.isdir(directory_path):
        return f"Error: The path '{directory_path}' is not a valid directory."

    if not image_classifier and not face_app:
        return "Error: Neither AI model loaded successfully. Check terminal logs."

    log_output = f"Starting scan of: {directory_path}\nMode selected: {sort_mode}\n"
    if preview_only:
        log_output += "PREVIEW ONLY: no files are written or moved.\n"
    log_output += "\n"

    face_embeddings = []
    face_metadata = []
    all_metadata = {}       # filename -> metadata; written to the sidecars only when not previewing
    processed_count = 0

    # Phase 1: Scanning and Classification
    for filename in sorted(os.listdir(directory_path)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        processed_count += 1
        image_path = os.path.join(directory_path, filename)
        json_path = sidecar_path(image_path)

        metadata = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    pass

        # A. General Categorization (SigLIP 2)
        if image_classifier:
            try:
                with Image.open(image_path) as pil_image:   # closed again, so the file can be moved
                    outputs = image_classifier(pil_image, candidate_labels=candidate_labels)
                top_result = outputs[0]
                metadata["auto_category"] = top_result['label']
                metadata["category_confidence"] = round(top_result['score'], 4)
            except Exception as e:
                log_output += f"[{filename}] Image Read Error: {e}\n"

        # B. Face Detection (InsightFace)
        if face_app:
            img_bgr = cv2.imread(image_path)
            if img_bgr is not None:
                faces = face_app.get(img_bgr)
                for face_index, face in enumerate(faces):
                    face_embeddings.append(face.embedding)
                    face_metadata.append({"filename": filename, "face_index": face_index})

                metadata["people_detected"] = []

        all_metadata[filename] = metadata

    # Phase 2: Clustering Phase
    if face_app and len(face_embeddings) > 0:
        log_output += f"\nClustering {len(face_embeddings)} total faces...\n"
        embeddings_array = np.array(face_embeddings)

        clusterer = DBSCAN(eps=strictness_slider, min_samples=2, metric="cosine")
        cluster_labels = clusterer.fit_predict(embeddings_array)

        for idx, label in enumerate(cluster_labels):
            person_id = f"Person_{label}" if label != -1 else "Unknown_Person"
            metadata = all_metadata[face_metadata[idx]["filename"]]
            people = metadata.setdefault("people_detected", [])
            if person_id not in people:
                people.append(person_id)

    # Phase 3: File Moving Phase
    log_output += "\nPlanned moves (preview):\n" if preview_only else "\nMoving files into folders...\n"
    moved = 0
    for filename, metadata in all_metadata.items():
        image_path = os.path.join(directory_path, filename)
        json_path = sidecar_path(image_path)

        folder_name = choose_folder(metadata, sort_mode)
        if folder_name is None:
            log_output += f"Skipped {filename} (No faces detected)\n"
        safe_folder_name = sanitize_folder_name(folder_name) if folder_name else ""

        if preview_only:
            if folder_name is not None:
                log_output += f"Would move {filename} -> {safe_folder_name}/\n"
                moved += 1
            continue

        try:
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        except OSError as e:
            log_output += f"Error writing {os.path.basename(json_path)}: {e}\n"
            continue
        if folder_name is None:
            continue

        destination_dir = os.path.join(directory_path, safe_folder_name)
        os.makedirs(destination_dir, exist_ok=True)

        try:
            target = unique_destination(destination_dir, filename)
            shutil.move(image_path, target)
            shutil.move(json_path, sidecar_path(target))
            renamed = "" if os.path.basename(target) == filename else f" (as {os.path.basename(target)})"
            log_output += f"Moved {filename} -> {safe_folder_name}/{renamed}\n"
            moved += 1
        except Exception as e:
            log_output += f"Error moving {filename}: {e}\n"

    if preview_only:
        log_output += (f"\nPreview finished: {processed_count} images checked, {moved} would be moved. "
                       "Untick 'Preview only' and click the button again to move them.")
    else:
        log_output += f"\nFinished! Processed {processed_count} images and moved {moved}."
    return log_output


# ==============================================================================
# --- 3. THE GRADIO BROWSER UI ---
# ==============================================================================
with gr.Blocks(title="AI Photo Sorter") as demo:
    gr.Markdown("# 📸 AI Photo Sorter & Facial Clustering")
    gr.Markdown("Paste the absolute path to your local photo directory below. The script will read the files directly from your hard drive, generate `.json` sidecars, and move them into sorted folders. Run it first with **Preview only** ticked to see what would happen.")
    
    with gr.Row():
        with gr.Column(scale=2):
            folder_input = gr.Textbox(
                label="Directory Path", 
                placeholder="e.g., C:\\Users\\Name\\Pictures\\Vacation",
                info="The absolute path to the folder on your desktop containing your JPG/PNG files."
            )
            
            sort_mode = gr.Radio(
                choices=[
                    "Sort by Broad Category (Nature, Pets, etc.)", 
                    "Sort by Specific People (Person ID)",
                    "Faces Only (Ignore everything else)"
                ],
                value="Faces Only (Ignore everything else)",
                label="Sorting Mode",
                info="Choose how the final subfolders are created."
            )

            strictness = gr.Slider(
                minimum=0.2, maximum=0.6, value=0.4, step=0.05, 
                label="Face Clustering Strictness (Eps)",
                info="Only affects 'Sort by Specific People' and 'Faces Only'. Lower = Stricter (less likely to group different people together)."
            )
            preview = gr.Checkbox(
                value=True,
                label="Preview only (don't write or move any files)",
                info="Shows where each photo would go. Untick it and run again to really move the photos."
            )
            run_btn = gr.Button("Start Sorting & Clustering", variant="primary")
            
        with gr.Column(scale=3):
            output_log = gr.Textbox(label="Processing Log", lines=20, max_lines=30)
            
    run_btn.click(
        fn=process_photo_folder,
        inputs=[folder_input, sort_mode, strictness, preview],
        outputs=[output_log]
    )

if __name__ == "__main__":
    logging.info("Launching UI...")
    demo.launch(inbrowser=True)
