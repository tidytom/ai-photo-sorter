# AI Photo Sorter

A local, private tool that sorts a folder of photos into subfolders by what is in them (people, pets, scenery, objects) or by which person is in them, for anyone with a large unsorted photo folder and an NVIDIA graphics card.

## What it does
- Labels every JPG and PNG photo with **Google SigLIP 2** (one of five labels, such as "A photo of a pet or animal").
- Finds faces with **InsightFace** and groups photos of the same person together (`Person0`, `Person1`, ...).
- Saves what it found in a small `.json` file next to each photo (for example `IMG_1.jpg.json`).
- Moves the photos into subfolders. A **Preview only** mode shows every planned move first and changes nothing.
- Runs completely on your own PC in a browser page (Gradio). No photo is uploaded.

## What you need
- Windows 10/11 or Linux.
- Python 3.10 or newer.
- An NVIDIA graphics card is strongly recommended. Without one it works, but slowly.
- For the GPU: the NVIDIA driver, **CUDA Toolkit 12.x** and **cuDNN 9** for CUDA 12.
- The SigLIP 2 model `google/siglip2-so400m-patch14-384` saved on your PC (a few GB, see Install step 4). InsightFace downloads its `buffalo_l` face model by itself on the first run.

## Install
1. Download this project (green **Code** button, then **Download ZIP**) and unzip it.
2. Open a terminal (Command Prompt) in the project folder.
3. Install PyTorch with GPU support, then the other packages:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
   If you have no NVIDIA card, skip the first line.
4. Download the SigLIP 2 model once (the sorter itself works offline and does not download it):
   ```
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/siglip2-so400m-patch14-384')"
   ```

## How to use
1. In the project folder, run:
   ```
   python ai_photo_sorter.py
   ```
2. Wait until the models are loaded. A page opens in your web browser.
3. In **Directory Path**, paste the full path of your photo folder, for example `C:\Users\Name\Pictures\Vacation`.
4. Choose a **Sorting Mode** (see below).
5. Leave **Preview only (don't write or move any files)** ticked and click **Start Sorting & Clustering**.
6. Read the **Processing Log**. It lists where each photo would go.
7. If it looks right, untick **Preview only** and click **Start Sorting & Clustering** again. This time the photos are moved.

## Settings and options

**Sorting Mode**

| Mode | Folders it makes |
|---|---|
| Sort by Broad Category (Nature, Pets, etc.) | One folder per label (see the list below). `Unsorted` if a photo could not be labelled. |
| Sort by Specific People (Person ID) | `Person0`, `Person1`, ... for photos with one known person, `UnknownPerson` for one face that matches no one else, `Group Photos` for photos with more than one person, and `No People Detected`. |
| Faces Only (Ignore everything else) | Like the mode above, but photos without a face stay where they are. This is the default. |

The folder names in **Broad Category** mode are the full label text, with commas and other punctuation removed:
- `A photo of exactly one person`
- `A photo of two or more people`
- `A photo of a pet or animal`
- `A photo of nature or scenery`
- `A photo of an object building or other subject`

**Face Clustering Strictness (Eps)**, from 0.2 to 0.6 (default 0.4): lower values are stricter, so two different people are less likely to end up in the same folder. A person needs at least two photos to get a `Person` folder.

## Safety notes
- **What it moves:** in a real run (Preview only unticked), photos move into subfolders inside the folder you chose. Nothing is deleted.
- **No overwriting:** if a file with the same name is already in the target folder, the photo gets a new name such as `IMG_1 (1).jpg`, and the log says so.
- **What it writes:** a sidecar file named after the full photo name (`IMG_1.jpg.json`, `IMG_1.png.json`) with the label, its confidence and the people found. It moves together with its photo. Preview mode writes nothing.
- **Undo:** move the photos back by hand from the subfolders and delete the `.json` files if you do not want them.
- **Privacy:** the models run on your PC. The SigLIP 2 model is loaded in offline mode (`HF_HUB_OFFLINE=1`), and the browser page is served only on your own computer.
- **Installs:** the program never installs or removes packages itself. If something is missing, it tells you to run `pip install -r requirements.txt`.

## Troubleshooting
- **"Failed to load offline SigLIP 2 model"**: the model is not saved on your PC yet. Run the command from Install step 4.
- **"PyTorch can't use an NVIDIA GPU"**: you have the CPU version of PyTorch. Run `pip uninstall torch torchvision`, then the first command from Install step 3.
- **"ONNX Runtime can't use the GPU" or "Failed to create CUDAExecutionProvider"**: check that CUDA 12 and cuDNN 9 are installed, restart the PC, and make sure only the GPU version is installed (`pip uninstall onnxruntime`, then `pip install onnxruntime-gpu==1.25.1`).
- **"Some Python packages are missing"**: run `pip install -r requirements.txt` in the project folder.
- **"Error: The path ... is not a valid directory"**: paste the folder path without quote marks.

## How it works
Each photo is labelled with zero-shot image classification (Hugging Face `transformers` pipeline with SigLIP 2) against five text labels. InsightFace (ONNX Runtime, `buffalo_l`) finds faces and computes a face embedding for each one. All embeddings are grouped with DBSCAN (scikit-learn) using cosine distance, so each cluster becomes one person. The plan is shown in a Gradio web page, and files are only moved when preview mode is off.

## License

MIT. See [LICENSE](LICENSE).
