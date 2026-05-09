# TAMOD — Task-Aware Object Detection Pipeline

**TAMOD** (Task-Adaptive Multimodal Object Detection) is a two-stage, CPU-based inference pipeline that addresses a core limitation of conventional object detectors: they are task-agnostic, dedicating equal computational resources to every detected class regardless of what the user actually needs.

Instead of simply finding a "cup," TAMOD can identify *"an object used to serve wine"* — evaluating detected objects against a natural language task prompt and returning only what's functionally relevant.

---

## How It Works

The pipeline decouples detection and semantic matching into two stages:

**Stage 1 — Open-Vocabulary Detection (YOLO-World)**
YOLO-World (`yolov8s-worldv2`) accepts a custom list of class names at inference time via `set_classes()`. For each task, the pipeline provides a curated list of task-relevant object names (e.g., for "extinguish fire": `["fire extinguisher", "bucket", "blanket", "hose"]`). This overcomes the fixed 80-class COCO vocabulary of standard YOLO, enabling detection of objects like fire extinguishers, tongs, and trowels. Detections run at a confidence threshold of 0.10, with each region cropped and padded by 10 pixels for context.

**Stage 2 — Semantic Matching (CLIP Ensemble)**
Each cropped region is passed to CLIP ViT-B/32 (`openai/clip-vit-base-patch32`). Rather than a single prompt, an ensemble of 4 hand-crafted task-specific prompts is scored and averaged per crop:

```
S = normalize(f_image) · normalize(f_text)
```

This ensemble strategy consistently outperforms single-prompt matching, especially for ambiguous or compound tasks. Detections scoring below a similarity threshold of **0.20** are discarded; the remainder are ranked and the top match is returned.

---

## Supported Tasks

| # | Task |
|---|------|
| 1 | Step on something |
| 2 | Sit comfortably |
| 3 | Place flowers |
| 4 | Get potatoes out of fire |
| 5 | Water plant |
| 6 | Get lemon out of tea |
| 7 | Dig hole |
| 8 | Open bottle of beer |
| 9 | Open parcel |
| 10 | Serve wine |
| 11 | Pour sugar |
| 12 | Smear butter |
| 13 | Extinguish fire |
| 14 | Pound carpet |

---

## Output Visualisation

Bounding boxes are colour-coded by rank:

- 🟢 **Green** — Best match
- 🟡 **Yellow** — 2nd ranked
- 🟠 **Orange** — 3rd ranked
- ⬜ **Grey dashed** — All YOLO-World detections (pre-threshold)

---

## Project Structure

```
.
├── code/
│   ├── images/               # Input images (auto-created)
│   ├── results/              # Output annotated images (auto-created)
│   └── tamod_pipeline.py     # Main pipeline
├── .gitignore
├── README.md
└── project.code-workspace
```

---

## Installation

```bash
pip install ultralytics transformers torch torchvision pillow matplotlib requests gradio
```

---

## Usage

```bash
python tamod_pipeline.py
```

A Gradio UI will open at `http://127.0.0.1:7860`. Upload an image, select a task from the dropdown, adjust the CLIP similarity threshold if needed, and click **Run Detection**.

- Input images are saved to `images/`
- Annotated result images are saved to `results/`

---

## Key Design Decisions

**CPU-only inference** — `DEVICE = 'cpu'` is hardcoded throughout. All tensor operations run exclusively on CPU, making the pipeline compatible with edge deployment targets lacking a discrete GPU.

**Low YOLO threshold (0.10)** — Safe to use because YOLO-World's search space is already constrained to the task-specific class list. This captures partially visible or occluded objects without excessive false positives.

**CLIP threshold (0.20)** — Empirically determined as the minimum meaningful cosine similarity for zero-shot task matching with CLIP ViT-B/32. Below this, matches are statistically indistinguishable from random noise.

**Ensemble prompting (4 prompts/task)** — Averaging scores across multiple phrasings of each task significantly improves robustness, particularly when detected object labels don't perfectly match the canonical task vocabulary (e.g., a watering can detected as "bucket" is still correctly resolved by CLIP scoring).

---

## Dependencies

| Component | Library / Model |
|-----------|----------------|
| Object detection | `ultralytics` — `yolov8s-worldv2.pt` |
| Semantic matching | `transformers` — `openai/clip-vit-base-patch32` |
| UI | `gradio` |
| Inference device | CPU only |