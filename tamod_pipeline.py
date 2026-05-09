import os
import io
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Gradio compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import warnings
import gradio as gr
warnings.filterwarnings('ignore')

# ============================================================
# Device — CPU for inference as per Stage 2A mandate
# ============================================================
DEVICE = 'cpu'

# ============================================================
# 14 Tasks from DVCon Problem Statement
# ============================================================
TASKS = {
    1:  'step on something',
    2:  'sit comfortably',
    3:  'place flowers',
    4:  'get potatoes out of fire',
    5:  'water plant',
    6:  'get lemon out of tea',
    7:  'dig hole',
    8:  'open bottle of beer',
    9:  'open parcel',
    10: 'serve wine',
    11: 'pour sugar',
    12: 'smear butter',
    13: 'extinguish fire',
    14: 'pound carpet'
}

# ============================================================
# Ensemble Prompts — multiple prompts per task, scores averaged
# This significantly improves CLIP accuracy for ambiguous tasks
# ============================================================
TASK_PROMPTS = {
    'step on something': [
        'an object used to step on something',
        'a stool or step or footstool',
        'something a person stands on or steps onto',
        'a platform or step for climbing',
    ],
    'sit comfortably': [
        'an object used to sit comfortably',
        'a chair or sofa or couch for sitting',
        'comfortable seating furniture',
        'a seat or bench to rest on',
    ],
    'place flowers': [
        'an object used to place flowers in',
        'a vase or pot or container for flowers',
        'a decorative vessel for holding flowers',
        'a flower vase or planter',
    ],
    'get potatoes out of fire': [
        'an object used to get potatoes out of fire',
        'tongs or a fork or tool for handling hot food',
        'a kitchen utensil for picking up hot items from fire',
        'oven mitts or tongs for grabbing food from flames',
    ],
    'water plant': [
        'an object used to water a plant',
        'a watering can or hose or spray bottle',
        'a container for pouring water onto plants',
        'a watering device for gardening',
    ],
    'get lemon out of tea': [
        'an object used to get lemon out of tea',
        'a spoon or tongs for removing lemon from a cup',
        'a small utensil for fishing out lemon from a drink',
        'a teaspoon or strainer for a cup of tea',
    ],
    'dig hole': [
        'an object used to dig a hole',
        'a shovel or spade or trowel for digging',
        'a garden tool for digging into soil',
        'a digging implement like a spade or fork',
    ],
    'open bottle of beer': [
        'an object used to open a bottle of beer',
        'a bottle opener or corkscrew',
        'a tool for removing a bottle cap',
        'a beer bottle opener',
    ],
    'open parcel': [
        'an object used to open a parcel or package',
        'scissors or a knife or box cutter',
        'a cutting tool for opening boxes or packages',
        'a blade or scissors for cutting tape',
    ],
    'serve wine': [
        'an object used to serve wine',
        'a wine glass or carafe or decanter',
        'glassware for serving or drinking wine',
        'a wine bottle or wine glass',
    ],
    'pour sugar': [
        'an object used to pour sugar',
        'a sugar bowl or dispenser or spoon',
        'a container for holding and pouring sugar',
        'a sugar shaker or bowl',
    ],
    'smear butter': [
        'an object used to smear or spread butter',
        'a butter knife or spreader',
        'a flat utensil for spreading butter on bread',
        'a knife for spreading food on toast',
    ],
    'extinguish fire': [
        'an object used to extinguish fire',
        'a fire extinguisher or bucket of water or blanket',
        'a device for putting out flames or fire',
        'a fire suppression tool like an extinguisher',
    ],
    'pound carpet': [
        'an object used to pound or beat a carpet',
        'a carpet beater or broom or stick',
        'a tool for beating dust out of a rug or carpet',
        'a paddle or beater for cleaning carpets',
    ],
}


# ============================================================
# Download a test image if no local image is available
# ============================================================
def download_test_image(save_path='test_image.jpg'):
    if os.path.exists(save_path):
        return save_path
    print('Downloading test image...')
    url = 'https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f'Test image saved: {save_path}')
    return save_path


# ============================================================
# Load Models (once at startup)
# ============================================================
def load_models():
    print('Loading YOLOv8n...')
    yolo = YOLO('yolov8n.pt')

    print('Loading CLIP...')
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip.eval()

    print('Both models ready.\n')
    return yolo, clip, processor


# ============================================================
# Step 1: Load Image
# ============================================================
def load_image(source):
    """Load image from local path or URL."""
    if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://')):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(source, timeout=15, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
    elif isinstance(source, np.ndarray):
        image = Image.fromarray(source).convert('RGB')
    else:
        image = Image.open(source).convert('RGB')
    return image


# ============================================================
# Step 2: YOLO Object Detection
# ============================================================
def detect_objects(image, yolo_model, conf_threshold=0.25):
    """Run YOLOv8n. Returns list of {label, confidence, bbox, crop}."""
    results = yolo_model(image, conf=conf_threshold, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            pad = 10
            w, h = image.size
            crop = image.crop((
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(w, x2 + pad),
                min(h, y2 + pad)
            ))

            detections.append({
                'label': label,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'crop': crop
            })

    return detections


# ============================================================
# Step 3: Ensemble CLIP Scoring
# Uses multiple prompts per task and averages the scores
# ============================================================
def compute_clip_scores(detections, task_text, clip_model, processor):
    """
    For each detection crop, compute CLIP similarity against
    ALL prompts for the task and average them into one score.
    """
    if not detections:
        return detections

    crops = [d['crop'] for d in detections]
    prompts = TASK_PROMPTS.get(task_text, [f'an object used to {task_text}'])

    # Encode all prompts at once
    text_inputs = processor(
        text=prompts,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(DEVICE)

    # Encode all crops at once
    image_inputs = processor(
        images=crops,
        return_tensors='pt',
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        text_out = clip_model.get_text_features(**text_inputs)
        image_out = clip_model.get_image_features(**image_inputs)

        # Handle both tensor and object output
        text_features = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
        image_features = image_out if isinstance(image_out, torch.Tensor) else image_out.pooler_output

        # L2 normalize
        text_features = torch.nn.functional.normalize(text_features, dim=-1)  # [num_prompts, D]
        image_features = torch.nn.functional.normalize(image_features, dim=-1)  # [num_crops, D]

        # scores[i][j] = similarity of crop i to prompt j
        scores_matrix = (image_features @ text_features.T).cpu().numpy()  # [num_crops, num_prompts]

        # Average across all prompts for final score
        avg_scores = scores_matrix.mean(axis=1)  # [num_crops]

    for i, d in enumerate(detections):
        d['clip_score'] = float(avg_scores[i])
        d['prompt_scores'] = scores_matrix[i].tolist()  # individual prompt scores for debug

    return detections


# ============================================================
# Step 4: Filter and Rank
# ============================================================
def filter_and_rank(detections, clip_threshold=0.20):
    """Filter below threshold, sort by CLIP score descending."""
    filtered = [d for d in detections if d.get('clip_score', 0) >= clip_threshold]
    return sorted(filtered, key=lambda x: x['clip_score'], reverse=True)


# ============================================================
# Step 5: Render Result Image (returns PIL for Gradio)
# ============================================================
def render_result(image, all_detections, ranked_detections, task_text, top_k=3):
    """
    Draw bounding boxes on image and return as PIL Image.
    Grey dashed = all YOLO detections
    Green/Yellow/Orange = top CLIP matches
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # All detections in grey
    for det in all_detections:
        x1, y1, x2, y2 = det['bbox']
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor='grey', facecolor='none', linestyle='--'
        ))

    # Top-K highlighted
    colors = ['#00FF00', '#FFFF00', '#FFA500']
    for i, det in enumerate(ranked_detections[:top_k]):
        x1, y1, x2, y2 = det['bbox']
        color = colors[i] if i < len(colors) else '#FF0000'
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3 if i == 0 else 2,
            edgecolor=color, facecolor='none'
        ))
        ax.text(
            x1, max(y1 - 5, 10),
            f"#{i+1} {det['label']} ({det['clip_score']:.3f})",
            color=color, fontsize=9, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.7, pad=2)
        )

    top_label = ranked_detections[0]['label'] if ranked_detections else 'No match found'
    top_score = f'{ranked_detections[0]["clip_score"]:.3f}' if ranked_detections else 'N/A'
    ax.set_title(
        f'Task: "{task_text}"  |  Best match: {top_label} (CLIP score: {top_score})',
        fontsize=13, pad=12, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()

    # Convert matplotlib figure to PIL image (for Gradio)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    result_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()

    return result_img


# ============================================================
# Build Results Text Summary
# ============================================================
def build_summary(all_detections, ranked_detections, task_text, top_k=3):
    lines = []
    lines.append(f'Task: "{task_text}"')
    lines.append(f'Prompts used: {len(TASK_PROMPTS.get(task_text, [task_text]))} (ensemble averaged)')
    lines.append(f'YOLO detections: {len(all_detections)}')
    lines.append(f'Objects above threshold: {len(ranked_detections)}')
    lines.append('')
    if ranked_detections:
        lines.append('Top Matches:')
        for i, det in enumerate(ranked_detections[:top_k]):
            lines.append(
                f'  #{i+1}  {det["label"]:20s}  '
                f'CLIP: {det["clip_score"]:.4f}  '
                f'YOLO conf: {det["confidence"]:.3f}'
            )
    else:
        lines.append('No objects passed the similarity threshold.')
        lines.append('Try lowering the threshold or using a different image.')
    return '\n'.join(lines)


# ============================================================
# Core Pipeline Function (called by Gradio)
# ============================================================
def run_pipeline(input_image, task_label, clip_threshold, yolo_model, clip_model, processor):
    """
    Full pipeline: image → YOLO → CLIP ensemble → ranked results.
    Accepts numpy array from Gradio image input.
    """
    if input_image is None:
        return None, 'Please upload an image.'

    task_text = task_label.split(': ', 1)[1]  # extract task text from dropdown label

    # Load image
    image = load_image(input_image)

    # Detect objects
    detections = detect_objects(image, yolo_model)
    if not detections:
        return image, 'YOLO found no objects in this image. Try a different image.'

    # CLIP ensemble scoring
    detections = compute_clip_scores(detections, task_text, clip_model, processor)

    # Filter and rank
    ranked = filter_and_rank(detections, clip_threshold)

    # Render output
    result_img = render_result(image, detections, ranked, task_text)
    summary = build_summary(detections, ranked, task_text)

    return result_img, summary


# ============================================================
# Gradio UI
# ============================================================
def build_gradio_ui(yolo_model, clip_model, processor):

    # Dropdown choices: "Task 1: step on something", etc.
    task_choices = [f'Task {k}: {v}' for k, v in TASKS.items()]

    with gr.Blocks(
        title='TAMOD 2.0 — DVCon India 2026',
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # TAMOD 2.0 — Task-Aware Object Detection
        ### DVCon India 2026 Design Contest | Stage 2A
        **Pipeline:** YOLOv8n (object detection) → CLIP Ensemble (semantic matching)

        Upload an image, select a task, and the system will identify the most relevant object for that task.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label='Input Image',
                    type='numpy',
                    height=350
                )
                task_dropdown = gr.Dropdown(
                    choices=task_choices,
                    value='Task 2: sit comfortably',
                    label='Select Task (1–14)'
                )
                threshold_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.50,
                    value=0.20,
                    step=0.01,
                    label='CLIP Similarity Threshold',
                    info='Lower = more permissive, Higher = stricter matching'
                )
                run_btn = gr.Button('Run Detection', variant='primary', size='lg')

            with gr.Column(scale=2):
                output_image = gr.Image(
                    label='Detection Result',
                    type='pil',
                    height=400
                )
                output_text = gr.Textbox(
                    label='Results Summary',
                    lines=10,
                    interactive=False
                )

        gr.Markdown("""
        ---
        **Color coding:** 🟢 Green = Best match &nbsp;|&nbsp; 🟡 Yellow = 2nd match &nbsp;|&nbsp; 🟠 Orange = 3rd match &nbsp;|&nbsp; ⬜ Grey dashed = all YOLO detections

        **Ensemble prompting:** Each task uses 4 carefully crafted prompts. CLIP scores are averaged across all prompts for higher accuracy.
        """)

        # Example images
        gr.Examples(
            examples=[
                [download_test_image('test_image.jpg'), 'Task 2: sit comfortably', 0.20],
            ],
            inputs=[input_image, task_dropdown, threshold_slider],
            label='Example'
        )

        # Wire up the button
        run_btn.click(
            fn=lambda img, task, thresh: run_pipeline(
                img, task, thresh, yolo_model, clip_model, processor
            ),
            inputs=[input_image, task_dropdown, threshold_slider],
            outputs=[output_image, output_text]
        )

    return demo


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    # Download test image
    download_test_image('test_image.jpg')

    # Load models
    yolo_model, clip_model, clip_processor = load_models()

    # Launch Gradio UI
    print('Starting Gradio UI...')
    demo = build_gradio_ui(yolo_model, clip_model, clip_processor)
    demo.launch(
        server_name='127.0.0.1',
        server_port=7860,
        share=False,        # set True to get a public URL
        inbrowser=True      # auto-opens browser
    )
