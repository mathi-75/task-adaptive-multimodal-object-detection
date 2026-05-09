"""
TAMOD 2.0 - Task-Aware Object Detection Pipeline
DVCon India 2026 | Stage 2A

How to run:
  1. pip install ultralytics transformers torch torchvision pillow matplotlib requests gradio
  2. python tamod_pipeline.py
  3. Browser opens at http://127.0.0.1:7860
"""

import os
import io
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLOWorld
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import warnings
import gradio as gr
warnings.filterwarnings('ignore')

# create the images and results folders next to this script if they don't exist
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(BASE_DIR, 'images')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(IMAGES_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f'Input images  → {IMAGES_DIR}')
print(f'Result images → {RESULTS_DIR}')

# we run everything on CPU as required by Stage 2A
DEVICE = 'cpu'

# the 14 tasks from the DVCon problem statement
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

# for each task, we tell YOLO-World exactly what objects to look for
# this is the key advantage over standard YOLO — we can ask for anything by name
TASK_CLASSES = {
    'step on something':        ['stool', 'step stool', 'footstool', 'ladder', 'box', 'platform', 'chair'],
    'sit comfortably':          ['chair', 'sofa', 'couch', 'armchair', 'bench', 'seat', 'recliner'],
    'place flowers':            ['vase', 'pot', 'flower pot', 'jar', 'bowl', 'container', 'planter'],
    'get potatoes out of fire': ['tongs', 'oven mitt', 'glove', 'fork', 'spatula', 'ladle', 'potato'],
    'water plant':              ['watering can', 'hose', 'spray bottle', 'bucket', 'jug', 'pitcher'],
    'get lemon out of tea':     ['spoon', 'teaspoon', 'tongs', 'fork', 'strainer', 'ladle'],
    'dig hole':                 ['shovel', 'spade', 'trowel', 'pickaxe', 'hoe', 'garden fork'],
    'open bottle of beer':      ['bottle opener', 'corkscrew', 'knife', 'lighter', 'key'],
    'open parcel':              ['scissors', 'knife', 'box cutter', 'blade', 'cutter'],
    'serve wine':               ['wine glass', 'glass', 'carafe', 'decanter', 'bottle', 'cup'],
    'pour sugar':               ['sugar bowl', 'bowl', 'spoon', 'jar', 'dispenser', 'container'],
    'smear butter':             ['knife', 'butter knife', 'spatula', 'spreader', 'spoon'],
    'extinguish fire':          ['fire extinguisher', 'extinguisher', 'bucket', 'blanket', 'hose'],
    'pound carpet':             ['broom', 'stick', 'carpet beater', 'paddle', 'brush', 'mop'],
}

# instead of using one prompt per task, we use 4 different ways to describe each task
# CLIP scores all 4 and we average them — gives much better accuracy than a single prompt
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


def save_input_image(image_array, task_text):
    # saves the uploaded image to images/ with a task-based filename
    slug = re.sub(r'[^a-z0-9]+', '_', task_text.lower()).strip('_')
    existing = [f for f in os.listdir(IMAGES_DIR) if f.startswith(slug)]
    idx = len(existing) + 1
    filename = f'{slug}_{idx:03d}.jpg'
    save_path = os.path.join(IMAGES_DIR, filename)
    Image.fromarray(image_array).convert('RGB').save(save_path)
    print(f'Input saved → images/{filename}')
    return save_path, filename


def save_result_image(result_pil, input_filename):
    # saves the result image to results/ with a matching filename
    base = os.path.splitext(input_filename)[0]
    filename = f'{base}_result.png'
    save_path = os.path.join(RESULTS_DIR, filename)
    result_pil.save(save_path)
    print(f'Result saved → results/{filename}')
    return save_path, filename


def download_test_image():
    # downloads a living room image to use as the default example
    save_path = os.path.join(IMAGES_DIR, 'test_living_room.jpg')
    if os.path.exists(save_path):
        return save_path
    print('Downloading test image...')
    url = 'https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800'
    response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f'Test image saved → {save_path}')
    return save_path


def load_models():
    # loads YOLO-World and CLIP — both download automatically on first run
    # YOLO-World is the open-vocabulary version — it can detect any object by name
    print('Loading YOLO-World (yolov8s-world.pt)...')
    yolo = YOLOWorld('yolov8s-worldv2.pt')

    print('Loading CLIP...')
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip.eval()

    print('Both models ready.\n')
    return yolo, clip, processor


def load_image(source):
    # handles numpy arrays (from Gradio), local paths, and URLs
    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert('RGB')
    if isinstance(source, str) and source.startswith('http'):
        response = requests.get(source, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    return Image.open(source).convert('RGB')


def detect_objects(image, yolo_model, task_text, conf_threshold=0.10):
    # tell YOLO-World what to look for based on the current task
    # this is what makes it open-vocabulary — we set custom classes per task
    classes = TASK_CLASSES.get(task_text, ['object'])
    yolo_model.set_classes(classes)

    results = yolo_model(image, conf=conf_threshold, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            label = yolo_model.names[int(box.cls[0])]

            # small padding around each crop gives CLIP better context
            pad = 10
            w, h = image.size
            crop = image.crop((
                max(0, x1 - pad), max(0, y1 - pad),
                min(w, x2 + pad), min(h, y2 + pad)
            ))

            detections.append({
                'label': label,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'crop': crop
            })

    return detections


def compute_clip_scores(detections, task_text, clip_model, processor):
    # scores each detected crop against all 4 task prompts and averages the result
    if not detections:
        return detections

    crops   = [d['crop'] for d in detections]
    prompts = TASK_PROMPTS.get(task_text, [f'an object used to {task_text}'])

    text_inputs  = processor(text=prompts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    image_inputs = processor(images=crops, return_tensors='pt', padding=True).to(DEVICE)

    with torch.no_grad():
        text_out  = clip_model.get_text_features(**text_inputs)
        image_out = clip_model.get_image_features(**image_inputs)

        # some versions of transformers return an object instead of a tensor — handle both
        text_features  = text_out  if isinstance(text_out,  torch.Tensor) else text_out.pooler_output
        image_features = image_out if isinstance(image_out, torch.Tensor) else image_out.pooler_output

        text_features  = torch.nn.functional.normalize(text_features,  dim=-1)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)

        # each row is one crop, each column is one prompt
        scores_matrix = (image_features @ text_features.T).cpu().numpy()
        avg_scores    = scores_matrix.mean(axis=1)

    for i, d in enumerate(detections):
        d['clip_score']    = float(avg_scores[i])
        d['prompt_scores'] = scores_matrix[i].tolist()

    return detections


def filter_and_rank(detections, clip_threshold=0.20):
    # drops anything below the threshold and sorts the rest by score
    filtered = [d for d in detections if d.get('clip_score', 0) >= clip_threshold]
    return sorted(filtered, key=lambda x: x['clip_score'], reverse=True)


def render_result(image, all_detections, ranked_detections, task_text, top_k=3):
    # draws bounding boxes on the image and returns it as a PIL image for Gradio
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # all detections in grey dashed
    for det in all_detections:
        x1, y1, x2, y2 = det['bbox']
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor='grey', facecolor='none', linestyle='--'
        ))

    # top matches in green, yellow, orange
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

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    result_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()

    return result_img


def build_summary(all_detections, ranked_detections, task_text, input_filename, result_filename, top_k=3):
    lines = [
        f'Task: "{task_text}"',
        f'YOLO-World classes searched: {TASK_CLASSES.get(task_text, [])}',
        f'CLIP prompts used: {len(TASK_PROMPTS.get(task_text, [task_text]))} (ensemble averaged)',
        f'YOLO detections: {len(all_detections)}',
        f'Objects above threshold: {len(ranked_detections)}',
        f'Input saved  : images/{input_filename}',
        f'Result saved : results/{result_filename}',
        '',
    ]

    if ranked_detections:
        lines.append('Top Matches:')
        for i, det in enumerate(ranked_detections[:top_k]):
            lines.append(
                f'  #{i+1}  {det["label"]:20s}  '
                f'CLIP: {det["clip_score"]:.4f}  '
                f'YOLO conf: {det["confidence"]:.3f}'
            )
    else:
        lines.append('No objects passed the threshold.')
        lines.append('Try lowering the threshold slider or using a different image.')

    return '\n'.join(lines)


def run_pipeline(input_image, task_label, clip_threshold, yolo_model, clip_model, processor):
    if input_image is None:
        return None, 'Please upload an image.'

    task_text = task_label.split(': ', 1)[1]

    # save the uploaded image first
    input_path, input_filename = save_input_image(input_image, task_text)
    image = load_image(input_path)

    # run YOLO-World with task-specific classes
    detections = detect_objects(image, yolo_model, task_text)
    if not detections:
        return image, (
            f'YOLO-World found none of these objects: {TASK_CLASSES.get(task_text, [])}\n'
            'Try a different image or lower the threshold.'
        )

    # run CLIP ensemble scoring
    detections = compute_clip_scores(detections, task_text, clip_model, processor)
    ranked     = filter_and_rank(detections, clip_threshold)

    # render and save result
    result_img = render_result(image, detections, ranked, task_text)
    _, result_filename = save_result_image(result_img, input_filename)

    summary = build_summary(detections, ranked, task_text, input_filename, result_filename)

    return result_img, summary


def build_gradio_ui(yolo_model, clip_model, processor):
    task_choices    = [f'Task {k}: {v}' for k, v in TASKS.items()]
    test_image_path = download_test_image()

    with gr.Blocks(title='TAMOD 2.0 — DVCon India 2026', theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # TAMOD 2.0 — Task-Aware Object Detection
        ### DVCon India 2026 Design Contest | Stage 2A
        **Pipeline:** YOLO-World (open-vocabulary detection) → CLIP Ensemble (4 prompts per task)

        Upload an image, pick a task, hit Run. Input images go to `images/`, results go to `results/`.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label='Input Image', type='numpy', height=350)
                task_dropdown = gr.Dropdown(
                    choices=task_choices,
                    value='Task 2: sit comfortably',
                    label='Select Task (1–14)'
                )
                threshold_slider = gr.Slider(
                    minimum=0.05, maximum=0.50, value=0.20, step=0.01,
                    label='CLIP Similarity Threshold',
                    info='Lower = more results, Higher = stricter matching'
                )
                run_btn = gr.Button('Run Detection', variant='primary', size='lg')

            with gr.Column(scale=2):
                output_image = gr.Image(label='Result (saved to results/)', type='pil', height=400)
                output_text  = gr.Textbox(label='Summary', lines=14, interactive=False)

        gr.Markdown("""
        ---
        🟢 Green = best match &nbsp;|&nbsp; 🟡 Yellow = 2nd &nbsp;|&nbsp; 🟠 Orange = 3rd &nbsp;|&nbsp; ⬜ Grey dashed = all YOLO-World detections
        """)

        gr.Examples(
            examples=[[test_image_path, 'Task 2: sit comfortably', 0.20]],
            inputs=[input_image, task_dropdown, threshold_slider],
            label='Quick Example'
        )

        run_btn.click(
            fn=lambda img, task, thresh: run_pipeline(img, task, thresh, yolo_model, clip_model, processor),
            inputs=[input_image, task_dropdown, threshold_slider],
            outputs=[output_image, output_text]
        )

    return demo


if __name__ == '__main__':
    yolo_model, clip_model, clip_processor = load_models()

    print('Starting Gradio UI...')
    demo = build_gradio_ui(yolo_model, clip_model, clip_processor)
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False, inbrowser=True)