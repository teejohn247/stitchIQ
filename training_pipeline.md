# StitchIQ — African Design Training Pipeline

### Teaching Your AI Models to Understand African Fashion

> **Version:** 1.1 | **Platform:** Kaggle Notebooks (GPU T4 x2) | **Last updated:** June 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [How Each Model Learns](#2-how-each-model-learns)
3. [Prerequisites](#3-prerequisites)
4. [Step 1 — Collect Your Images](#4-step-1--collect-your-images)
5. [Step 2 — Auto-Caption Images for SDXL Training](#5-step-2--auto-caption-images-for-sdxl-training)
6. [Step 3 — Train the SDXL LoRA](#6-step-3--train-the-sdxl-lora)
7. [Step 4 — Build the CLIP Embedding Index](#7-step-4--build-the-clip-embedding-index)
8. [Step 5 — Load Both into the AI Worker](#8-step-5--load-both-into-the-ai-worker)
9. [Step 6 — Test Your Models](#9-step-6--test-your-models)
10. [Retraining Schedule](#10-retraining-schedule)
11. [Folder Structure](#11-folder-structure)
12. [Troubleshooting](#12-troubleshooting)
13. [Cost Estimates](#13-cost-estimates)

---

## 1. Overview

StitchIQ uses two AI models that benefit directly from African fashion training data:

| Model | What it does | How it learns |
|-------|-------------|---------------|
| **SDXL + LoRA** | Generates images (occasion stylist, alterations) | Fine-tuning on labelled African fashion images |
| **CLIP** | Finds visually similar fabrics in the marketplace | Embedding indexing of all fabric photos |
| **SAM 2** | Segments garment regions | Zero-shot — no training needed |
| **IDM-VTON** | Virtual try-on draping | Zero-shot — no training needed |
| **Claude API** | Text reasoning | Prompt engineering only |

> **Key insight:** You only actively train SDXL and index CLIP. The rest just work out of the box.

---

## 2. How Each Model Learns

### SDXL LoRA Fine-tuning (Path A)
LoRA (Low-Rank Adaptation) is a technique that adds a small adapter layer on top of the existing SDXL model. Instead of retraining the entire 6.9B parameter model (which would take weeks), you train a tiny adapter (~50MB) that steers image generation toward African fashion aesthetics.

```
Before LoRA → SDXL generates generic Western fashion
After LoRA  → SDXL generates Ankara prints, Kente cloth, Aso-oke, lace iro & buba
```

**Trigger word:** You add `africanfashion` to every training caption. At inference time, including this word in any prompt activates the LoRA influence.

### CLIP Embedding Index (Path B)
CLIP converts every image into a 512-dimensional vector (a list of numbers that encodes visual meaning). When a user uploads a fabric photo to search, CLIP encodes that too and finds the mathematically closest matches in your index.

```
User uploads photo of blue Ankara → CLIP encodes it → FAISS finds 5 nearest vectors
→ Returns matching fabrics from your marketplace
```

This is not training — it is **indexing**. It is fast, cheap, and can be rebuilt in 30 minutes whenever you add new fabrics.

---

## 3. Prerequisites

### Kaggle Notebook setup
- Create a notebook at [kaggle.com](https://www.kaggle.com) → **Create → New Notebook**
- **Settings → Accelerator → GPU T4 x2**
- **Settings → Internet → On** (for pip, scraping, Claude API)
- Workspace path: **`/kaggle/working/stitchiq`** (auto-detected by `ai-training/paths.py`)
- Override anywhere with env var: `export STITCHIQ_BASE=/your/path`
- **Save outputs** as a Kaggle Dataset before the session ends (working dir is ephemeral)

### Initialize workspace (first cell)

```python
# Upload or clone ai-training/ into the notebook, then:
!cd /kaggle/working && git clone <your-repo> tailor-app  # or upload files
!python /kaggle/working/tailor-app/ai-training/setup_drive.py
```

Or manually:

```python
import os
base = '/kaggle/working/stitchiq'
for folder in ['raw_images', 'captioned_dataset', 'lora_output', 'clip_index', 'worker_models']:
    os.makedirs(f'{base}/{folder}', exist_ok=True)
print("Folders ready:", os.listdir(base))
```

### Install all dependencies (run once per session)

```bash
# Core ML libraries
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers diffusers accelerate safetensors
pip install -q open-clip-torch faiss-gpu
pip install -q fastapi uvicorn pyngrok cloudinary Pillow requests

# BLIP-2 for auto-captioning
pip install -q salesforce-lavis

# kohya_ss LoRA trainer (clone once per workspace)
KOHYA_PATH = '/kaggle/working/stitchiq/kohya_ss'
if not os.path.exists(KOHYA_PATH):
    !git clone https://github.com/bmaltais/kohya_ss.git {KOHYA_PATH}
    !cd {KOHYA_PATH} && pip install -q -r requirements.txt

print("All dependencies installed")
```

---

## 4. Step 1 — Collect Your Images

### Target quantities

| Purpose | Minimum | Ideal |
|---------|---------|-------|
| SDXL LoRA training | 500 images | 2,000 images |
| CLIP fabric index | 1,000 images | 10,000+ images |

### Source 1 — Web scraping (Pinterest / Instagram)

```python
# Install gallery-dl for ethical scraping of public boards
!pip install -q gallery-dl

# Scrape a public Pinterest board (replace URL with your target board)
# NOTE: Do not limit to these styles. Ensure you scrape a wide variety of African styles, especially Nigerian styles!
!gallery-dl --dest /kaggle/working/stitchiq/raw_images/ankara \
    "https://www.pinterest.com/search/pins/?q=ankara+styles+nigeria"

!gallery-dl --dest /kaggle/working/stitchiq/raw_images/kente \
    "https://www.pinterest.com/search/pins/?q=kente+cloth+ghana"

!gallery-dl --dest /kaggle/working/stitchiq/raw_images/asoke \
    "https://www.pinterest.com/search/pins/?q=aso+oke+fabric+nigerian"
    
!gallery-dl --dest /kaggle/working/stitchiq/raw_images/nigerian_styles \
    "https://www.pinterest.com/search/pins/?q=latest+nigerian+fashion+styles"
```

> **Ethics note:** Only scrape publicly shared images. At launch, add a Terms of Service clause that allows StitchIQ to use vendor-uploaded images for model improvement, with an opt-out option.

### Source 2 — Vendor upload pipeline

Add this endpoint to your Node.js API to flag images for training:

```javascript
// routes/vendor.js — tag new listings for training queue
router.post('/fabrics', authenticate, async (req, res) => {
  const fabric = await Fabric.create({
    ...req.body,
    vendorId: req.user._id,
    taggedForTraining: true,   // new field — marks image for CLIP indexing
    trainingStatus: 'pending'  // pending | indexed | rejected
  });
  
  // Enqueue CLIP indexing job immediately
  await clipIndexQueue.add('index-fabric', {
    fabricId: fabric._id,
    imageUrl: fabric.imageUrls[0]
  });

  res.json(fabric);
});
```

### Source 3 — Public datasets

```python
import requests, zipfile, io

# DeepFashion (filter for African styles after download)
# Dataset page: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
# Note: requires registration — download manually and upload to Drive

# iMaterialist Fashion (Kaggle)
# kaggle competitions download -c imaterialist-fashion-2020-fgvc7
# Filter by fabric_type tags after download

# African Fashion Dataset (HuggingFace — community contributed)
from datasets import load_dataset
ds = load_dataset("fashion-mnist")  # Replace with African fashion dataset when available
```

### Step 1b — Filter and clean images

```python
from PIL import Image
import os, shutil

RAW_DIR     = '/kaggle/working/stitchiq/raw_images'
CLEAN_DIR   = '/kaggle/working/stitchiq/clean_images'
os.makedirs(CLEAN_DIR, exist_ok=True)

MIN_SIZE = 512   # minimum width and height in pixels
MAX_SIZE = 4096

kept, rejected = 0, 0

for root, dirs, files in os.walk(RAW_DIR):
    for fname in files:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
        fpath = os.path.join(root, fname)
        try:
            img = Image.open(fpath)
            w, h = img.size
            # Reject if too small, too large, or not RGB
            if w < MIN_SIZE or h < MIN_SIZE or max(w, h) > MAX_SIZE:
                rejected += 1
                continue
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save to clean directory
            dest = os.path.join(CLEAN_DIR, fname)
            img.save(dest, 'JPEG', quality=95)
            kept += 1
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            rejected += 1

print(f"Kept: {kept} | Rejected: {rejected}")
```

---

## 5. Step 2 — Auto-Caption Images for SDXL Training

SDXL LoRA requires each image paired with a descriptive text caption. You generate these automatically using BLIP-2 and then enrich them with Claude API to add African fashion context.

### 5a — Generate base captions with BLIP-2

```python
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import json, os

# Load BLIP-2 (runs on Colab GPU)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

CLEAN_DIR   = '/kaggle/working/stitchiq/clean_images'
CAPTION_DIR = '/kaggle/working/stitchiq/captioned_dataset'
os.makedirs(CAPTION_DIR, exist_ok=True)

captions = {}

for fname in os.listdir(CLEAN_DIR)[:2000]:  # process up to 2000 images
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    fpath = os.path.join(CLEAN_DIR, fname)
    img   = Image.open(fpath).convert('RGB')

    inputs = processor(images=img, return_tensors="pt").to("cuda", torch.float16)
    ids    = blip_model.generate(**inputs, max_new_tokens=60)
    cap    = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # Copy image to captioned dataset folder
    img.save(os.path.join(CAPTION_DIR, fname))
    captions[fname] = cap
    print(f"{fname} → {cap}")

# Save base captions
with open('/kaggle/working/stitchiq/base_captions.json', 'w') as f:
    json.dump(captions, f, indent=2)

print(f"Captioned {len(captions)} images")
```

### 5b — Enrich captions with Claude API

```python
import anthropic, json

client  = anthropic.Anthropic(api_key="YOUR_CLAUDE_API_KEY")

with open('/kaggle/working/stitchiq/base_captions.json') as f:
    base_captions = json.load(f)

# Fabric type hints — pre-tagged from your folder structure
# We support a wide variety of African styles, especially Nigerian styles.
FABRIC_HINTS = {
    'ankara': 'Ankara wax print fabric',
    'kente':  'Kente cloth',
    'asoke':  'Aso-oke handwoven fabric',
    'lace':   'African lace fabric',
    'adire':  'Adire indigo tie-dye fabric',
    'iro':    'Nigerian iro and buba',
    'agbada': 'Nigerian Agbada robe',
    'dashiki': 'Dashiki print shirt',
    'george': 'George fabric wrapper'
}

enriched_captions = {}

for fname, base_cap in base_captions.items():
    # Guess fabric type from folder name
    fabric_hint = next(
        (hint for key, hint in FABRIC_HINTS.items() if key in fname.lower()),
        'African fashion fabric'
    )

    prompt = f"""You are a fashion expert specialising in African textiles.
Given this base image caption: "{base_cap}"
And this fabric context: "{fabric_hint}"

Write a single enriched caption (max 75 words) for SDXL LoRA training that:
1. Starts with "africanfashion" (the LoRA trigger word)
2. Describes the garment style, silhouette, and neckline
3. Mentions the fabric type and its pattern/colours
4. Notes the occasion suitability (casual, wedding, owambe etc.)
5. Is written in plain descriptive English

Return ONLY the caption text, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    enriched = response.content[0].text.strip()
    enriched_captions[fname] = enriched
    print(f"{fname}:\n  {enriched}\n")

# Save enriched captions
with open('/kaggle/working/stitchiq/enriched_captions.json', 'w') as f:
    json.dump(enriched_captions, f, indent=2)

print(f"Enriched {len(enriched_captions)} captions")
```

### 5c — Write caption .txt files (kohya_ss format)

kohya_ss expects each image to have a matching `.txt` file with the same filename.

```python
import json, os, shutil

CAPTION_DIR = '/kaggle/working/stitchiq/captioned_dataset'

with open('/kaggle/working/stitchiq/enriched_captions.json') as f:
    enriched = json.load(f)

for fname, caption in enriched.items():
    base   = os.path.splitext(fname)[0]
    txt_path = os.path.join(CAPTION_DIR, f"{base}.txt")
    with open(txt_path, 'w') as tf:
        tf.write(caption)

print(f"Written {len(enriched)} caption files to {CAPTION_DIR}")

# Verify — should show pairs of .jpg and .txt files
files = os.listdir(CAPTION_DIR)
images = [f for f in files if f.endswith(('.jpg','.png'))]
txts   = [f for f in files if f.endswith('.txt')]
print(f"Images: {len(images)} | Caption files: {len(txts)}")
```

---

## 6. Step 3 — Train the SDXL LoRA

### Training configuration

```python
import os

KOHYA       = '/kaggle/working/stitchiq/kohya_ss'
DATASET_DIR = '/kaggle/working/stitchiq/captioned_dataset'
OUTPUT_DIR  = '/kaggle/working/stitchiq/lora_output'
LOG_DIR     = '/kaggle/working/stitchiq/lora_logs'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# kohya_ss expects the dataset inside a numbered repeat folder
# Format: {repeat_count}_{trigger_word}
# 10 repeats means each image is seen 10x per epoch
TRAIN_DIR = f'{DATASET_DIR}_train/10_africanfashion'
os.makedirs(TRAIN_DIR, exist_ok=True)

# Symlink (or copy) your captioned images into this folder
import shutil
for f in os.listdir(DATASET_DIR):
    src = os.path.join(DATASET_DIR, f)
    dst = os.path.join(TRAIN_DIR, f)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

print(f"Training folder ready: {len(os.listdir(TRAIN_DIR))} files")
```

### Run the LoRA training

```bash
%%bash
cd /kaggle/working/stitchiq/kohya_ss

accelerate launch \
  --num_cpu_threads_per_process=2 \
  sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="/kaggle/working/stitchiq/captioned_dataset_train" \
  --output_dir="/kaggle/working/stitchiq/lora_output" \
  --output_name="stitchiq_african_v1" \
  --save_model_as="safetensors" \
  --resolution="1024,1024" \
  --train_batch_size=2 \
  --max_train_steps=2000 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=200 \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --save_every_n_steps=500 \
  --logging_dir="/kaggle/working/stitchiq/lora_logs" \
  --caption_extension=".txt" \
  --shuffle_caption \
  --seed=42
```

> **Time estimate on Colab Pro A100:** ~2–3 hours for 2,000 images at 2,000 steps

### Verify the LoRA output

```python
import os

OUTPUT_DIR = '/kaggle/working/stitchiq/lora_output'
lora_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.safetensors')]
print("LoRA files saved:")
for f in lora_files:
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
    print(f"  {f} — {size_mb:.1f} MB")
```

### Quick test — generate a sample image

```python
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load your African fashion LoRA
LORA_PATH = '/kaggle/working/stitchiq/lora_output/stitchiq_african_v1.safetensors'
pipe.load_lora_weights(LORA_PATH)

# Test prompts — the trigger word "africanfashion" activates the LoRA
test_prompts = [
    "africanfashion woman wearing a blue and gold Ankara wrap dress, puff sleeves, owambe party, full length",
    "africanfashion man wearing Kente cloth agbada, traditional Ghanaian style, formal occasion",
    "africanfashion woman in ivory Aso-oke iro and buba, Nigerian wedding guest, elegant",
]

for i, prompt in enumerate(test_prompts):
    image = pipe(
        prompt=prompt,
        negative_prompt="western fashion, jeans, suit, low quality, blurry",
        num_inference_steps=40,
        guidance_scale=7.5
    ).images[0]
    
    out = f'/kaggle/working/stitchiq/test_output_{i+1}.jpg'
    image.save(out)
    print(f"Saved: {out}")

print("Test generation complete — check your Drive for results")
```

---

## 7. Step 4 — Build the CLIP Embedding Index

### 4a — Encode all fabric images

```python
import open_clip
import torch
import faiss
import numpy as np
from PIL import Image
import os, json

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)
model = model.to('cuda').eval()

FABRIC_DIR = '/kaggle/working/stitchiq/clean_images'
INDEX_DIR  = '/kaggle/working/stitchiq/clip_index'
os.makedirs(INDEX_DIR, exist_ok=True)

embeddings  = []
image_ids   = []  # tracks which image each embedding belongs to

image_files = [
    f for f in os.listdir(FABRIC_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"Encoding {len(image_files)} fabric images...")

BATCH_SIZE = 32
for i in range(0, len(image_files), BATCH_SIZE):
    batch_files = image_files[i:i+BATCH_SIZE]
    batch_tensors = []

    for fname in batch_files:
        try:
            img = Image.open(os.path.join(FABRIC_DIR, fname)).convert('RGB')
            batch_tensors.append(preprocess(img))
            image_ids.append(fname)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    if not batch_tensors:
        continue

    batch = torch.stack(batch_tensors).to('cuda')
    with torch.no_grad(), torch.cuda.amp.autocast():
        embs = model.encode_image(batch)
        embs = embs / embs.norm(dim=-1, keepdim=True)  # normalise
    embeddings.append(embs.cpu().numpy())

    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Processed {i + len(batch_files)}/{len(image_files)}")

# Stack all embeddings
all_embeddings = np.vstack(embeddings).astype('float32')
print(f"Embedding matrix shape: {all_embeddings.shape}")  # (N, 512)
```

### 4b — Build and save the FAISS index

```python
# Build FAISS inner-product index (cosine similarity after normalisation)
dim   = all_embeddings.shape[1]   # 512 for ViT-B-32
index = faiss.IndexFlatIP(dim)    # IP = inner product = cosine sim after normalisation
index.add(all_embeddings)

# Save index to Drive
FAISS_PATH  = os.path.join(INDEX_DIR, 'fabric_clip.index')
MAPPING_PATH = os.path.join(INDEX_DIR, 'fabric_id_map.json')

faiss.write_index(index, FAISS_PATH)

# Save the mapping: index position → image filename
with open(MAPPING_PATH, 'w') as f:
    json.dump(image_ids, f, indent=2)

print(f"FAISS index saved: {FAISS_PATH}")
print(f"ID mapping saved:  {MAPPING_PATH}")
print(f"Total vectors in index: {index.ntotal}")
```

### 4c — Test fabric search

```python
def search_similar_fabrics(query_image_path, top_k=5):
    """Given an image path, find the top-k most similar fabrics."""
    img = Image.open(query_image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to('cuda')

    with torch.no_grad(), torch.cuda.amp.autocast():
        query_emb = model.encode_image(tensor)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

    query_np = query_emb.cpu().numpy().astype('float32')
    scores, indices = index.search(query_np, top_k)

    print(f"\nTop {top_k} similar fabrics:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  {rank+1}. {image_ids[idx]}  (similarity: {score:.3f})")

    return [(image_ids[i], float(s)) for s, i in zip(scores[0], indices[0])]

# Test with any image from your dataset
test_img = os.path.join(FABRIC_DIR, image_files[0])
results  = search_similar_fabrics(test_img)
```

---

## 8. Step 5 — Load Both into the AI Worker

This is the FastAPI worker that runs on Colab Pro and is called by your Node.js API via a Bull queue job.

### AI Worker startup (`worker.py`)

```python
# worker.py — Python FastAPI AI Worker
# Runs on Google Colab Pro, exposed via ngrok

import os, json, time, logging
import torch
import faiss
import numpy as np
import open_clip
from PIL import Image
from io import BytesIO
import requests
import cloudinary
import cloudinary.uploader
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stitchiq-worker")

# ── Config ─────────────────────────────────────────────────────────
WORKER_TOKEN   = os.environ.get("WORKER_TOKEN", "your-secret-token")
LORA_PATH      = "/kaggle/working/stitchiq/lora_output/stitchiq_african_v1.safetensors"
FAISS_PATH     = "/kaggle/working/stitchiq/clip_index/fabric_clip.index"
MAPPING_PATH   = "/kaggle/working/stitchiq/clip_index/fabric_id_map.json"

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(title="StitchIQ AI Worker")

# ── Model registry (loaded once on startup) ─────────────────────────
models = {}

@app.on_event("startup")
async def load_models():
    logger.info("Loading models — this takes ~3–5 minutes on first boot...")

    # 1. SDXL + African LoRA
    logger.info("Loading SDXL + African LoRA...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.load_lora_weights(LORA_PATH)
    models["sdxl"] = pipe

    # 2. SDXL Inpainting + LoRA
    logger.info("Loading SDXL Inpainting + LoRA...")
    inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16
    ).to("cuda")
    inpaint_pipe.load_lora_weights(LORA_PATH)
    models["sdxl_inpaint"] = inpaint_pipe

    # 3. CLIP + FAISS index
    logger.info("Loading CLIP and FAISS index...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    models["clip"]            = clip_model.to("cuda").eval()
    models["clip_preprocess"] = clip_preprocess
    models["faiss_index"]     = faiss.read_index(FAISS_PATH)
    with open(MAPPING_PATH) as f:
        models["fabric_id_map"] = json.load(f)

    logger.info("All models loaded and ready!")

# ── Auth middleware ──────────────────────────────────────────────────
def verify_token(x_worker_token: str = Header(...)):
    if x_worker_token != WORKER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid worker token")

# ── Health check ─────────────────────────────────────────────────────
@app.get("/worker/health")
def health():
    return {
        "status":  "ok",
        "gpu":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU",
        "models":  list(models.keys()),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2)
    }

# ── SDXL image generation ────────────────────────────────────────────
class SDXLRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry, western fashion, jeans, suit"
    steps: int = 40
    guidance: float = 7.5
    width: int = 1024
    height: int = 1024

@app.post("/worker/sdxl")
def generate_image(req: SDXLRequest, x_worker_token: str = Header(...)):
    verify_token(x_worker_token)
    # Automatically inject trigger word for African LoRA
    prompt = f"africanfashion {req.prompt}" if "africanfashion" not in req.prompt else req.prompt

    image = models["sdxl"](
        prompt=prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        width=req.width,
        height=req.height
    ).images[0]

    # Upload to Cloudinary and return URL
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    buffer.seek(0)
    result = cloudinary.uploader.upload(buffer, folder="stitchiq/generated")
    return {"url": result["secure_url"], "public_id": result["public_id"]}

# ── SDXL Inpainting (alterations) ────────────────────────────────────
class InpaintRequest(BaseModel):
    image_url: str
    mask_url: str
    prompt: str
    negative_prompt: str = "low quality, blurry"
    steps: int = 40

@app.post("/worker/sdxl-inpaint")
def inpaint_image(req: InpaintRequest, x_worker_token: str = Header(...)):
    verify_token(x_worker_token)

    def url_to_pil(url):
        r = requests.get(url)
        return Image.open(BytesIO(r.content)).convert("RGB").resize((1024, 1024))

    image = url_to_pil(req.image_url)
    mask  = url_to_pil(req.mask_url).convert("L")

    prompt = f"africanfashion {req.prompt}" if "africanfashion" not in req.prompt else req.prompt

    result_image = models["sdxl_inpaint"](
        prompt=prompt,
        negative_prompt=req.negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=req.steps
    ).images[0]

    buffer = BytesIO()
    result_image.save(buffer, format="JPEG", quality=92)
    buffer.seek(0)
    res = cloudinary.uploader.upload(buffer, folder="stitchiq/alterations")
    return {"url": res["secure_url"]}

# ── CLIP fabric search ────────────────────────────────────────────────
class CLIPSearchRequest(BaseModel):
    image_url: str
    top_k: int = 5

@app.post("/worker/clip-search")
def clip_search(req: CLIPSearchRequest, x_worker_token: str = Header(...)):
    verify_token(x_worker_token)

    r   = requests.get(req.image_url)
    img = Image.open(BytesIO(r.content)).convert("RGB")
    tensor = models["clip_preprocess"](img).unsqueeze(0).to("cuda")

    with torch.no_grad(), torch.cuda.amp.autocast():
        emb = models["clip"].encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    query = emb.cpu().numpy().astype("float32")
    scores, indices = models["faiss_index"].search(query, req.top_k)

    results = [
        {"filename": models["fabric_id_map"][int(i)], "score": float(s)}
        for s, i in zip(scores[0], indices[0])
    ]
    return {"matches": results}

# ── Run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Start the worker with ngrok tunnel (Colab cell)

```python
import subprocess, os
from pyngrok import ngrok

# Set your ngrok auth token (free at ngrok.com)
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

# Kill any existing workers
os.system("pkill -f uvicorn")

# Start FastAPI in background
subprocess.Popen(
    ["python", "worker.py"],
    env={**os.environ, "WORKER_TOKEN": "your-secret-token"}
)

import time; time.sleep(5)  # wait for server to boot

# Open public tunnel
tunnel = ngrok.connect(8000, "http")
WORKER_URL = tunnel.public_url

print(f"\n{'='*50}")
print(f"  AI Worker URL: {WORKER_URL}")
print(f"  Health check:  {WORKER_URL}/worker/health")
print(f"{'='*50}")
print("\nCopy the Worker URL above into your Node.js .env:")
print(f"  AI_WORKER_URL={WORKER_URL}")
```

### Update your Node.js `.env`

```bash
# .env (Node.js API server)
AI_WORKER_URL=https://xxxx-xx-xxx.ngrok.io   # paste from Colab output
WORKER_TOKEN=your-secret-token                 # must match worker.py
```

---

## 9. Step 6 — Test Your Models

### Test from Node.js (Bull worker call)

```javascript
// workers/aiWorker.js
const axios  = require('axios');
const dotenv = require('dotenv');
dotenv.config();

const WORKER_URL   = process.env.AI_WORKER_URL;
const WORKER_TOKEN = process.env.WORKER_TOKEN;

const headers = { 'x-worker-token': WORKER_TOKEN };

// Test 1: Health check
async function healthCheck() {
    const res = await axios.get(`${WORKER_URL}/worker/health`, { headers });
    console.log('Worker health:', res.data);
}

// Test 2: Generate an occasion stylist image
async function testSDXL() {
    const res = await axios.post(`${WORKER_URL}/worker/sdxl`, {
        prompt: "woman wearing a red and gold Ankara ballgown, Nigerian owambe party, full length, elegant",
        steps: 30
    }, { headers });
    console.log('Generated image URL:', res.data.url);
}

// Test 3: Fabric search by image
async function testCLIPSearch(imageUrl) {
    const res = await axios.post(`${WORKER_URL}/worker/clip-search`, {
        image_url: imageUrl,
        top_k: 5
    }, { headers });
    console.log('Similar fabrics:', res.data.matches);
}

(async () => {
    await healthCheck();
    await testSDXL();
})();
```

---

## 10. Retraining Schedule

| Trigger | Action | Time on Colab Pro | Cost |
|---------|--------|-------------------|------|
| 500 new vendor fabric uploads | Rebuild CLIP index only | ~30 minutes | ~$1 Colab credit |
| 1,000 new style images added | Retrain LoRA from checkpoint | ~1.5 hours | ~$3 Colab credit |
| User satisfaction drops below 4.0 | Full LoRA retrain on cleaned dataset | ~3 hours | ~$6 Colab credit |
| New fabric type added (e.g. Kanga) | Add 100+ tagged images, fine-tune LoRA | ~2 hours | ~$4 Colab credit |
| Monthly (routine) | Rebuild CLIP index with all new fabrics | ~45 minutes | ~$1.50 Colab credit |

### Automate CLIP index rebuilds (run monthly)

```python
# rebuild_clip_index.py — run in Colab to refresh the index

import open_clip, faiss, torch, numpy as np, json, os
from PIL import Image
import cloudinary, cloudinary.api

# Fetch all fabric image URLs from your MongoDB
import pymongo
client = pymongo.MongoClient(os.environ["MONGODB_URI"])
fabrics = list(client.stitchiq.fabrics.find({}, {"imageUrls": 1, "_id": 1}))

print(f"Indexing {len(fabrics)} fabrics from MongoDB...")

# Re-run embedding encoding (same as Step 4a above)
# ... [same encoding loop] ...

# Save and replace old index
faiss.write_index(index, FAISS_PATH)
print("CLIP index rebuilt successfully")
```

---

## 11. Folder Structure

```
/kaggle/working/stitchiq/
├── raw_images/                    # Original scraped/uploaded images
│   ├── ankara/
│   ├── kente/
│   ├── asoke/
│   └── lace/
├── clean_images/                  # Filtered, resized images ready for use
├── captioned_dataset/             # Images + .txt caption pairs for LoRA training
├── captioned_dataset_train/       # kohya_ss training folder
│   └── 10_africanfashion/         # 10x repeat, trigger word = africanfashion
├── lora_output/                   # Trained LoRA weights
│   ├── stitchiq_african_v1.safetensors
│   └── stitchiq_african_v1-500.safetensors   # checkpoint at step 500
├── lora_logs/                     # TensorBoard training logs
├── clip_index/                    # CLIP search index
│   ├── fabric_clip.index          # FAISS binary index
│   └── fabric_id_map.json         # [index_position] → filename
├── base_captions.json             # Raw BLIP-2 captions
├── enriched_captions.json         # Claude-enriched captions
└── test_output_*.jpg              # Test generation outputs
```

---

## 12. Troubleshooting

### Colab session disconnected mid-training

```python
# kohya_ss saves checkpoints every 500 steps — resume from latest
# Add --resume_from_checkpoint flag pointing to last checkpoint folder:

# In your training command, add:
# --resume_from_checkpoint="/kaggle/working/stitchiq/lora_output/stitchiq_african_v1-1500"
```

### Out of VRAM during training

```bash
# Reduce batch size and enable gradient checkpointing (already in command)
# Change in training command:
--train_batch_size=1          # reduce from 2 to 1
--gradient_accumulation_steps=4   # accumulate to compensate
```

### ngrok URL changes after session reconnect

```python
# In your Node.js API, add a health-check loop that reads the current URL from Redis
# On Colab reconnect, the notebook updates Redis with the new ngrok URL

import redis, os
r = redis.from_url(os.environ["REDIS_URL"])

# In Colab — run this after getting the new tunnel URL:
r.set("ai_worker_url", WORKER_URL)
r.set("ai_worker_url_updated_at", str(time.time()))
print(f"Worker URL updated in Redis: {WORKER_URL}")
```

```javascript
// In Node.js Bull worker — always fetch current URL from Redis before calling worker
const workerUrl = await redis.get('ai_worker_url');
const res = await axios.post(`${workerUrl}/worker/sdxl`, payload, { headers });
```

### Poor image quality from SDXL LoRA

- Increase training steps to 3,000–4,000
- Check that caption `.txt` files are present for every image
- Remove low-quality images from dataset (blurry, dark, watermarked)
- Reduce LoRA strength at inference: `pipe.set_adapters(["default"], adapter_weights=[0.7])`

### CLIP returning irrelevant fabric matches

- Ensure all fabric images are properly cropped to show the fabric clearly (no background clutter)
- Rebuild the index after cleaning — remove images where the fabric is < 50% of the frame
- Increase top_k and re-rank results by MongoDB fabric type filter

---

## 13. Cost Estimates

| Task | Colab Pro GPU time | Estimated cost |
|------|-------------------|----------------|
| Initial LoRA training (2,000 images, 2,000 steps) | ~3 hours A100 | ~$6 of Colab credits |
| CLIP index build (5,000 images) | ~45 minutes | ~$1.50 |
| Monthly CLIP index rebuild | ~45 minutes | ~$1.50 |
| Quarterly LoRA retrain | ~3 hours | ~$6 |
| **Total Year 1 training cost** | — | **~$40–60** |
| Colab Pro+ subscription | — | **$50/month** |
| **Total AI infrastructure (Year 1)** | — | **~$650 ($50×12 + training)** |

> Compare to fully paid API route: **~$6,000–10,000 Year 1**

---

*StitchIQ AI Training Pipeline — v1.0 — May 2026*
*For questions or contributions open an issue on the StitchIQ GitHub repository.*
