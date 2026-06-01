import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import json
import re
import time
import logging
from io import BytesIO
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stitchiq-worker")

# ── Config ─────────────────────────────────────────────────────────
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "your-secret-token")
MOCK_ML_MODELS = os.environ.get("MOCK_ML_MODELS", "true").lower() == "true"


def _should_load_heavy_models():
    """SDXL/CLIP need NVIDIA GPU. auto=skip when no CUDA (local Mac-friendly)."""
    pref = os.environ.get("LOAD_HEAVY_MODELS", "auto").lower()
    if pref in ("0", "false", "no"):
        return False
    if pref in ("1", "true", "yes"):
        return True
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


LOAD_HEAVY_MODELS = _should_load_heavy_models()


def _stitchiq_base():
    override = os.environ.get("STITCHIQ_BASE")
    if override:
        return override.rstrip("/")
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/stitchiq"
    return os.path.expanduser("~/stitchiq")

_BASE        = _stitchiq_base()
LORA_PATH    = os.path.join(_BASE, "lora_output/stitchiq_african_v1.safetensors")
FAISS_PATH   = os.path.join(_BASE, "clip_index/fabric_clip.index")
MAPPING_PATH = os.path.join(_BASE, "clip_index/fabric_id_map.json")

if not MOCK_ML_MODELS and LOAD_HEAVY_MODELS:
    import torch
    import faiss
    import numpy as np
    import open_clip
    from PIL import Image
    import cloudinary
    import cloudinary.uploader
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline

    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "mock"),
        api_key=os.environ.get("CLOUDINARY_API_KEY", "mock"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET", "mock")
    )
elif not MOCK_ML_MODELS:
    logger.info(
        "Real API mode without SDXL/CLIP (no CUDA or LOAD_HEAVY_MODELS=false). "
        "Pattern analysis + sketches use Gemini/Anthropic."
    )
else:
    logger.info("Running in MOCK_ML_MODELS mode. Heavy PyTorch dependencies will not be loaded.")

# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(title="StitchIQ AI Worker")

def load_image(url):
    from PIL import Image
    if url.startswith("data:image"):
        import base64
        try:
            header, encoded = url.split(",", 1)
            data = base64.b64decode(encoded)
            return Image.open(BytesIO(data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")
    else:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return Image.open(BytesIO(r.content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")

# ── Model registry (loaded once on startup) ─────────────────────────
models = {}

def _load_gemini_client():
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        logger.warning("GEMINI_API_KEY not set — pattern analysis unavailable")
        return
    try:
        from google import genai as google_genai
        models["gemini_client"] = google_genai.Client(api_key=gemini_key)
        logger.info("Gemini 2.5 Flash loaded")
    except Exception as e:
        logger.error(f"Failed to load Gemini: {e}")


@app.on_event("startup")
async def load_models():
    if MOCK_ML_MODELS:
        logger.info("Mock models loaded instantly.")
        models["sdxl"] = "mock_sdxl"
        models["sdxl_inpaint"] = "mock_sdxl_inpaint"
        models["clip"] = "mock_clip"
        return

    _load_gemini_client()

    if not LOAD_HEAVY_MODELS:
        logger.info("Light mode ready (Gemini/Anthropic). SDXL/CLIP not loaded.")
        return

    logger.info("Loading models — this takes ~3–5 minutes on first boot...")

    # 1. SDXL + African LoRA
    logger.info("Loading SDXL + African LoRA (Optimized fp16 variant)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    if os.path.exists(LORA_PATH):
        pipe.load_lora_weights(LORA_PATH)
    else:
        logger.warning(f"LoRA weights not found at {LORA_PATH}, using base model")
        
    # Enable attention/VAE slicing to keep memory footprint low
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    models["sdxl"] = pipe

    # 2. SDXL Inpainting + LoRA
    logger.info("Loading SDXL Inpainting + LoRA (Optimized fp16 variant)...")
    inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    if os.path.exists(LORA_PATH):
        inpaint_pipe.load_lora_weights(LORA_PATH)
        
    # Enable attention/VAE slicing to keep memory footprint low
    inpaint_pipe.enable_attention_slicing()
    inpaint_pipe.enable_vae_slicing()
    models["sdxl_inpaint"] = inpaint_pipe

    # 3. CLIP + FAISS index
    logger.info("Loading CLIP and FAISS index...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    models["clip"] = clip_model.to("cuda").eval()
    models["clip_preprocess"] = clip_preprocess
    
    if os.path.exists(FAISS_PATH):
        models["faiss_index"] = faiss.read_index(FAISS_PATH)
    else:
        logger.warning(f"FAISS index not found at {FAISS_PATH}")
        
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH) as f:
            models["fabric_id_map"] = json.load(f)
    else:
        logger.warning(f"ID mapping not found at {MAPPING_PATH}")

    logger.info("All models loaded and ready!")

def activate_model(model_name: str):
    if MOCK_ML_MODELS or not LOAD_HEAVY_MODELS:
        return

    import torch
    logger.info(f"Activating model '{model_name}' on GPU...")
    
    if model_name == "sdxl":
        active_key = "sdxl"
        passive_key = "sdxl_inpaint"
    elif model_name == "sdxl_inpaint":
        active_key = "sdxl_inpaint"
        passive_key = "sdxl"
    else:
        return

    active_pipe = models[active_key]
    passive_pipe = models[passive_key]

    # Offload the passive pipeline to CPU if it is on CUDA
    try:
        if hasattr(passive_pipe, "unet") and passive_pipe.unet.device.type == "cuda":
            logger.info(f"Offloading passive model '{passive_key}' to CPU...")
            passive_pipe.to("cpu")
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Failed to offload passive model: {e}")

    # Move active pipeline to GPU if not already there
    try:
        if hasattr(active_pipe, "unet") and active_pipe.unet.device.type != "cuda":
            logger.info(f"Loading active model '{active_key}' onto GPU...")
            active_pipe.to("cuda")
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Failed to load active model: {e}")

# ── Auth middleware ──────────────────────────────────────────────────
def verify_token(x_worker_token: str = Header(...)):
    if x_worker_token != WORKER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid worker token")

# ── Health check ─────────────────────────────────────────────────────
@app.get("/worker/health")
def health():
    gpu_name = "no GPU"
    vram_gb = 0.0
    
    if not MOCK_ML_MODELS and LOAD_HEAVY_MODELS:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)
        
    if MOCK_ML_MODELS:
        mode = "mock"
    elif not LOAD_HEAVY_MODELS:
        mode = "light"
    else:
        mode = "production"
    return {
        "status": "ok",
        "mode": mode,
        "gpu": gpu_name,
        "models": list(models.keys()),
        "vram_gb": vram_gb
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
    
    if MOCK_ML_MODELS:
        time.sleep(2)
        url = "https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg"
        prompt_lower = req.prompt.lower()
        if "ankara" in prompt_lower or "nigerian" in prompt_lower:
            url = "https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg"
        return {
            "url": url, 
            "public_id": "mock_id",
            "mocked": True
        }

    if "sdxl" not in models:
        raise HTTPException(
            status_code=503,
            detail="SDXL not loaded. Use a CUDA machine with LOAD_HEAVY_MODELS=true, or test pattern_analysis in light mode.",
        )

    prompt_lower = req.prompt.lower()
    is_armless = True
    sleeve_keywords = ["sleeve", "sleeves", "long-sleeve", "puff-sleeve", "arm", "arms", "hand", "hands", "shoulder loop"]
    if any(kw in prompt_lower for kw in sleeve_keywords) and "armless" not in prompt_lower and "sleeveless" not in prompt_lower:
        is_armless = False

    if is_armless:
        mannequin_details = "on an armless tailors canvas mannequin with a beige linen torso, polished gold neck cap metal finial, elegant tailor-studio background"
    else:
        mannequin_details = "on a premium tailors mannequin with articulated wood arms and hands, elegant tailor-studio background"

    base_prompt = f"africanfashion {req.prompt}" if "africanfashion" not in req.prompt else req.prompt
    prompt = f"{base_prompt}, {mannequin_details}"

    activate_model("sdxl")
    image = models["sdxl"](
        prompt=prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        width=req.width,
        height=req.height
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    buffer.seek(0)
    result = cloudinary.uploader.upload(buffer, folder="stitchiq/generated")
    
    if not MOCK_ML_MODELS:
        torch.cuda.empty_cache()
        
    return {"url": result["secure_url"], "public_id": result["public_id"]}

# ── SDXL Inpainting (alterations) ────────────────────────────────────
class InpaintRequest(BaseModel):
    image_url: str
    mask_url: str = ""
    prompt: str
    negative_prompt: str = "low quality, blurry"
    steps: int = 40

@app.post("/worker/sdxl-inpaint")
def inpaint_image(req: InpaintRequest, x_worker_token: str = Header(...)):
    verify_token(x_worker_token)

    if MOCK_ML_MODELS:
        time.sleep(2)
        return {
            "url": "https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg",
            "mocked": True
        }

    if "sdxl_inpaint" not in models:
        raise HTTPException(status_code=503, detail="SDXL inpaint not loaded (light mode).")

    def url_to_pil(url):
        return load_image(url).convert("RGB").resize((1024, 1024))

    image = url_to_pil(req.image_url)
    if req.mask_url:
        mask = url_to_pil(req.mask_url).convert("L")
    else:
        mask = Image.new("L", (1024, 1024), 255)

    # Determine if it's armless or has sleeves
    prompt_lower = req.prompt.lower()
    is_armless = True
    
    sleeve_keywords = ["sleeve", "sleeves", "long-sleeve", "puff-sleeve", "arm", "arms", "hand", "hands", "shoulder loop"]
    if any(kw in prompt_lower for kw in sleeve_keywords) and "armless" not in prompt_lower and "sleeveless" not in prompt_lower:
        is_armless = False
        
    if is_armless:
        mannequin_details = "on an armless tailors canvas mannequin with a beige linen torso, polished gold neck cap metal finial"
    else:
        mannequin_details = "on a premium tailors mannequin with articulated wood arms and hands"
        
    base_prompt = f"africanfashion {req.prompt}" if "africanfashion" not in req.prompt else req.prompt
    prompt = f"{base_prompt}, {mannequin_details}"

    activate_model("sdxl_inpaint")
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
    
    if not MOCK_ML_MODELS:
        torch.cuda.empty_cache()
        
    return {"url": res["secure_url"]}

# ── CLIP fabric search ────────────────────────────────────────────────
class CLIPSearchRequest(BaseModel):
    image_url: str
    top_k: int = 5

@app.post("/worker/clip-search")
def clip_search(req: CLIPSearchRequest, x_worker_token: str = Header(...)):
    verify_token(x_worker_token)

    if MOCK_ML_MODELS:
        return {
            "matches": [
                {"filename": "ankara_mock_1.jpg", "score": 0.95},
                {"filename": "ankara_mock_2.jpg", "score": 0.88}
            ],
            "mocked": True
        }

    if "clip" not in models:
        raise HTTPException(status_code=503, detail="CLIP/FAISS not loaded (light mode).")

    import torch
    img = load_image(req.image_url).convert("RGB")
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

# ── Pattern Analysis ─────────────────────────────────────────────────
class PatternAnalysisRequest(BaseModel):
    image_url: str

def parse_json_from_text(raw: str) -> dict:
    """
    Five-strategy JSON extractor.
    Handles markdown fences, preamble, truncation, trailing commas.
    """
    if not raw or not raw.strip():
        raise ValueError("Model returned empty response")

    original = raw

    # ── Strategy 1: strip markdown fences and parse directly ──────
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: bracket-counter to find outermost { } ─────────
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i, ch in enumerate(cleaned[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end != -1:
            fragment = cleaned[start:end]
            # remove trailing commas
            fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                pass

    # ── Strategy 3: truncated JSON — close open brackets ──────────
    # When Gemini cuts off mid-JSON, close the open brackets
    if start != -1:
        fragment = cleaned[start:]
        # remove trailing commas and incomplete last line
        lines = fragment.rstrip().split("\n")
        # drop last line if it looks incomplete (no closing quote/bracket)
        last = lines[-1].strip()
        if last and not any(last.endswith(c) for c in ('"', ']', '}', ',')):
            lines = lines[:-1]
        fragment = "\n".join(lines)
        # remove trailing commas
        fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
        # count unclosed brackets and close them
        open_braces   = fragment.count("{") - fragment.count("}")
        open_brackets = fragment.count("[") - fragment.count("]")
        # close any open string first
        if fragment.count('"') % 2 != 0:
            fragment += '"'
        fragment += "]" * max(0, open_brackets)
        fragment += "}" * max(0, open_braces)
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            pass

    # ── Strategy 4: extract just the flat spec fields ──────────────
    # If assembly/draft sections are corrupted, return what we can
    flat_fields = ["style_name", "difficulty", "silhouette",
                   "neckline", "sleeves", "closure", "fabric", "detail"]
    result = {}
    for field in flat_fields:
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', cleaned)
        if m:
            result[field] = m.group(1)

    if len(result) >= 4:          # got at least half the flat fields
        result.setdefault("construction_notes", [
            "See tailor for construction details"
        ])
        result.setdefault("assembly_steps", [
            {"step": 1, "title": "Consult tailor",
             "detail": "Full assembly instructions not parsed — retry."}
        ])
        result.setdefault("draft_cuts", [])
        return result

    # ── Strategy 5: give up with a clear error ────────────────────
    raise ValueError(
        f"Could not parse JSON after 4 strategies.\n"
        f"Raw response (first 400 chars):\n{original[:400]}"
    )

def analyse_image(img) -> dict:
    VISION_PROMPT = """You are a professional fashion pattern maker.
Look at this garment photo carefully.

Return ONLY raw JSON. Start with { and end with }.
No markdown. No ```json. No explanation before or after. Just the JSON object.

{
  "style_name": "descriptive name of what you see",
  "difficulty": "Beginner or Intermediate or Advanced or Expert",
  "silhouette": "exact silhouette you see",
  "neckline": "exact neckline you see",
  "sleeves": "describe or say Sleeveless",
  "closure": "closure visible",
  "fabric": "fabric and texture you see",
  "detail": "trim or structural detail",
  "construction_notes": ["note 1", "note 2", "note 3"],
  "assembly_steps": [
    {"step": 1, "title": "title", "detail": "instruction"},
    {"step": 2, "title": "title", "detail": "instruction"},
    {"step": 3, "title": "title", "detail": "instruction"},
    {"step": 4, "title": "title", "detail": "instruction"}
  ],
  "draft_cuts": [
    {"label": "FRONT BODICE", "note": "cut note", "seam": "seam note"},
    {"label": "BACK BODICE",  "note": "cut note", "seam": "seam note"},
    {"label": "SKIRT PANEL",  "note": "cut note", "seam": "seam note"}
  ]
}"""

    if "gemini_client" not in models:
        raise HTTPException(status_code=503, detail="No vision model loaded")

    logger.info("Using Gemini 2.5 Flash vision for pattern analysis")

    from google.genai import types
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    image_bytes = buf.getvalue()

    response = models["gemini_client"].models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=VISION_PROMPT)
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,          # ← was 1500, now 8192
            thinking_config=types.ThinkingConfig(
                thinking_budget=0            # ← disable thinking overhead
            )
        )
    )

    raw = response.text
    logger.info(f"Gemini raw (first 300 chars): {raw[:300]}")
    return parse_json_from_text(raw)

def analyse_image_minimal(img) -> dict:
    """
    Fallback: ask only for flat string fields.
    Used when the full JSON prompt causes truncation.
    """
    MINIMAL_PROMPT = """Look at this garment. Return ONLY this JSON, nothing else:
{
  "style_name": "name",
  "difficulty": "Beginner or Intermediate or Advanced or Expert",
  "silhouette": "silhouette type",
  "neckline": "neckline type",
  "sleeves": "sleeves description or Sleeveless",
  "closure": "closure type",
  "fabric": "fabric type",
  "detail": "main detail",
  "construction_notes": ["note 1", "note 2"],
  "assembly_steps": [
    {"step": 1, "title": "Main step", "detail": "Cut and assemble main panels"},
    {"step": 2, "title": "Join seams", "detail": "Join all major seams"}
  ],
  "draft_cuts": [
    {"label": "FRONT", "note": "Cut 1", "seam": "1.5cm"},
    {"label": "BACK",  "note": "Cut 2", "seam": "1.5cm"}
  ]
}"""

    if "gemini_client" not in models:
        raise HTTPException(status_code=503, detail="No vision model loaded")

    from google.genai import types
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    image_bytes = buf.getvalue()

    response = models["gemini_client"].models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=MINIMAL_PROMPT)
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    return parse_json_from_text(response.text)

@app.post("/worker/pattern-analysis")
def pattern_analysis(req: PatternAnalysisRequest, x_worker_token: str = Header(...)):
    verify_token(x_worker_token)

    if MOCK_ML_MODELS:
        time.sleep(1)
        return {
            "specs": {
                "style_name": "Black Column Gown with Painted Chiffon Panel",
                "difficulty": "Advanced",
                "silhouette": "Column / Mermaid",
                "neckline": "One-shoulder asymmetric with structural loop",
                "sleeves": "Sleeveless",
                "closure": "Hidden back zipper, centre placement",
                "fabric": "Matte black bodycon blend with printed chiffon panel",
                "detail": "Abstract watercolour painted chiffon side panel, floor-length train"
            },
            "construction_notes": [
                "Spiral boning required in bodice for column structure",
                "Chiffon panel attached at left princess seam — allow free movement at hem",
                "Invisible zip from mid-back to waist, matched to fabric grain"
            ],
            "assembly_steps": [
                {"step": 1, "title": "Bodice Construction",
                 "detail": "Cut front and back bodice panels on grain. Insert boning channels along princess seams. Press flat."},
                {"step": 2, "title": "Chiffon Panel Attachment",
                 "detail": "Baste chiffon panel to left princess seam. French seam finish. Allow panel to flow freely below hip."},
                {"step": 3, "title": "Skirt Assembly",
                 "detail": "Sew front and back skirt panels. Join at side seams. Clip curves at hip for smooth fit."},
                {"step": 4, "title": "Zipper & Finish",
                 "detail": "Baste invisible zip centred on back seam from waist to mid-back. Attach shoulder structural loop last."}
            ],
            "draft_cuts": [
                {"label": "FRONT BODICE", "note": "Cut 1 on fold", "seam": "1.5cm seam allowance included"},
                {"label": "BACK BODICE", "note": "Cut 2 (mirror)", "seam": "Match centre back grainline"},
                {"label": "CHIFFON PANEL", "note": "Cut 1 on bias", "seam": "French seam — 2cm allowance"}
            ],
            "mocked": True
        }

    img = load_image(req.image_url).convert("RGB")

    # ── Attempt 1: full analysis ───────────────────────────────────
    try:
        analysis = analyse_image(img)
    except Exception as e1:
        logger.warning(f"Full analysis failed ({e1}), retrying with minimal prompt...")

        # ── Attempt 2: minimal flat-fields-only prompt ─────────────
        try:
            analysis = analyse_image_minimal(img)
        except Exception as e2:
            logger.error(f"Minimal analysis also failed: {e2}")
            raise HTTPException(
                status_code=500,
                detail=f"Vision analysis failed: {e2}"
            )

    spec_keys = {"style_name", "difficulty", "silhouette", "neckline",
                 "sleeves", "closure", "fabric", "detail"}
    specs = {k: v for k, v in analysis.items() if k in spec_keys}

    return {
        "specs":              specs,
        "construction_notes":  analysis.get("construction_notes", []),
        "assembly_steps":      analysis.get("assembly_steps", []),
        "draft_cuts":          analysis.get("draft_cuts", [])
    }

# ── Fabric Price Surfer ──────────────────────────────────────────────
class FabricPriceRequest(BaseModel):
    prompt: str
    fabric_name: str
    base_price: float

@app.post("/worker/fabric-price")
def fabric_price(req: FabricPriceRequest, x_worker_token: str = Header(...)):
    verify_token(x_worker_token)
    prompt_lower = req.prompt.lower()
    variance = 0.0
    
    if "premium" in prompt_lower or "lace" in prompt_lower or "silk" in prompt_lower:
        variance += 0.07
    if "embroidery" in prompt_lower or "beading" in prompt_lower or "stones" in prompt_lower:
        variance += 0.05
    if "simple" in prompt_lower or "casual" in prompt_lower:
        variance -= 0.04
        
    import random
    variance += random.uniform(-0.03, 0.04)
    
    final_price = round(req.base_price * (1 + variance) / 100) * 100
    return {"pricePerYd": max(1000, final_price), "variance": variance}


# ── Pattern piece SVG sketch generator ───────────────────────────────
class SketchRequest(BaseModel):
    draft_cuts: list        # the draft_cuts array from Gemini analysis
    silhouette: str = ""    # e.g. "Mermaid/Trumpet" — helps Claude draw accurately
    fabric: str = ""        # e.g. "stretch crepe" — affects drape notes

def _build_piece_svg(p: dict) -> str:
    """
    Renders a professional technical pattern piece in white-on-dark style,
    matching industry standard pattern drafting sheets.
    """
    W, H   = 220, 260
    BG     = "#0a1a0f"
    LINE   = "#ffffff"
    SEAM   = "#ffffff"
    DIM    = "rgba(255,255,255,0.45)"
    GOLD   = "#D4A843"
    GRID   = "rgba(255,255,255,0.06)"

    label    = p.get("label", "PIECE")
    is_bias  = p.get("is_bias", False)
    outer    = p.get("outer_path", "M30 30 L190 30 L200 220 L20 220 Z")
    inner    = p.get("inner_path", outer)
    gx       = p.get("grain_x", W // 2)
    gy1      = p.get("grain_y1", 80)
    gy2      = p.get("grain_y2", 180)
    ly       = p.get("label_y", 150)
    seam_mm  = p.get("seam_mm", "15")
    width_cm = p.get("width_cm", "")
    height_cm= p.get("height_cm", "")

    # ── Grid ─────────────────────────────────────────────────────────
    grid_lines = ""
    for x in range(0, W + 1, 20):
        grid_lines += f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="{GRID}" stroke-width="0.5"/>'
    for y in range(0, H + 1, 20):
        grid_lines += f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{GRID}" stroke-width="0.5"/>'

    # ── Grain line ───────────────────────────────────────────────────
    if is_bias:
        dx, dy = 22, 22
        x1, y1 = gx - dx, gy2
        x2, y2 = gx + dx, gy1
        grain = f"""
  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="1"/>
  <polygon points="{x2},{y2} {x2-5},{y2+9} {x2+5},{y2+9}" fill="{LINE}"/>
  <polygon points="{x1},{y1} {x1-5},{y1-9} {x1+5},{y1-9}" fill="{LINE}"/>
  <text x="{x2+6}" y="{(y1+y2)//2}" font-size="7" fill="{DIM}" font-family="monospace">BIAS</text>"""
    else:
        grain = f"""
  <line x1="{gx}" y1="{gy1}" x2="{gx}" y2="{gy2}" stroke="{LINE}" stroke-width="1"/>
  <polygon points="{gx},{gy1} {gx-5},{gy1+10} {gx+5},{gy1+10}" fill="{LINE}"/>
  <polygon points="{gx},{gy2} {gx-5},{gy2-10} {gx+5},{gy2-10}" fill="{LINE}"/>"""

    # ── Notches (double triangle marks) ──────────────────────────────
    notches = f"""
  <polygon points="{W//2},{16} {W//2-5},{26} {W//2+5},{26}" fill="{LINE}" opacity="0.9"/>
  <polygon points="{W//2},{10} {W//2-4},{18} {W//2+4},{18}" fill="none" stroke="{LINE}" stroke-width="0.8"/>"""

    # ── Dimension lines ───────────────────────────────────────────────
    dim_lines = ""
    # Width dimension at top
    if width_cm:
        dim_lines += f"""
  <line x1="22" y1="8" x2="{W-22}" y2="8" stroke="{DIM}" stroke-width="0.6" marker-start="url(#arr)" marker-end="url(#arr)"/>
  <text x="{W//2}" y="6" text-anchor="middle" font-size="7" fill="{DIM}" font-family="monospace">{width_cm} cm</text>"""
    # Height dimension on right
    if height_cm:
        dim_lines += f"""
  <line x1="{W-8}" y1="22" x2="{W-8}" y2="{H-22}" stroke="{DIM}" stroke-width="0.6"/>
  <text x="{W-4}" y="{H//2}" text-anchor="middle" font-size="7" fill="{DIM}" font-family="monospace" transform="rotate(90,{W-4},{H//2})">{height_cm} cm</text>"""

    # ── Seam allowance label ──────────────────────────────────────────
    seam_label = f"""
  <text x="6" y="{H-6}" font-size="6.5" fill="{DIM}" font-family="monospace">SA {seam_mm}mm</text>"""

    # ── Piece label ───────────────────────────────────────────────────
    # Truncate long labels
    display = label if len(label) <= 22 else label[:20] + "…"
    piece_label = f"""
  <text x="{W//2}" y="{ly}" text-anchor="middle" font-size="9" font-weight="600"
        fill="{LINE}" font-family="monospace" letter-spacing="0.5">{display}</text>"""

    # ── Cut instruction ───────────────────────────────────────────────
    cut_text = "CUT ON BIAS" if is_bias else "CUT ON FOLD" if "fold" in label.lower() else "CUT 1"
    cut_instruction = f"""
  <text x="{W//2}" y="{ly+14}" text-anchor="middle" font-size="6.5"
        fill="{GOLD}" font-family="monospace" opacity="0.85">{cut_text}</text>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}" style="background:{BG};display:block">
  <defs>
    <marker id="arr" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto">
      <path d="M0,0 L4,2 L0,4 Z" fill="{DIM}"/>
    </marker>
  </defs>

  {grid_lines}

  <!-- Cut line (outer) -->
  <path d="{outer}"
        fill="rgba(255,255,255,0.04)"
        stroke="{LINE}" stroke-width="1.8" stroke-linejoin="round"/>

  <!-- Seam allowance (inner dashed) -->
  <path d="{inner}"
        fill="none"
        stroke="{SEAM}" stroke-width="0.8"
        stroke-dasharray="5 3" opacity="0.5"/>

  {grain}
  {notches}
  {dim_lines}
  {seam_label}
  {piece_label}
  {cut_instruction}
</svg>"""

    return svg


def _mock_svg(label: str, i: int) -> str:
    """
    Realistic pattern piece shapes for each garment section.
    Uses proper fashion pattern geometry — bodices, panels, collars, etc.
    """
    lbl = label.upper()

    # Pick shape based on label keywords
    if any(k in lbl for k in ["FRONT BODICE", "BACK BODICE"]):
        outer = "M30 25 C45 22 135 22 150 25 L158 130 C158 145 148 158 140 162 L80 170 L20 162 C12 158 2 145 2 130 Z"
        inner = "M38 33 C52 30 128 30 142 33 L150 128 C150 142 142 153 134 157 L80 164 L26 157 C18 153 10 142 10 128 Z"
        gx, gy1, gy2, ly = 80, 60, 145, 148
        w, h = "42 cm", "58 cm"
    elif any(k in lbl for k in ["SKIRT", "PANEL"]):
        outer = "M20 20 C50 15 150 15 180 20 L188 200 C185 235 160 250 110 252 L80 254 L50 252 C20 250 -5 235 -8 200 Z"
        inner = "M28 28 C57 23 143 23 172 28 L180 198 C177 230 154 244 107 246 L80 248 L53 246 C26 244 3 230 0 198 Z"
        gx, gy1, gy2, ly = 90, 60, 220, 205
        w, h = "68 cm", "95 cm"
    elif any(k in lbl for k in ["SLEEVE"]):
        outer = "M80 15 C120 18 158 50 162 85 C165 108 155 128 140 138 L130 155 L80 165 L30 155 L20 138 C5 128 -5 108 -2 85 C2 50 40 18 80 15 Z"
        inner = "M80 24 C116 27 150 56 154 88 C157 109 148 127 134 136 L124 152 L80 161 L36 152 L26 136 C12 127 3 109 6 88 C10 56 44 27 80 24 Z"
        gx, gy1, gy2, ly = 80, 55, 145, 130
        w, h = "38 cm", "62 cm"
    elif any(k in lbl for k in ["COLLAR", "NECKLINE", "FACING"]):
        outer = "M10 80 C10 30 60 12 110 15 L165 20 C195 25 205 45 200 65 L195 85 C190 100 175 108 150 108 L80 110 L25 105 C12 100 10 92 10 80 Z"
        inner = "M18 78 C18 36 65 20 113 23 L160 28 C188 33 197 50 192 68 L187 86 C183 99 169 106 145 106 L80 108 L28 103 C18 98 18 88 18 78 Z"
        gx, gy1, gy2, ly = 105, 50, 90, 88
        w, h = "52 cm", "18 cm"
    elif any(k in lbl for k in ["LINING"]):
        outer = "M28 22 L152 22 L160 155 C158 168 148 175 130 176 L80 178 L30 176 C12 175 2 168 0 155 Z"
        inner = "M36 30 L144 30 L152 152 C150 163 142 169 126 170 L80 172 L34 170 C18 169 10 163 8 152 Z"
        gx, gy1, gy2, ly = 80, 65, 148, 145
        w, h = "40 cm", "52 cm"
    elif any(k in lbl for k in ["STRAP", "LOOP", "BAND"]):
        outer = "M50 20 L130 20 L132 240 L48 240 Z"
        inner = "M58 28 L122 28 L124 232 L56 232 Z"
        gx, gy1, gy2, ly = 90, 60, 200, 175
        w, h = "8 cm", "55 cm"
    else:
        # Generic panel
        outer = "M25 22 L155 22 L162 185 C160 200 148 210 125 212 L80 214 L35 212 C12 210 0 200 -2 185 Z"
        inner = "M33 30 L147 30 L154 182 C152 196 141 205 119 207 L80 209 L41 207 C19 205 8 196 6 182 Z"
        gx, gy1, gy2, ly = 80, 70, 175, 162
        w, h = "46 cm", "68 cm"

    is_bias = any(k in lbl for k in ["BIAS", "PANEL", "LOOP"])
    return _build_piece_svg({
        "label": label,
        "outer_path": outer,
        "inner_path": inner,
        "grain_x": gx, "grain_y1": gy1, "grain_y2": gy2,
        "label_y": ly,
        "is_bias": is_bias,
        "width_cm": w,
        "height_cm": h,
        "seam_mm": "15"
    })

@app.post("/worker/pattern-sketches")
def pattern_sketches(req: SketchRequest, x_worker_token: str = Header(...)):
    verify_token(x_worker_token)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if MOCK_ML_MODELS or not anthropic_key:
        logger.info("Mock mode active or ANTHROPIC_API_KEY not set. Returning high-fidelity mock SVG sketches.")
        return {
            "sketches": [
                {
                    "label": cut.get("label", f"Piece {i+1}"),
                    "svg": _mock_svg(cut.get("label","PIECE"), i)
                }
                for i, cut in enumerate(req.draft_cuts)
            ],
            "mocked": True
        }

    def _mock_fallback(reason: str):
        logger.warning(f"Claude SVG fallback: {reason}")
        return {
            "sketches": [
                {"label": cut.get("label", f"Piece {i+1}"), "svg": _mock_svg(cut.get("label", "PIECE"), i)}
                for i, cut in enumerate(req.draft_cuts)
            ],
            "mocked": True
        }

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_key)

        pieces_desc = "\n".join([
            f"- {c.get('label','?')}: {c.get('note','')} | seam: {c.get('seam','')}"
            for c in req.draft_cuts
        ])

        prompt = f"""You are a professional fashion pattern drafting expert creating technical flat pattern pieces.
Generate SVG path data that looks like industry-standard sewing pattern sheets (think Vogue Patterns, Galia Lahav technical sheets).

Garment silhouette: {req.silhouette}
Fabric type: {req.fabric}

Pattern pieces:
{pieces_desc}

Return a JSON array. Each item must have:
- "label": exact piece name from the list above
- "outer_path": SVG path `d` string for the outer cut line
  - viewBox is 0 0 220 260
  - MUST be a closed path ending with Z
  - Use anatomically correct fashion pattern shapes:
    * Bodice: narrower at shoulder/top, wider at waist, curved side seams with subtle waist curve
    * Skirt panel: narrow at waist, generous sweep at hem with a curved hem line
    * Back bodice: similar to front but with center-back straight edge if needed
    * Sleeves: bell-curve cap at top, tapered towards cuff, slightly curved undersea
    * Collar/facing: flat crescent or arc shape, much wider than tall
    * Lining: mirrors the main piece but slightly smaller
    * Straps/bands/loops: long narrow rectangle or strip
  - Centre shapes within the 220×260 canvas with at least 15px margin
  - Use cubic bezier curves (C) for organic edges, not just straight lines
- "inner_path": same outline inset ~8px all around for seam allowance dashed line
- "grain_x": x for grain line (usually canvas centre, ~110)
- "grain_y1": y for top grain arrow (inside shape, upper quarter)
- "grain_y2": y for bottom grain arrow (inside shape, lower quarter)
- "label_y": y for label text placement (centre of shape)
- "is_bias": true only if piece is explicitly bias-cut
- "width_cm": estimated finished width in cm (e.g. "42 cm")
- "height_cm": estimated finished height in cm (e.g. "58 cm")
- "seam_mm": seam allowance in mm as string (usually "15")

Return ONLY a valid JSON array. No markdown fences. No explanation. Start with ["""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        import re as _re, json as _json
        raw = _re.sub(r"```(?:json)?|```", "", raw).strip()
        if not raw.startswith("["):
            match = _re.search(r"\[.*\]", raw, _re.DOTALL)
            if match:
                raw = match.group()

        pieces_data = _json.loads(raw)
        sketches = []
        for piece in pieces_data:
            svg = _build_piece_svg(piece)
            sketches.append({"label": piece["label"], "svg": svg})

        return {"sketches": sketches}

    except Exception as e:
        return _mock_fallback(str(e))

# ── Run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
    if ngrok_token:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(ngrok_token)
            tunnel = ngrok.connect(8000)
            print(f"\n🚀 StitchIQ AI Worker live on NGROK: {tunnel.public_url}\n")
        except Exception as e:
            print(f"⚠️ Could not start pyngrok: {e}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
