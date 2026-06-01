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
    GOLD   = "#D4A843"
    DIM    = "rgba(255,255,255,0.35)"
    DASH   = "rgba(212,168,67,0.5)"
    BG     = "#0D2318"
    
    gx  = p.get("grain_x", 80)
    gy1 = p.get("grain_y1", 70)
    gy2 = p.get("grain_y2", 114)
    ly  = p.get("label_y", 130)
    label = p.get("label", "PIECE")
    is_bias = p.get("is_bias", False)
    
    # Grain line — diagonal if bias cut
    if is_bias:
        grain = f"""
  <line x1="{gx-20}" y1="{gy2}" x2="{gx+20}" y2="{gy1}" 
        stroke="{GOLD}" stroke-width="1"/>
  <path d="M{gx-20+2} {gy2-6} L{gx-20} {gy2} L{gx-20+6} {gy2-2}" 
        stroke="{GOLD}" stroke-width="1" fill="none"/>
  <path d="M{gx+20-2} {gy1+6} L{gx+20} {gy1} L{gx+20-6} {gy1+2}" 
        stroke="{GOLD}" stroke-width="1" fill="none"/>
  <text x="{gx+24}" y="{(gy1+gy2)//2}" font-size="7" fill="{DIM}" 
        font-family="sans-serif">BIAS</text>"""
    else:
        grain = f"""
  <line x1="{gx}" y1="{gy1}" x2="{gx}" y2="{gy2}" 
        stroke="{GOLD}" stroke-width="1"/>
  <path d="M{gx-4} {gy1+8} L{gx} {gy1} L{gx+4} {gy1+8}" 
        stroke="{GOLD}" stroke-width="1" fill="none"/>
  <path d="M{gx-4} {gy2-8} L{gx} {gy2} L{gx+4} {gy2-8}" 
        stroke="{GOLD}" stroke-width="1" fill="none"/>"""

    # Notch at top centre
    notch = f"""
  <path d="M76 18 L80 12 L84 18" 
        stroke="{GOLD}" stroke-width="1" fill="{GOLD}"/>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 200" 
     width="160" height="200" style="background:{BG}">
  
  <path d="{p['outer_path']}" 
        fill="rgba(212,168,67,0.08)" 
        stroke="{GOLD}" stroke-width="1.5"/>
  
  <path d="{p.get('inner_path', p['outer_path'])}" 
        fill="none" 
        stroke="{DASH}" stroke-width="0.8" 
        stroke-dasharray="4 3"/>
  {grain}
  {notch}
  
  <text x="80" y="{ly}" 
        text-anchor="middle" font-size="9.5" font-weight="500" 
        fill="{GOLD}" font-family="sans-serif">
    {label}
  </text>
</svg>"""

    return svg

def _mock_svg(label: str, i: int) -> str:
    """Return a simple mock SVG for testing without GPU."""
    shapes = [
        "M22 20 L138 20 L148 160 L12 160 Z",   # bodice
        "M30 20 Q80 24 130 20 L148 100 Q148 150 80 180 Q12 150 12 100 Z",  # skirt
        "M20 140 Q20 30 155 30 L155 52 Q42 52 42 140 Z",  # facing
        "M50 20 Q80 18 110 30 L118 60 Q122 120 100 170 Q80 182 60 170 Q38 120 42 60 Z",  # loop
        "M80 12 C120 18 148 50 145 90 C142 130 118 160 80 175 C42 160 18 130 15 90 C12 50 40 18 80 12 Z",  # panel
    ]
    inners = [
        "M28 26 L132 26 L141 154 L19 154 Z",
        "M36 26 Q80 30 124 26 L141 100 Q141 144 80 170 Q19 144 19 100 Z",
        "M26 136 Q26 36 149 36 L149 46 Q36 46 36 136 Z",
        "M54 26 Q80 24 106 34 L113 62 Q116 118 94 164 Q80 174 66 164 Q44 118 47 62 Z",
        "M80 20 C115 26 138 54 135 90 C132 126 110 152 80 166 C50 152 28 126 25 90 C22 54 45 26 80 20 Z",
    ]
    idx = i % len(shapes)
    return _build_piece_svg({
        "label": label,
        "outer_path": shapes[idx],
        "inner_path": inners[idx],
        "grain_x": 80, "grain_y1": 70, "grain_y2": 114,
        "label_y": 132,
        "is_bias": "bias" in label.lower() or "loop" in label.lower()
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

    import anthropic
    client = anthropic.Anthropic(api_key=anthropic_key)

    pieces_desc = "\n".join([
        f"- {c.get('label','?')}: {c.get('note','')} | seam: {c.get('seam','')}"
        for c in req.draft_cuts
    ])

    prompt = f"""You are a fashion pattern drafting expert. 
I need you to generate SVG path data for flat technical pattern piece sketches.

Garment silhouette: {req.silhouette}
Fabric type: {req.fabric}

Pattern pieces to draw:
{pieces_desc}

For each piece, return a JSON array. Each item must have:
- "label": the piece name (exact, from the list above)
- "outer_path": an SVG path `d` attribute string for the outer cut line
  - viewBox is 0 0 160 200
  - Path must be closed (end with Z)
  - Use realistic fashion pattern shapes:
    * Bodice pieces: trapezoid shape, wider at bottom, princess seam curves if structured
    * Skirt panels: wider at hem, curved side seams for mermaid/trumpet silhouettes
    * Facings: thin arc or curved strip shapes
    * Panels: organic shapes following the described cut
    * Sleeves: cap curve at top, tapered sides
    * Loops/straps: long narrow rectangles or ovals
  - Place shapes centred around x=80, y=100, with 15px margin from all edges
- "inner_path": same shape inset ~7px for seam allowance dashed line
- "grain_x": x coordinate for grain line centre (usually 80)
- "grain_y1": top of grain line arrow
- "grain_y2": bottom of grain line arrow  
- "label_y": y coordinate to place the label text (should be inside the shape)
- "is_bias": true if cut on bias (diagonal grain line needed)

Return ONLY a valid JSON array. No markdown. No explanation. Start with ["""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    
    try:
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
        logger.warning(f"Claude SVG generation failed ({e}). Falling back to mock SVGs.")
        return {
            "sketches": [
                {"label": cut.get("label", f"Piece {i+1}"), "svg": _mock_svg(cut.get("label", "PIECE"), i)}
                for i, cut in enumerate(req.draft_cuts)
            ],
            "mocked": True
        }

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
