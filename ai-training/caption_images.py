import os
import json
import argparse
from PIL import Image

def generate_base_captions(base_path, max_images=2000):
    """Loads BLIP-2 model on CUDA GPU and generates initial descriptive captions."""
    print("\n--- Phase 1: Generating Base Captions with BLIP-2 ---")
    
    try:
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
    except ImportError:
        print("Required PyTorch/Transformers dependencies are missing. Run this script in Google Colab Pro.")
        return
        
    clean_dir = os.path.join(base_path, 'clean_images')
    caption_dir = os.path.join(base_path, 'captioned_dataset')
    os.makedirs(caption_dir, exist_ok=True)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for running BLIP-2 model.")
        
    print("Loading BLIP-2 to GPU...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    captions = {}
    image_files = [f for f in os.listdir(clean_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_images]
    
    print(f"Generating base descriptions for {len(image_files)} images...")
    for idx, fname in enumerate(image_files):
        fpath = os.path.join(clean_dir, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            
            inputs = processor(images=img, return_tensors="pt").to("cuda", torch.float16)
            ids = blip_model.generate(**inputs, max_new_tokens=60)
            cap = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            
            # Save a copy in captioned_dataset directory
            img.save(os.path.join(caption_dir, fname))
            captions[fname] = cap
            print(f"[{idx+1}/{len(image_files)}] {fname} -> {cap}")
        except Exception as e:
            print(f"Error processing image {fname}: {e}")
            
    # Save to disk
    base_captions_path = os.path.join(base_path, 'base_captions.json')
    with open(base_captions_path, 'w') as f:
        json.dump(captions, f, indent=2)
    print(f"Base captions saved to {base_captions_path}")

def enrich_captions_with_claude(base_path, api_key):
    """Enriches standard BLIP-2 captions with high-fidelity African fashion keywords using Claude."""
    print("\n--- Phase 2: Enriching Captions with Claude API ---")
    
    try:
        import anthropic
    except ImportError:
        print("anthropic SDK not installed. Run 'pip install anthropic'.")
        return
        
    client = anthropic.Anthropic(api_key=api_key)
    
    base_captions_path = os.path.join(base_path, 'base_captions.json')
    if not os.path.exists(base_captions_path):
        print(f"No base captions found at {base_captions_path}. Please run BLIP-2 first.")
        return
        
    with open(base_captions_path) as f:
        base_captions = json.load(f)
        
    FABRIC_HINTS = {
        'ankara': 'Ankara wax print fabric',
        'kente':  'Kente cloth',
        'asoke':  'Aso-oke handwoven fabric',
        'lace':   'African lace fabric',
        'adire':  'Adire indigo tie-dye fabric',
        'iro':    'Nigerian iro and buba style',
        'agbada': 'Nigerian Agbada robe style',
        'dashiki': 'Dashiki print shirt style',
        'george': 'George fabric wrapper style'
    }
    
    enriched_captions = {}
    print(f"Enriching {len(base_captions)} captions with professional fashion tags...")
    
    for idx, (fname, base_cap) in enumerate(base_captions.items()):
        # Infer fabric hint from filename
        fabric_hint = next(
            (hint for key, hint in FABRIC_HINTS.items() if key in fname.lower()),
            'African fashion garment'
        )
        
        prompt = f"""You are a fashion expert specialising in African textiles.
Given this base image caption: "{base_cap}"
And this fabric context: "{fabric_hint}"

Write a single enriched caption (max 75 words) for SDXL LoRA training that:
1. Starts with "africanfashion" (the LoRA trigger word)
2. Describes the garment style, silhouette, and neckline
3. Mentions the fabric type and its pattern/colours
4. Notes the suitability for special occasions (casual, wedding, owambe etc.)
5. Is written in plain descriptive English

Return ONLY the caption text, nothing else."""

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            enriched = response.content[0].text.strip()
            enriched_captions[fname] = enriched
            print(f"[{idx+1}/{len(base_captions)}] {fname}:\n  {enriched}\n")
        except Exception as e:
            print(f"API Error on {fname}: {e}")
            
    # Save enriched captions
    enriched_path = os.path.join(base_path, 'enriched_captions.json')
    with open(enriched_path, 'w') as f:
        json.dump(enriched_captions, f, indent=2)
    print(f"Enriched captions saved to {enriched_path}")

def generate_txt_sidecars(base_path):
    """Generates the .txt sidecar files required by kohya_ss training script."""
    print("\n--- Phase 3: Writing kohya_ss Sidecar .txt Files ---")
    
    enriched_path = os.path.join(base_path, 'enriched_captions.json')
    caption_dir = os.path.join(base_path, 'captioned_dataset')
    
    if not os.path.exists(enriched_path):
        print(f"No enriched captions found at {enriched_path}.")
        return
        
    with open(enriched_path) as f:
        enriched = json.load(f)
        
    for fname, caption in enriched.items():
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(caption_dir, f"{base}.txt")
        with open(txt_path, 'w') as tf:
            tf.write(caption)
            
    print(f"Successfully wrote {len(enriched)} sidecar caption files to {caption_dir}.")

if __name__ == "__main__":
    from paths import stitchiq_base

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=stitchiq_base())
    parser.add_argument("--claude-api-key", help="Claude API key for caption enrichment")
    parser.add_argument("--skip-blip", action="store_true", help="Skip BLIP-2 base captions generation")
    args = parser.parse_args()
    
    if not args.skip_blip:
        generate_base_captions(args.base)
        
    if args.claude_api_key:
        enrich_captions_with_claude(args.base, args.claude_api_key)
        generate_txt_sidecars(args.base)
    else:
        print("\n[Warning] Claude API key not provided. Skipping Phase 2 & 3.")
        print("To enrich captions, run: python caption_images.py --claude-api-key YOUR_KEY")
