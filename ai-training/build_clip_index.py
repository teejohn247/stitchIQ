import os
import json
import argparse

def build_index(base_path, batch_size=32):
    """Encodes all clean fabric images to CLIP embeddings and compiles a FAISS search index."""
    print("\n--- Compile CLIP Embedding Index ---")
    
    try:
        import torch
        import open_clip
        import faiss
        import numpy as np
        from PIL import Image
    except ImportError:
        print("Required ML libraries are missing (torch, open_clip, faiss, numpy, PIL). Run in Colab Pro+.")
        return
        
    fabric_dir = os.path.join(base_path, 'clean_images')
    index_dir = os.path.join(base_path, 'clip_index')
    os.makedirs(index_dir, exist_ok=True)
    
    if not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Running indexing on CPU, this might be slow.")
        device = 'cpu'
    else:
        device = 'cuda'
        
    print(f"Loading CLIP (ViT-B-32) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    model = model.to(device).eval()
    
    image_files = [
        f for f in os.listdir(fabric_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if not image_files:
        print(f"No clean fabric images found at {fabric_dir}. Please load images into the directory first.")
        return
        
    print(f"Encoding {len(image_files)} fabrics...")
    embeddings = []
    image_ids = []
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        batch_tensors = []
        
        for fname in batch_files:
            try:
                img = Image.open(os.path.join(fabric_dir, fname)).convert('RGB')
                batch_tensors.append(preprocess(img))
                image_ids.append(fname)
            except Exception as e:
                print(f"  Skipping {fname} due to load error: {e}")
                
        if not batch_tensors:
            continue
            
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    embs = model.encode_image(batch)
            else:
                embs = model.encode_image(batch)
                
            # Normalize embeddings for cosine similarity
            embs = embs / embs.norm(dim=-1, keepdim=True)
            
        embeddings.append(embs.cpu().numpy())
        print(f"  Processed {min(i + batch_size, len(image_files))}/{len(image_files)}")
        
    all_embeddings = np.vstack(embeddings).astype('float32')
    dim = all_embeddings.shape[1]
    
    print(f"Embedding matrix compiled. Shape: {all_embeddings.shape}")
    
    # Build FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    
    # Save files
    faiss_path = os.path.join(index_dir, 'fabric_clip.index')
    mapping_path = os.path.join(index_dir, 'fabric_id_map.json')
    
    faiss.write_index(index, faiss_path)
    with open(mapping_path, 'w') as f:
        json.dump(image_ids, f, indent=2)
        
    print(f"\nFAISS index compiled successfully!")
    print(f"  Index File: {faiss_path}")
    print(f"  ID Map:     {mapping_path}")
    print(f"  Total Vectors: {index.ntotal}")

if __name__ == "__main__":
    from paths import stitchiq_base

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=stitchiq_base())
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    build_index(args.base, args.batch_size)
