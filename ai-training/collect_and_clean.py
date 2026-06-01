import os
import shutil
import subprocess
from PIL import Image

def run_scraping(base_path):
    """Ethically scrapes public Pinterest fashion boards using gallery-dl."""
    print("\n--- Phase 1: Ethical Scraping ---")
    
    # Check if gallery-dl is installed
    try:
        subprocess.run(["gallery-dl", "--version"], stdout=subprocess.DEVNULL, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("gallery-dl is not installed. Installing it now...")
        subprocess.run(["pip", "install", "gallery-dl"], check=True)
        
    targets = {
        'ankara': "https://www.pinterest.com/search/pins/?q=ankara+styles+nigeria",
        'kente': "https://www.pinterest.com/search/pins/?q=kente+cloth+ghana",
        'asoke': "https://www.pinterest.com/search/pins/?q=aso+oke+fabric+nigerian",
        'nigerian_styles': "https://www.pinterest.com/search/pins/?q=latest+nigerian+fashion+styles"
    }
    
    for category, url in targets.items():
        dest_dir = os.path.join(base_path, 'raw_images', category)
        print(f"Scraping {category} to: {dest_dir}")
        cmd = ["gallery-dl", "--dest", dest_dir, url]
        # Run in background/try to scrape a small sample size if running locally
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Failed to scrape category {category}: {e}")

def clean_images(base_path, min_size=512, max_size=4096):
    """Filters, resizes, and converts images to RGB JPEGs."""
    print("\n--- Phase 2: Image Cleaning and Verification ---")
    raw_dir = os.path.join(base_path, 'raw_images')
    clean_dir = os.path.join(base_path, 'clean_images')
    os.makedirs(clean_dir, exist_ok=True)
    
    kept, rejected = 0, 0
    
    if not os.path.exists(raw_dir):
        print(f"Raw images directory not found at {raw_dir}. Please run scraping first.")
        return
        
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath)
                w, h = img.size
                
                # Filter criteria
                if w < min_size or h < min_size or max(w, h) > max_size:
                    print(f"  [Rejected] {fname} - size {w}x{h} out of bounds")
                    rejected += 1
                    continue
                    
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                dest_path = os.path.join(clean_dir, f"{os.path.basename(root)}_{fname}")
                img.save(dest_path, 'JPEG', quality=95)
                kept += 1
            except Exception as e:
                print(f"  [Error] Skipping {fname}: {e}")
                rejected += 1
                
    print(f"\nProcessing Complete!")
    print(f"  Kept:     {kept} images saved to {clean_dir}")
    print(f"  Rejected: {rejected} images due to resolution or file errors")

if __name__ == "__main__":
    import argparse
    from paths import stitchiq_base

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=stitchiq_base(), help="Path to StitchIQ training folder")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping and go straight to cleaning")
    args = parser.parse_args()
    
    if not args.skip_scrape:
        run_scraping(args.base)
    clean_images(args.base)
