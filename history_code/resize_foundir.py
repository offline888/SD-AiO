"""
Resize FoundIR dataset to 768 (short edge) for faster training.
Usage: python resize_foundir.py
"""
import os
import gc
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Try to import pillow-simd for much faster processing
try:
    from PIL import Image
    import PIL.ImageResize as use_pil
    USE_PIL_SIMD = True
except ImportError:
    USE_PIL_SIMD = False

# Configuration
SRC_ROOT = Path("/home/yhmi/data/Datadir/FoundIR")
DST_ROOT = Path("/home/yhmi/data/Datadir/FoundIR_768")
TARGET_SHORT_EDGE = 768

# JPEG quality: keep original 95 for best quality
JPEG_QUALITY = 95

# Folders to process - compound degradation datasets
FOLDERS = [
    "18Lowlight_Blur_Noise"
]
SUBFOLDERS = ["LQ_train", "LQ_val", "GT"]

# Check for CUDA support
HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False
print(f"[Info] CUDA available: {HAS_CUDA}")
print(f"[Info] Using PIL-SIMD: {USE_PIL_SIMD}")

def get_output_size(w, h, target=768):
    """Calculate new size keeping aspect ratio, short edge = target"""
    if w < h:
        new_w = target
        new_h = int(h * target / w)
    else:
        new_h = target
        new_w = int(w * target / h)
    return (new_w, new_h)

def resize_image(args):
    """Resize single image, save as JPG for smaller size"""
    src_path, dst_path, target, use_pil_simd = args
    try:
        os.makedirs(dst_path.parent, exist_ok=True)
        
        # Change extension to .jpg for output
        dst_path = dst_path.with_suffix('.jpg')
        
        # Skip if already exists
        if dst_path.exists():
            return True, str(src_path)
        
        if use_pil_simd:
            # Use PIL (much faster than cv2)
            with Image.open(str(src_path)) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                w, h = img.size
                new_w, new_h = get_output_size(w, h, target)
                
                # Use LANCZOS for high quality, or BILINEAR for speed
                # NEAREST is fastest but lower quality
                img_resized = img.resize((new_w, new_h), Image.LANCZOS)
                
                # Save with optimized settings
                img_resized.save(str(dst_path), 'JPEG', quality=JPEG_QUALITY, optimize=True)
                
                # Explicitly clean up
                img.close()
                img_resized.close()
        else:
            # Use cv2
            img = cv2.imread(str(src_path))
            if img is None:
                return False, str(src_path)
            
            h, w = img.shape[:2]
            new_w, new_h = get_output_size(w, h, target)
            
            # Use INTER_AREA for downsampling, INTER_CUBIC for upsampling
            if new_w < w or new_h < h:
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            cv2.imwrite(str(dst_path), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            
            # Explicitly release cv2 memory
            del img
            del resized
        
        return True, str(src_path)
    except Exception as e:
        return False, f"{src_path}: {e}"

def process_folder(src_folder, dst_folder, target, skip_existing=True):
    """Process all images in a folder with resume support"""
    tasks = []
    
    if not src_folder.exists():
        return 0, 0
    
    for subfolder in SUBFOLDERS:
        src_sub = src_folder / subfolder
        dst_sub = dst_folder / subfolder
        
        if not src_sub.exists():
            continue
        
        for img_path in src_sub.glob("*"):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Change extension to .jpg for output
                dst_path = dst_sub / (img_path.stem + '.jpg')
                
                # Skip if already exists (resume support)
                if skip_existing and dst_path.exists():
                    continue
                    
                tasks.append((img_path, dst_path, target, USE_PIL_SIMD))
    
    if not tasks:
        return 0, 0
    
    success = 0
    failed = 0
    
    # Use multiprocessing - reduced workers to lower IO congestion
    # 16 workers is a good balance for most storage systems
    n_workers = 16
    
    # Use chunksize for better performance with many small tasks
    chunksize = max(1, len(tasks) // (n_workers * 4))
    
    # Periodic cleanup interval (every N images)
    cleanup_interval = 500
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(resize_image, task) for task in tasks]
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(tasks), 
                         desc=f"{src_folder.name}", leave=False)):
            ok, _ = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            
            # Periodic cleanup to prevent memory buildup
            if (i + 1) % cleanup_interval == 0:
                gc.collect()
                # Also try to drop caches if available
                try:
                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                        f.write('1')
                except (PermissionError, FileNotFoundError):
                    pass  # Not root or not Linux
    
    # Final cleanup after batch completes
    gc.collect()
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('1')
    except (PermissionError, FileNotFoundError):
        pass
    
    return success, failed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Resize FoundIR dataset')
    parser.add_argument('--reset', action='store_true', help='Reset and start from scratch')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality (default: 95)')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers (0=auto)')
    args = parser.parse_args()
    
    global JPEG_QUALITY
    if args.quality:
        JPEG_QUALITY = args.quality
    
    n_workers = args.workers if args.workers > 0 else 16
    
    print(f"FoundIR Dataset Resizer")
    print(f"=" * 50)
    print(f"Source: {SRC_ROOT}")
    print(f"Destination: {DST_ROOT}")
    print(f"Target short edge: {TARGET_SHORT_EDGE}")
    print(f"Resume mode: {not args.reset}")
    print(f"=" * 50)
    

    est_size_per_img = 0.1  # MB
    est_total = 320000 * est_size_per_img / 1024  # GB (approx for 5 compound degradation folders)
    print(f"Estimated storage: ~{est_total:.1f} GB (JPEG @ {JPEG_QUALITY} quality)")
    print(f"Workers: {n_workers}")
    print()
    
    # Remove destination if --reset is specified
    if args.reset and DST_ROOT.exists():
        print(f"Reset mode: Removing existing {DST_ROOT}...")
        shutil.rmtree(DST_ROOT)
    
    # Show current progress
    for folder in FOLDERS:
        dst_folder = DST_ROOT / folder
        if dst_folder.exists():
            total = sum(1 for _ in dst_folder.rglob("*.jpg"))
            print(f"  {folder}: {total} images already processed")
    
    total_success = 0
    total_failed = 0
    
    for folder in FOLDERS:
        src_folder = SRC_ROOT / folder
        dst_folder = DST_ROOT / folder
        
        print(f"Processing {folder}...")
        success, failed = process_folder(src_folder, dst_folder, TARGET_SHORT_EDGE)
        total_success += success
        total_failed += failed
        print(f"  Success: {success}, Failed: {failed}")
        
        # Release memory after each folder
        gc.collect()
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('1')
            print(f"  [Memory] Cache dropped")
        except (PermissionError, FileNotFoundError):
            pass  # Not root or not Linux
    
    print()
    print(f"=" * 50)
    print(f"Done! Total: {total_success} images processed, {total_failed} failed")
    
    # Calculate actual size
    if DST_ROOT.exists():
        actual_size = shutil.disk_usage(DST_ROOT).used / (1024**3)
        print(f"Actual size: {actual_size:.2f} GB")

if __name__ == "__main__":
    main()
