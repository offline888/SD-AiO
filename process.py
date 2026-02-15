import os, sys, cv2
from concurrent.futures import ProcessPoolExecutor

def process(args):
    src, dst, target_size = args
    if os.path.exists(dst): return # 1. 断点续传：存在则跳过
    try:
        img = cv2.imread(src)
        if img is None: return
        
        # 2. 短边 Resize 逻辑
        h, w = img.shape[:2]
        scale = target_size / min(h, w) # 计算缩放比例
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 3. 执行缩放并保存 (保持原格式)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(dst, resized)
    except: pass

if __name__ == '__main__':
    if len(sys.argv) < 4: sys.exit("Usage: python script.py <src> <dst> <short_edge_size>")
    src_root, dst_root, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    
    tasks = []
    print("Scanning files...")
    # 4. 扫描文件并保持目录结构
    for root, _, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        out_dir = os.path.join(dst_root, rel)
        if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)
        
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
                tasks.append((os.path.join(root, f), os.path.join(out_dir, name + ext.lower()), size))

    # 5. 分片处理 (每2万张一组，释放内存)
    BATCH_SIZE = 20000
    total = len(tasks)
    print(f"Total: {total} images. Target Short Edge: {size}px")

    for i in range(0, total, BATCH_SIZE):
        batch = tasks[i : i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}...")
        with ProcessPoolExecutor() as exe: list(exe.map(process, batch))

    print("Done.")