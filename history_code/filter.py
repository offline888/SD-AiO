import os
import random
import shutil
import base64
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
from openai import OpenAI

# ================= 极简配置区 =================
API_KEY = "sk-dtszcvakzlyweppbkzgwfyrbbjgmyemwhpgxiwfyjleuryrd" 
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "kimi-k2.5"

OUTPUT_DIR = "/home/yhmi/data/deg_syn/2_3_order_GT"
TARGET_QUOTA = 300
MAX_IMG_SIZE = 512  

IMAGE_PATHS = [
    "/home/yhmi/data/Datadir/FoundIR_768/00clean/", 
]

TARGET_TASKS = [
    'rainy+hazy', 'snowy+hazy', 'rainy+low-light', 'snowy+low-light', 
    'rainy+low-light+hazy', 'snowy+low-light+hazy'
]

DEGRADATION_RULES = {
    'rainy': "- RAINY: Outdoor open space. Ground can hold puddles. Overcast/night lighting. NO direct sunlight/blue sky.",
    'snowy': "- SNOWY: Has horizontal surfaces for snow. Cold diffuse lighting. NO warm sunlight.",
    'hazy': "- HAZY: Large-scale outdoor depth. Night scenes need point lights. MUST be very clear currently.",
    'low-light': "- LOW-LIGHT: Balanced dynamic range. NO strong direct sunlight. Shadows have texture."
}

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

GLOBAL_USED_IMAGES = set()
LOCK_STATE = threading.Lock()     
LOCK_RATE = threading.Lock()      

# 官方接口限流保护 (如果你的 Tier 较高，可以把这个间隔调小)
MIN_REQUEST_INTERVAL = 0.15  # 限制请求间隔，防止并发过高被官方临时封禁
last_request_time = 0.0

def wait_for_rate_limit():
    global last_request_time
    with LOCK_RATE:
        current_time = time.time()
        elapsed = current_time - last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        last_request_time = time.time()

# ================= 核心工具函数 =================
def get_all_images_fast(paths):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = []
    for p in paths:
        for root, _, files in os.walk(p):
            for f in files:
                if f.lower().endswith(exts):
                    images.append(os.path.join(root, f))
    return images

def encode_image(image_path):
    """读取并压缩为 JPEG Base64，与官方示例的 image_url 无缝对接"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            img.thumbnail((MAX_IMG_SIZE, MAX_IMG_SIZE), Image.Resampling.NEAREST)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=75)
            # 因为统一存为 JPEG，所以 MIME 类型固定为 image/jpeg
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception:
        return None

def build_instruction(task):
    degs = task.split('+')
    rules = "\n".join([DEGRADATION_RULES[d] for d in degs])
    deg_names = " AND ".join([d.upper() for d in degs])
    return (
        f"Evaluate if this image is highly suitable as a clean ground truth to add [{deg_names}] degradation.\n"
        f"Rules must satisfy ALL:\n{rules}\n\n"
        f"Output EXACTLY like this:\n"
        f"Analysis: <Check contents and rules step-by-step>\n"
        f"Judgment: <Highly Suitable or Unsuitable>"
    )

# ================= 线程执行体 =================
def process_image(img_path, task, instruction, task_out_dir, task_state):
    with LOCK_STATE:
        if task_state['count'] >= TARGET_QUOTA or img_path in GLOBAL_USED_IMAGES:
            return False

    base64_img = encode_image(img_path)
    if not base64_img: return False
    
    image_url = f"data:image/jpeg;base64,{base64_img}"

    wait_for_rate_limit()

    try:
        # 完全采用官方示例的 messages 结构
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert computer vision data evaluator."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": instruction,
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=100,  
            timeout=20.0     
        )
        res_text = response.choices[0].message.content.lower()

        if "judgment: highly suitable" in res_text:
            with LOCK_STATE:
                if task_state['count'] < TARGET_QUOTA and img_path not in GLOBAL_USED_IMAGES:
                    GLOBAL_USED_IMAGES.add(img_path)
                    task_state['count'] += 1
                    shutil.copy(img_path, os.path.join(task_out_dir, os.path.basename(img_path)))
                    print(f"[HIT] {task} -> {task_state['count']}/{TARGET_QUOTA} | {os.path.basename(img_path)}")
                    return True
                    
    except Exception as e:
        # 调试时可以把这行取消注释来看看是不是报了 Rate Limit 错误
        # print(f"API Error: {e}")
        pass
    return False

# ================= 主控制流 =================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Initializing Fast Scan with Official Moonshot API...")
    all_images = get_all_images_fast(IMAGE_PATHS)
    random.shuffle(all_images)
    
    # 建议先用 8 个线程跑跑看，如果非常稳定不报错，可以往上加
    WORKER_THREADS = 8 
    print(f"[INFO] Valid Images: {len(all_images)} | Concurrency: {WORKER_THREADS}")

    for task in TARGET_TASKS:
        print(f"\n{'='*40}\n>>> Task: [{task.upper()}]\n{'='*40}")
        task_out_dir = os.path.join(OUTPUT_DIR, task)
        os.makedirs(task_out_dir, exist_ok=True)
        
        instruction = build_instruction(task)
        task_state = {'count': 0} 

        with ThreadPoolExecutor(max_workers=WORKER_THREADS) as executor:
            futures = []
            for img_path in all_images:
                if task_state['count'] >= TARGET_QUOTA: break
                if img_path in GLOBAL_USED_IMAGES: continue

                futures.append(executor.submit(
                    process_image, img_path, task, instruction, task_out_dir, task_state
                ))
                
                if len(futures) >= WORKER_THREADS * 3:
                    for f in as_completed(futures):
                        if task_state['count'] >= TARGET_QUOTA: break
                    futures = [] 

    print("\n[INFO] 筛选结束！你可以去整理 PPT 准备明天的组会了！")

if __name__ == "__main__":
    main()