from pathlib import Path
import cv2
from PIL import Image, ImageFile

# 关键：允许截断的 PNG
ImageFile.LOAD_TRUNCATED_IMAGES = True

TARGET_SHORT_EDGE = 768
JPEG_QUALITY = 95

files = [
    "/home/yhmi/data/Datadir/FoundIR/01Blur/GT/0003147.png",
    "/home/yhmi/data/Datadir/FoundIR/01Blur/GT/0004090.png",
    "/home/yhmi/data/Datadir/FoundIR/01Blur/GT/0007791.png",
]

for src_path in files:
    src = Path(src_path)
    # 中间修复后的临时 PNG（可以放在同目录或 /tmp）
    tmp_png = src.with_suffix(".fixed.png")
    dst = Path(str(src).replace("FoundIR", "FoundIR_768").replace(".png", ".jpg"))
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 用 PIL 读并重存一遍（修复 PNG 结构）
        img = Image.open(src).convert("RGB")
        img.save(tmp_png, "PNG")
    except Exception as e:
        print(f"PIL 仍然读不动: {src} -> {e}")
        continue

    # 2. 用 cv2 读修复后的 PNG，再按你的 resize 逻辑处理
    img_cv = cv2.imread(str(tmp_png))
    if img_cv is None:
        print(f"cv2 仍然读不动: {tmp_png}")
        continue

    h, w = img_cv.shape[:2]
    if w < h:
        new_w, new_h = TARGET_SHORT_EDGE, int(h * TARGET_SHORT_EDGE / w)
    else:
        new_h, new_w = TARGET_SHORT_EDGE, int(w * TARGET_SHORT_EDGE / h)

    resized = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(dst), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print(f"Saved: {dst}")