#!/usr/bin/env python3
import cv2
import numpy as np
import os

# val 图片
val_rain0 = cv2.imread('/home/yhmi/data/output/flux2_convnext_ft_3/val_monitor/step_000020/rain_0.png')
print(f"val_rain0 shape: {val_rain0.shape}")
print(f"val_rain0 mean: {val_rain0.mean():.4f}")

# 从 val 图片中提取 LQ, PRED, GT
# val_monitor 保存的格式是: [LQ | PRED | GT] 横向拼接
w = val_rain0.shape[1] // 3
lq_extracted = val_rain0[:, :w]
pred_extracted = val_rain0[:, w:2*w]
gt_extracted = val_rain0[:, 2*w:]

print(f"\n=== step_000020 rain_0 像素分析 ===")
print(f"  LQ 部分: mean={lq_extracted.mean():.4f}, std={lq_extracted.std():.2f}")
print(f"  PRED部分: mean={pred_extracted.mean():.4f}, std={pred_extracted.std():.2f}")
print(f"  GT  部分: mean={gt_extracted.mean():.4f}, std={gt_extracted.std():.2f}")

# 计算逐像素差异
diff_pred_lq = np.abs(pred_extracted.astype(float) - lq_extracted.astype(float)).mean()
diff_pred_gt = np.abs(pred_extracted.astype(float) - gt_extracted.astype(float)).mean()
diff_lq_gt = np.abs(lq_extracted.astype(float) - gt_extracted.astype(float)).mean()

print(f"\n  PRED vs LQ: {diff_pred_lq:.4f}")
print(f"  PRED vs GT: {diff_pred_gt:.4f}")
print(f"  LQ   vs GT: {diff_lq_gt:.4f}")

# 逐通道分析
for ch, name in enumerate(['B', 'G', 'R']):
    pred_ch = pred_extracted[:,:,ch].astype(float)
    lq_ch = lq_extracted[:,:,ch].astype(float)
    gt_ch = gt_extracted[:,:,ch].astype(float)
    print(f"  {name}: PRED_mean={pred_ch.mean():.2f}, LQ_mean={lq_ch.mean():.2f}, GT_mean={gt_ch.mean():.2f}")
    print(f"       PRED vs LQ diff={np.abs(pred_ch - lq_ch).mean():.4f}")

# 检查是否完全相等 (浮点误差)
is_equal_to_lq = np.allclose(pred_extracted, lq_extracted, atol=1)
is_equal_to_gt = np.allclose(pred_extracted, gt_extracted, atol=1)
print(f"\n  PRED 完全等于 LQ (atol=1): {is_equal_to_lq}")
print(f"  PRED 完全等于 GT (atol=1): {is_equal_to_gt}")

# 检查最大差异
max_diff_lq = np.abs(pred_extracted.astype(float) - lq_extracted.astype(float)).max()
max_diff_gt = np.abs(pred_extracted.astype(float) - gt_extracted.astype(float)).max()
print(f"  PRED vs LQ max_diff: {max_diff_lq}")
print(f"  PRED vs GT max_diff: {max_diff_gt}")

# =============================================================
# 对比 step_000010 的图片
# =============================================================
val_rain0_s10 = cv2.imread('/home/yhmi/data/output/flux2_convnext_ft_3/val_monitor/step_000010/rain_0.png')
if val_rain0_s10 is not None:
    print(f"\n\n=== step_000010 rain_0 ===")
    w = val_rain0_s10.shape[1] // 3
    lq_s10 = val_rain0_s10[:, :w]
    pred_s10 = val_rain0_s10[:, w:2*w]
    gt_s10 = val_rain0_s10[:, 2*w:]
    print(f"  LQ: mean={lq_s10.mean():.4f}")
    print(f"  PRED: mean={pred_s10.mean():.4f}")
    print(f"  GT: mean={gt_s10.mean():.4f}")
    print(f"  PRED vs LQ: {np.abs(pred_s10.astype(float) - lq_s10.astype(float)).mean():.4f}")
    print(f"  PRED vs GT: {np.abs(pred_s10.astype(float) - gt_s10.astype(float)).mean():.4f}")

# =============================================================
# 检查其他退化类型的图片
# =============================================================
print("\n\n=== 检查其他退化类型 (step_000020) ===")
for deg in ['haze', 'lowlight', 'lowlight_haze', 'rain_haze']:
    for idx in range(3):
        path = f'/home/yhmi/data/output/flux2_convnext_ft_3/val_monitor/step_000020/{deg}_{idx}.png'
        if os.path.exists(path):
            img = cv2.imread(path)
            w = img.shape[1] // 3
            lq = img[:, :w]
            pred = img[:, w:2*w]
            gt = img[:, 2*w:]
            diff_pl = np.abs(pred.astype(float) - lq.astype(float)).mean()
            diff_pg = np.abs(pred.astype(float) - gt.astype(float)).mean()
            print(f"  {deg}_{idx}: PREDvsLQ={diff_pl:.4f}, PREDvsGT={diff_pg:.4f}, LQmean={lq.mean():.2f}, PREDmean={pred.mean():.2f}")
            break

# =============================================================
# 最终诊断
# =============================================================
print("\n\n=== 最终诊断 ===")
if diff_pred_lq < 0.5:
    print("⚠️  严重问题: PRED 与 LQ 几乎完全相同!")
    print("   这意味着模型输出或解码过程有严重 bug")
elif diff_pred_lq < 5:
    print("⚠️  问题: PRED 与 LQ 非常接近")
    print("   可能模型输出接近 0，或者解码有问题")
else:
    print("正常: PRED 与 LQ 有显著差异")
