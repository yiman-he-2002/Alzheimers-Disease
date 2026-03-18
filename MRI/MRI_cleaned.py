# %%
import os
import pydicom
import numpy as np
import cv2
import hashlib
from pathlib import Path

# Root folders (adjust path!)
root = "/Users/yimanhe/Desktop/COURSE/2025 S2/COMP5703/adni/MRI data"
output_root = "/Users/yimanhe/Desktop/COURSE/2025 S2/COMP5703/adni/Cleaned MRI"

# %%
# Groups and labels
groups = {"AD": 0, "MCI": 1, "NC": 2}

def preprocess_image(img, target_size):
    """
    Normalize and resize/pad to target_size (h, w) while maintaining aspect ratio
    """
    img = img.astype("float32")
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # scale 0–1

    h, w = img.shape
    target_h, target_w = target_size  # keep consistent: (h, w)

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    # pad on black canvas
    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img_resized

    return (canvas * 255).astype("uint8")

def get_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()


def is_low_quality(img, black_thresh=10,
                   mean_thresh=25.0,
                   pct_above_30_thresh=0.20,
                   black_ratio_high_thresh=0.6):
                   
    #Decide whether to skip an image. Returns True to skip.
    arr = img.astype(np.float32)
    black_ratio = float(np.mean(arr < black_thresh))
    mean_val = float(arr.mean())
    pct_above_30 = float(np.mean(arr > 30))

    # debug prints (uncomment for tuning)
    # print(f"mean={mean_val:.2f}, pct>30={pct_above_30:.3f}, black_ratio={black_ratio:.3f}")

    if mean_val < mean_thresh:
        return True
    if pct_above_30 < pct_above_30_thresh:
        return True
    if black_ratio > black_ratio_high_thresh:
        return True
    return False

# %%
target_size = (224, 224)

# %%
# Process all images recursively
for group in groups.keys():
    group_path = os.path.join(root, group)
    out_group_path = os.path.join(output_root, group)
    Path(out_group_path).mkdir(parents=True, exist_ok=True)

    seen_hashes = set()

    for ptid in os.listdir(group_path):
        ptid_path = os.path.join(group_path, ptid)
        if not os.path.isdir(ptid_path):
            continue

        # Recursively search all DICOM files under ptid_path
        dcm_files = [
            os.path.join(dirpath, f)
            for dirpath, _, files in os.walk(ptid_path)
            for f in files if f.lower().endswith(".dcm")
        ]


        print(len(dcm_files))

        for i, dcm_path in enumerate(dcm_files):
            try:
                dcm = pydicom.dcmread(dcm_path, force=True)

                # Skip non-image DICOMs (e.g., headers or incomplete files)
                if not hasattr(dcm, "PixelData"):
                    continue

                img = dcm.pixel_array

                # Skip multi-frame or weirdly shaped arrays
                if img.ndim != 2:
                    continue

                img_proc = preprocess_image(img, target_size)

                # New: combined low-quality check
                if is_low_quality(img_proc,
                                  black_thresh=10,
                                  mean_thresh=25.0,
                                  pct_above_30_thresh=0.20,
                                  black_ratio_high_thresh=0.6):
                    continue

                # Check duplicates
                h = get_hash(img_proc)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                out_name = f"{ptid}_slice{i:04d}.png"
                out_path = os.path.join(out_group_path, out_name)
                cv2.imwrite(out_path, img_proc)

            except Exception as e:
                print(f"Skipping {dcm_path}: {e}")

print("Cleaning finished. Processed dataset saved in:", output_root)




