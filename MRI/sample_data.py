import os
import random
from pathlib import Path
import shutil

cleaned_root = output_root = "/Users/yimanhe/Desktop/COURSE/2025 S2/COMP5703/adni/Cleaned MRI"
sampled_root = "/Users/yimanhe/Desktop/COURSE/2025 S2/COMP5703/adni/Sampled MRI"
Path(sampled_root).mkdir(parents=True, exist_ok=True)

num_samples = 3000

for group in ["AD", "MCI", "NC"]:
    group_folder = os.path.join(cleaned_root, group)
    out_folder = os.path.join(sampled_root, group)
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    # list all processed images
    files = [f for f in os.listdir(group_folder)
             if not f.startswith('.') and f.lower().endswith('.png')]

    # sample 5000 (or fewer if not enough images)
    sample_count = min(num_samples, len(files))
    sampled_files = random.sample(files, sample_count)

    # copy sampled images
    for fname in sampled_files:
        src = os.path.join(group_folder, fname)
        dst = os.path.join(out_folder, fname)
        shutil.copy2(src, dst)

    print(f"{group}: sampled {sample_count} images → {out_folder}")

print("Sampling finished.")




