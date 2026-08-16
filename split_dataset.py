import os, random, shutil

random.seed(42)
img_dir = "frames"
label_dir = "shuttle_dataset/labels_raw"
out_base = "shuttle_dataset"

for split in ["train", "val"]:
    os.makedirs(f"{out_base}/images/{split}", exist_ok=True)
    os.makedirs(f"{out_base}/labels/{split}", exist_ok=True)

labeled = [f[:-4] for f in os.listdir(label_dir) if f.endswith(".txt")]
random.shuffle(labeled)
split_idx = int(len(labeled) * 0.8)
train_ids, val_ids = labeled[:split_idx], labeled[split_idx:]

for split, ids in [("train", train_ids), ("val", val_ids)]:
    for name in ids:
        shutil.copy(f"{img_dir}/{name}.jpg", f"{out_base}/images/{split}/{name}.jpg")
        shutil.copy(f"{label_dir}/{name}.txt", f"{out_base}/labels/{split}/{name}.txt")

print(f"Train: {len(train_ids)} images, Val: {len(val_ids)} images")