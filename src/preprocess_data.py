# src/preprocess_data.py
import os, cv2, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ✅ Paths
RAW_DIR = "../data/raw/chestxray14/images/images"   # <-- updated path
CSV_PATH = "../data/raw/chestxray14/Data_Entry_subset.csv"
OUT_DIR = "../data/processed"
SPLIT_DIR = "../data/splits"
IMAGE_SIZE = 224

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# ✅ Load metadata
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={'Image Index':'image_id','Finding Labels':'labels'})

# ✅ Only keep images from the 001 archive
df = df[df['image_id'].str.startswith('0000')]  # 00000001_000.png etc.

# ✅ Limit classes to simpler subset
CLASSES = ['Pneumonia','Fibrosis','Consolidation','No Finding']

for c in CLASSES:
    df[c] = df['labels'].apply(lambda x: 1 if c in x else 0)

# ✅ Keep only rows where at least one of these labels exists
df = df[(df[CLASSES].sum(axis=1) > 0)]

# ✅ Make sure only images that exist are used
available_imgs = set(os.listdir(RAW_DIR))
df = df[df['image_id'].isin(available_imgs)]

df['path'] = df['image_id']

print(f"Total images after filtering: {len(df)}")

# ✅ Resize & save to processed/
for img_name in tqdm(df['image_id'], desc="Resizing"):
    src = os.path.join(RAW_DIR, img_name)
    dst = os.path.join(OUT_DIR, img_name)
    try:
        img = cv2.imread(src)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print("Error with:", img_name, e)
        continue

# ✅ Split into train/val/test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

for name, d in zip(["train","val","test"], [train_df,val_df,test_df]):
    d[['image_id','path']+CLASSES].to_csv(
        os.path.join(SPLIT_DIR, f"{name}.csv"), index=False)

print("✅ Done! Processed images + CSV splits ready.")
