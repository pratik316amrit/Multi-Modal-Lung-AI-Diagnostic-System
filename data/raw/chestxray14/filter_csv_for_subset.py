import os
import pandas as pd

# paths
CSV_PATH = "Data_Entry_2017_v2020.csv"
IMG_DIR = "images/images"  # where your extracted images are
OUT_CSV = "Data_Entry_subset.csv"

# load all available images
available_imgs = set(os.listdir(IMG_DIR))
print(f"Found {len(available_imgs)} images in {IMG_DIR}")

# load CSV
df = pd.read_csv(CSV_PATH)
print(f"Original CSV entries: {len(df)}")

# keep only rows where image file exists
df_filtered = df[df['Image Index'].isin(available_imgs)]

print(f"Filtered entries: {len(df_filtered)}")

# save new CSV
df_filtered.to_csv(OUT_CSV, index=False)
print(f"✅ Saved filtered CSV as {OUT_CSV}")
