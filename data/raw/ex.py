import os
import tarfile
from tqdm import tqdm

tar_file = "images_001.tar.gz"
output_dir = os.path.join("chestxray14", "images")

os.makedirs(output_dir, exist_ok=True)

print(f"🫧 Extracting {tar_file} → {output_dir}")

# open the tar.gz file in gzip mode
with tarfile.open(tar_file, "r:gz") as tar:
    members = tar.getmembers()
    for member in tqdm(members, desc="Extracting"):
        tar.extract(member, path=output_dir)

print("✅ Extraction complete! All images are now inside chestxray14/images/")
