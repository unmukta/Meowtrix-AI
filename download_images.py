import os
import shutil
import urllib.request
import tarfile
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import requests
import time

NUM_REAL = 5000
NUM_FAKE = 5000

real_dir = "data/real"
fake_dir = "data/fake"
lfw_extract_dir = "data/real_lfw"

def reset_folders():
    for folder in [real_dir, fake_dir, lfw_extract_dir]:
        if os.path.exists(folder):
            print(f"[INFO] Removing folder: {folder}")
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    print("[INFO] Fresh folders created.\n")

def download_real_faces():
    print("[1/2] Downloading REAL images from LFW...")
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    lfw_tar = "lfw.tgz"

    try:
        urllib.request.urlretrieve(lfw_url, lfw_tar)
    except Exception as e:
        print("[❌ ERROR] Failed to download LFW dataset:", e)
        return

    with tarfile.open(lfw_tar) as tar:
        tar.extractall(path=lfw_extract_dir)

    count = 0
    print("[INFO] Extracting and copying real face images...")
    for root, _, files in os.walk(os.path.join(lfw_extract_dir, "lfw-deepfunneled")):
        for file in files:
            if file.endswith(".jpg"):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(real_dir, f"real_{count+1}.jpg")
                try:
                    Image.open(src_path).save(dest_path)
                    count += 1
                    if count >= NUM_REAL:
                        print(f"[✅] Downloaded {count} real images.\n")
                        return
                except:
                    continue
    print(f"[⚠️ WARNING] Only {count} real images copied.\n")

def download_fake_faces():
    print(f"[2/2] Downloading {NUM_FAKE} FAKE images from ThisPersonDoesNotExist.com...")
    url = "https://thispersondoesnotexist.com/"
    headers = {"User-Agent": "Mozilla/5.0"}

    for i in tqdm(range(NUM_FAKE), desc="Fake Images"):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(os.path.join(fake_dir, f"fake_{i+1}.jpg"))
            time.sleep(0.1)  # prevent being blocked
        except Exception as e:
            print(f"[ERROR] Fake image {i+1} failed: {e}")

    print(f"[✅] Saved {NUM_FAKE} fake images to {fake_dir}\n")

if __name__ == "__main__":
    reset_folders()
    download_real_faces()   # ✅ REAL FIRST
    download_fake_faces()   # ✅ THEN FAKE
    print("[🎉] All downloads complete.")
