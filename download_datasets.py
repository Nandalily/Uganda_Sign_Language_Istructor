#!/usr/bin/env python3
"""
Kaggle Dataset Downloader for ASL Sign Language
================================================
Downloads the two recommended datasets:
  1. ASL Alphabet: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
  2. ASL Numbers:  https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers

SETUP STEPS:
  1. Create a Kaggle account at kaggle.com
  2. Go to Account → API → Create New API Token
  3. Place the downloaded 'kaggle.json' in ~/.kaggle/kaggle.json
  4. chmod 600 ~/.kaggle/kaggle.json
  5. pip install kaggle
  6. Run this script: python3 download_datasets.py
"""

import os
import sys
import subprocess

def check_kaggle():
    try:
        import kaggle
        return True
    except ImportError:
        print("❌ kaggle package not found. Run: pip install kaggle")
        return False

def check_credentials():
    creds = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(creds):
        print(f"❌ Kaggle credentials not found at: {creds}")
        print("  → Log in at kaggle.com → Account → API → Create New Token")
        return False
    print(f"✅ Kaggle credentials found at: {creds}")
    return True

def download_dataset(dataset_slug, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📥 Downloading: {dataset_slug}")
    cmd = f"kaggle datasets download -d {dataset_slug} -p {output_dir} --unzip"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Saved to: {output_dir}")
    else:
        print(f"❌ Error: {result.stderr}")

def main():
    print("="*55)
    print("  Uganda ASL – Kaggle Dataset Downloader")
    print("="*55)

    if not check_kaggle():
        sys.exit(1)
    if not check_credentials():
        sys.exit(1)

    # Dataset 1: USL Alphabet (A-Z) – 87,000 images
    download_dataset(
        dataset_slug="grassknoted/asl-alphabet",
        output_dir="./datasets/asl_alphabet"
    )

    # Dataset 2: USL Numbers (0-9)
    download_dataset(
        dataset_slug="lexset/synthetic-asl-numbers",
        output_dir="./datasets/asl_numbers"
    )

    print("\n" + "="*55)
    print("   Downloads complete!")
    print("  Replace generate_synthetic_landmark_data() in asl_trainer.py")
    print("  with load_real_dataset() to use these images.")
    print("="*55)

    print("""
─── How to use real data in asl_trainer.py ───────────────

def load_real_dataset(dataset_path):
    import os
    from PIL import Image
    import mediapipe as mp
    
    mp_hands = mp.solutions.hands
    rows = []
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        for label in os.listdir(dataset_path):
            label_dir = os.path.join(dataset_path, label)
            if not os.path.isdir(label_dir): continue
            for img_file in os.listdir(label_dir)[:100]:   # limit per class
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path)
                if img is None: continue
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0].landmark
                    coords = [c for p in lm for c in (p.x, p.y, p.z)]
                    rows.append({'label': label, **{f'lm_{i}': coords[i] 
                                                    for i in range(63)}})
    return pd.DataFrame(rows)

# Then in main():
#   df = load_real_dataset("./datasets/asl_alphabet/asl_alphabet_train")
""")

if __name__ == "__main__":
    main()
