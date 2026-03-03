#!/usr/bin/env python3
"""
Run EDA only – generates all 26 visualizations without webcam.
Run this first to verify your environment and see the analysis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from usl_trainer import generate_synthetic_landmark_data, run_eda, OUTPUT_DIR

print("Uganda USL Sign Language – EDA Runner")
print("="*50)
df = generate_synthetic_landmark_data(n_samples_per_class=80)
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols | {df['label'].nunique()} classes")
run_eda(df)
print(f"\nAll visualizations saved to: ./{OUTPUT_DIR}/")
print("Open the folder to view all 26 PNG files.")
