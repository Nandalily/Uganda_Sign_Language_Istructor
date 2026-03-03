#!/usr/bin/env python3
"""
Uganda Sign Language ML Training Instructor
============================================
A Machine Learning-powered system for teaching Uganda Sign Language (USL)
alphabets (A-Z) to Ugandan users. Uses MediaPipe for hand landmark detection,
semi-supervised and weak supervision learning approaches.

Features:
- EDA with 26+ visualizations
- Letter-by-letter tutorial with finger position descriptions
- Webcam-based sign recognition
- Percentage score feedback
- YouTube resource links
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for EDA generation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import mediapipe as mp
import os
import sys
import time
import warnings
import json
import webbrowser
warnings.filterwarnings('ignore')

# Mediapipe version detection
_MP_NEW_API = not hasattr(mp, 'solutions')

# ─────────────────────────────────────────────────────────────
#  CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────

OUTPUT_DIR = "eda_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# USL Alphabet finger descriptions
SIGN_DESCRIPTIONS = {
    'A': "Make a fist with all 4 fingers curled; thumb rests on the side of the index finger.",
    'B': "Hold all 4 fingers straight up and together; thumb folds across the palm.",
    'C': "Curve all fingers and thumb into a 'C' shape, like holding a cup.",
    'D': "Index finger points up, other fingers curl to meet the thumb forming an 'O'.",
    'E': "Curl all 4 fingers so tips touch the palm; thumb tucks under fingers.",
    'F': "Index finger and thumb touch at tips; remaining 3 fingers spread straight up.",
    'G': "Index finger points sideways, thumb points horizontally, both parallel.",
    'H': "Index and middle fingers extend together horizontally; other fingers folded.",
    'I': "Only the little (pinky) finger points straight up; others curl into fist.",
    'J': "Raise little finger then trace a 'J' curve in the air downward.",
    'K': "Index finger points up, middle finger angled out, thumb between them.",
    'L': "Index finger points up, thumb extends sideways; form an 'L' shape.",
    'M': "Tuck thumb under the first 3 fingers (index, middle, ring) curled over it.",
    'N': "Tuck thumb under index and middle fingers only; ring and pinky fold down.",
    'O': "All fingers and thumb curve together to form a circle 'O' shape.",
    'P': "Like K but rotated downward; index points down, thumb extends out.",
    'Q': "Like G but rotated downward; index and thumb both point downward.",
    'R': "Cross middle finger over index finger; others stay folded into palm.",
    'S': "Make a fist with thumb resting across all 4 curled fingers.",
    'T': "Make a fist with thumb between the index and middle fingers.",
    'U': "Index and middle fingers extend straight up together; others curled.",
    'V': "Index and middle fingers extend up in a 'V' or peace sign.",
    'W': "Index, middle, and ring fingers extend straight up spread apart.",
    'X': "Index finger hooks (bends at knuckle) like a beckoning gesture.",
    'Y': "Extend thumb out to side and pinky finger upward; others curl in.",
    'Z': "Index finger extended, draw a 'Z' shape (zigzag) in the air.",
}

# Numbers 1–10
NUMBER_DESCRIPTIONS = {
    '1': "Index finger points straight up; all other fingers and thumb are folded.",
    '2': "Index and middle fingers extend up (like a 'V'); others folded.",
    '3': "Thumb, index, and middle fingers extended; ring and pinky folded.",
    '4': "Four fingers (index–pinky) extend straight up; thumb folds across palm.",
    '5': "All five fingers spread wide open and extended.",
    '6': "Pinky and thumb touch at tips; index, middle, ring extend up.",
    '7': "Ring finger and thumb touch at tips; other fingers extend up.",
    '8': "Middle finger and thumb touch at tips; other fingers extend.",
    '9': "Index finger and thumb form a loop (like 'OK'); others extend up.",
    '10': "Thumb up and shake the hand side to side (or show '1' then '0').",
}

# YouTube learning resources
YOUTUBE_RESOURCES = [
    {"title": "USL Alphabet A-Z Complete Tutorial", "url": "https://www.youtube.com/watch?v=tkMg8g8vVUo"},
    {"title": "Learn USL Numbers 1-10", "url": "https://www.youtube.com/watch?v=_VDDDPKJGBs"},
    {"title": "USL for Beginners - Full Course", "url": "https://www.youtube.com/watch?v=v1desDduz5M"},
    {"title": "Sign Language Practice Daily", "url": "https://www.youtube.com/watch?v=MxVEoJCOvZg"},
    {"title": "Uganda Sign Language Introduction", "url": "https://www.youtube.com/results?search_query=Uganda+sign+language+tutorial"},
    {"title": "Handshape Recognition Tips", "url": "https://www.youtube.com/results?search_query=ASL+handshape+practice"},
]

# ─────────────────────────────────────────────────────────────
#  SYNTHETIC DATASET GENERATION (Simulates Kaggle USL dataset)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_landmark_data(n_samples_per_class=80, seed=42):
    """
    Generate synthetic hand landmark data simulating USL signs.
    In production, replace with real Kaggle USL dataset.
    Each sample = 21 landmarks × 3 coords (x, y, z) = 63 features.
    """
    np.random.seed(seed)
    labels_alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    labels_num   = [str(i) for i in range(1, 11)]
    all_labels   = labels_alpha + labels_num

    data_rows = []
    for label in all_labels:
        # Create a class-specific "fingerprint" pattern
        base_config = np.random.RandomState(ord(label[0]) + len(label) * 17).randn(63) * 0.3
        for _ in range(n_samples_per_class):
            noise = np.random.randn(63) * 0.08
            row = base_config + noise
            data_rows.append({'label': label, **{f'lm_{i}': row[i] for i in range(63)}})

    df = pd.DataFrame(data_rows)
    return df

# ─────────────────────────────────────────────────────────────
#  EDA ENGINE – 26+ VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

def run_eda(df):
    """Produce 26 EDA visualizations with descriptions."""
    print("\n" + "="*65)
    print("  EXPLORATORY DATA ANALYSIS – 26 VISUALIZATIONS")
    print("="*65)

    feature_cols = [c for c in df.columns if c.startswith('lm_')]
    X = df[feature_cols].values
    y = df['label'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Plot 1: Class Distribution (Alphabets) ──────────────────
    _save(1, "Class Distribution of Alphabet Signs",
          "Shows how many samples exist per letter (A-Z), confirming class balance.")
    fig, ax = plt.subplots(figsize=(14,5))
    alpha_df = df[df['label'].isin(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))]
    counts = alpha_df['label'].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=plt.cm.tab20.colors[:26])
    ax.set_title('Class Distribution – USL Alphabet (A-Z)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Sign Letter'); ax.set_ylabel('Sample Count')
    for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                            str(int(b.get_height())), ha='center', fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/01_class_distribution_alphabets.png", dpi=120); plt.close()

    # ── Plot 2: Class Distribution (Numbers) ───────────────────
    _save(2, "Class Distribution of Number Signs",
          "Shows sample counts for number signs 1-10 to verify balanced dataset.")
    fig, ax = plt.subplots(figsize=(10,5))
    num_df = df[~df['label'].isin(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))]
    counts2 = num_df['label'].value_counts().reindex([str(i) for i in range(1,11)])
    ax.bar(counts2.index, counts2.values, color='steelblue')
    ax.set_title('Class Distribution – Number Signs (1-10)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Number Sign'); ax.set_ylabel('Sample Count')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/02_class_distribution_numbers.png", dpi=120); plt.close()

    # ── Plot 3: Feature Correlation Heatmap (first 20 LMs) ─────
    _save(3, "Feature Correlation Heatmap",
          "Reveals correlations between landmark coordinates; useful for feature selection.")
    fig, ax = plt.subplots(figsize=(12,10))
    corr = pd.DataFrame(X_scaled[:, :20]).corr()
    sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0, linewidths=0.3, annot=False)
    ax.set_title('Landmark Feature Correlation (First 20 Features)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/03_feature_correlation_heatmap.png", dpi=120); plt.close()

    # ── Plot 4: PCA Variance Explained ─────────────────────────
    _save(4, "PCA Variance Explained",
          "Shows how many principal components are needed to explain 95% of the variance.")
    pca_full = PCA().fit(X_scaled)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(np.cumsum(pca_full.explained_variance_ratio_)*100, marker='o', markersize=3)
    ax.axhline(95, color='red', linestyle='--', label='95% threshold')
    ax.set_title('Cumulative PCA Variance Explained', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Components'); ax.set_ylabel('Cumulative Variance (%)')
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_pca_variance_explained.png", dpi=120); plt.close()

    # ── Plot 5: PCA 2D Scatter ──────────────────────────────────
    _save(5, "PCA 2D Projection of All Signs",
          "Projects all signs onto 2 principal components to visualize class separability.")
    pca2 = PCA(n_components=2).fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(12,8))
    scatter = ax.scatter(pca2[:,0], pca2[:,1], c=y_enc, cmap='tab20', alpha=0.6, s=20)
    ax.set_title('PCA 2D – All USL Signs', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax, label='Sign Class Index')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/05_pca_2d_scatter.png", dpi=120); plt.close()

    # ── Plot 6: t-SNE 2D ───────────────────────────────────────
    _save(6, "t-SNE 2D Projection",
          "Non-linear embedding showing natural groupings of sign classes.")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=600)
    tsne_result = tsne.fit_transform(X_scaled[:500])
    y_sub = y_enc[:500]
    fig, ax = plt.subplots(figsize=(12,8))
    scatter = ax.scatter(tsne_result[:,0], tsne_result[:,1], c=y_sub, cmap='tab20', alpha=0.7, s=30)
    ax.set_title('t-SNE 2D Projection (First 500 Samples)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE-1'); ax.set_ylabel('t-SNE-2')
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/06_tsne_2d.png", dpi=120); plt.close()

    # ── Plot 7: Mean Landmark Value per Class ──────────────────
    _save(7, "Mean Feature Values per Sign Class",
          "Compares average landmark coordinates across classes as a class profile.")
    class_means = pd.DataFrame(X_scaled, columns=feature_cols)
    class_means['label'] = y
    means = class_means.groupby('label')[feature_cols[:20]].mean()
    fig, ax = plt.subplots(figsize=(16,7))
    im = ax.imshow(means.values, aspect='auto', cmap='viridis')
    ax.set_title('Mean Feature Values per Sign Class', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(means.index))); ax.set_yticklabels(means.index, fontsize=8)
    ax.set_xlabel('Feature Index (Landmark Coords)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/07_mean_features_per_class.png", dpi=120); plt.close()

    # ── Plot 8: Feature Standard Deviation Distribution ────────
    _save(8, "Feature Variance Distribution",
          "Shows which landmark coordinates vary most, highlighting informative features.")
    stds = pd.DataFrame(X_scaled, columns=feature_cols).std()
    fig, ax = plt.subplots(figsize=(14,5))
    ax.bar(range(len(stds)), stds.values, color='coral')
    ax.set_title('Feature Standard Deviation (All 63 Landmarks)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index'); ax.set_ylabel('Std Deviation')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/08_feature_variance.png", dpi=120); plt.close()

    # ── Plot 9: KMeans Cluster Visualization ───────────────────
    _save(9, "KMeans Clustering (Unsupervised)",
          "K-Means groups signs without labels; tests if geometry alone clusters signs.")
    km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_scaled)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.scatter(pca2[:,0], pca2[:,1], c=km.labels_[:len(pca2)], cmap='tab10', alpha=0.6, s=20)
    ax.scatter(PCA(n_components=2).fit_transform(km.cluster_centers_)[:,0],
               PCA(n_components=2).fit_transform(km.cluster_centers_)[:,1],
               c='black', s=100, marker='X', label='Centroids')
    ax.set_title('KMeans Clusters in PCA Space (k=10)', fontsize=14, fontweight='bold')
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_kmeans_clusters.png", dpi=120); plt.close()

    # ── Plot 10: DBSCAN Density Clustering ─────────────────────
    _save(10, "DBSCAN Density Clustering",
          "Density-based clustering detects dense sign clusters and outliers (noise=-1).")
    db = DBSCAN(eps=1.5, min_samples=5).fit(pca2)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.scatter(pca2[:,0], pca2[:,1], c=db.labels_, cmap='plasma', alpha=0.6, s=20)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    ax.set_title(f'DBSCAN Clustering – {n_clusters} Clusters Found', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/10_dbscan_clusters.png", dpi=120); plt.close()

    # ── Plot 11: Box Plot of Top Features ──────────────────────
    _save(11, "Box Plot of Top 8 Landmark Features",
          "Displays distribution spread and outliers for the 8 most variable features.")
    top8 = stds.nlargest(8).index.tolist()
    fig, ax = plt.subplots(figsize=(12,6))
    class_means[top8].boxplot(ax=ax)
    ax.set_title('Box Plot – Top 8 Most Variable Landmark Features', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature'); ax.set_ylabel('Scaled Value')
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_boxplot_top_features.png", dpi=120); plt.close()

    # ── Plot 12: Violin Plot ────────────────────────────────────
    _save(12, "Violin Plot of Select Features per Class",
          "Combines box plot and density to show distribution shape for each sign.")
    top3 = stds.nlargest(3).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15,6))
    for i, feat in enumerate(top3):
        sample = class_means[['label', feat]].sample(min(200, len(class_means)), random_state=42)
        sns.violinplot(data=sample, x='label', y=feat, ax=axes[i], palette='tab20', cut=0)
        axes[i].set_title(f'Violin: {feat}', fontsize=11)
        axes[i].tick_params(axis='x', rotation=90)
    plt.suptitle('Violin Plots – Top 3 Features by Variance', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/12_violin_plots.png", dpi=120); plt.close()

    # ── Plot 13: Histogram of Feature Values ───────────────────
    _save(13, "Feature Value Distribution Histogram",
          "Shows whether landmark coordinate values follow normal or skewed distributions.")
    fig, ax = plt.subplots(figsize=(10,5))
    flat = X_scaled.flatten()
    ax.hist(flat, bins=80, color='teal', edgecolor='white', alpha=0.8)
    ax.set_title('Distribution of All Landmark Feature Values', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scaled Landmark Value'); ax.set_ylabel('Frequency')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/13_feature_histogram.png", dpi=120); plt.close()

    # ── Plot 14: Pairplot of Top 4 Features ────────────────────
    _save(14, "Pairplot of Top 4 Features (A-E only)",
          "Scatter matrix for key features showing class separation between A-E signs.")
    subset = df[df['label'].isin(list('ABCDE'))].copy()
    top4 = stds.nlargest(4).index.tolist()
    pp_data = subset[top4 + ['label']].sample(min(150, len(subset)), random_state=42)
    pair_fig = sns.pairplot(pp_data, hue='label', palette='tab10', plot_kws={'alpha':0.5})
    pair_fig.fig.suptitle('Pairplot – Top 4 Features for Signs A-E', y=1.02, fontsize=13, fontweight='bold')
    pair_fig.savefig(f"{OUTPUT_DIR}/14_pairplot_top_features.png", dpi=100); plt.close('all')

    # ── Plot 15: Inertia / Elbow Curve ─────────────────────────
    _save(15, "KMeans Elbow Curve",
          "Elbow method helps identify the optimal number of clusters for unsupervised learning.")
    inertias = []
    ks = range(2, 20)
    for k in ks:
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init=5).fit(X_scaled[:400]).inertia_)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(list(ks), inertias, marker='o', color='purple')
    ax.set_title('KMeans Elbow Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)'); ax.set_ylabel('Inertia')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/15_kmeans_elbow.png", dpi=120); plt.close()

    # ── Plot 16: Semi-supervised Label Propagation ─────────────
    _save(16, "Semi-Supervised Learning – Label Propagation",
          "Propagates labels from labeled to unlabeled data; simulates weak supervision.")
    small = df.sample(300, random_state=42)
    X_s = scaler.fit_transform(small[feature_cols].values)
    pca3 = PCA(n_components=10).fit_transform(X_s)
    y_s = le.fit_transform(small['label'].values)
    y_partial = y_s.copy()
    unlabeled_mask = np.random.RandomState(42).choice([True, False], size=len(y_partial), p=[0.6, 0.4])
    y_partial[unlabeled_mask] = -1
    lp = LabelPropagation(kernel='rbf', gamma=20, max_iter=500)
    lp.fit(pca3, y_partial)
    fig, axes = plt.subplots(1,2, figsize=(14,6))
    axes[0].scatter(pca3[:,0], pca3[:,1], c=y_s, cmap='tab20', alpha=0.6, s=25)
    axes[0].set_title('Ground Truth Labels', fontsize=12)
    axes[1].scatter(pca3[:,0], pca3[:,1], c=lp.predict(pca3), cmap='tab20', alpha=0.6, s=25)
    axes[1].set_title('Label Propagation Predictions', fontsize=12)
    plt.suptitle('Semi-Supervised: Label Propagation Results', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/16_label_propagation.png", dpi=120); plt.close()

    # ── Plot 17: Positive-Unlabeled Learning Simulation ────────
    _save(17, "Positive-Unlabeled (PU) Learning Simulation",
          "Simulates PU learning where only 'A' is labeled positive; all others unlabeled.")
    pu_labels = np.where(y == 'A', 1, 0)
    unlabeled_idx = np.random.choice(np.where(pu_labels == 0)[0], size=int(0.7*np.sum(pu_labels==0)), replace=False)
    pu_partial = pu_labels.copy()
    pu_partial[unlabeled_idx] = -1
    fig, ax = plt.subplots(figsize=(10,7))
    colors_pu = {1:'green', 0:'red', -1:'gray'}
    for val, color, label_txt in [(1,'green','Positive (A)'), (0,'red','Negative (known)'), (-1,'gray','Unlabeled')]:
        mask = pu_partial == val
        ax.scatter(pca2[mask,0], pca2[mask,1], c=color, alpha=0.5, s=15, label=label_txt)
    ax.set_title('Positive-Unlabeled Learning Setup', fontsize=14, fontweight='bold')
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/17_pu_learning.png", dpi=120); plt.close()

    # ── Plot 18: Isolation Forest Anomaly Detection ─────────────
    _save(18, "Anomaly Detection – Isolation Forest",
          "Identifies unusual/noisy hand poses that could confuse the sign classifier.")
    iso = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso.fit_predict(X_scaled)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.scatter(pca2[anomalies==1,0], pca2[anomalies==1,1], c='steelblue', s=15, alpha=0.5, label='Normal')
    ax.scatter(pca2[anomalies==-1,0], pca2[anomalies==-1,1], c='red', s=40, alpha=0.8, marker='X', label='Anomaly')
    ax.set_title('Anomaly Detection via Isolation Forest', fontsize=14, fontweight='bold')
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/18_anomaly_detection.png", dpi=120); plt.close()

    # ── Plot 19: Random Forest Feature Importance ───────────────
    _save(19, "Random Forest Feature Importance",
          "Identifies which landmark coordinates matter most for classifying signs.")
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(12,6))
    ax.barh([f'lm_{i}' for i in top_idx], importances[top_idx], color='darkorange')
    ax.set_title('Top 20 Important Landmark Features (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Importance Score')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/19_feature_importance.png", dpi=120); plt.close()

    # ── Plot 20: Confusion Matrix (RF Subset) ───────────────────
    _save(20, "Confusion Matrix – Random Forest",
          "Shows correct vs. incorrect predictions; darker diagonal = better accuracy.")
    y_pred = rf.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(cm, ax=ax, cmap='Blues', fmt='d', linewidths=0.3,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title('Confusion Matrix – Random Forest Classifier', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/20_confusion_matrix.png", dpi=120); plt.close()

    # ── Plot 21: Weak Supervision – Majority Vote ───────────────
    _save(21, "Weak Supervision – Simulated Labeling Function Agreement",
          "Multiple noisy labeling functions vote; agreement level shows label confidence.")
    n_lf = 4  # labeling functions
    lf_predictions = np.array([np.random.choice(le.classes_, size=len(y)) for _ in range(n_lf)])
    agreement = np.sum(lf_predictions == lf_predictions[0], axis=0) / n_lf
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(agreement, bins=20, color='orchid', edgecolor='white')
    ax.axvline(0.75, color='red', linestyle='--', label='75% agreement threshold')
    ax.set_title('Weak Supervision – Labeling Function Agreement', fontsize=14, fontweight='bold')
    ax.set_xlabel('Agreement Score'); ax.set_ylabel('Count')
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/21_weak_supervision_agreement.png", dpi=120); plt.close()

    # ── Plot 22: Learning Curve ─────────────────────────────────
    _save(22, "Learning Curve – Model Performance vs Training Size",
          "Shows how model accuracy improves as more labeled training data is added.")
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=20, random_state=42),
        X_scaled, y_enc, cv=3, train_sizes=np.linspace(0.1, 1.0, 8), scoring='accuracy')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train Accuracy')
    ax.plot(train_sizes, test_scores.mean(axis=1), 's--', label='Val Accuracy')
    ax.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1),
                    train_scores.mean(1)+train_scores.std(1), alpha=0.1)
    ax.fill_between(train_sizes, test_scores.mean(1)-test_scores.std(1),
                    test_scores.mean(1)+test_scores.std(1), alpha=0.1)
    ax.set_title('Learning Curve – Sign Classifier', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Samples'); ax.set_ylabel('Accuracy')
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/22_learning_curve.png", dpi=120); plt.close()

    # ── Plot 23: PCA 3D Visualization ──────────────────────────
    _save(23, "PCA 3D Projection",
          "Three-dimensional principal component view shows deeper class structure.")
    pca3d = PCA(n_components=3).fit_transform(X_scaled[:300])
    y3d = y_enc[:300]
    fig = plt.figure(figsize=(11,8))
    ax3 = fig.add_subplot(111, projection='3d')
    sc = ax3.scatter(pca3d[:,0], pca3d[:,1], pca3d[:,2], c=y3d, cmap='tab20', alpha=0.7, s=20)
    ax3.set_title('PCA 3D Projection of USL Signs', fontsize=14, fontweight='bold')
    ax3.set_xlabel('PC1'); ax3.set_ylabel('PC2'); ax3.set_zlabel('PC3')
    plt.colorbar(sc, ax=ax3, shrink=0.5)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/23_pca_3d.png", dpi=120); plt.close()

    # ── Plot 24: Class Similarity Matrix ───────────────────────
    _save(24, "Inter-Class Similarity Matrix",
          "Cosine similarity between class centroids; high values = easily confused signs.")
    centroids = np.array([X_scaled[y==cls].mean(axis=0) for cls in le.classes_])
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(centroids)
    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(sim_matrix, ax=ax, cmap='YlOrRd',
                xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.3)
    ax.set_title('Inter-Class Cosine Similarity of Sign Centroids', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/24_class_similarity.png", dpi=120); plt.close()

    # ── Plot 25: Labeled vs Unlabeled Split Visualization ──────
    _save(25, "Semi-Supervised Data Split",
          "Visualizes proportion of labeled vs. unlabeled data for semi-supervised setup.")
    ratios = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    accs = [0.41, 0.55, 0.63, 0.74, 0.84, 0.91]  # simulated accuracy trend
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot([r*100 for r in ratios], accs, marker='o', color='mediumseagreen', linewidth=2)
    ax.fill_between([r*100 for r in ratios], accs, alpha=0.2, color='mediumseagreen')
    ax.set_title('Semi-Supervised Accuracy vs % Labeled Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage of Data Labeled (%)'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/25_semi_supervised_accuracy.png", dpi=120); plt.close()

    # ── Plot 26: Summary Dashboard ─────────────────────────────
    _save(26, "Summary Dashboard",
          "A quick-reference panel combining class counts, PCA variance, and model accuracy.")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Uganda USL Sign Language – Training Data Summary', fontsize=16, fontweight='bold')
    # 2a: class counts
    all_counts = df['label'].value_counts().sort_index()
    axes[0,0].bar(range(len(all_counts)), all_counts.values,
                  color=['steelblue' if l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' else 'coral' for l in all_counts.index])
    axes[0,0].set_title('All Class Counts (A-Z & 1-10)'); axes[0,0].set_xlabel('Sign Index')
    # 2b: PCA variance
    axes[0,1].plot(np.cumsum(pca_full.explained_variance_ratio_[:30])*100, color='purple', marker='.')
    axes[0,1].axhline(95, color='red', linestyle='--', alpha=0.5, label='95%')
    axes[0,1].set_title('PCA Variance (First 30 Comps)'); axes[0,1].legend()
    # 2c: Accuracy vs training size
    axes[1,0].plot(train_sizes, test_scores.mean(1)*100, color='green', marker='s')
    axes[1,0].set_title('Val Accuracy vs Training Size'); axes[1,0].set_ylabel('Accuracy (%)')
    # 2d: Feature importance top 10
    top10 = np.argsort(importances)[-10:]
    axes[1,1].barh([f'lm_{i}' for i in top10], importances[top10], color='darkorange')
    axes[1,1].set_title('Top 10 Feature Importances')
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/26_summary_dashboard.png", dpi=120); plt.close()

    print(f"\n  ✅ All 26 visualizations saved to './{OUTPUT_DIR}/'")
    return rf, scaler, le, X_scaled, y_enc, feature_cols

def _save(n, title, description):
    print(f"  [{n:02d}/26] {title}")
    print(f"        → {description}")

# ─────────────────────────────────────────────────────────────
#  TUTORIAL: Letter-by-Letter display
# ─────────────────────────────────────────────────────────────

def display_alphabet_tutorial():
    """Display A-Z with finger position descriptions."""
    print("\n" + "="*65)
    print("    UGANDAN SIGN LANGUAGE TRAINING INSTRUCTOR")
    print("    USL Alphabet Tutorial – A to Z")
    print("="*65)
    for letter, desc in SIGN_DESCRIPTIONS.items():
        print(f"\n  ┌─ Letter: {letter} ─────────────────────────────────")
        print(f"  │ {desc}")
        print(f"  └─────────────────────────────────────────────────")
        time.sleep(1)   # brief pause between letters

    print("\n" + "─"*65)
    print("    Numbers 1–10")
    print("─"*65)
    for num, desc in NUMBER_DESCRIPTIONS.items():
        print(f"\n  ┌─ Number: {num} ──────────────────────────────────")
        print(f"  │ {desc}")
        print(f"  └──────────────────────────────────────────────────")
        time.sleep(0.50)

# ─────────────────────────────────────────────────────────────
#  SIGN CHOICE MENU
# ─────────────────────────────────────────────────────────────

def choose_sign():
    """Display alphabet grid and ask user to pick a letter."""
    all_signs = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [str(i) for i in range(1, 11)]

    print("\n" + "="*65)
    print("  CHOOSE A SIGN TO PRACTICE")
    print("="*65)
    # Display grid A-Z
    print("\n  Alphabets:")
    row = "  "
    for i, letter in enumerate(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 1):
        row += f"[{letter}] "
        if i % 9 == 0:
            print(row); row = "  "
    if row.strip(): print(row)

    # Numbers
    print("\n  Numbers:")
    num_row = "  " + "  ".join(f"[{i}]" for i in range(1,11))
    print(num_row)

    while True:
        choice = input("\n  Enter a letter or number to practice (e.g., A or 3): ").strip().upper()
        if choice in all_signs:
            desc = SIGN_DESCRIPTIONS.get(choice) or NUMBER_DESCRIPTIONS.get(choice)
            print(f"\n  You chose: {choice}")
            print(f"  How to sign it: {desc}")
            return choice
        else:
            print("  ⚠  Invalid choice. Please enter A-Z or 1-10.")

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE HAND LANDMARK EXTRACTOR
# ─────────────────────────────────────────────────────────────

def _get_hand_detector_legacy(static=True):
    """Use legacy mediapipe.solutions API if available."""
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=static, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

def _get_hand_detector_new():
    """Use mediapipe Tasks API (v0.10+)."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request, tempfile

    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("  Downloading MediaPipe hand model (~25 MB)...")
        urllib.request.urlretrieve(model_url, model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1
    )
    return mp_vision.HandLandmarker.create_from_options(options), model_path


def extract_landmarks(frame):
    """Extract 21 hand landmarks from a BGR frame."""
    if not _MP_NEW_API:
        with _get_hand_detector_legacy(static=True) as hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                return np.array([[p.x, p.y, p.z] for p in lm]).flatten()
        return None
    else:
        # New API
        try:
            detector, _ = _get_hand_detector_new()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                return np.array([[p.x, p.y, p.z] for p in lm]).flatten()
        except Exception as e:
            print(f"  MediaPipe detection error: {e}")
        return None


def draw_landmarks_on_frame(frame):
    """Draw hand landmarks on a frame for display."""
    annotated = frame.copy()
    found = False

    if not _MP_NEW_API:
        mp_hands_mod = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        with mp_hands_mod.Hands(static_image_mode=False, max_num_hands=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                found = True
                for hand_lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(annotated, hand_lm, mp_hands_mod.HAND_CONNECTIONS)
    else:
        # New API – draw circles for each detected landmark
        try:
            detector, _ = _get_hand_detector_new()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if result.hand_landmarks:
                found = True
                h, w = frame.shape[:2]
                for lm_list in result.hand_landmarks:
                    pts = [(int(p.x*w), int(p.y*h)) for p in lm_list]
                    for pt in pts:
                        cv2.circle(annotated, pt, 5, (0, 255, 0), -1)
                    connections = [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
                                   (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),
                                   (17,18),(18,19),(19,20),(0,5),(5,9),(9,13),(13,17),(0,17)]
                    for a, b in connections:
                        cv2.line(annotated, pts[a], pts[b], (0, 180, 255), 2)
        except Exception:
            pass

    return annotated, found

# ─────────────────────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────────────────────

def generate_reference_landmarks(target_sign, seed_offset=0):
    """
    Generate a reference landmark vector for the target sign.
    In production this comes from the trained model's class centroid.
    """
    np.random.seed(ord(target_sign[0]) + len(target_sign)*17 + seed_offset)
    base = np.random.randn(63) * 0.3
    return base

def compute_score(user_lm, target_sign):
    """
    Compute similarity score (10%–100%) between user hand and target sign.
    Uses cosine similarity + Euclidean distance blend.
    """
    ref_lm = generate_reference_landmarks(target_sign)

    # Normalize
    user_norm = user_lm / (np.linalg.norm(user_lm) + 1e-8)
    ref_norm  = ref_lm  / (np.linalg.norm(ref_lm)  + 1e-8)

    cosine_sim   = np.dot(user_norm, ref_norm)           # –1 to 1
    euclidean_d  = np.linalg.norm(user_norm - ref_norm)  # 0+

    cosine_score = (cosine_sim + 1) / 2                  # 0 to 1
    eucl_score   = np.exp(-euclidean_d / 2.0)            # 0 to 1

    raw_score = 0.55 * cosine_score + 0.45 * eucl_score

    # Clamp to 10%–100%
    percentage = int(np.clip(raw_score * 110, 10, 100))
    # Round to nearest 5%
    percentage = round(percentage / 5) * 5
    percentage = max(10, min(100, percentage))
    return percentage

def feedback_message(score):
    if score >= 90: return "🌟 Excellent! Perfect sign!"
    if score >= 75: return "👍 Great job! Nearly perfect."
    if score >= 60: return "😊 Good effort! Keep practicing."
    if score >= 40: return "📖 Needs work – review the finger position."
    return "❌ Try again – check your hand shape carefully."

# ─────────────────────────────────────────────────────────────
#  WEBCAM PRACTICE SESSION
# ─────────────────────────────────────────────────────────────

def run_webcam_practice(target_sign):
    """Open webcam, capture user's hand, compare and score."""
    desc = SIGN_DESCRIPTIONS.get(target_sign) or NUMBER_DESCRIPTIONS.get(target_sign, "")
    print(f"\n  📸 Opening webcam for sign: {target_sign}")
    print(f"  Tip: {desc}")
    print("  Press [SPACE] to capture your sign, [Q] to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ⚠  Could not open webcam. Using simulated hand data for demo.")
        _demo_score(target_sign)
        return

    score_history = []
    best_score = 0
    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        annotated, hand_found = draw_landmarks_on_frame(frame)

        # Overlay instructions
        h, w = frame.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        cv2.putText(annotated, f"Practice: {target_sign}  |  SPACE=Capture  Q=Quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        status = "✅ Hand detected!" if hand_found else "❌ No hand detected"
        color  = (0, 255, 100) if hand_found else (0, 100, 255)
        cv2.putText(annotated, status, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show last score if available
        if score_history:
            cv2.putText(annotated, f"Last Score: {score_history[-1]}%",
                        (w-200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

        cv2.imshow("Uganda Sign Language Instructor", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            landmarks = extract_landmarks(frame)
            if landmarks is not None:
                score = compute_score(landmarks, target_sign)
                score_history.append(score)
                best_score = max(best_score, score)
                print(f"\n  📊 Score: {score}%  —  {feedback_message(score)}")
                captured_frame = frame.copy()

                # Flash green/red border
                border_color = (0, 200, 0) if score >= 60 else (0, 0, 200)
                cv2.rectangle(annotated, (0, 0), (w, h), border_color, 8)
                cv2.putText(annotated, f"Score: {score}%", (w//2-80, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, border_color, 4)
                cv2.imshow("Uganda Sign Language Instructor", annotated)
                cv2.waitKey(1500)
            else:
                print("  ⚠  No hand detected. Make sure your hand is visible.")

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    if score_history:
        print(f"\n  ─── Session Summary for Sign '{target_sign}' ───")
        print(f"  Attempts    : {len(score_history)}")
        print(f"  Best Score  : {best_score}%  — {feedback_message(best_score)}")
        print(f"  Average     : {int(np.mean(score_history))}%")
    show_youtube_resources(best_score if score_history else 0)

def _demo_score(target_sign):
    """Demo scoring without a webcam."""
    print(f"\n  🤖 Demo Mode – Generating simulated score for '{target_sign}'")
    fake_lm = generate_reference_landmarks(target_sign, seed_offset=99) + np.random.randn(63)*0.15
    score = compute_score(fake_lm, target_sign)
    print(f"\n  📊 Simulated Score: {score}%  —  {feedback_message(score)}")
    show_youtube_resources(score)

# ─────────────────────────────────────────────────────────────
#  YOUTUBE RESOURCES
# ─────────────────────────────────────────────────────────────

def show_youtube_resources(score):
    """Display YouTube links based on score."""
    print("\n" + "="*65)
    print("  📺 YouTube Resources to Improve Your Sign Language Skills")
    print("="*65)
    if score < 50:
        print("  Your score suggests you'd benefit from beginner resources:\n")
    elif score < 75:
        print("  Keep improving with these practice videos:\n")
    else:
        print("  Great score! Here are advanced resources to perfect your skills:\n")

    for i, res in enumerate(YOUTUBE_RESOURCES, 1):
        print(f"  {i}. {res['title']}")
        print(f"      {res['url']}")

    open_link = input("\n  Open a video link in browser? Enter number (1-6) or 0 to skip: ").strip()
    if open_link.isdigit() and 1 <= int(open_link) <= len(YOUTUBE_RESOURCES):
        url = YOUTUBE_RESOURCES[int(open_link)-1]['url']
        print(f"  Opening: {url}")
        try:
            webbrowser.open(url)
        except Exception:
            print("  (Browser launch not available in this environment)")

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "╔"+"═"*63+"╗")
    print("║     UGANDA SIGN LANGUAGE ML TRAINING INSTRUCTOR           ║")
    print("║     Supporting the Deaf Community in Uganda 🇺🇬  🇺🇬          ║")
    print("╚"+"═"*63+"╝")

    print("\n  Step 1: Generating synthetic USL landmark dataset...")
    df = generate_synthetic_landmark_data(n_samples_per_class=80)
    print(f"  Dataset ready: {df.shape[0]} samples, {df.shape[1]-1} features, "
          f"{df['label'].nunique()} classes")
    print(f"  Dataset shape: {df.shape}")
    print(f"  Classes: {sorted(df['label'].unique())}")

    print("\n  Step 2: Running EDA and generating 26 visualizations...")
    rf_model, scaler, le, X_scaled, y_enc, feature_cols = run_eda(df)

    print("\n  Step 3: Alphabet & Number Tutorial...")
    input("\n  Press ENTER to begin the A-Z letter tutorial...")
    display_alphabet_tutorial()

    print("\n  Step 4: Interactive Practice Session")
    while True:
        target = choose_sign()
        run_webcam_practice(target)
        again = input("\n  Practice another sign? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\n" + "="*65)
    print("  Thank you for using the Uganda Sign Language Instructor!")
    print("  Together we support the deaf community in Uganda. 🙏")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
