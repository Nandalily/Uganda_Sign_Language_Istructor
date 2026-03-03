"""
=============================================================
UGANDA SIGN LANGUAGE — EDA & VISUALIZATION (Script 02)
=============================================================
Performs full Exploratory Data Analysis on collected datasets:
  - Class distribution plots
  - PCA  2D/3D scatter
  - t-SNE scatter
  - UMAP scatter  (if umap-learn installed, else skipped gracefully)
  - Cluster quality metrics
  - Anomaly detection heatmap
  - Sample landmark visualisations

Run after collecting data:
  python3 02_eda_analysis.py --mode alphabet
  python3 02_eda_analysis.py --mode numbers
  python3 02_eda_analysis.py --mode both
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import os
import json
import argparse
from pathlib import Path
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

DATA_DIR    = Path("data")
REPORT_DIR  = Path("reports/eda")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = plt.cm.tab20.colors


# ── Helpers ────────────────────────────────────────────────────────────────

def load_dataset(base_dir: Path):
    """Load all *_landmarks.npy files under base_dir, return X, y arrays."""
    X_list, y_list = [], []
    for cls_dir in sorted(base_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.glob("*_landmarks.npy"):
            arr = np.load(f)
            label = f.stem.replace("_landmarks", "")
            X_list.append(arr)
            y_list.extend([label] * len(arr))
    if not X_list:
        return None, None
    return np.vstack(X_list), np.array(y_list)


def load_images(base_dir: Path, label):
    """Load thumbnail images for a class."""
    for cls_dir in base_dir.iterdir():
        if cls_dir.is_dir():
            f = cls_dir / f"{label}_images.npy"
            if f.exists():
                return np.load(f)
    return None


# ── EDA Functions ──────────────────────────────────────────────────────────

def plot_class_distribution(y, title, save_path):
    counts = Counter(y)
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 5))
    bars = ax.bar(labels, values,
                  color=[PALETTE[i % 20] for i in range(len(labels))],
                  edgecolor="white", linewidth=0.5)
    ax.set_title(f"Class Distribution — {title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")
    ax.axhline(np.mean(values), color="red", linestyle="--", alpha=0.6,
               label=f"Mean = {np.mean(values):.1f}")
    ax.legend()
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Class distribution → {save_path}")


def plot_feature_stats(X, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # mean per feature
    axes[0].plot(X.mean(axis=0), color="steelblue")
    axes[0].fill_between(range(X.shape[1]),
                         X.mean(0) - X.std(0),
                         X.mean(0) + X.std(0),
                         alpha=0.3, color="steelblue")
    axes[0].set_title("Feature Mean ± Std")
    axes[0].set_xlabel("Feature Index")
    # variance
    axes[1].bar(range(X.shape[1]), X.var(axis=0), color="coral", alpha=0.7)
    axes[1].set_title("Feature Variance")
    axes[1].set_xlabel("Feature Index")
    # correlation matrix (sampled)
    if X.shape[1] <= 63:
        im = axes[2].imshow(np.corrcoef(X.T), cmap="coolwarm",
                            vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=axes[2])
        axes[2].set_title("Feature Correlation Matrix")
    else:
        axes[2].text(0.5, 0.5, "Too many features\nfor corr matrix",
                     ha="center", va="center")
    fig.suptitle(f"Feature Statistics — {title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Feature stats → {save_path}")


def plot_pca(X, y, le, title, save_path_2d, save_path_3d=None):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(3, X.shape[1]))
    Xp = pca.fit_transform(Xs)
    unique_labels = np.unique(y)
    colors = {l: PALETTE[i % 20] for i, l in enumerate(unique_labels)}

    # 2D
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        mask = y == label
        ax.scatter(Xp[mask, 0], Xp[mask, 1], c=[colors[label]],
                   label=label, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA 2D — {title}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
              ncol=2, markerscale=1.5)
    plt.tight_layout()
    fig.savefig(save_path_2d, dpi=150)
    plt.close(fig)
    print(f"  ✓ PCA 2D → {save_path_2d}")

    # Scree plot
    pca_full = PCA().fit(Xs)
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax.bar(range(1, min(21, len(cumvar)+1)),
           pca_full.explained_variance_ratio_[:20],
           color="steelblue", alpha=0.7, label="Individual")
    ax.plot(range(1, min(21, len(cumvar)+1)),
            cumvar[:20], "ro-", label="Cumulative")
    ax.axhline(0.95, linestyle="--", color="green", label="95% threshold")
    ax.set_title(f"PCA Scree Plot — {title}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.legend()
    plt.tight_layout()
    scree_path = str(save_path_2d).replace("pca_2d", "pca_scree")
    fig.savefig(scree_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ PCA Scree → {scree_path}")

    return Xp


def plot_tsne(X, y, title, save_path, perplexity=30):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Reduce first with PCA for speed
    n_pca = min(50, Xs.shape[1], Xs.shape[0]-1)
    Xr = PCA(n_components=n_pca).fit_transform(Xs)
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X)-1),
                random_state=42, max_iter=1000)
    Xt = tsne.fit_transform(Xr)
    unique_labels = np.unique(y)
    colors = {l: PALETTE[i % 20] for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        mask = y == label
        ax.scatter(Xt[mask, 0], Xt[mask, 1], c=[colors[label]],
                   label=label, alpha=0.65, s=22, edgecolors="none")
    ax.set_title(f"t-SNE — {title}")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
              ncol=2, markerscale=1.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ t-SNE → {save_path}")
    return Xt


def plot_umap(X, y, title, save_path):
    if not UMAP_AVAILABLE:
        print("  ⚠ UMAP not installed (pip install umap-learn) — skipping")
        return
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reducer = umap.UMAP(n_components=2, random_state=42)
    Xu = reducer.fit_transform(Xs)
    unique_labels = np.unique(y)
    colors = {l: PALETTE[i % 20] for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        mask = y == label
        ax.scatter(Xu[mask, 0], Xu[mask, 1], c=[colors[label]],
                   label=label, alpha=0.65, s=22)
    ax.set_title(f"UMAP — {title}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ UMAP → {save_path}")


def cluster_analysis(X, y, title, save_path):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xr = PCA(n_components=min(20, Xs.shape[1])).fit_transform(Xs)
    n_classes = len(np.unique(y))

    # K-Means with varying k
    k_range = range(2, min(n_classes + 3, 15))
    sil_scores, db_scores, inertias = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        preds = km.fit_predict(Xr)
        sil_scores.append(silhouette_score(Xr, preds))
        db_scores.append(davies_bouldin_score(Xr, preds))
        inertias.append(km.inertia_)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(list(k_range), sil_scores, "bo-")
    axes[0].axvline(n_classes, color="red", linestyle="--",
                    label=f"True k={n_classes}")
    axes[0].set_title("Silhouette Score")
    axes[0].set_xlabel("k")
    axes[0].legend()

    axes[1].plot(list(k_range), db_scores, "ro-")
    axes[1].axvline(n_classes, color="blue", linestyle="--",
                    label=f"True k={n_classes}")
    axes[1].set_title("Davies-Bouldin Score (lower=better)")
    axes[1].set_xlabel("k")
    axes[1].legend()

    axes[2].plot(list(k_range), inertias, "gs-")
    axes[2].set_title("KMeans Inertia (Elbow)")
    axes[2].set_xlabel("k")

    fig.suptitle(f"Cluster Quality — {title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Cluster analysis → {save_path}")

    # Best silhouette
    best_k = list(k_range)[np.argmax(sil_scores)]
    return {"best_k": best_k, "best_silhouette": max(sil_scores)}


def anomaly_detection(X, y, title, save_path):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xr = PCA(n_components=min(10, Xs.shape[1])).fit_transform(Xs)

    iso = IsolationForest(contamination=0.05, random_state=42)
    anomaly_labels = iso.fit_predict(Xr)  # -1 = anomaly

    n_anomalies = (anomaly_labels == -1).sum()
    anomaly_rate = n_anomalies / len(Xr) * 100

    # Plot anomaly scores per class
    scores = iso.decision_function(Xr)
    unique_labels = np.unique(y)
    class_scores = {l: scores[y == l] for l in unique_labels}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].boxplot([class_scores[l] for l in unique_labels],
                    labels=unique_labels, patch_artist=True,
                    medianprops=dict(color="red"))
    axes[0].axhline(0, color="orange", linestyle="--", label="Decision boundary")
    axes[0].set_title("Anomaly Scores per Class")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Anomaly Score")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].hist(scores, bins=40, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--",
                    label=f"Anomalies: {n_anomalies} ({anomaly_rate:.1f}%)")
    axes[1].set_title("Anomaly Score Distribution")
    axes[1].set_xlabel("Anomaly Score")
    axes[1].legend()

    fig.suptitle(f"Anomaly Detection — {title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Anomaly detection → {save_path} ({n_anomalies} anomalies = {anomaly_rate:.1f}%)")

    return {"n_anomalies": int(n_anomalies), "anomaly_rate": anomaly_rate}


def run_eda(mode: str):
    if mode in ("alphabet", "both"):
        alpha_dir = DATA_DIR / "alphabets"
        if alpha_dir.exists():
            print("\n" + "─" * 50)
            print("  EDA: ALPHABET dataset")
            print("─" * 50)
            X, y = load_dataset(alpha_dir)
            if X is not None:
                _run_full_eda(X, y, "Alphabet", REPORT_DIR / "alphabet")
            else:
                print("  ⚠ No alphabet data found. Run 01_collect_dataset.py first.")
        else:
            print("  ⚠ data/alphabets/ not found. Generating synthetic demo data...")
            X, y = generate_synthetic(list("ABCDE"), 50)
            _run_full_eda(X, y, "Alphabet (synthetic)", REPORT_DIR / "alphabet")

    if mode in ("numbers", "both"):
        num_dir = DATA_DIR / "numbers"
        if num_dir.exists():
            print("\n" + "─" * 50)
            print("  EDA: NUMBERS dataset")
            print("─" * 50)
            X, y = load_dataset(num_dir)
            if X is not None:
                _run_full_eda(X, y, "Numbers", REPORT_DIR / "numbers")
            else:
                print("  ⚠ No numbers data found.")
        else:
            print("  ⚠ data/numbers/ not found. Generating synthetic demo data...")
            X, y = generate_synthetic([str(i) for i in range(1, 6)], 50)
            _run_full_eda(X, y, "Numbers (synthetic)", REPORT_DIR / "numbers")


def generate_synthetic(classes, n_per_class):
    """Generate fake landmark data for demo/testing purposes."""
    np.random.seed(42)
    X_list, y_list = [], []
    for i, cls in enumerate(classes):
        center = np.random.randn(63) * 0.5 + i * 0.3
        samples = center + np.random.randn(n_per_class, 63) * 0.15
        X_list.append(samples)
        y_list.extend([cls] * n_per_class)
    return np.vstack(X_list), np.array(y_list)


def _run_full_eda(X, y, title, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n  Dataset shape : {X.shape}")
    print(f"  Classes       : {sorted(np.unique(y))}")
    print(f"  Total samples : {len(y)}")

    stats = {
        "shape": list(X.shape),
        "n_classes": int(len(np.unique(y))),
        "classes": sorted(np.unique(y).tolist()),
    }

    plot_class_distribution(y, title, out_dir / "01_class_distribution.png")
    plot_feature_stats(X, title, out_dir / "02_feature_stats.png")
    plot_pca(X, y, le, title,
             out_dir / "03_pca_2d.png",
             out_dir / "04_pca_3d.png")
    plot_tsne(X, y, title, out_dir / "05_tsne.png")
    plot_umap(X, y, title, out_dir / "06_umap.png")
    cluster_stats = cluster_analysis(X, y, title, out_dir / "07_cluster_quality.png")
    anomaly_stats = anomaly_detection(X, y, title, out_dir / "08_anomaly_detection.png")

    stats.update(cluster_stats)
    stats.update(anomaly_stats)

    with open(out_dir / "eda_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\n  📊 EDA complete! Reports saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alphabet", "numbers", "both"],
                        default="both")
    args = parser.parse_args()
    run_eda(args.mode)
    print("\n✅ EDA finished. Check reports/eda/ for all plots.")


if __name__ == "__main__":
    main()



import os    #ets pyhon interact with folders and files
import cv2   #cv library for reading images
import numpy as np    #used for numerical operations, arrays, matrices,math. statistics etc...
import pandas as pd   #storing data in tables
import matplotlib.pyplot as plt  #for ploting graphs and visualization
import seaborn as sns #''tion''ploting & visualiza
from tqdm import tqdm   #adds a progress bar when looping through files
from sklearn.decomposition import PCA    #tool for dimension reduction
from sklearn.cluster import KMeans  #for clustering
from sklearn.manifold import TSNE   #advanced visualization

DATASET_PATH = "asl_alphabet_train/asl_alphabet_train"

classes = os.listdir(DATASET_PATH)    #reads all folder names

data = []  #empty list to store extracted data

print("Scanning dataset...")    #progress message

for label in tqdm(classes):    #loop through each class folder
    folder = os.path.join(DATASET_PATH, label)  #creates path to the folder
    for imgname in os.listdir(folder):   #loops through every image inside the class
        path = os.path.join(folder, imgname)    #full path to an image

        img = cv2.imread(path)    #loads image in an array
        if img is None:      #if broken or unreadable, skip
            continue

        h,w,_ = img.shape    #extracts height and width and channels, ignores _
        mean = img.mean()  #average imagge brightness
        std = img.std()     #pixel intensity variation

        data.append([label,h,w,mean,std])    #store all info. about image

df = pd.DataFrame(data,columns=["label","height","width","mean","std"])  #creates structured data tables

os.makedirs("outputs",exist_ok=True)   #creates the output folder if it doesnt exist


#VISUALIZATION
print("Generating plots...")  #status message

# 1 Class distribution
plt.figure()
df["label"].value_counts().plot(kind="bar")  #counts images per class
plt.title("Class Distribution")
plt.savefig("outputs/1_class_distribution.png")

# 2 Height distribution (histogram of image height)
plt.figure()
sns.histplot(df["height"],kde=True)
plt.savefig("outputs/2_height.png")

# 3 Width distribution (histogram of image width)
plt.figure()
sns.histplot(df["width"],kde=True)
plt.savefig("outputs/3_width.png")

# 4 Mean (brighteness) pixel distribution  (shows lighting distribution)
plt.figure()
sns.histplot(df["mean"],kde=True)
plt.savefig("outputs/4_mean.png")

# 5 Std distribution (for contrast variability)
plt.figure()
sns.histplot(df["std"],kde=True)
plt.savefig("outputs/5_std.png")

# 6 Boxplot mean per class (brightness difference between classes)
plt.figure(figsize=(12,5))
sns.boxplot(x="label",y="mean",data=df)
plt.xticks(rotation=90)
plt.savefig("outputs/6_box_mean.png")

# 7 Violin (shows spread of contrast per class)
plt.figure(figsize=(12,5))
sns.violinplot(x="label",y="std",data=df)
plt.xticks(rotation=90)
plt.savefig("outputs/7_violin_std.png")

# 8 Correlation heatmap (for relationship between variables)
plt.figure()
sns.heatmap(df[["height","width","mean","std"]].corr(),annot=True)
plt.savefig("outputs/8_corr.png")

# 9 Scatter mean vs std (mean vs std colored by class)
plt.figure()
sns.scatterplot(x="mean",y="std",hue="label",data=df,legend=False)
plt.savefig("outputs/9_scatter.png")

# 10 Pairplot (ploting every feature against every other feature)
sns.pairplot(df[["height","width","mean","std"]])
plt.savefig("outputs/10_pair.png")

# PCA for clustering
X=df[["height","width","mean","std"]]   #selects numerical features
pca=PCA(n_components=2)   #reduce to 2 dimensions
X2=pca.fit_transform(X)   #transforms data into compression version

# 11 PCA plot     #vsualizes data structure in 2D
plt.figure()
plt.scatter(X2[:,0],X2[:,1])
plt.title("PCA Projection")
plt.savefig("outputs/11_pca.png")

# KMeans clustering    groups images into 10 clusters
kmeans=KMeans(n_clusters=10)
clusters=kmeans.fit_predict(X)  #assigning cluster number to each image

# 12 Cluster plot
plt.figure()
plt.scatter(X2[:,0],X2[:,1],c=clusters)
plt.savefig("outputs/12_clusters.png")

# t-SNE   maps data to 2D while preserving the structure
tsne=TSNE(n_components=2,perplexity=30)
X3=tsne.fit_transform(X)

# 13 TSNE   #shows hidden patterns visually
plt.figure()
plt.scatter(X3[:,0],X3[:,1],c=clusters)
plt.savefig("outputs/13_tsne.png")

# More distributions

# 14 CDF shows cumulative brightness distribution
plt.figure()
df["mean"].hist(cumulative=True,bins=50)
plt.savefig("outputs/14_cdf.png")

# 15 Density   for smoth distribution curve
plt.figure()
df["mean"].plot(kind="density")
plt.savefig("outputs/15_density.png")

# 16 Countplot is a class count chart
plt.figure(figsize=(12,5))
sns.countplot(x="label",data=df)
plt.xticks(rotation=90)
plt.savefig("outputs/16_count.png")

# 17 KDE is a smoothed std distribution
plt.figure()
sns.kdeplot(df["std"])
plt.savefig("outputs/17_kde.png")

# 18 Histogram stacked is a histogram for all numeric feature
plt.figure()
df.hist(figsize=(8,6))
plt.savefig("outputs/18_multi_hist.png")

# 19 Area plot  sorted brightness as areas chat
plt.figure()
df["mean"].sort_values().reset_index(drop=True).plot.area()
plt.savefig("outputs/19_area.png")

# 20 Line plot
plt.figure()
df["mean"].sort_values().plot()
plt.savefig("outputs/20_line.png")

# 21 Rolling mean   moving average of brightness
plt.figure()
df["mean"].rolling(100).mean().plot()
plt.savefig("outputs/21_roll.png")

# 22 Box all  mean vs std spread
plt.figure()
sns.boxplot(data=df[["mean","std"]])
plt.savefig("outputs/22_box_all.png")

# 23 Jointplot   is a combination of scatter and histogram
sns.jointplot(x="mean",y="std",data=df)
plt.savefig("outputs/23_joint.png")

# 24 Hexbin   density scatter
plt.figure()
plt.hexbin(df["mean"],df["std"],gridsize=30)
plt.savefig("outputs/24_hex.png")

# 25 Final distribution    high resolution brighness distribution
plt.figure()
sns.histplot(df["mean"],bins=100)
plt.savefig("outputs/25_final.png")

print("Done. All plots saved in outputs folder.")


#This helps you verify dataset quality before training a model, which prevents poor accuracy later.