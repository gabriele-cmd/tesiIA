"""
Test reversibile per migliorare le UMAP di KITM:
  1. Normalizzazioni diverse (L2, MaxAbs, nessuna)
  2. Soft assignment (vettori densi invece di sparsi)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import normalize, StandardScaler, MaxAbsScaler
import umap

COMPOSER_COLORS = {
    'mozart':    '#2196F3',
    'chopin':    '#E91E63',
    'debussy':   '#4CAF50',
    'bach':      '#FF9800',
    'beethoven': '#9C27B0',
}

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Soft assignment K-means
def soft_kitm_kmeans(pch_data: np.ndarray, interval_data: np.ndarray,
                     kmeans, normalizer: str = 'l2') -> np.ndarray:
    """
    Versione soft di KITM con K-means.
    Invece di assegnare al cluster più vicino, distribuisce il peso
    su tutti i cluster proporzionalmente alla distanza inversa.
    """
    if normalizer == 'l2':
        pch_norm = normalize(pch_data, norm='l2')
    else:
        pch_norm = pch_data

    centroids = kmeans.cluster_centers_  # (12, 12)
    results = []

    for i in range(len(pch_norm)):
        pch = pch_norm[i]
        interval = interval_data[i]

        # Distanze euclidee da tutti i centroidi
        dists = np.linalg.norm(centroids - pch, axis=1)  # (12,)

        # Converti in pesi (inverso della distanza + epsilon)
        eps = 1e-8
        weights = 1.0 / (dists + eps)
        weights /= weights.sum()  # normalizza a somma 1

        # Costruisce matrice 12x12 pesata (densa)
        matrix = np.outer(weights, interval)  # (12, 12)
        results.append(matrix.flatten())

    return np.array(results, dtype=np.float32)

# Soft assignment KS
def soft_kitm_ks(pch_data: np.ndarray, interval_data: np.ndarray) -> np.ndarray:
    """
    Versione soft di KITM con KS.
    Usa le correlazioni KS come pesi invece di assegnare a una sola tonalità.
    """
    results = []

    for i in range(len(pch_data)):
        pch = pch_data[i]
        interval = interval_data[i]

        # Correlazioni KS con tutte le 12 tonalità
        correlations = np.zeros(12)
        for k in range(12):
            profile = np.roll(KS_MAJOR, k)
            if pch.std() > 0:
                correlations[k] = np.corrcoef(pch, profile)[0, 1]

        # Clip negativi e normalizza
        weights = np.clip(correlations, 0, None)
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(12) / 12.0

        # Costruisce matrice 12x12 pesata (densa)
        matrix = np.outer(weights, interval)  # (12, 12)
        results.append(matrix.flatten())

    return np.array(results, dtype=np.float32)

def plot_umap(X: np.ndarray, labels: np.ndarray, composers: list,
              title: str, output_path: str, seed: int = 42):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        random_state=seed, verbose=False)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for composer in composers:
        mask = labels == composer
        color = COMPOSER_COLORS.get(composer, '#888888')
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=8, alpha=0.5, color=color,
                   label=f'{composer.capitalize()} ({mask.sum()})')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir",    type=str, default="data/features")
    parser.add_argument("--composers",        nargs="+",
                        default=["mozart", "chopin", "debussy"])
    parser.add_argument("--model_path",       type=str,
                        default="models/kmeans_key_final.pkl")
    parser.add_argument("--n_per_composer",   type=int, default=500)
    parser.add_argument("--output_dir",       type=str,
                        default="results/umap/kitm_test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    features_dir = Path(args.features_dir)

    # Carica modello K-means
    with open(args.model_path, 'rb') as f:
        model_data = pickle.load(f)
    kmeans     = model_data['kmeans']
    normalizer = model_data.get('normalizer', 'l2')
    print(f"Modello caricato: k={model_data['n_clusters']}")

    # Carica PCH e intervalli
    print("\nCaricamento dati...")
    pch_list, interval_list, kitm_hard_list, labels_list = [], [], [], []
    rng = np.random.default_rng(42)

    for composer in args.composers:
        pch_all = np.load(features_dir / f"{composer}_pch.npy").astype(np.float32)
        kitm_all = np.load(features_dir / f"{composer}_kitm_kmeans.npy").astype(np.float32)

        n_available = min(len(pch_all), len(kitm_all))
        pch_all = pch_all[:n_available]
        kitm_all = kitm_all[:n_available]

        n = min(args.n_per_composer, n_available)
        idx = rng.choice(n_available, n, replace=False)

        pch_list.append(pch_all[idx])
        kitm_hard_list.append(kitm_all[idx])
        labels_list.extend([composer] * n)
        print(f"  {composer}: {n} chunk")

    pch_data  = np.vstack(pch_list)
    kitm_hard = np.vstack(kitm_hard_list)
    labels    = np.array(labels_list)

    # Costruisce intervalli dal KITM hard (estrae riga non-zero)
    print("\nEstrazione intervalli dal KITM hard...")
    interval_data = np.zeros((len(kitm_hard), 12), dtype=np.float32)
    for i in range(len(kitm_hard)):
        matrix = kitm_hard[i].reshape(12, 12)
        row_sums = matrix.sum(axis=1)
        nonzero = np.where(row_sums > 0)[0]
        if len(nonzero) > 0:
            interval_data[i] = matrix[nonzero[0]]

    print(f"  Intervalli estratti: {len(interval_data)}")

    #TEST 1: Normalizzazioni diverse su KITM hard
    print("\nTest normalizzazioni su KITM hard...")

    for norm_name, X in [
        ("standard",  StandardScaler().fit_transform(kitm_hard)),
        ("l2",        normalize(kitm_hard, norm='l2')),
        ("maxabs",    MaxAbsScaler().fit_transform(kitm_hard)),
        ("none",      kitm_hard),
    ]:
        plot_umap(
            X.astype(np.float32), labels, args.composers,
            f"KITM hard — normalizzazione: {norm_name}",
            os.path.join(args.output_dir, f"umap_kitm_hard_{norm_name}.png")
        )

    #TEST 2: Soft assignment K-means
    print("\nCalcolo soft KITM K-means...")
    kitm_soft_km = soft_kitm_kmeans(pch_data, interval_data, kmeans, normalizer)
    print(f"  Shape: {kitm_soft_km.shape}, non-zero medio per vettore: "
          f"{(kitm_soft_km != 0).sum(axis=1).mean():.1f}/144")

    for norm_name, X in [
        ("standard", StandardScaler().fit_transform(kitm_soft_km)),
        ("l2",       normalize(kitm_soft_km, norm='l2')),
    ]:
        plot_umap(
            X.astype(np.float32), labels, args.composers,
            f"KITM soft K-means — normalizzazione: {norm_name}",
            os.path.join(args.output_dir, f"umap_kitm_soft_kmeans_{norm_name}.png")
        )

    #TEST 3: Soft assignment KS
    print("\nCalcolo soft KITM KS...")
    kitm_soft_ks = soft_kitm_ks(pch_data, interval_data)
    print(f"  Shape: {kitm_soft_ks.shape}, non-zero medio per vettore: "
          f"{(kitm_soft_ks != 0).sum(axis=1).mean():.1f}/144")

    for norm_name, X in [
        ("standard", StandardScaler().fit_transform(kitm_soft_ks)),
        ("l2",       normalize(kitm_soft_ks, norm='l2')),
    ]:
        plot_umap(
            X.astype(np.float32), labels, args.composers,
            f"KITM soft KS — normalizzazione: {norm_name}",
            os.path.join(args.output_dir, f"umap_kitm_soft_ks_{norm_name}.png")
        )

    print(f"\nCompletato — {8} grafici in {args.output_dir}")