"""
Addestra K-means sui PCH dei compositori specificati e salva il modello.
Il modello include anche la mappa cluster→tonalità calcolata con argmax e KS.

Default: maestro_train_val, k=12, L2
python scripts/train_kmeans_key.py \
    --features_dir data/features_real \
    --composers maestro_train_val \
    --n_clusters 12 \
    --output_path models/kmeans_key_final.pkl \
    --plot_path results/kmeans_centroids_final.png
"""

import argparse
import numpy as np
import pickle
import os
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profili Krumhansl-Schmuckler per tonalità maggiori (Krumhansl 1990)
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])


def ks_key(centroid: np.ndarray) -> tuple:
    """Identifica tonalità del centroide tramite correlazione KS. Restituisce (key_idx, r)."""
    best_r, best_key = -np.inf, 0
    for k in range(12):
        profile = np.roll(KS_MAJOR, k)
        r = np.corrcoef(centroid, profile)[0, 1]
        if r > best_r:
            best_r, best_key = r, k
    return best_key, best_r


def build_cluster_mappings(kmeans: KMeans) -> tuple:
    """
    Costruisce due mappe cluster→tonalità:
      - argmax: tonalità = nota con peso massimo nel centroide
      - ks:     tonalità = correlazione KS massima con profili psicoacustici
    Restituisce (cluster_to_key_argmax, cluster_to_key_ks).
    """
    cluster_to_key_argmax = {}
    cluster_to_key_ks = {}
    for i, centroid in enumerate(kmeans.cluster_centers_):
        cluster_to_key_argmax[i] = int(np.argmax(centroid))
        cluster_to_key_ks[i], _ = ks_key(centroid)
    return cluster_to_key_argmax, cluster_to_key_ks


def load_pch(features_dir: str, composers: list) -> np.ndarray:
    arrays = []
    for composer in composers:
        path = Path(features_dir) / f"{composer}_pch.npy"
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        arr = np.load(path).astype(np.float32)
        arrays.append(arr)
        print(f"  {composer}: {len(arr)} frammenti")
    return np.vstack(arrays)


def plot_centroids(kmeans: KMeans, labels: np.ndarray, output_path: str,
                   composers: list, n_clusters: int,
                   cluster_to_key_argmax: dict, cluster_to_key_ks: dict):
    cols = 4
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, centroid in enumerate(kmeans.cluster_centers_):
        n = np.sum(labels == i)
        key_argmax = NOTE_NAMES[cluster_to_key_argmax[i]]
        key_ks, r_ks = ks_key(centroid)
        key_ks_name = NOTE_NAMES[key_ks]
        axes[i].bar(NOTE_NAMES, centroid, color='steelblue')
        axes[i].set_title(
            f'Cluster {i} | argmax: {key_argmax} | KS: {key_ks_name} (r={r_ks:.2f})\nn={n}',
            fontsize=7
        )
        axes[i].set_ylim(0, 0.65)
        axes[i].tick_params(labelsize=7)

    for j in range(n_clusters, len(axes)):
        axes[j].set_visible(False)

    composers_str = "+".join(composers)
    fig.suptitle(
        f'K-means k={n_clusters}, normalizzazione L2\n'
        f'Compositori: {composers_str}',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Grafico salvato: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/features_real")
    parser.add_argument("--composers", nargs="+", default=["maestro_train_val"])
    parser.add_argument("--n_clusters", type=int, default=12)
    parser.add_argument("--normalizer", type=str, default="l2",
                        choices=["l2", "none"])
    parser.add_argument("--n_init", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_path", type=str,
                        default="models/kmeans_key_final.pkl")
    parser.add_argument("--plot_path", type=str,
                        default="results/kmeans_centroids_final.png")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Subsample casuale prima del training (default: usa tutto)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)

    print("=" * 50)
    print("Training K-means per key detection")
    print(f"  Compositori:  {args.composers}")
    print(f"  n_clusters:   {args.n_clusters}")
    print(f"  normalizer:   {args.normalizer}")
    print("=" * 50)

    print("\nCaricamento PCH:")
    pch_all = load_pch(args.features_dir, args.composers)
    print(f"  Totale: {len(pch_all)} chunk")

    if args.normalizer == 'l2':
        pch_fit = normalize(pch_all, norm='l2')
        print("  Normalizzazione L2 applicata")
    else:
        pch_fit = pch_all

    if args.n_samples is not None and args.n_samples < len(pch_fit):
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(len(pch_fit), args.n_samples, replace=False)
        pch_fit = pch_fit[idx]
        print(f"  Subsample: {len(pch_fit)} chunk")

    print(f"\nTraining K-means (n_init={args.n_init})...")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        n_init=args.n_init,
    )
    kmeans.fit(pch_fit)
    print(f"  Inertia: {kmeans.inertia_:.4f}")

    # Costruisce mappe cluster→tonalità
    cluster_to_key_argmax, cluster_to_key_ks = build_cluster_mappings(kmeans)

    # Verifica biiezione
    keys_argmax = list(cluster_to_key_argmax.values())
    keys_ks     = list(cluster_to_key_ks.values())
    bij_argmax  = len(set(keys_argmax)) == 12
    bij_ks      = len(set(keys_ks)) == 12
    print(f"\n  Biiezione argmax: {bij_argmax} ({len(set(keys_argmax))}/12 tonalità distinte)")
    print(f"  Biiezione KS:     {bij_ks} ({len(set(keys_ks))}/12 tonalità distinte)")

    model_data = {
        'kmeans':               kmeans,
        'normalizer':           args.normalizer,
        'n_clusters':           args.n_clusters,
        'composers':            args.composers,
        'random_state':         args.random_state,
        'cluster_to_key_argmax': cluster_to_key_argmax,
        'cluster_to_key_ks':    cluster_to_key_ks,
    }
    with open(args.output_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModello salvato: {args.output_path}")

    plot_centroids(kmeans, kmeans.labels_, args.plot_path,
                   args.composers, args.n_clusters,
                   cluster_to_key_argmax, cluster_to_key_ks)

    print('\nCluster → tonalità (argmax | KS):')
    for i, centroid in enumerate(kmeans.cluster_centers_):
        ka = NOTE_NAMES[cluster_to_key_argmax[i]]
        kk = NOTE_NAMES[cluster_to_key_ks[i]]
        _, r = ks_key(centroid)
        n = np.sum(kmeans.labels_ == i)
        match = "✓" if cluster_to_key_argmax[i] == cluster_to_key_ks[i] else "✗"
        print(f'  Cluster {i:2d}: argmax={ka:3s} | KS={kk:3s} (r={r:.3f}) {match}  n={n}')