"""
Visualizza in 2D con UMAP i chunk reali di MAESTRO insieme ai chunk
generati dai diversi modelli, per confrontare visivamente quanto
ciascun modello si avvicina alla distribuzione reale.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import umap

# Colori: reali in grigio, generati in colori vivaci
REAL_COLOR = '#888888'
GEN_COLORS = {
    'maestro_huang':      '#E91E63',
    'maestro_plasser':    '#4CAF50',
    'maestro_nc':         '#2196F3',
    'scarlatti_nc':       '#2196F3',
    'scarlatti_plasser':  '#4CAF50',
    'pop909_nc':          '#2196F3',
    'pop909_plasser':     '#4CAF50',
}

GEN_LABELS = {
    'maestro_huang':      'Huang (generato)',
    'maestro_plasser':    'Plasser (generato)',
    'maestro_nc':         'NC (generato)',
    'scarlatti_nc':       'NC (generato)',
    'scarlatti_plasser':  'Plasser (generato)',
    'pop909_nc':          'NC (generato)',
    'pop909_plasser':     'Plasser (generato)',
}


def load_and_subsample(path: Path, n: int, seed: int = 42) -> np.ndarray:
    data = np.load(path).astype(np.float32)
    rng  = np.random.default_rng(seed)
    n    = min(n, len(data))
    idx  = rng.choice(len(data), n, replace=False)
    return data[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir_real", type=str, default="data/features")
    parser.add_argument("--features_dir_gen",  type=str, default="data/features_generated")
    parser.add_argument("--real_composers",    nargs="+",
                        default=['mozart', 'chopin', 'debussy'])
    parser.add_argument("--gen_datasets",      nargs="+",
                        default=['maestro_huang', 'maestro_plasser', 'maestro_nc'])
    parser.add_argument("--feature_type",      type=str,
                        default="pctm_w3.0_h1.5_concat")
    parser.add_argument("--n_per_group",       type=int, default=500)
    parser.add_argument("--n_neighbors",       type=int, default=15)
    parser.add_argument("--min_dist",          type=float, default=0.1)
    parser.add_argument("--output_dir",        type=str, default="results/umap")
    args = parser.parse_args()

    features_dir_real = Path(args.features_dir_real)
    features_dir_gen  = Path(args.features_dir_gen)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Feature: {args.feature_type}")
    print(f"Campioni per gruppo: {args.n_per_group}")

    # ── Carica reali ──────────────────────────────────────────────
    real_arrays = []
    for composer in args.real_composers:
        path = features_dir_real / f"{composer}_{args.feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        real_arrays.append(
            load_and_subsample(path, args.n_per_group // len(args.real_composers))
        )
    X_real = np.vstack(real_arrays)
    print(f"  Reali: {len(X_real)} chunk")

    # ── Carica generati ───────────────────────────────────────────
    gen_data = {}
    for dataset in args.gen_datasets:
        path = features_dir_gen / f"{dataset}_{args.feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        gen_data[dataset] = load_and_subsample(path, args.n_per_group)
        print(f"  {dataset}: {len(gen_data[dataset])} chunk")

    # ── Allinea dimensioni (troncamento al minimo) ────────────────
    all_arrays = [X_real] + list(gen_data.values())
    min_dim    = min(a.shape[1] for a in all_arrays)
    X_real     = X_real[:, :min_dim]
    for k in gen_data:
        gen_data[k] = gen_data[k][:, :min_dim]
    print(f"  Dimensione allineata: {min_dim}")

    # ── Costruisce dataset totale con etichette ───────────────────
    X_list      = [X_real]
    labels      = ['reale'] * len(X_real)

    for dataset, arr in gen_data.items():
        X_list.append(arr)
        labels += [dataset] * len(arr)

    X      = np.vstack(X_list)
    labels = np.array(labels)

    print(f"\nDataset totale: {len(X)} chunk, dim={X.shape[1]}")
    print("Calcolo UMAP...")

    reducer   = umap.UMAP(
        n_neighbors  = args.n_neighbors,
        min_dist     = args.min_dist,
        random_state = 42,
        verbose      = False,
    )
    embedding = reducer.fit_transform(X)

    # ── Grafico ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    # Plotta reali per primi (sotto) in grigio semitrasparente
    mask = labels == 'reale'
    ax.scatter(
        embedding[mask, 0], embedding[mask, 1],
        s=8, alpha=0.3, color=REAL_COLOR,
        label=f'Reale ({mask.sum()})',
        zorder=1,
    )

    # Plotta generati sopra con colori vivaci
    for dataset in args.gen_datasets:
        mask  = labels == dataset
        color = GEN_COLORS.get(dataset, '#FF9800')
        label = GEN_LABELS.get(dataset, dataset)
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=10, alpha=0.6, color=color,
            label=f'{label} ({mask.sum()})',
            zorder=2,
        )

    ax.set_title(
        f"UMAP — Reale vs Generato\n"
        f"feature: {args.feature_type}, {args.n_per_group} campioni/gruppo",
        fontsize=12
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    composers_str = "_".join(args.real_composers)
    datasets_str  = "_".join(d.replace('maestro_', '') for d in args.gen_datasets)
    fname         = f"umap_real_vs_gen_{args.feature_type}_{datasets_str}.png"
    path          = os.path.join(args.output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nGrafico salvato: {path}")
    plt.close()