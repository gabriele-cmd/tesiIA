"""
Testa la stabilità dei cluster UMAP al variare del subsample.
Per ogni trial viene estratto un subsample diverso dei chunk,
mantenendo il random_state di UMAP fisso.
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

COMPOSER_COLORS = {
    'mozart':    '#2196F3',
    'chopin':    '#E91E63',
    'debussy':   '#4CAF50',
    'bach':      '#FF9800',
    'beethoven': '#9C27B0',
}

def load_and_subsample(features_dir: Path, composer: str,
                       feature_type: str, n: int, seed: int) -> np.ndarray:
    path = features_dir / f"{composer}_{feature_type}.npy"
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    data = np.load(path).astype(np.float32)
    rng  = np.random.default_rng(seed)
    n    = min(n, len(data))
    idx  = rng.choice(len(data), n, replace=False)
    return data[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/features")
    parser.add_argument("--composers", nargs="+", default=["mozart", "chopin", "debussy"])
    parser.add_argument("--feature_type", type=str, default="pctm_w3.0_h1.5_concat")
    parser.add_argument("--n_per_composer", type=int, default=500)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--umap_seed", type=int, default=42,
                        help="Seed fisso per UMAP — varia solo il subsample")
    parser.add_argument("--output_dir", type=str, default="results/umap/stability")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Feature:      {args.feature_type}")
    print(f"Compositori:  {args.composers}")
    print(f"Trial:        {args.n_trials}")
    print(f"n/compositore: {args.n_per_composer}")
    print(f"UMAP seed:    {args.umap_seed} (fisso)")
    print()

    # Carica tutti i dati una volta sola
    all_data = {}
    for composer in args.composers:
        path = features_dir / f"{composer}_{args.feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        all_data[composer] = np.load(path).astype(np.float32)
        print(f"  {composer}: {len(all_data[composer])} chunk, dim={all_data[composer].shape[1]}")

    # Allinea dimensioni
    min_dim = min(d.shape[1] for d in all_data.values())
    all_data = {c: d[:, :min_dim] for c, d in all_data.items()}
    print(f"  Dimensione allineata: {min_dim}\n")

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.umap_seed,
        verbose=False,
    )

    for trial in range(args.n_trials):
        print(f"Trial {trial + 1}/{args.n_trials}...")

        # Subsample con seed diverso per ogni trial
        trial_seed = trial * 1000 + 7
        X_list, labels = [], []

        for composer in args.composers:
            data = all_data[composer]
            rng = np.random.default_rng(trial_seed + hash(composer) % 1000)
            n = min(args.n_per_composer, len(data))
            idx = rng.choice(len(data), n, replace=False)
            X_list.append(data[idx])
            labels += [composer] * n

        X = np.vstack(X_list)
        labels = np.array(labels)

        embedding = reducer.fit_transform(X)

        # Grafico
        fig, ax = plt.subplots(figsize=(8, 6))

        for composer in args.composers:
            mask = labels == composer
            color = COMPOSER_COLORS.get(composer, '#888888')
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                s=8, alpha=0.5, color=color,
                label=f"{composer.capitalize()} ({mask.sum()})",
            )

        composers_str = "_".join(args.composers)
        ax.set_title(
            f"UMAP Stability Test — Trial {trial + 1}/{args.n_trials}\n"
            f"feature: {args.feature_type}, seed subsample: {trial_seed}",
            fontsize=11
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=2, fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        fname = (f"stability_{args.feature_type}_{composers_str}"
                 f"_trial{trial + 1:02d}.png")
        path = os.path.join(args.output_dir, fname)
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Salvato: {path}")

    print(f"\nCompletato — {args.n_trials} grafici in {args.output_dir}")