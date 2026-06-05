"""
Visualizza come la distribuzione Q collassa progressivamente nello spazio UMAP
al variare del parametro K di Dirichlet

Per ogni valore di K genera un gragico UMAP con P (reale, in grigio) e Q (collassato, colorato per compositore dominante)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import umap

COMPOSER_COLORS = {
    'mozart':    '#2196F3',
    'chopin':    '#E91E63',
    'debussy':   '#4CAF50',
    'bach':      '#FF9800',
    'beethoven': '#9C27B0',
}

K_VALUES = [100.0, 1.274, 0.100]

#Campiona n_points dalla distribuzione collassata Q con param. K.
#Restituisce i vettori campionati e l'indice del compostiore dominante.
def dirichlet_collapse(composers_data: dict, K: float, n_points: int, seed: int = 42) -> tuple:
    composers = list(composers_data.keys())
    n_composers = len(composers)
    rng = np.random.default_rng(seed)

    alpha = np.ones(n_composers) * K
    weights = rng.dirichlet(alpha)

    n_per_composer = (weights * n_points).astype(int)
    n_per_composer[-1] = n_points - n_per_composer[:-1].sum()

    samples = []
    composer_labels = []
    for i, composer in enumerate(composers):
        data = composers_data[composer]
        n = n_per_composer[i]
        if n > 0:
            idx = rng.choice(len(data), n, replace=True)
            samples.append(data[idx])
            composer_labels.extend([composer] * n)

    return np.vstack(samples), composer_labels, weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/features")
    parser.add_argument("--composers", nargs="+",
                        default=["mozart", "chopin", "debussy"])
    parser.add_argument("--feature_type", type=str, default="pctm_w3.0_h1.5_concat")
    parser.add_argument("--n_per_group", type=int, default=300)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--umap_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="results/umap/collapse")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Carica dati
    composers_data = {}
    for composer in args.composers:
        path = features_dir / f"{composer}_{args.feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        composers_data[composer] = np.load(path).astype(np.float32)
        print(f"  {composer}: {len(composers_data[composer])} chunk")

    # Allinea dimensioni
    min_dim = min(d.shape[1] for d in composers_data.values())
    composers_data = {c: d[:, :min_dim] for c, d in composers_data.items()}
    print(f"  Dimensione allineata: {min_dim}")

    # Campiona P reale
    rng = np.random.default_rng(args.umap_seed)
    P_list = []
    for composer in args.composers:
        data = composers_data[composer]
        n = min(args.n_per_group, len(data))
        idx = rng.choice(len(data), n, replace=False)
        P_list.append(data[idx])
    P = np.vstack(P_list)
    print(f"\nP reale: {len(P)} chunk")

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.umap_seed,
        verbose=False,
    )

    print("Fitting UMAP su P...")
    emb_P = reducer.fit_transform(P)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    for idx_k, K in enumerate(K_VALUES):
        Q, q_labels, weights = dirichlet_collapse(
            composers_data, K, len(P), seed=args.umap_seed + idx_k * 17
        )

        emb_Q = reducer.transform(Q)

    for idx_k, K in enumerate(K_VALUES):
        collapse = 1.0 - 1.0 / (1.0 + np.exp(-2.0 * (K - 1.0)))
        # Collapse semplificato: proporzione approssimativa
        # Usa formula diretta
        n_comp = len(args.composers)
        max_w = 1.0 / (1.0 + (n_comp - 1) * (K / (K + 1)))
        collapse_approx = max(0, (max_w - 1 / n_comp) / (1 - 1 / n_comp))

        Q, q_labels, weights = dirichlet_collapse(
            composers_data, K, len(P), seed=args.umap_seed + idx_k * 17
        )
        ax = axes[idx_k]

        # Plotta P in grigio
        ax.scatter(emb_P[:, 0], emb_P[:, 1],
                   s=6, alpha=0.25, color='#AAAAAA', label='P reale', zorder=1)

        # Plotta Q colorato per compositore
        q_labels_arr = np.array(q_labels)
        for composer in args.composers:
            mask = q_labels_arr == composer
            color = COMPOSER_COLORS.get(composer, '#888888')
            n_c = mask.sum()
            if n_c > 0:
                ax.scatter(emb_Q[mask, 0], emb_Q[mask, 1],
                           s=8, alpha=0.5, color=color,
                           label=f'{composer.capitalize()} ({n_c})', zorder=2)

        w_str = " ".join([f"{w:.2f}" for w in weights])
        ax.set_title(f"K={K:.3f}  |  pesi: [{w_str}]", fontsize=9)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

        if idx_k == 0:
            ax.legend(markerscale=2, fontsize=7, loc='lower right')

    composers_str = "_".join(args.composers)
    fig.suptitle(
        f"Mode Collapse Progressivo — {args.feature_type}\n"
        f"P reale (grigio) vs Q collassato (colorato per compositore)",
        fontsize=12
    )
    plt.tight_layout()

    fname = f"umap_collapse_{args.feature_type}_{composers_str}.png"
    path = os.path.join(args.output_dir, fname)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nGrafico salvato: {path}")