"""
scripts/umap_check.py
Il file esegue un SAFETY CHECK: visualizza i frammenti dei compositori in 2D con UMAP
per verificare che PCH e PCTHM formino cluster separati e ben distinguibili.

Se questi cluster sono ben separati per compositore, allora le feature sono sufficientemente
discriminative. Se i cluster si sovrappongono, le feature non distinguono correttamente i compositori.

Esecuzione:
    python scripts/umap_check.py --features_dir data/features
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
import sys
if '--interactive' in sys.argv:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import umap
from sklearn.preprocessing import StandardScaler

from utils import attach_interactive_picker, load_manifest

COMPOSER_COLORS = {
    'mozart':       '#2196F3',
    'chopin':       '#E91E63',
    'debussy':      '#4CAF50',
    'bach':         '#FF9800',
    'beethoven':    '#9C27B0',
    'schubert':     '#00BCD4',
    'liszt':        '#FF5722',
    'rachmaninoff': '#795548',
    'schumann':     '#607D8B',
    'haydn':        '#8BC34A',
}

N_MAX = 2000 #massimo di campioni per compositore (rispecchia il compositore con meno brani chunks disponibili, Mozart con 2055

#Carica le feature di tutti i compositori e costruisce il dataset con le etichette
def load_features(features_dir: Path, feature_type: str, composers: list) -> tuple:
    X_list = [] #[n_total, d] - feature di tutti i compositori
    labels = [] #[n_total] - indice del compositore (0,1,2)
    idx_per_composer = {}

    for i, composer in enumerate(composers):
        path = features_dir / f"{composer}_{feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

        data = np.load(path)

        #Bilanciamento usando al massimo N_MAX campioni per compositore
        if len(data) > N_MAX:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(data), N_MAX, replace=False)
            data = data[idx]
        else:
            idx = np.arange(len(data))

        idx_per_composer[composer] = idx
        X_list.append(data)
        labels.extend([i] * len(data))
        print(f" {composer} ({feature_type}): {len(data)} campioni")

    X = np.vstack(X_list)
    X = StandardScaler().fit_transform(X)
    return X, np.array(labels), idx_per_composer
    #return np.vstack(X_list), np.array(labels)

#Plotta la proiezione UMAP
def plot_umap(
        embedding: np.ndarray,
        labels: np.ndarray,
        title: str,
        output_path: Path,
        composers: list,
        colors: list,
        interactive: bool = False,
        all_paths: list = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_to_global = {}

    for i, (composer, color) in enumerate(zip(composers, colors)):
        mask = labels == i
        sc = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=5,
            alpha=0.4,
            color=color,
            label=composer.capitalize(),
            picker=True, pickradius=5,
        )
        scatter_to_global[sc] = np.where(mask)[0]

    ax.set_title(title, fontsize=13)
    ax.legend(markerscale=3, fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if interactive and all_paths:
        attach_interactive_picker(fig, ax, embedding, scatter_to_global, all_paths)
        plt.show()
    else:
        plt.savefig(output_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"Grafico salvato: {output_path}")

#Esegue la riduzione dimensionale UMAP da d a 2 dimensioni
def run_umap(
        X: np.ndarray, #[n, d]
        n_neighbors: int = 15, #controlla quanto è locale la struttura catturata. valori piccoli = struttura locale fine
        min_dist: float = 0.1 #distanza minima tra punti nella proiezione 2D. valori piccoli = cluster più compatti
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True, help="Cartella con i file .npy delle feature")
    parser.add_argument("--output_dir", default="results/umap", help="Cartella output per i grafici")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--composers", nargs="+", default=['mozart', 'chopin', 'debussy'], help="Lista compositori da visualizzare")
    parser.add_argument("--feature_type", type=str, default=None, help="Se specificato esegue UMAP solo su questa feature")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--manifest_dir", type=str, default=None)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    composers = args.composers
    colors = [COMPOSER_COLORS[c] for c in composers]
    composers_str = "_".join(c for c in composers)

    print("=" * 55)
    print("UMAP Safety Check — Cluster per compositore")
    print("=" * 55)

    composers_str = "_".join(c for c in composers)

    if args.feature_type is not None:
        # Modalità singola feature
        print(f"\nCaricamento feature '{args.feature_type}'...")
        X, labels, idx_per_composer = load_features(features_dir, args.feature_type, composers)

        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else features_dir
        all_paths = []
        for composer in composers:
            raw = load_manifest(manifest_dir, composer, f'_{args.feature_type}')
            if raw is None:
                #fallback: manifest senza prefisso tipo feature
                suffix_parts = args.feature_type.split('_', 1)
                if len(suffix_parts) > 1:
                    raw = load_manifest(manifest_dir, composer, f'_{suffix_parts[1]}')
            if raw is not None:
                all_paths.extend([raw[i] for i in idx_per_composer[composer]])
            else:
                print(f"  ATTENZIONE: manifest non trovato per {composer}")
                all_paths.extend([''] * len(idx_per_composer[composer]))

        print(f"  Dataset totale: {X.shape}")
        print(f"  Calcolo UMAP su {args.feature_type}...")
        emb = run_umap(X, args.n_neighbors, args.min_dist)
        plot_umap(
            emb, labels,
            title=f"UMAP — {args.feature_type.upper()}",
            output_path=output_dir / f"umap_{args.feature_type}_{composers_str}.png",
            composers=composers,
            colors=colors,
            interactive=args.interactive,
            all_paths=all_paths,
        )
    else:
        # Modalità default: PCH, PCTM e both
        print("\nCaricamento feature PCH...")
        X_pch, labels_pch, idx_per_composer = load_features(features_dir, 'pch', composers)

        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else features_dir
        all_paths = []
        for composer in composers:
            raw = load_manifest(manifest_dir, composer, '_pch')
            if raw:
                all_paths.extend([raw[i] for i in idx_per_composer[composer]])
            else:
                all_paths.extend([''] * len(idx_per_composer[composer]))

        print(f"  Dataset totale: {X_pch.shape}")
        print("  Calcolo UMAP su PCH...")
        emb_pch = run_umap(X_pch, args.n_neighbors, args.min_dist)
        plot_umap(
            emb_pch, labels_pch,
            title="UMAP — Pitch Class Histogram (PCH)",
            output_path=output_dir / f"umap_pch_{composers_str}.png",
            composers=composers,
            colors=colors,
            interactive=args.interactive,
            all_paths=all_paths,
        )

        print("\nCaricamento feature PCTM...")
        X_pctm, labels_pctm = load_features(features_dir, 'pctm', composers)
        print(f"  Dataset totale: {X_pctm.shape}")
        print("  Calcolo UMAP su PCTM...")
        emb_pctm = run_umap(X_pctm, args.n_neighbors, args.min_dist)
        plot_umap(
            emb_pctm, labels_pctm,
            title="UMAP — Pitch Class Transition Matrix (PCTM)",
            output_path=output_dir / f"umap_pctm_{composers_str}.png",
            composers=composers,
            colors=colors,
            interactive=args.interactive,
            all_paths=all_paths,
        )

        print("\nCalcolo UMAP su PCH + PCTM concatenati...")
        X_both = np.hstack([X_pch, X_pctm])
        print(f"  Dataset totale: {X_both.shape}")
        emb_both = run_umap(X_both, args.n_neighbors, args.min_dist)
        plot_umap(
            emb_both, labels_pch,
            title="UMAP — PCH + PCTM (concatenati)",
            output_path=output_dir / f"umap_both_{composers_str}.png",
            composers=composers,
            colors=colors,
            interactive=args.interactive,
            all_paths=all_paths,
        )

    print("\n" + "=" * 55)
    print("Safety check completato.")
    print("Verifica i grafici in:", output_dir)
    print("=" * 55)