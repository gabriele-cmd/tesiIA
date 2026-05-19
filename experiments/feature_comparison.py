"""
experiments/feature_comparison.py
Confronto tra approcci di combinazione feature multiple nel calcolo OA
1. OA media per feature: normalizza ogni feature separatamente e calcola OA per ciascuna, poi ne fa la media
2. OA su vettore concatenato: concatena i vettori normalizzati e ci calcola sopra l'OA

Normalizzazione supportata:
    - minmax:    x_norm = (x - min) / (max - min)  ->  ogni feature in [0,1]
    - standard:  x_norm = (x - mean) / std          ->  media=0, std=1

Esecuzione:
    python experiments/feature_comparison.py --features_dir data/features --composers mozart chopin debussy --normalizer minmax --n_samples 10000 --output_dir results/comparison
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import overlap_area_kde

#Feature vettoriali prese da Yang & Lerch
VECTOR_FEATURES = ['pch', 'pctm', 'nlh', 'nltm']

#Caricamento feature dei compositori
def load_features(
        features_dir: Path, #cartella con i file .npy
        composers: str, #nome del compositore
        feature_names: list, #lista di feature da caricare
        n_max: int = 2000, #massimo campioni
        seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    all_features = {}

    print(f"\nCaricamento feature {feature_names}:")

    for composer in composers:
        all_features[composer] = {}
        first_path = features_dir / f"{composer}_{feature_names[0]}.npy"
        first_data = np.load(first_path)
        n = min(len(first_data), n_max)
        idx = rng.choice(len(first_data), size=n, replace=False)

        for feat in feature_names:
            path = features_dir / f"{composer}_{feat}.npy"
            if not path.exists():
                raise FileNotFoundError(f"File non trovato: {path}")
            data = np.load(path).astype(np.float32)
            all_features[composer][feat] = data[idx]

        dims = [all_features[composer][f].shape[1] for f in feature_names]
        print(f" {composer}: {n} chunk, dims={dims}")

    return all_features

#Normalizza P e Q per ogni feature usando lo stesso scaler fittato sulla concatenazione di entrambi
def normalize_features(
        P_dict: dict, #{feat: [n, d]] primo composer
        Q_dict: dict, #{feat: [n, d]} secondo composer
        normalizer: str = 'minmax', #'minmax' o 'scaler'
) -> tuple:
    P_norm = {}
    Q_norm = {}

    for feat in P_dict:
        P = P_dict[feat]
        Q = Q_dict[feat]
        #fitta lo scaler su entrambi i dataset combinati
        combined = np.vstack([P, Q])

        if normalizer == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        scaler.fit(combined)
        P_norm[feat] = scaler.transform(P)
        Q_norm[feat] = scaler.transform(Q)

    return P_norm, Q_norm

#1. OA media
def oa_mean(
        P_norm: dict,
        Q_norm: dict,
        feature_names: list,
        seed: int = 0,
) -> dict:
    oa_per_feature = {}
    for feat in feature_names:
        oa_per_feature[feat] = overlap_area_kde(P_norm[feat], Q_norm[feat], seed=seed)

    oa_per_feature['mean'] = float(np.mean(list(oa_per_feature.values())))
    return oa_per_feature

#2. OA concatenato
def oa_concat(
        P_norm: dict,
        Q_norm: dict,
        feature_names: list,
        seed: int = 0,
) -> float:
    P_cat = np.hstack([P_norm[f] for f in feature_names])
    Q_cat = np.hstack([Q_norm[f] for f in feature_names])
    return float(overlap_area_kde(P_cat, Q_cat, seed=seed))

#Test
#Per ogni coppia di compositori e per la baseline (stesso compositore)
#calcola OA con entrambi gli approcci mediando su n_trials
def comparison(
        all_features: dict,
        feature_names: list,
        normalizer: str = 'minmax',
        n_samples: int = 10000,
        n_trials: int = 10,
        seed: int = 0,
) -> dict:
    composers = list(all_features.keys())
    pairs = list(combinations(composers, 2))
    baselines = [(c,c) for c in composers]
    all_pairs = pairs + baselines
    results = {}

    print(f"\n{'Coppia':<35} {'OA_mean':>9} {'OA_concat':>10}")
    print(" " + "-"*56)

    for comp_a, comp_b in all_pairs:
        label = (f"{comp_a} vs {comp_b} (baseline)"
                 if comp_a == comp_b
                 else f"{comp_a} vs {comp_b}")

        oa_mean_trials = []
        oa_concat_trials = []
        oa_feat_trials = {feat: [] for feat in feature_names}

        for trial in range(n_trials):
            trial_seed = seed + trial * 100
            rng = np.random.default_rng(trial_seed)

            if comp_a == comp_b:
                #Baseline: divide chunk in due meta'
                data_size = len(all_features[comp_a][feature_names[0]])
                half = data_size // 2
                idx_all = rng.permutation(data_size)
                idx_p = idx_all[:half]
                idx_q = idx_all[half:2*half]

                P_dict = {feat: all_features[comp_a][feat][idx_p] for feat in feature_names}
                Q_dict = {feat: all_features[comp_a][feat][idx_q] for feat in feature_names}
            else:
                #Coppia reale: subsample da ciascun compositore
                n = min(n_samples // 2, len(all_features[comp_a][feature_names[0]]))
                idx_a = rng.choice(len(all_features[comp_a][feature_names[0]]), size=n, replace=False)
                idx_b = rng.choice(len(all_features[comp_b][feature_names[0]]), size=n, replace=False)
                P_dict = {feat: all_features[comp_a][feat][idx_a] for feat in feature_names}
                Q_dict = {feat: all_features[comp_b][feat][idx_b] for feat in feature_names}

            #Normalizza
            P_norm, Q_norm = normalize_features(P_dict, Q_dict, normalizer)

            #Approccio 1
            res1 = oa_mean(P_norm, Q_norm, feature_names, seed=trial_seed)
            oa_mean_trials.append(res1['mean'])

            for feat in feature_names:
                oa_feat_trials[feat].append(res1[feat])

            #Approccio 2
            res2 = oa_concat(P_norm, Q_norm, feature_names, seed=trial_seed)
            oa_concat_trials.append(res2)

        oa_mean_final = float(np.mean(oa_mean_trials))
        oa_concat_final = float(np.mean(oa_concat_trials))
        oa_feat_final = {feat: float(np.mean(oa_feat_trials[feat])) for feat in feature_names}

        results[label] = {
            'oa_mean':        oa_mean_final,
            'oa_concat':      oa_concat_final,
            'oa_per_feature': oa_feat_final,
        }
        print(f"  {label:<33} {oa_mean_final:>9.4f} {oa_concat_final:>10.4f}")
    return results

#Grafici
def plot_results(
        results: dict,
        feature_names: list,
        composers: list,
        normalizer: str,
        n_trials: int,
        output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    composers_str = "_".join(composers)
    labels = list(results.keys())
    oa_means = [results[l]['oa_mean'] for l in labels]
    oa_concats = [results[l]['oa_concat'] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Confronto OA media vs OA concatenato — MAESTRO\n"
        f"compositori: {' / '.join(c.capitalize() for c in composers)}, "
        f"normalizer: {normalizer}, n_trials={n_trials}",
        fontsize=11
    )

    # Grafico 1: OA media vs OA concatenato per coppia
    ax = axes[0]
    ax.bar(x - width / 2, oa_means, width,
           label='OA media', color='darkorange', alpha=0.8)
    ax.bar(x + width / 2, oa_concats, width,
           label='OA concatenato', color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [l.replace(' vs ', '\nvs\n') for l in labels], fontsize=8
    )
    ax.set_ylabel("Overlap Area")
    ax.set_title("OA media vs OA concatenato per coppia")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    # Grafico 2: OA per feature individuale per coppia
    ax = axes[1]
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']
    n_feat = len(feature_names)
    width2 = 0.8 / n_feat

    for j, (feat, color) in enumerate(zip(feature_names, colors)):
        oa_vals = [results[l]['oa_per_feature'][feat] for l in labels]
        offset = (j - n_feat / 2 + 0.5) * width2
        ax.bar(x + offset, oa_vals, width2,
               label=feat.upper(), color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [l.replace(' vs ', '\nvs\n') for l in labels], fontsize=8
    )
    ax.set_ylabel("Overlap Area")
    ax.set_title("OA per feature individuale")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"comparison_{normalizer}_{composers_str}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Grafico salvato: {path}")
    plt.close()

#Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/features")
    parser.add_argument("--composers", nargs="+",
                        default=['mozart', 'chopin', 'debussy'])
    parser.add_argument("--features", nargs="+",
                        default=VECTOR_FEATURES)
    parser.add_argument("--normalizer", type=str, default="minmax",
                        choices=["minmax", "standard"])
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_max", type=int, default=2000)
    parser.add_argument("--output_dir", type=str,
                        default="results/comparison")
    args = parser.parse_args()

    print("=" * 60)
    print("Feature Comparison -- OA media vs OA concatenato")
    print("=" * 60)

    features_dir = Path(args.features_dir)

    all_features = load_features(
        features_dir, args.composers, args.features,
        n_max=args.n_max
    )

    results = comparison(
        all_features=all_features,
        feature_names=args.features,
        normalizer=args.normalizer,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
    )

    plot_results(
        results,
        feature_names=args.features,
        composers=args.composers,
        normalizer=args.normalizer,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
    )

    # Riepilogo finale
    print(f"\n{'=' * 60}")
    print("RIEPILOGO")
    print(f"\n{'Coppia':<35} {'OA_mean':>9} {'OA_concat':>10} {'diff':>8}")
    print("-" * 64)
    for label, res in results.items():
        oa_m = res['oa_mean']
        oa_c = res['oa_concat']
        print(f"  {label:<33} {oa_m:>9.4f} {oa_c:>10.4f} {oa_c - oa_m:>+8.4f}")

    print(f"\nOA per feature individuale:")
    print(f"  {'Coppia':<33}", end="")
    for feat in args.features:
        print(f"  {feat.upper():>8}", end="")
    print()
    print("  " + "-" * 60)
    for label, res in results.items():
        print(f"  {label:<33}", end="")
        for feat in args.features:
            print(f"  {res['oa_per_feature'][feat]:>8.4f}", end="")
        print()

    print("\nAnalisi completata.")