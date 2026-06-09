"""
Confronto OA media vs OA concatenato usando P=reale e Q=generato.
Risponde alla domanda: su dataset diversi, OA concatenato è sempre
più discriminativo di OA media per distinguere reale da generato?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from device import get_device, move_to_device
from mmd import compute_mmd2
from utils.utils_oa import overlap_area_kde


def load_features(features_dir: str, dataset: str, feature: str) -> np.ndarray:
    path = Path(features_dir) / f"{dataset}_{feature}.npy"
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    return np.load(path).astype(np.float32)


def normalize(P: np.ndarray, Q: np.ndarray, normalizer: str) -> tuple:
    combined = np.vstack([P, Q])
    scaler = StandardScaler() if normalizer == 'standard' else MinMaxScaler()
    scaler.fit(combined)
    return scaler.transform(P), scaler.transform(Q)


def compute_oa_mmd(P: np.ndarray, Q: np.ndarray, device, normalizer: str,
                   n_samples: int, n_trials: int, seed: int = 0) -> dict:
    oa_trials, mmd_trials = [], []
    rng = np.random.default_rng(seed)

    # Allinea dimensioni
    min_dim = min(P.shape[1], Q.shape[1])
    P = P[:, :min_dim]
    Q = Q[:, :min_dim]

    for trial in range(n_trials):
        trial_seed = seed + trial * 100
        rng_t = np.random.default_rng(trial_seed)
        n = min(n_samples, len(P), len(Q))
        p = P[rng_t.choice(len(P), n, replace=False)]
        q = Q[rng_t.choice(len(Q), n, replace=False)]

        p_norm, q_norm = normalize(p, q, normalizer)

        X = torch.from_numpy(p_norm)
        Y = torch.from_numpy(q_norm)
        X, Y = move_to_device(X, Y, device=device)
        mmd2, _ = compute_mmd2(X, Y, device=device)
        mmd_trials.append(float(mmd2))
        oa_trials.append(float(overlap_area_kde(p_norm, q_norm, seed=trial_seed)))

    return {
        'oa':   float(np.mean(oa_trials)),
        'oa_std': float(np.std(oa_trials)),
        'mmd2': float(np.mean(mmd_trials)),
        'mmd2_std': float(np.std(mmd_trials)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir_real", type=str, default="data/features_real")
    parser.add_argument("--features_dir_gen",  type=str, default="data/features_generated")
    parser.add_argument("--real_dataset",      type=str, required=True)
    parser.add_argument("--gen_dataset",       type=str, required=True)
    parser.add_argument("--features",          nargs="+", default=["pch", "pctm"])
    parser.add_argument("--normalizer",        type=str, default="standard",
                        choices=["standard", "minmax"])
    parser.add_argument("--n_samples",         type=int, default=1000)
    parser.add_argument("--n_trials",          type=int, default=20)
    parser.add_argument("--output_dir",        type=str,
                        default="results/comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()

    print("=" * 60)
    print(f"Feature Comparison — OA media vs OA concatenato")
    print(f"  P (reale):   {args.real_dataset}")
    print(f"  Q (generato): {args.gen_dataset}")
    print("=" * 60)

    results = {}

    for feat in args.features:
        # Statico
        P_static = load_features(args.features_dir_real, args.real_dataset, feat)
        Q_static = load_features(args.features_dir_gen,  args.gen_dataset,  feat)
        res_static = compute_oa_mmd(P_static, Q_static, device,
                                    args.normalizer, args.n_samples, args.n_trials)
        results[f"{feat}_static"] = res_static

        # Cerca concat disponibili
        for suffix in ["_w2.0_h1.0_concat", "_w3.0_h1.5_concat",
                        "_b2_h1_concat", "_b4_h2_concat"]:
            try:
                P_c = load_features(args.features_dir_real, args.real_dataset,
                                    feat + suffix)
                Q_c = load_features(args.features_dir_gen,  args.gen_dataset,
                                    feat + suffix)
                res_c = compute_oa_mmd(P_c, Q_c, device, args.normalizer,
                                       args.n_samples, args.n_trials)
                results[f"{feat}{suffix}"] = res_c
            except FileNotFoundError:
                pass

    # Stampa riepilogo
    print(f"\n{'Feature':<30} {'OA':>8} {'±':>6} {'MMD²':>10}")
    print("-" * 58)
    for feat_key, res in results.items():
        print(f"  {feat_key:<28} {res['oa']:>8.4f} {res['oa_std']:>6.4f} "
              f"{res['mmd2']:>10.6f}")

    print("=" * 60)