"""
Confronto tra distribuzione reale (P) e distribuzione generata (Q) di Maestro usando MMD e OA
P = frammenti reali di MAESTRO (Mozart + Chopin + Debussy combinati)
Q = chunk generati dal modello (huang, plasser, nc)

Calcola:
- MMD tra P e Q
- OA tra P e Q
- OA media per feature
- OA su vettore concatenato
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from device import get_device, move_to_device
from mmd import compute_mmd2
from utils import overlap_area_kde

#Caricamento feature reali e fa subsample
def load_real(
        features_dir: str,
        composers: list,
        feature_type: str,
        n_samples: int,
        seed: int = 0,
) -> np.ndarray:
    features_dir = Path(features_dir)
    arrays = []

    for composer in composers:
        path = features_dir / f"{composer}_{feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        arrays.append(np.load(path).astype(np.float32))

    P_all = np.vstack(arrays)
    print(f"Reali totali: {len(P_all)} frammenti")

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(P_all))
    idx = rng.choice(len(P_all), n, replace=False)
    P = P_all[idx]
    print(f"Subsample P: {len(P)} frammenti, dim={P.shape[1]}")
    return P

def load_generated(
        features_dir: str,
        dataset: str,
        feature_type: str,
        n_samples: int,
        seed: int = 0,
) -> np.ndarray:
    features_dir = Path(features_dir)
    path = features_dir / f"{dataset}_{feature_type}.npy"
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    Q_all = np.load(path).astype(np.float32)
    print(f"Generali totali: {len(Q_all)} frammenti")

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(Q_all))
    idx = rng.choice(len(Q_all), n, replace=False)
    Q = Q_all[idx]
    print(f"Subsample P: {len(Q)} frammenti, dim={Q.shape[1]}")
    return Q

#Normalizzazione
def normalize(P: np.ndarray, Q: np.ndarray, normalizer: str) -> tuple:
    combined = np.vstack([P, Q])
    if normalizer == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(combined)
    return scaler.transform(P), scaler.transform(Q)

#Test principale - Calcola MMD e OA tra P e Q su n_trials
def experiment(
        device: torch.device,
        P_raw: np.ndarray,
        Q_raw: np.ndarray,
        normalizer: str = 'standard',
        n_samples: int = 2000,
        n_outer: int = 5,
        n_inner: int = 10,
        baseline: bool = False,
        seed: int = 0,
) -> dict:
    # Allinea le dimensioni
    min_dim = min(P_raw.shape[1], Q_raw.shape[1])
    P_raw = P_raw[:, :min_dim]
    Q_raw = Q_raw[:, :min_dim]
    print(f"  Dimensione allineata: {min_dim}")

    outer_oa_means   = []
    outer_mmd2_means = []

    for outer in range(n_outer):
        outer_seed = seed + outer * 1000
        rng_outer  = np.random.default_rng(outer_seed)

        if baseline:
            idx_all = rng_outer.permutation(len(P_raw))
            Q_fixed = P_raw[idx_all[n_samples:n_samples*2]]
            P_pool  = P_raw[idx_all[:n_samples*4]]
        else:
            n_q     = min(n_samples, len(Q_raw))
            idx_q   = rng_outer.choice(len(Q_raw), n_q, replace=False)
            Q_fixed = Q_raw[idx_q]
            P_pool  = P_raw

        inner_oa   = []
        inner_mmd2 = []

        for inner in range(n_inner):
            inner_seed = outer_seed + inner * 100
            rng_inner  = np.random.default_rng(inner_seed)

            n_p   = min(n_samples, len(P_pool))
            idx_p = rng_inner.choice(len(P_pool), n_p, replace=False)
            P     = P_pool[idx_p]
            Q     = Q_fixed[:n_p]

            P_norm, Q_norm = normalize(P, Q, normalizer)

            X = torch.from_numpy(P_norm.astype(np.float32))
            Y = torch.from_numpy(Q_norm.astype(np.float32))
            X, Y = move_to_device(X, Y, device=device)
            mmd2_val, _ = compute_mmd2(X, Y, device=device)
            inner_mmd2.append(float(mmd2_val))

            oa = overlap_area_kde(P_norm, Q_norm, seed=inner_seed)
            inner_oa.append(float(oa))

        outer_oa_means.append(float(np.mean(inner_oa)))
        outer_mmd2_means.append(float(np.mean(inner_mmd2)))

    return {
        'mmd2':     float(np.mean(outer_mmd2_means)),
        'mmd2_std': float(np.std(outer_mmd2_means)),
        'oa':       float(np.mean(outer_oa_means)),
        'oa_std':   float(np.std(outer_oa_means)),
    }

#Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir_real", type=str, default="data/features")
    parser.add_argument("--features_dir_gen", type=str, default="data/features_generated")
    parser.add_argument("--real_composers", nargs="+",
                        default=['mozart', 'chopin', 'debussy'])
    parser.add_argument("--gen_dataset", type=str, default="maestro_huang")
    parser.add_argument("--feature_type", type=str, default="pctm")
    parser.add_argument("--normalizer", type=str, default="standard",
                        choices=["minmax", "standard"])
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--n_inner", type=int, default=10)
    parser.add_argument("--output_dir", type=str,
                        default="results/real_vs_generated")
    parser.add_argument("--baseline", action="store_true",
                        help="Modalità baseline: confronta P_reale vs P_reale diviso in due metà")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Reale vs Generato — MMD e OA")
    print("=" * 60)

    device = get_device()

    print(f"\nCaricamento feature '{args.feature_type}':")
    if args.baseline:
        print("\nModalità baseline — P vs P (stesso dataset diviso in due metà)")
        P_all = load_real(
            args.features_dir_real,
            args.real_composers,
            args.feature_type,
            n_samples=999999,
        )
        Q = P_all
        P = P_all
    else:
        Q = load_generated(
            args.features_dir_gen,
            args.gen_dataset,
            args.feature_type,
            args.n_samples,
        )
        P = load_real(
            args.features_dir_real,
            args.real_composers,
            args.feature_type,
            args.n_samples,
        )

    print(f"\nEsperimento (outer={args.n_outer}, inner={args.n_inner})...")
    results = experiment(
        device=device,
        P_raw=P,
        Q_raw=Q,
        normalizer=args.normalizer,
        n_samples=args.n_samples,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        baseline=args.baseline,
    )

    print(f"\n{'=' * 60}")
    print("RISULTATI")
    print(f"  P: MAESTRO reale ({', '.join(args.real_composers)})")
    print(f"  Q: {args.gen_dataset}")
    print(f"  Feature: {args.feature_type}")
    print(f"  Normalizer: {args.normalizer}")
    print(f"\n  MMD²: {results['mmd2']:.6f} ± {results['mmd2_std']:.6f}")
    print(f"  OA:   {results['oa']:.4f} ± {results['oa_std']:.4f}")
    print("=" * 60)