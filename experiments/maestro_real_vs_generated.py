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
        n_trials: int = 10,
        seed: int = 0,
) -> dict:
    mmd2_trials = []
    oa_trials = []

    for trial in range(n_trials):
        trial_seed = seed + trial * 100
        rng = np.random.default_rng(trial_seed)

        #Subsample per trial
        n = min(len(P_raw), len(Q_raw))
        idx_p = rng.choice(len(P_raw), n, replace=False)
        idx_q = rng.choice(len(Q_raw), n, replace=False)
        P = P_raw[idx_p]
        Q = Q_raw[idx_q]

        #Normalizza
        P_norm, Q_norm = normalize(P, Q, normalizer)

        #MMD
        X = torch.from_numpy(P_norm.astype(np.float32))
        Y = torch.from_numpy(Q_norm.astype(np.float32))
        X, Y = move_to_device(X, Y, device=device)
        mmd2_val, _ = compute_mmd2(X, Y, device=device)
        mmd2_trials.append(float(mmd2_val))

        #OA
        oa = overlap_area_kde(P_norm, Q_norm, seed=trial_seed)
        oa_trials.append(float(oa))

    return {
        'mmd2': float(np.mean(mmd2_trials)),
        'mmd2_std': float(np.std(mmd2_trials)),
        'oa': float(np.mean(oa_trials)),
        'oa_std': float(np.std(oa_trials)),
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
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--output_dir", type=str,
                        default="results/real_vs_generated")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Reale vs Generato — MMD e OA")
    print("=" * 60)

    device = get_device()

    print(f"\nCaricamento feature '{args.feature_type}':")
    P = load_real(
        args.features_dir_real,
        args.real_composers,
        args.feature_type,
        args.n_samples,
    )
    Q = load_generated(
        args.features_dir_gen,
        args.gen_dataset,
        args.feature_type,
        args.n_samples,
    )

    print(f"\nEsperimento ({args.n_trials} trial)...")
    results = experiment(
        device=device,
        P_raw=P,
        Q_raw=Q,
        normalizer=args.normalizer,
        n_trials=args.n_trials,
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