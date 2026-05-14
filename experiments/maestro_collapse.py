"""
experiments/maestro_collapse.py

Studio empirico della sensibilità di MMD e OA al mode collapse su dati musicali reali da dataset MAESTRO
Setup:
    P: distribuzione reale bilanciata - pesi uniformi [1/3, 1/3, 1/3]
       campiona chunk reali con probabilità uguale da ogni compositore
    Q: stessa struttura ma pesi variati secondo Dir(K * c)
       simula un modello generativo che "dimentica" alcuni compositori (mode collapse)

Esecuzione: python experiments/maestro_collapse.py --features_dir data/features --feature_type pch --n_points 300 --n_samples 10000 --n_trials 50 --output_dir results/maestro_collapse
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

from device import get_device, move_to_device
from mmd import compute_mmd2
from utils import overlap_area_kde

#1. Caricamento feature dataset reale
def load_features(
        features_dir: str, #cartella con i file .npy
        feature_type: str, #'pch, 'pctm', o qualsiasi altra feature disponibile
        composers: list,
        n_max: int = 2000, #massimo chunk per compositore (per bilanciamento)
        seed: int = 0, #riproducibilità del subsample
) -> dict:
    features_dir = Path(features_dir)
    rng = np.random.default_rng(seed)
    features = {}

    print(f"\nCaricamento feature '{feature_type}':")
    for composer in composers:
        path = features_dir / f"{composer}_{feature_type}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"File non trovato: {path}\n"
                f"Assicurati di aver eseguito scripts/extract_features.py"
            )
        data = np.load(path).astype(np.float32)

        #Bilanciamento: usa al massimo n_max chunk per compositore
        if len(data) > n_max:
            idx = rng.choice(len(data), n_max, replace=False)
            data = data[idx]

        features[composer] = data
        print(f"  {composer}: {len(data)} chunk, "
              f"feature dim={data.shape[1]}")

    return features

#2. Campionamento pesato dal dataset tramite dado virtuale
def sample_from_dataset(
        features: dict, #{composer: [n_chunks, d]}
        weights: np.ndarray, #[n_composers] — pesi della distribuzione, sommano a 1
        n: int, #numero di chunk da campionare
        seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    composers = list(features.keys())

    #Conta quanti chunk prendere da ciascun compositore
    counts = rng.multinomial(n, weights)

    samples = []
    for composer, count in zip(composers, counts):
        if count == 0:
            continue
        pool = features[composer]
        #Campiona con rimpiazzo se count > pool disponibile
        replace = count > len(pool)
        idx = rng.choice(len(pool), count, replace=replace)
        samples.append(pool[idx])

    return np.vstack(samples)

#3. Collapse graduale dei pesi indotto con Dirichlet
def sample_dirichlet_weights(
        base_weights: np.ndarray,
        K: float,
        seed: int = 0,
) -> np.ndarray:
    """
    K grande -> pesi vicini a base_weights -> no collapse
    K piccolo -> pesi spinti verso un vertice -> collapse su un compositore
    """
    rng = np.random.default_rng(seed)
    alpha = K * base_weights
    return rng.dirichlet(alpha)

def collapse_degree(weights: np.ndarray) -> float:
    """
    Grado collapse - entropia normalizzata
    0 = nessun collapse, 1 = collapse totale
    """
    n = len(weights)
    w = np.clip(weights, 1e-10, 1.0)
    H = -np.sum(w * np.log(w))
    return float(1.0 - H / np.log(n))

#4. Test
def test_maestro_collapse(
        device: torch.device,
        features: dict,
        base_weights: np.ndarray,
        n_points: int = 300, #chunk per gruppo per MMD
        n_samples_oa: int = 10000, #chunk per la stima OA campionati dal dataset
        n_trials: int = 50, #ripetizioni per ogni K
        seed: int = 0,
) -> dict:
    K_values = np.logspace(2, -1, 20)

    results = {
        'K_values': [],
        'mmd2': [],
        'oa': [],
        'collapse_degree': [],
    }

    print(f"\n{'K':>8}  {'collapse':>10}  {'OA':>8}  "
          f"{'MMD²':>10}  {'pesi Q (media)'}")
    print("  " + "-" * 65)

    for i, K in enumerate(K_values):
        mmd2_trials = []
        oa_trials = []
        cd_trials = []
        w_trials = []

        for trial in range(n_trials):
            trial_seed = seed + i * 1000 + trial

            #Campiona pesi collassati da Dirichlet
            w_q = sample_dirichlet_weights(
                base_weights, K, seed=trial_seed
            )
            w_trials.append(w_q)
            cd_trials.append(collapse_degree(w_q))

            #Campiona P (pesi uniformi) e Q (pesi collassati) per MMD
            X_np = sample_from_dataset(
                features, base_weights, n_points, seed=trial_seed
            )
            Y_np = sample_from_dataset(
                features, w_q, n_points, seed=trial_seed + 500000
            )

            #MMD
            X = torch.from_numpy(X_np)
            Y = torch.from_numpy(Y_np)
            X, Y = move_to_device(X, Y, device=device)
            mmd2_val, _ = compute_mmd2(X, Y, device=device)
            mmd2_trials.append(float(mmd2_val))

            #Campiona P e Q per OA
            P_oa = sample_from_dataset(
                features, base_weights, n_samples_oa // 2, seed=trial_seed + 1000000
            )
            Q_oa = sample_from_dataset(
                features, w_q, n_samples_oa // 2, seed=trial_seed + 1500000
            )

            #OA (con KDE)
            oa = overlap_area_kde(P_oa, Q_oa, seed=trial_seed)
            oa_trials.append(oa)

        #Media sui trial
        mmd2_mean = float(np.mean(mmd2_trials))
        oa_mean = float(np.mean(oa_trials))
        cd_mean = float(np.mean(cd_trials))
        w_mean = np.mean(w_trials, axis=0)

        results['K_values'].append(float(K))
        results['mmd2'].append(mmd2_mean)
        results['oa'].append(oa_mean)
        results['collapse_degree'].append(cd_mean)

        if i % 4 == 0 or i == len(K_values) - 1:
            print(f"  {K:>8.3f}  {cd_mean:>10.3f}  {oa_mean:>8.4f}  "
                  f"{mmd2_mean:>10.6f}  {w_mean.round(3)}")

    return results

#5. Grafici
def plot_results(
        results: dict,
        feature_type: str,
        composers: list,
        n_trials: int,
        output_dir: str = "results/maestro_collapse",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cd = results['collapse_degree']
    mmd = results['mmd2']
    oa = results['oa']
    K = results['K_values']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Sensibilità di MMD² e OA al Mode Collapse — MAESTRO\n"
        f"(feature: {feature_type.upper()}, "
        f"compositori: {' / '.join(c.capitalize() for c in composers)}, "
        f"n_trials={n_trials})",
        fontsize=11
    )

    #Grafico 1: vs grado di collapse
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(cd, mmd, 'o-', color='steelblue', linewidth=2,
            markersize=4, label='MMD²')
    ax2.plot(cd, oa, 's--', color='darkorange', linewidth=2,
             markersize=4, label='OA')
    ax.set_xlabel("Grado di collapse  (0=nessuno, 1=totale)")
    ax.set_ylabel("MMD²", color='steelblue')
    ax2.set_ylabel("Overlap Area", color='darkorange')
    ax.set_title("MMD² e OA vs Grado di Collapse")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    #Annotazione r
    corr = float(np.corrcoef(oa, mmd)[0, 1])
    ax.text(0.05, 0.95, f"r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=13, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.15))

    #Grafico 2: vs K
    ax = axes[1]
    ax2 = ax.twinx()
    ax.semilogx(K, mmd, 'o-', color='steelblue', linewidth=2,
                markersize=4, label='MMD²')
    ax2.semilogx(K, oa, 's--', color='darkorange', linewidth=2,
                 markersize=4, label='OA')
    ax.set_xlabel("K (parametro Dirichlet, scala log)")
    ax.set_ylabel("MMD²", color='steelblue')
    ax2.set_ylabel("Overlap Area", color='darkorange')
    ax.set_title("MMD² e OA vs Parametro K")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    composers_str = "_".join(c for c in composers)
    path = os.path.join(output_dir, f"maestro_collapse_{feature_type}_{composers_str}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Grafico salvato: {path}")
    plt.close()

#6. MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/features", help="Cartella con i file .npy delle feature")
    parser.add_argument("--feature_type", type=str, default="pch", choices=["pch", "pctm", "avg_pitch", "pitch_range", "avg_interval", "interval_histogram"], help="Tipo di feature da usare: pch o pctm")
    parser.add_argument("--n_points", type=int, default=300)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_max", type=int, default=2000, help="Max chunk per compositore")
    parser.add_argument("--output_dir", type=str, default="results/maestro_collapse")
    parser.add_argument("--composers", nargs="+", default=['mozart', 'chopin', 'debussy'], help="Lista compositori da usare")
    args = parser.parse_args()

    print("=" * 60)
    print("Mode Collapse su MAESTRO — Dirichlet")
    print("=" * 60)

    device = get_device()

    COMPOSERS = args.composers
    n = len(COMPOSERS)
    BASE_WEIGHTS = np.array([1/n] * n)
    #Carica feature dal dataset reale
    features = load_features(
        args.features_dir,
        args.feature_type,
        composers = COMPOSERS,
        n_max = args.n_max,
    )

    results = test_maestro_collapse(
        device = device,
        features = features,
        base_weights = BASE_WEIGHTS,
        n_points = args.n_points,
        n_samples_oa = args.n_samples,
        n_trials = args.n_trials,
    )

    #Grafici
    plot_results(
        results,
        feature_type = args.feature_type,
        composers = COMPOSERS,
        n_trials = args.n_trials,
        output_dir = args.output_dir,
    )

    #Correlazione finale
    corr = float(np.corrcoef(results['oa'], results['mmd2'])[0, 1])
    print(f"\n  r(OA, MMD²) = {corr:.4f}")
    print(f"\n{'=' * 60}")
    print("Analisi completata.")
    print("=" * 60)