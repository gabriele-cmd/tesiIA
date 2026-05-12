"""
ESEGUIRE: python3 experiments/square_collapse.py
================================
Sensibilità di MMD e OA al mode collapse su distribuzione
a 4 gaussiane poste sui vertici di un quadrato (2D).

Basato sul setup di Eleonora, con collapse indotto tramite
distribuzione di Dirichlet invece che rimozione discreta delle mode.

P: distribuzione reale, 4 gaussiane bilanciate sui vertici
Q: stessa struttura ma pesi variati con Dir(K · c)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from device import get_device, move_to_device
from mmd import compute_mmd2
from utils import overlap_area_kde

#1. Setup Quadrato
#4 gaussiane isotropiche sui vertici di un quadrato centrato nell'origine
VERTICES = [
    np.array([-4.0, -4.0]),
    np.array([4.0, -4.0]),
    np.array([-4.0, 4.0]),
    np.array([4.0, 4.0]),
]
SIGMA_DATA = 0.5 #std delle gaussiane - invariato
N_SAMPLES = 400 #campioni totali - invariato
BASE_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25]) #pesi base uniformi per non privilegiare nessuna moda in P (distr. reale)

#2. Generazione campioni e funzioni di supporto
def sample_square(
        weights: np.ndarray, #[4] — pesi delle componenti, devono sommare a 1
        n: int, #numero di campioni totali
        seed: int = 0
) -> np.ndarray:
    """
    Campiona n punti dalla miscela di 4 gaussiane con pesi dati.
    """
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, weights)
    samples = []
    for mu, count in zip(VERTICES, counts):
        if count > 0:
            pts = mu + SIGMA_DATA * rng.standard_normal((count, 2))
            samples.append(pts)
    return np.vstack(samples)

def sample_dirichlet_weights(
        base_weights: np.ndarray,
        K: float, #parametro variabile
        seed: int = 0,
) -> np.ndarray:
    """
    Campiona pesi collassati da Dir(K · base_weights).
    Stesso di mode_collapse_analysis.py.
    """
    rng = np.random.default_rng(seed)
    alpha = K * base_weights #iperparametro
    return rng.dirichlet(alpha)

def collapse_degree(weights: np.ndarray) -> float:
    """
    Grado di collapse: 1 - entropia normalizzata.
    0 = nessun collapse, 1 = collapse totale.
    Stesso di mode_collapse_analysis.py.
    """
    n = len(weights)
    w = np.clip(weights, 1e-10, 1.0)
    H = -np.sum(w * np.log(w)) #Hentropy
    return float(1.0 - H / np.log(n))

#3. Overlap Area
def overlap_area_square(
        weights_q: np.ndarray, #pesi di Q DOPO COLLAPSE
        n_samples: int = 10000, #campioni Monte Carlo
        seed: int = 0, #riproducibilità
) -> float:
    """
    Stima OA tra P (pesi uniformi) e Q (pesi collassati)
    nello spazio 2D con campionamento Monte Carlo.
    """
    P = sample_square(BASE_WEIGHTS, n_samples // 2, seed=seed)
    Q = sample_square(weights_q, n_samples // 2, seed=seed + 99999)
    return overlap_area_kde(P, Q, seed = seed)

#4. Esperimento collapse con Dirichlet
def experiment_square_collapse(
        device: torch.device,
        n_points: int = 400, #campioni per gruppo per MMD
        n_samples_oa: int = 10000, #campioni Monte Carlo per OA
        n_trials: int = 50, #ripetizioni per ogni K
        seed: int = 0, #per riproducibilità
) -> dict:
    """
    Calcola MMD e OA al variare del parametro K di Dirichlet
    sulla distribuzione a 4 gaussiane sul quadrato.
    """
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
            #Campioni pesi collassati
            w_q = sample_dirichlet_weights(
                BASE_WEIGHTS, K, seed=seed + i * 100 + trial
            )
            w_trials.append(w_q)

            cd = collapse_degree(w_q)
            cd_trials.append(cd)

            #OA nello spazio 2D
            oa = overlap_area_square(
                w_q,
                n_samples = n_samples_oa,
                seed = seed + i * 100 + trial,
            )
            oa_trials.append(oa)

            #Genera campioni P e Q per MMD
            X_np = sample_square(
                BASE_WEIGHTS, n_points,
                seed = seed + i * 100 + trial,
            ).astype(np.float32)

            Y_np = sample_square(
                w_q, n_points,
                seed=seed + i * 100 + trial + 50000
            ).astype(np.float32)

            X = torch.from_numpy(X_np)
            Y = torch.from_numpy(Y_np)
            X, Y = move_to_device(X, Y, device=device)

            mmd2_val, _ = compute_mmd2(X, Y, device=device)
            mmd2_trials.append(mmd2_val)

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

#5. Grafici e Main
#GRAFICO ANDAMENTO COLLAPSE DEGREE
def plot_square_collapse(results: dict, output_dir: str = "results", n_trials: int = 100) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cd  = results['collapse_degree']
    mmd = results['mmd2']
    oa  = results['oa']
    K   = results['K_values']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Sensibilità di MMD² e OA al Mode Collapse\n"
        f"(Quadrato 2D, 4 gaussiane, Dirichlet — n_trials={n_trials})",
        fontsize=12
    )

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
    path = os.path.join(output_dir, "square_collapse.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Grafico salvato in: {path}")
    plt.close()

#GRAFICO SCATTER 2D + KDE DISTANZE
def plot_kde(results: dict, output_dir: str = "results", seed: int = 0, n_trials: int = 100) -> None:
    """
    Per 4 valori rappresentativi di K mostra:
        - sinistra: scatter P (blu) e Q (colorato) nello spazio 2D
        - destra:   KDE di D_PP, D_QQ, D_PQ con area overlap colorata
    Identico allo stile del file di Eleonora.
    """
    from scipy.stats import gaussian_kde

    os.makedirs(output_dir, exist_ok=True)
    #Sceglie 4 valori di K rappresentativi
    K_all = results['K_values']
    #indici: primo (nessun collapse), due intermedi, ultimo (collapse forte)
    indices = [0,
               len(K_all) // 2,
               len(K_all) -1,]
    colors = ['#34d399', '#f472b6', '#f87171']
    labels = ['Nessun collapse', 'Collapse parziale', 'Collapse forte']

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        "Mode Collapse sul Quadrato 2D\n"
        f"(Dirichlet, median heuristic σ — n_trials={n_trials})",
        fontsize=12
    )

    P = sample_square(BASE_WEIGHTS, N_SAMPLES, seed=seed).astype(np.float32)

    for row, (idx, color, label) in enumerate(zip(indices, colors, labels)):
        K = K_all[idx]
        cd = results['collapse_degree'][idx]
        oa = results['oa'][idx]
        mmd = results['mmd2'][idx]

        #Campiona Q con pesi Dirichlet per questo K
        w_q = sample_dirichlet_weights(BASE_WEIGHTS, K, seed=seed + idx)
        Q = sample_square(w_q, N_SAMPLES,
                          seed=seed + idx + 50000).astype(np.float32)

        #Scatter 2D
        ax = axes[row, 0]
        ax.scatter(P[:, 0], P[:, 1], s=10, alpha=0.3,
                   color='steelblue', label='P (reale)')
        ax.scatter(Q[:, 0], Q[:, 1], s=10, alpha=0.5,
                   color=color, label='Q (generata)')
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_title(f"{label}  (K={K:.3f}, collapse={cd:.3f})")
        ax.legend(markerscale=2)
        ax.grid(True, alpha=0.2)

        #KDE distanze
        ax = axes[row, 1]

        #Distanze intra-P
        i_idx, j_idx = np.triu_indices(len(P), k=1)
        d_pp = np.linalg.norm(P[i_idx] - P[j_idx], axis=1)

        #Distanze intra-Q
        i_idx, j_idx = np.triu_indices(len(Q), k=1)
        d_qq = np.linalg.norm(Q[i_idx] - Q[j_idx], axis=1)

        #Distanze inter P-Q (subsample per velocità)
        rng = np.random.default_rng(seed + idx)
        p_sub = P[rng.choice(len(P), min(200, len(P)), replace=False)]
        q_sub = Q[rng.choice(len(Q), min(200, len(Q)), replace=False)]
        d_pq = np.linalg.norm(
            p_sub[:, None, :] - q_sub[None, :, :], axis=2
        ).ravel()

        grid = np.linspace(0, max(d_pp.max(), d_qq.max(), d_pq.max()), 500)

        kde_pp = gaussian_kde(d_pp)(grid)
        kde_qq = gaussian_kde(d_qq)(grid)
        kde_pq = gaussian_kde(d_pq)(grid)

        ax.plot(grid, kde_pp, label='D_PP', color='steelblue')
        ax.plot(grid, kde_qq, label='D_QQ', color=color)
        ax.plot(grid, kde_pq, label='D_PQ', color='gray', linestyle='--')
        ax.fill_between(grid, np.minimum(kde_pp, kde_pq), alpha=0.2,
                        color='steelblue')
        ax.set_title(f"OA={oa:.3f}   |   MMD²={mmd:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "square_collapse_eleonora_style.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Grafico scatter salvato in: {path}")
    plt.close()

if __name__ == "__main__":
    import time
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_points",   type=int, default=400)
    parser.add_argument("--n_samples",  type=int, default=10000)
    parser.add_argument("--n_trials",   type=int, default=50)
    parser.add_argument("--output_dir", type=str,
                        default="results/square_collapse")
    args = parser.parse_args()

    print("="*55)
    print("Analisi Mode Collapse — Quadrato 2D con Dirichlet")
    print("="*55)

    device = get_device()

    results = experiment_square_collapse(
        device       = device,
        n_points     = args.n_points,
        n_samples_oa = args.n_samples,
        n_trials     = args.n_trials,
    )

    plot_square_collapse(results, output_dir=args.output_dir)
    plot_kde(results, output_dir=args.output_dir)

    corr = float(np.corrcoef(results['oa'], results['mmd2'])[0, 1])
    print(f"\n  r(OA, MMD²) = {corr:.4f}")
    print(f"\nTempo totale: {time.time() - t0:.1f}s")
    print("\n" + "="*55)
    print("Analisi completata.")
    print("="*55)