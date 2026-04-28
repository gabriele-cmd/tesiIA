"""
Studio empirico della sensibilità di MMD e Overlap Area
al fenomeno di mode collapse su una GMM a 3 componenti.

-P: distribuzione reale, GMM a 3 componenti fittata su Breast Cancer
-Q: stessa GMM ma con pesi modificati tramite distribuzione di Dirichlet
-Al diminuire del parametro K i pesi di Q si concentrano su una componente (vertice) sola
-Si ha mode collapse
-Osserviamo come MMD e OA reagiscono al fenomeno

Riferimento:
    Distribuzione di Dirichlet: Dir(α) con α = K · c
    dove c = pesi base della GMM e K = parametro di concentrazione
"""

import sys, os

from networkx.algorithms.bipartite.basic import density

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

from device import get_device, move_to_device
from mmd import compute_mmd2
from bandwidth import median_bandwidth

#1. Fit della GMM base
def fit_base_gmm(n_components: int = 3, seed: int = 0):
    """
    Carica Breast Cancer, lo normalizza e fitta una GMM a n_components.

    Ritorna:
        gmm: oggetto fittato
        scaler: StandardScaler fittato
        data: array normalizzato [569, 30]
    """

    raw = load_breast_cancer()
    scaler = StandardScaler()
    data = scaler.fit_transform(raw.data).astype(np.float64)

    gmm = GaussianMixture(
        n_components = n_components,
        covariance_type = 'diag',
        random_state = seed,
        n_init = 10,
        max_iter = 300,
        reg_covar = 1e-4,
    )

    gmm.fit(data)

    print(f"GMM fittata su Breast Cancer ({data.shape[0]} campioni, "
          f"{data.shape[1]} feature)")
    print(f"  Componenti:    {n_components}")
    print(f"  Pesi base:     {gmm.weights_.round(3)}")
    print(f"  BIC:           {gmm.bic(data):.1f}")
    return gmm, scaler, data

#2. Campionamento pesi con distribuzione di Dirichlet
#La distribuzione di Dirichlet Dir(α) genera vettori di probabilità con α = K · c
#Il "grado di collapse" si misura con l'entropia dei pesi:
#entropia alta → pesi bilanciati → nessun collapse
#entropia bassa → un peso domina → collapse
def sample_collapsed_weights(
        base_weights: np.ndarray, #pesi base della GMM
        K: float, #parametro di concentrazione. K >> 1 nessun collapse, K << 1 collapse forte
        seed: int = 0, #per riproducibilità
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha = K * base_weights
    #Dirichlet campiona dal simplesso
    weights = rng.dirichlet(alpha)
    return weights

def collapse_degree(weights: np.ndarray) -> float:
    """
    Misura il grado di collapse come 1 - entropia normalizzata.

    Entropia normalizzata = H(w) / log(K)
    dove K è il numero di componenti.
        collapse = 0 -> pesi perfettamente bilanciati
        collapse = 1 -> tutto il peso su una componente
    """
    n = len(weights)
    #Evita log(0)
    w = np.clip(weights, 1e-10, 1.0)
    H = -np.sum(w * np.log(w)) #Entropia di Shannon
    H_max = np.log(n) #Entropia Massima (uniforme)
    return float(1.0 - H / H_max)

#3. Overlap Area tra GMM con pesi diversi
#P e Q hanno stesse medie e covarianze - solo pesi diversi, questo è mode collapse
#Valuta la densità di una GMM in x dati parametri espliciti
def gmm_pdf_from_params(
        x: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        weights: np.ndarray,
) -> np.ndarray:
    density = np.zeros(x.shape[0])
    for k in range(len(weights)):
        diff = x - means[k]
        sq_dist = (diff ** 2).sum(axis=1)
        norm = (2 * np.pi * stds[k]**2) ** (x.shape[1] / 2)
        density += weights[k] * np.exp(-sq_dist / (2 * stds[k]**2)) / norm
    return density

#Stima OA tra P (pesi base GMM) e Q (pesi collassati)
def overlap_area_collpase(
        gmm: GaussianMixture,
        weights_q: np.ndarray, #pesi di Q dopo il collapse
        n_samples: int = 10000, #campioni Monte Carlo
        seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    means = gmm.means_
    stds = np.array([np.sqrt(gmm.covariances_[k]).mean()
                     for k in range(gmm.n_components)])
    weights_p = gmm.weights_
    d = means.shape[1]

    #Campiona dalla miscela M = (P + Q) / 2
    n_half = n_samples // 2
    samples = []

    counts_p = rng.multinomial(n_half, weights_p)
    for k, count in enumerate(counts_p):
        if count > 0:
            samples.append(means[k] + stds[k] * rng.standard_normal((count, d)))

    counts_q = rng.multinomial(n_half, weights_q)
    for k, count in enumerate(counts_q):
        if count > 0:
            samples.append(means[k] + stds[k] * rng.standard_normal((count, d)))

    x = np.vstack(samples)

    p_x = gmm_pdf_from_params(x, means, stds, weights_p)
    q_x = gmm_pdf_from_params(x, means, stds, weights_q)

    denom = p_x + q_x
    mask = denom > 1e-300
    ratio = np.zeros_like(denom)
    ratio[mask] = 2 * np.minimum(p_x[mask], q_x[mask]) / denom[mask]

    return float(ratio.mean())

#4. Esperimento collapse
#Per ogni valore di K: 1. Campiona pesi collassati da Dir(K*c)
#                      2. Costruisce Q con quei pesi
#                      3. Calcola MMD(P, Q) e OA(P, Q)
#                      4. Calcola il grado di collapse
#ripete n_trials volte per ogni K (pesi diversi dalla stessa Dir) e prende la media (Dir Stocastica)
def experiment_collapse(
        device: torch.device,
        gmm: GaussianMixture,
        n_points: int = 300, #campioni per gruppo per MMD
        n_samples_oa: int = 10000, #campioni Monte Carlo per OA
        n_trials: int = 10, #ripetizioni per ogni K (media su realizzazioni Dirichlet)
        seed: int = 0,
) -> dict:
    base_weights = gmm.weights_
    means = gmm.means_
    stds = np.array([np.sqrt(gmm.covariances_[k]).mean()
                     for k in range(gmm.n_components)])

    #Valori di K da testare:
    #da K grande (nessun collapse) a K piccolo (collapse forte)
    #scala logaritmica perché l'effetto è moltiplicativo
    K_values = np.logspace(2, -2, 30) #da 100 a 0.01

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
            #Campiona pesi collassati da Dirichlet
            w_q = sample_collapsed_weights(
                base_weights, K, seed = seed + i * 100 + trial
            )
            w_trials.append(w_q)

            #Grado di collapse
            cd = collapse_degree(w_q)
            cd_trials.append(cd)

            #OA tra P (pesi base) e Q (pesi collassati)
            oa = overlap_area_collpase(
                gmm,w_q,
                n_samples = n_samples_oa,
                seed = seed + i * 100 + trial,
            )
            oa_trials.append(oa)

            #Genera campioni da P e Q per MMD
            rng = np.random.default_rng(seed + i * 100 + trial)

            #P: campiona con pesi base
            counts_p = rng.multinomial(n_points, base_weights)
            X_list = [means[k] + stds[k] * rng.standard_normal((c, means.shape[1]))
                      for k, c in enumerate(counts_p) if c > 0]
            X_np = np.vstack(X_list).astype(np.float32)

            #Q: stesse medie/covarianze, pesi collassati
            counts_q = rng.multinomial(n_points, w_q)
            Y_list = [means[k] + stds[k] * rng.standard_normal((c, means.shape[1]))
                      for k, c in enumerate(counts_q) if c > 0]
            Y_np = np.vstack(Y_list).astype(np.float32)

            X = torch.from_numpy(X_np)
            Y = torch.from_numpy(Y_np)
            X, Y = move_to_device(X, Y, device=device)

            mmd2_val, _ = compute_mmd2(X, Y, device=device)
            mmd2_trials.append(mmd2_val)

        #Media sui trial
        mmd2_mean = float(np.mean(mmd2_trials))
        oa_mean = float(np.mean(oa_trials))
        cd_mean = float(np.mean(cd_trials))
        w_mean = np.mean(w_trials, axis=0)

        results['K_values'].append(float(K))
        results['mmd2'].append(mmd2_mean)
        results['oa'].append(oa_mean)
        results['collapse_degree'].append(cd_mean)

        if i % 5 == 0 or i == len(K_values) - 1:
            print(f"  {K:>8.3f}  {cd_mean:>10.3f}  {oa_mean:>8.4f}  "
                  f"{mmd2_mean:>10.6f}  {w_mean.round(3)}")

    return results

#5. Grafici e Main
def plot_collapse(results: dict, output_dir: str = "results") -> None:
    os.makedirs(output_dir, exist_ok=True)

    cd = results["collapse_degree"]
    oa = results["oa"]
    mmd = results["mmd2"]
    K = results["K_values"]

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle("Sensibilità di MMD² e OA al Mode Collapse\n"
                 "(Breast Cancer, GMM 3 componenti, Dirichlet)", fontsize=12)

    #Grafico 1: MMD e OA vs grado di collapse
    ax = axes[0]
    ax2 = ax.twinx() #secondo asse Y per OA
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

    #Grafico 2: MMD e OA vs K (scala logaritmica)
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
    ax.invert_xaxis()  #K grande a sinistra = nessun collapse
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    path = os.path.join(output_dir, "mode_collapse_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Grafico salvato in: {path}")
    plt.close()

#MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_points",    type=int, default=300)
    parser.add_argument("--n_samples",   type=int, default=10000)
    parser.add_argument("--n_trials",    type=int, default=10)
    parser.add_argument("--n_components",type=int, default=3)
    parser.add_argument("--output_dir",  type=str, default="results/mode_collapse")
    args = parser.parse_args()

    print("="*55)
    print("Analisi sensibilità MMD e OA al Mode Collapse")
    print("="*55)

    device = get_device()

    # Fit GMM una volta sola
    gmm, scaler, data = fit_base_gmm(
        n_components = args.n_components,
        seed         = 0,
    )

    # Esperimento
    results = experiment_collapse(
        device       = device,
        gmm          = gmm,
        n_points     = args.n_points,
        n_samples_oa = args.n_samples,
        n_trials     = args.n_trials,
    )

    # Grafici
    plot_collapse(results, output_dir=args.output_dir)

    # Correlazione finale
    corr = float(np.corrcoef(results['oa'], results['mmd2'])[0, 1])
    print(f"\n  r(OA, MMD²) = {corr:.4f}")
    print("\n" + "="*55)
    print("Analisi completata.")
    print("="*55)