"""
ESECUZIONE:
    python experiments/mmd_overlap_analysis.py
    Eseguire sulla macchina remota per dataset grandi:
    python3 experiments/mmd_overlap_analysis.py --n_samples 500 --n_points 200


ANALISI EMPIRICA DELLA RELAZIONE TRA MMD E OVERLAP AREA
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') #backend non interattivo, salva su file
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from device import get_device, move_to_device
from mmd import compute_mmd2

#Funzioni di densità
def gaussian_pdf(x: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    """
    Per calcolare l'overlap area bisogna valutare le densità p(x) e q(x)
    in ogni punto campionato. Si implementa poi la densità di una miscela gaussiana multivariata con covarianza diagonale (isotropica)
    Si usa: p(x) = (2π σ²)^(-d/2) exp( -‖x - μ‖² / 2σ² )

    """
    d = x.shape[1] #dimensione di X [n, d], punti dove valutare la densità
    diff = x - mean #mean centro della gaussiana, dimensione [d]
    sq_dist = (diff ** 2).sum(axis=1)
    norm = (2 * np.pi * std**2) ** (d / 2) #std deviazione standard (uguale per ogni dimensione)
    return np.exp(-sq_dist / (2 * std**2)) / norm #[n] calcola e ritorna il valore della densità in ogni punto

def gmm_pdf(
        x: np.ndarray, #[n, d] punti dove valutare la densità
        means: list[np.ndarray], #lista di [d] centri delle componenti
        stds: list[float], #lista di float, std di ogni compinent
        weights: list[float] | None = None, #pesi delle componenti (default: uniformi)
) -> np.ndarray:
    """
    Valuta la densità di una miscela gaussiana in x secondo la formula:
        p(x) = Σₖ wₖ · N(x | μₖ, σₖ²I)
    """
    n_comp = len(means)
    if weights is None:
        weights = [1.0 / n_comp] * n_comp

    density = np.zeros(x.shape[0])
    for w, mu, s in zip(weights, means, stds):
        density += w * gaussian_pdf(x, mu, s)
    return density #calcola e ritorna la densità

"""
L'overlap area tra P e Q è:
    OA(P, Q) = ∫ min( p(x), q(x) ) dx
Non ha una formula chiusa per miscele gaussiane in alta dimensione.

Si può stimare con il metodo Monte Carlo:
    OA(P, Q) = E_{x ~ M} [ 2 · min(p(x), q(x)) / (p(x) + q(x)) ]
    
con M = (P + Q) / 2 MISCELA delle due distribuzioni
"""
#Overlap Area con campionamento Monte Carlo tra due miscele gaussiane
def overlap_area_mc(
        #parametri della distribuzione P
        means_p: list[np.ndarray],
        stds_p: list[float],
        weights_p: list[float],
        #parametri della distribuzione Q
        means_q: list[np.ndarray],
        stds_q: list[float],
        weights_q: list[float],
        n_samples: int = 10000, #campioni Monte Carlo, più è alto più è preciso
        seed: int = 0, #riproducibilità
) -> float:
    rng = np.random.default_rng(seed)
    d = len(means_p[0])

    #Campiona n_samples / 2 punti da P e n_samples / 2 da Q per formare campioni dalla miscela M = (P+Q)/2
    n_half = n_samples // 2
    samples = []

    #Campiona da P
    counts_p = rng.multinomial(n_half, weights_p)
    for mu, s, count in zip(means_p, stds_p, counts_p):
        if count > 0:
            samples.append(mu + s * rng.standard_normal((count, d)))

    x = np.vstack(samples) #[n_samples, d]

    #Valuta le densità in ogni punto campionato
    p_x = gmm_pdf(x, means_p, stds_p, weights_p)
    q_x = gmm_pdf(x, means_q, stds_q, weights_q)

    #Stimatore importance sampling
    #Si evitano divisioni per zero con un epsilon piccolo
    denom = p_x + q_x
    mask = denom > 1e-300
    ratio = np.zeros_like(denom)
    ratio[mask] = 2 * np.minimum(p_x[mask], q_x[mask]) / denom[mask]

    return float(ratio.mean()) #ritorna un float in [0,1] che indica overlap (0 = distr. compl. separate, 1 = distr. identiche)

#Stima parametri con Expectation-Maximization da dati reali
def fit_gmm_em(
        data: np.ndarray, #[n, d] dati di addestramento
        n_components: int = 2, #numero di componenti della miscela
        covariance_type: str = 'diag', #tipo di covarianza. 'diag' è covarianza diagonale (isotropica)
        seed: int = 0, #per riproducibilità
) -> GaussianMixture:
    """
    L'algoritmo EM stima i parametri di una miscela gaussiana da dati.
    Alterna due passi fino alla convergenza:

        E-step (Expectation):
        Per ogni campione xᵢ calcola la probabilità che appartenga
        a ogni componente k:
            r_ik = wₖ · N(xᵢ|μₖ,Σₖ) / Σⱼ wⱼ · N(xᵢ|μⱼ,Σⱼ)
        dette "responsabilità".

        M-step (Maximization):
        Aggiorna i parametri usando le responsabilità come pesi:
            wₖ  = (1/n) Σᵢ r_ik
            μₖ  = Σᵢ r_ik · xᵢ / Σᵢ r_ik
            Σₖ  = Σᵢ r_ik · (xᵢ-μₖ)(xᵢ-μₖ)ᵀ / Σᵢ r_ik
    """
    gmm = GaussianMixture(
        n_components = n_components,
        covariance_type = covariance_type,
        random_state = seed,
        n_init = 5, #5 inizializzazioni diverse -> prende la migliore
        max_iter = 200, #200 iterazioni massime EM
    )
    gmm.fit(data)
    return gmm #ritorna un oggetto GaussianMixture fittato con attributi means, covariances e weights

def sample_from_gmm(
        gmm: GaussianMixture, #oggetto già fittato
        n: int, #numero di campioni
        seed: int = 0,
) -> np.ndarray:
    """
    Campiona n punti da una miscela gaussiana già fittata
    """
    samples, _ = gmm.sample(n)
    return samples.astype(np.float32) #[n, d]

def gmm_pdf_sklearn(
        x: np.ndarray,
        gmm: GaussianMixture,
) -> np.ndarray:
    """
    Valuta la densità di una GMM sklearn in x.
    Usa il metodo built-in score_samples che è numericamente stabile.
    """
    log_density = gmm.score_samples(x)
    return np.exp(log_density) #ritorna densità [n], valori di densità

#Esperimento sintetico: MMD vs OA su miscele controllate
def experiment_synthetic(
        device: torch.device,
        d: int = 8, #dimensioni delle feature
        n_points: int = 200, #campioni per gruppo per calcolo MMD
        n_samples_oa: int = 30, #campioni Monte Carlo per overlap
        n_shifts: int = 30, #numero di configurazioni diverse da testare
) -> tuple[list[float], list[float]]:
    """
    Calcola MMD e OA al variare della separazione tra distribuzioni
    """
    print("\n" + "=" * 57)
    print("ESPERIMENTO SINTETICO — MMD vs Overlap Area")
    print(f"  d={d}, n_points={n_points}, n_shifts={n_shifts}")
    print("=" * 57)

    #Varia lo shift da 0 (identiche) a 4 (separate)
    shifts = np.linspace(0.0, 4.0, n_shifts)
    mmd2_values = []
    oa_values = []

    for i, shift in enumerate(shifts):
        #P: miscela con due componenti in +-1
        means_p = [np.zeros(d), np.zeros(d)]
        means_p[0][0] = -1.0
        means_p[1][0] = +1.0
        stds_p = [0.8, 0.8]
        weights_p = [0.5, 0.5]

        #Q: stessa struttura spostata di shift
        means_q = [np.zeros(d), np.zeros(d)]
        means_q[0][0] = -1.0 + shift
        means_q[1][0] = +1.0 + shift
        stds_q = [0.8, 0.8]
        weights_q = [0.5, 0.5]

        #Calcolo Overlap Area
        oa = overlap_area_mc(
            means_p, stds_p, weights_p,
            means_q, stds_q, weights_q,
            n_samples = n_samples_oa,
            seed = i,
        )

        #Genera campioni e calcola MMD
        rng = np.random.default_rng(i)
        X_np = np.vstack([
            means_p[k] + stds_p[k] * rng.standard_normal((n_points//2, d))
            for k in range(2)
        ]).astype(np.float32)
        Y_np = np.vstack([
            means_q[k] + stds_q[k] * rng.standard_normal((n_points//2, d))
            for k in range(2)
        ]).astype(np.float32)

        X = torch.from_numpy(X_np)
        Y = torch.from_numpy(Y_np)
        X, Y = move_to_device(X, Y, device = device)

        mmd2, _ = compute_mmd2(X, Y, device=device)

        mmd2_values.append(float(mmd2))
        oa_values.append(float(oa))

        if (i+1) % 5 == 0 or i == 0:
            print(f" shift={shift:.2f} OA={oa:.4f} MMD²={mmd2:.6f}")

    return mmd2_values, oa_values