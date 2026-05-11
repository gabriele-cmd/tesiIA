"""
Funzioni di utilità condivise tra tutti gli esperimenti.
"""
import numpy as np
from scipy.stats import gaussian_kde

#Calcola l'Overlap Area tramite KDE delle distribuzioni delle distanze.
def overlap_area_kde(
    p: np.ndarray, #[n, d] — campioni dalla distribuzione reale P
    q: np.ndarray, #[m, d] — campioni dalla distribuzione generata Q
    n_grid: int = 500, #punti della griglia per l'integrazione numerica
    subsample: int = 500, #massimo campioni da usare per le distanze inter P-Q.
    seed: int = 0,
) -> float:

    #Distanze intra-P (triangolo superiore)
    n = len(p)
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_pp = np.linalg.norm(p[i_idx] - p[j_idx], axis=1)

    #Distanze inter P-Q (con subsample se necessario)
    rng = np.random.default_rng(seed)
    p_sub = p
    q_sub = q
    if len(p) > subsample:
        p_sub = p[rng.choice(len(p), subsample, replace=False)]
    if len(q) > subsample:
        q_sub = q[rng.choice(len(q), subsample, replace=False)]

    d_pq = np.linalg.norm(
        p_sub[:, None, :] - q_sub[None, :, :],
        axis=2
    ).ravel()

    #KDE delle due distribuzioni di distanze
    kde_pp = gaussian_kde(d_pp)
    kde_pq = gaussian_kde(d_pq)

    #Griglia comune su cui valutare le KDE
    grid = np.linspace(
        0,
        max(d_pp.max(), d_pq.max()),
        n_grid
    )

    pp = kde_pp(grid)
    pq = kde_pq(grid)

    #OA = area sotto min(KDE_PP, KDE_PQ)
    oa = float(np.trapz(np.minimum(pp, pq), grid))

    return oa