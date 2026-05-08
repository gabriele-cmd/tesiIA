import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

np.random.seed(42)

N = 400
SIGMA = 0.5
KERNEL_SIGMA = 0.5

REAL_MODES = [
    np.array([-4, -4]),
    np.array([ 4, -4]),
    np.array([-4,  4]),
    np.array([ 4,  4]),
]

MODELS = [
    {
        "name": "Good model",
        "modes": REAL_MODES,
        "color": "#34d399",
    },
    {
        "name": "Partial collapse",
        "modes": [
            np.array([-4, -4]),
            np.array([ 4, -4]),
        ],
        "color": "#f472b6",
    },
    {
        "name": "Total collapse",
        "modes": [
            np.array([-4, -4]),
        ],
        "color": "#f87171",
    },
]


def sample_mixture(modes, sigma, n):
    k = len(modes)
    counts = np.random.multinomial(n, [1 / k] * k)
    samples = []

    for mu, c in zip(modes, counts):
        pts = np.random.multivariate_normal(
            mu,
            sigma * np.eye(2),
            c
        )
        samples.append(pts)
    return np.vstack(samples)


def pairwise_distances(x):
    i, j = np.triu_indices(len(x), k=1)
    return np.linalg.norm(x[i] - x[j], axis=1)

def compute_metrics(p, q):
    d_pp = pairwise_distances(p)
    d_qq = pairwise_distances(q)

    d_pq = np.linalg.norm(
        p[:, None, :] - q[None, :, :],
        axis=2
    ).ravel()

    kde_pp = gaussian_kde(d_pp)
    kde_pq = gaussian_kde(d_pq)
    grid = np.linspace(
        0,
        max(d_pp.max(), d_qq.max(), d_pq.max()),
        500
    )

    pp = kde_pp(grid)
    pq = kde_pq(grid)
    oa = np.trapezoid(
        np.minimum(pp, pq),
        grid
    )

    def rbf(a, b):
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(
            -np.sum(diff**2, axis=2)
            / (2 * KERNEL_SIGMA**2)
        )
    n = len(p)
    m = len(q)
    Kxx = rbf(p, p)
    Kyy = rbf(q, q)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2 * rbf(p, q).mean()
    )
    mmd = np.sqrt(max(mmd2, 0))
    return {
        "oa": oa,
        "mmd": mmd,
        "d_pp": d_pp,
        "d_qq": d_qq,
        "d_pq": d_pq,
        "grid": grid,
    }

P = sample_mixture(REAL_MODES, SIGMA, N)
results = []
for model in MODELS:
    Q = sample_mixture(
        model["modes"],
        SIGMA,
        N
    )
    metrics = compute_metrics(P, Q)
    results.append({
        "model": model,
        "Q": Q,
        "metrics": metrics,
    })

fig, axes = plt.subplots(
    len(MODELS),
    2,
    figsize=(12, 12)
)

for row, result in enumerate(results):
    model = result["model"]
    Q = result["Q"]
    M = result["metrics"]
    color = model["color"]

    ax = axes[row, 0]
    ax.scatter(
        P[:, 0],
        P[:, 1],
        s=10,
        alpha=0.3,
        label="Training",
    )
    ax.scatter(
        Q[:, 0],
        Q[:, 1],
        s=10,
        alpha=0.5,
        label="Generated",
        color=color,
    )
    ax.set_title(model["name"])
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.legend()
    ax = axes[row, 1]
    grid = M["grid"]

    kde_pp = gaussian_kde(M["d_pp"])(grid)
    kde_qq = gaussian_kde(M["d_qq"])(grid)
    kde_pq = gaussian_kde(M["d_pq"])(grid)

    ax.plot(grid, kde_pp, label="D_PP")
    ax.plot(grid, kde_qq, label="D_QQ")
    ax.plot(grid, kde_pq, label="D_PQ")

    ax.fill_between(
        grid,
        np.minimum(kde_pp, kde_pq),
        alpha=0.2
    )

    ax.set_title(
        f"OA = {M['oa']:.3f}   |   MMD = {M['mmd']:.3f}"
    )
    ax.legend()

plt.tight_layout()
plt.show()