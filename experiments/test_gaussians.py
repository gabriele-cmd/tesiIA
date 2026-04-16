"""
ESECUZIONE:
    python experiments/test_gaussians.py per test di DEFAULT
    python experiments/test_gaussians.py --n 2000 --d 128 --n_perm 1000 --trials 100 per test VARIABILE (valori di esempio)
    USARE python3 SU MACCHINA REMOTA

Test di robustezza del codice MMD svolto su dataset sintetici derivati da miscele gaussiane

Svolge tre tipologie di esperimento:
    1. Separazione crescente - crescita MMD
    2. Overlap variabile - decrescita MMD
    3. Potenza test - frequenza di rifiuto corretto di H0

Per la miscela gaussiana si sfrutta la foruma della distribuzione:
        P = 0.5 · N(μ₁, σ²I) + 0.5 · N(μ₂, σ²I)
"""

import time #utile per avere una misura del tempo di esecuzione necessario
import argparse #utile per ricevere il dataset dinamicamente in input
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import Tensor
from device import get_device, move_to_device
from mmd import compute_mmd2, bootstrap_test

#Generatore di miscele gaussiane
def make_gaussian_mixture(
        n: int, #numero di campioni (es. brani simulati)
        d: int, #dimensioni delle feature (es. embeddings audio)
        means: list[Tensor] | None = None, #lista di Tensor [d] (centri dei componenti) se None le componenti sono equidistanti a distanza 1.0
        std: float = 1.0, #deviazione standard dei componenti
        n_components: int = 2, #numero di componenti della miscela
        seed: int | None = None, #utile per riproducibilità
) -> Tensor:

    if seed is not None:
        torch.manual_seed(seed)

    if means is None:
        #Componenti posti equidistanti lungo il primo asse
        means = [torch.zeros(d) for _ in range(n_components)]
        for k in range(n_components):
            means[k][0] = k * 1.0 #distanza 1.0 tra i centri successivi

    n_comp = len(means)
    samples = []

    #stabilisce numero di campioni per componente
    counts = [n // n_comp] * n_comp
    counts[-1] += n - sum(counts)

    for k, (mu, count) in enumerate(zip(means, counts)):
        #formula per la mistura: N(mu, std²·I): campiona dalla gaussiana standard e scala
        component_samples = mu + std * torch.randn(count, d)
        samples.append(component_samples)

    return torch.cat(samples, dim=0)

#Esperimento 1: Separazione crescente
#L'MMD cresce con la distanza tra le distribuzioni riuscendo a rilevare la diversità tra di esse?
def experiment_1_separation(device: torch.device, n: int = 150, d: int = 32, n_perm: int = 500):
    """
    Setup:
        P = N(0, I) - distribuzione REALE
        Q = N(Δ·e₁, I) - distribuzione GENERATA, media spostata di delta
        DELTA varia tra 0 e 3

    RISULTATO ATTESO:
        Δ=0: MMD² ≈ 0  (distribuzioni identiche)
        Δ cresce: MMD² cresce monotonamente
        p-value scende sotto 0.05 a partire da un certo Δ (soglia di rilevamento)
    """
    print("\n" + "="*55)
    print("ESPERIMENTO 1 — Separazione crescente")
    print(f"  n={n} campioni, d={d} dimensioni, n_perm=500")
    print("=" * 55)
    print(f"  {'Δ':>6}  {'MMD²':>10}  {'σ':>8}  {'p-value':>10}  {'H₀ rifiutata':>14}")
    print("  " + "-" * 52)

    deltas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    results = []

    for delta in deltas:
        #P: gaussiana centrata in 0
        mu_p = torch.zeros(d)
        #Q: gaussiana spostata di delta lungo l'asse
        mu_q = torch.zeros(d)
        mu_q[0] = delta

        X = make_gaussian_mixture(n, d, means=[mu_p], seed=0)
        Y = make_gaussian_mixture(n, d, means=[mu_q], seed=1)
        X, Y = move_to_device(X, Y, device=device)

        result = bootstrap_test(X, Y, n_perm=500, device=device, seed=42)
        results.append(result)

        reject_str = "SI ✓" if result.reject_h0 else "no"
        pval_str = f"{result.p_value:.4f}" if result.p_value >= 0.001 else "< 0.001"
        print(f"  {delta:>6.2f}  {result.mmd2:>10.6f}  "
              f"{result.sigma:>8.3f}  {pval_str:>10}  {reject_str:>14}")

    #Verifica monotonia: MMD deve crescere con Delta
    mmd2_vals = [r.mmd2 for r in results]
    is_monotone = all(mmd2_vals[i] <= mmd2_vals[i+1] + 0.001
                      for i in range(len(mmd2_vals) - 1))

    print(f"\n  Monotonia MMD²: {'✓ confermata' if is_monotone else '✗ violata'}")
    print(f"  Δ minimo rilevato (p<0.05): ", end="")
    detected = [d for d, r in zip(deltas, results) if r.reject_h0]
    print(f"Δ={min(detected):.2f}" if detected else "nessuno rilevato")

    return results

#Esperimento 2: Miscele gaussiane con overlap variabile
#L'MMD funziona anche con distribuzioni multimodali e riesce a catturare differenze con overlap parziale?
def experiment_2_overlap(device: torch.device, n: int = 150, d: int = 32, n_perm: int = 500):
    """
    Setup:
        P = 0.5·N(-1, I) + 0.5·N(+1, I)    — miscela fissa
        Q = 0.5·N(-1+s, I) + 0.5·N(+1+s, I) — miscela spostata di s

        (shift)
        s=0.0 → P=Q, overlap totale    → MMD² ≈ 0
        s=1.0 → overlap parziale       → MMD² moderato
        s=3.0 → quasi nessun overlap   → MMD² grande
    """
    print("\n" + "=" * 55)
    print("ESPERIMENTO 2 — Miscele gaussiane con overlap variabile")
    print(f"  n={n} campioni, d={d} dimensioni, n_perm=500")
    print("=" * 55)
    print(f"  {'shift':>8}  {'overlap':>10}  {'MMD²':>10}  "
          f"{'p-value':>10}  {'H₀ rifiutata':>14}")
    print("  " + "-" * 58)

    #Shift da 0 (distr. identiche) a 4 (completamente separate)
    shifts = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0]
    results = []

    for shift in shifts:
        #P: miscela centrata in +-1
        mu_p1 = torch.zeros(d); mu_p1[0] = -1.0
        mu_p2 = torch.zeros(d); mu_p2[0] = +1.0

        #Q: stessa miscela spostata di shift
        mu_q1 = torch.zeros(d); mu_q1[0] = -1.0 + shift
        mu_q2 = torch.zeros(d); mu_q2[0] = +1.0 + shift

        X = make_gaussian_mixture(n, d, means=[mu_p1, mu_p2], std=0.8, seed=0)
        Y = make_gaussian_mixture(n, d, means=[mu_q1, mu_q2], std=0.8, seed=1)
        X, Y = move_to_device(X, Y, device=device)

        result = bootstrap_test(X, Y, n_perm=500, device=device, seed=42)
        results.append(result)

        #Stima approssimativa dell'overlap (formula per 1D)
        #overlap decresce con shift, 1.0 quando shift=0
        import math
        overlap_approx = max(0.0, 1.0 - shift / 4.0)

        reject_str = "SI ✓" if result.reject_h0 else "no"
        pval_str   = f"{result.p_value:.4f}" if result.p_value >= 0.001 else "< 0.001"
        print(f"  {shift:>8.2f}  {overlap_approx:>10.2f}  {result.mmd2:>10.6f}  "
              f"{pval_str:>10}  {reject_str:>14}")

    print(f"\n  Shift=0 (P=Q): MMD²={results[0].mmd2:.6f}  "
          f"(atteso ≈ 0)")
    print(f"  Shift=4 (sep): MMD²={results[-1].mmd2:.6f}  "
          f"(atteso >> 0)")
    return results

#Esperimento 3: potenza del test (probabilità di rifiutare H0 quando questa è falsa)
#con quanta affidabilità il test rileva differenze reali?
#determina P(rifiutare H0 | H0 falsa)
def experiment_3_power(device: torch.device, n: int = 150, d: int = 32, n_perm: int = 300, n_trials: int = 50):
    """
    Idealmente P = 1.0 ma realmente dipende da:
        - n (numero campioni proporzionale alla potenza)
        - d (numero dimensioni inversamente proporzionale alla potenza)
        - entità della differenza (shift proporzionale alla potenza)
        - n_perm (numero permutazioni proporzionale all'accuratezza della stima di p_value)

    Per ogni shift si ripete il test per n_trials=50 volte con seed diversi
    Allora Potenza ≈ rejections (numero rifiuti) / n_trials (numero tentativi)

    Risultato atteso:
        shift=0.0 → potenza ≈ 0.05  (solo falsi positivi al livello α)
        shift=1.0 → potenza intermedia
        shift=3.0 → potenza ≈ 1.0   (rileva sempre)
    """
    print("\n" + "=" * 55)
    print("ESPERIMENTO 3 — Potenza del test")
    print(f"  n={n} campioni, d={d} dim, n_trials={n_trials}, n_perm={n_perm}")
    print("=" * 55)
    print(f"  {'shift':>8}  {'potenza':>10}  {'rifiuti/50':>12}  "
          f"{'MMD² medio':>12}")
    print("  " + "-" * 48)

    shifts = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    powers = []

    for shift in shifts:
        rejections = 0
        mmd2_vals = []

        for trial in range(n_trials): #Seed diverso per ogni tentativo (trials) così i dataset sono indipendenti tra loro
            mu_p = torch.zeros(d)
            mu_q = torch.zeros(d); mu_q[0] = shift

            X = make_gaussian_mixture(n, d, means=[mu_p], seed=trial + 200)
            Y = make_gaussian_mixture(n, d, means=[mu_q], seed=trial + 300)
            X, Y = move_to_device(X, Y, device=device)

            result = bootstrap_test(X, Y, n_perm=n_perm, device=device, seed=trial)

            if result.reject_h0:
                rejections += 1
            mmd2_vals.append(result.mmd2)

        power = rejections / n_trials #calcolo potenza
        mmd2_mean = sum(mmd2_vals) / len(mmd2_vals)
        powers.append(power)

        bar = "█" * int(power * 20)
        print(f"  {shift:>8.2f}  {power:>10.2f}  {rejections:>5}/{n_trials:<6}  "
              f"{mmd2_mean:>12.6f}  {bar}")

    #Verifica che la potenza cresca con lo shift
    is_increasing = all(powers[i] <= powers[i+1] + 0.1
                        for i in range(len(powers) - 1))
    print(f"\n  Potenza crescente con shift: "
          f"{'✓ confermata' if is_increasing else '✗ violata'}")
    print(f"  Potenza a shift=0 (tasso falsi positivi): {powers[0]:.2f}  "
          f"(atteso ≈ {0.05})")
    print(f"  Potenza a shift={shifts[-1]}: {powers[-1]:.2f}  "
          f"(atteso ≈ 1.0)")

    return powers

#Esecuzione esperimenti
if __name__ == "__main__":
    #Valori dataset fisso (DEFAULT)
    args_n = 150
    args_d = 32
    args_n_perm = 300
    args_trials = 50

    #COMMENTARE QUESTA SEZIONE PER TOGLIERE LA POSSIBILITA DI INPUT
    parser = argparse.ArgumentParser(description="Test MMD su miscele gaussiane")
    parser.add_argument("--n",       type=int, default=150,
                        help="numero di campioni per gruppo (default: 150)")
    parser.add_argument("--d",       type=int, default=32,
                        help="dimensioni delle feature (default: 32)")
    parser.add_argument("--n_perm",  type=int, default=300,
                        help="permutazioni bootstrap (default: 300)")
    parser.add_argument("--trials",  type=int, default=50,
                        help="trial per esperimento 3 (default: 50)")
    args = parser.parse_args()
    #COMMENTARE QUESTA SEZIONE PER TOGLIERE LA POSSIBILITA DI INPUT

    print("="*57)
    print("Test di robustezza MMD su miscele gaussiane")
    print(f"  n={args.n}, d={args.d}, n_perm={args.n_perm}, trials={args.trials}")
    print("="*57)

    device = get_device()

    import time

    t0 = time.time()
    experiment_1_separation(device, n=args.n, d=args.d, n_perm=args.n_perm)
    print(f"  Tempo: {time.time()-t0:.1f}s")

    t0 = time.time()
    experiment_2_overlap(device, n=args.n, d=args.d, n_perm=args.n_perm)
    print(f"  Tempo: {time.time()-t0:.1f}s")

    t0 = time.time()
    experiment_3_power(device, n=args.n, d=args.d,
                       n_perm=args.n_perm, n_trials=args.trials)
    print(f"  Tempo: {time.time()-t0:.1f}s")

    print("\n" + "="*57)
    print("Tutti gli esperimenti completati.")
    print("="*57)
