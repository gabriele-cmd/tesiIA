"""
    Esecuzione: python tests/test_mmd.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from device import get_device, move_to_device
from kernels import GaussianKernel
from mmd import compute_mmd2, bootstrap_test, MMDResult

def run():
    device = get_device()
    torch.manual_seed(42)
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✓  {name}")
            passed += 1
        else:
            print(f"  ✗  {name}" + (f"\n     → {detail}" if detail else ""))
            failed += 1
    d = 32

    #verifica proprietà fondamentali di compute_mmd2
    print("\n── compute_mmd2: Proprietà fondamentali ──")

    x_same = torch.randn(80, d)
    y_same = torch.randn(80, d) #stessa distribuzione N(0,1)
    x_diff = torch.randn(80, d)
    y_diff = torch.randn(80, d) + 2.0 #distribuzione spostata di 2

    m_same, s = compute_mmd2(x_same, y_same, device=device)
    m_diff, _ = compute_mmd2(x_diff, y_diff, device=device)
    m_self, _ = compute_mmd2(x_same, x_same, device=device)

    ok("Restituisce una tupla (float, float)",
       isinstance(m_same, float) and isinstance(s, float))

    ok("MMD²(X, X) ≈ 0  (stessa distribuzione esatta, O(1/n))",
       abs(m_self) < 0.05,
       f"valore = {m_self:.5f}")

    ok("MMD²(distrib. diverse) >> MMD²(distrib. simili)",
       m_diff > m_same * 3,
       f"diff={m_diff:.4f}, same={m_same:.4f}")

    ok("sigma restituito è > 0",
       s > 0, f"sigma = {s:.4f}")

    ok("sigma esplicito viene usato",
       compute_mmd2(x_same, y_same, sigma=5.0, device=device)[1] == 5.0)

    ok("kernel esplicito viene usato",
       compute_mmd2(x_same, y_same,
                    kernel=GaussianKernel(sigma=3.0),
                    device=device)[1] == 3.0)

    #analisi monotonia
    print("\n── compute_mmd2: MMD² cresce con la differenza ──")

    shifts = [0.0, 0.5, 1.0, 2.0, 3.0]
    vals = []

    for s_val in shifts:
        y_s = torch.randn(80, d) + s_val
        v, _ = compute_mmd2(x_same, y_s, device=device)
        vals.append(v)
        print(f"     shift={s_val:.1f}  MMD²={v:.4f}")

    ok("MMD² cresce monotonamente con lo shift",
       all(vals[i] <= vals[i + 1] + 0.01 for i in range(len(vals) - 1)),
       f"valori: {[f'{v:.3f}' for v in vals]}")

    #MMDResult
    print("\n── MMDResult ──")

    result_no_boot = MMDResult (mmd2 = 0.034, sigma = 8.3)
    ok("__str__ funziona senza bootstrap",
       "MMD²" in str(result_no_boot) and "p_value" not in str(result_no_boot))

    result_with_boot = MMDResult (mmd2 = 0.034, sigma = 8.3, p_value = 0.002, reject_h0 = True, n_perm = 1000)
    ok("__str__ funziona con bootstrap",
       "p_value" in str(result_with_boot) and "RIFIUTATA" in str(result_with_boot))

    #bootstrap_test
    print("\n── bootstrap_test ──")

    #Test su distribuzioni DIVERSE
    #PROVVISARIAMENTE RIDOTTO N_PERM PER TEST SU LAPTOP, INCREMENTARE A 1000
    result = bootstrap_test(x_diff, y_diff, n_perm=300, device=device, seed=0)

    ok("Restituisce MMDResult",
       isinstance(result, MMDResult))

    ok("p_value è un float in [0, 1]",
       isinstance(result.p_value, float) and 0.0 <= result.p_value <= 1.0,
       f"p_value = {result.p_value}")

    ok("reject_h0 è bool",
       isinstance(result.reject_h0, bool))

    ok("null_distribution ha n_perm valori",
       result.null_distribution is not None and len(result.null_distribution) == 300,
       f"len = {len(result.null_distribution) if result.null_distribution is not None else 'None'}")

    ok("null_distribution è su CPU (non in VRAM)",
       result.null_distribution.device.type == "cpu")

    ok("device_used è una stringa non vuota",
       isinstance(result.device_used, str) and len(result.device_used) > 0)

    ok("p_value < 0.05 per distribuzioni diverse (shift=2.0)",
       result.p_value < 0.05,
       f"p = {result.p_value:.4f}")

    ok("reject_h0 = True per distribuzioni diverse",
       result.reject_h0 is True)

    #Test su distribuzioni SIMILI
    #PROVVISARIAMENTE RIDOTTO N_PERM PER TEST SU LAPTOP, INCREMENTARE A 1000
    result_same = bootstrap_test(x_same, y_same, n_perm=300, device=device, seed=0)

    ok("p_value > 0.05 per distribuzioni simili  (tipicamente)",
       result_same.p_value > 0.05,
       f"p = {result_same.p_value:.4f}  (nota: può fallire raramente per varianza)")

    #Riproducibilità
    print("\n── Riproducibilità con seed ──")

    r1 = bootstrap_test(x_diff, y_diff, n_perm=100, device=device, seed=7)
    r2 = bootstrap_test(x_diff, y_diff, n_perm=100, device=device, seed=7)

    ok("Stesso seed → stesso MMD²",
       r1.mmd2 == r2.mmd2)
    ok("Stesso seed → stesso p_value",
       r1.p_value == r2.p_value)
    ok("Stesso seed → stessa null_distribution",
       torch.allclose(r1.null_distribution, r2.null_distribution))

    #Stampa risultati completi
    print(f"\n── Esempio output MMDResult ──")
    print(result)

    #Riepilogo testing
    print(f"\n{'=' * 45}")
    print(f"  {passed}/{passed + failed} test passati — device: {device}")
    print(f"{'=' * 45}\n")
    return failed == 0

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)