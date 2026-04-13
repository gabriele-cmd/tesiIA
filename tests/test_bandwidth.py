"""
Esecuzione: python tests/test_bandwidth.py
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from device import get_device, move_to_device
from bandwidth import median_bandwidth, silverman_bandwidth, bandwidth_grid

def run():
    device = get_device()
    torch.manual_seed(0)
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

    print("\n── median_bandwidth: proprietà fondamentali ──")
    x = torch.randn(100, d) * 2.0
    y = torch.randn(100, d) * 2.0 + 1.0
    x, y = move_to_device(x, y, device=device)
    sigma = median_bandwidth(x, y)

    ok("Restituisce un float Python", isinstance(sigma, float))
    ok("σ > 0", sigma > 0, f"sigma={sigma}")
    ok("σ è finito", sigma != float('inf') and sigma == sigma)

    print("\n── median_bandwidth: scala con la dispersione ──")
    x_g, y_g = move_to_device(torch.randn(100,d)*20, torch.randn(100,d)*20, device=device)
    x_p, y_p = move_to_device(torch.randn(100,d)*2, torch.randn(100,d)*2, device=device)
    sg, sp = median_bandwidth(x_g, y_g), median_bandwidth(x_p, y_p)

    ok("σ scala con la dispersione (std×10 → σ×~10)",
       7 < sg / sp < 15,
       f"sg={sg:.3f}, sp={sp:.3f}, rapporto={sg / sp:.1f}x")

    ok("y=None funziona (solo X)",
       median_bandwidth(x) > 0)

    print("\n── median_bandwidth: sottocampionamento ──")

    x5, y5 = move_to_device(torch.randn(5000, d), torch.randn(5000, d), device=device)
    s_sub = median_bandwidth(x5, y5, max_samples=500)
    s_full = median_bandwidth(x5, y5, max_samples=5000)

    ok("Sottocampionamento: errore < 20% rispetto al full",
       abs(s_sub - s_full) / s_full < 0.2,
       f"sub={s_sub:.4f}, full={s_full:.4f}")

    #CASO LIMITE: dati costanti
    print("\n── median_bandwidth: caso limite ──")

    x_c = torch.ones(20, d, device=device)
    y_c = torch.ones(20, d, device=device)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s_const = median_bandwidth(x_c, y_c)

    ok("Emette UserWarning per dati costanti",
       any(issubclass(wi.category, UserWarning) for wi in w))
    ok("Fallback a σ=1.0",
       s_const == 1.0, f"sigma={s_const}")

    #Silverman
    print("\n── silverman_bandwidth ──")

    s_sil = silverman_bandwidth(x)
    ok("Restituisce float > 0", isinstance(s_sil, float) and s_sil > 0)
    ok("Scala con std dei dati",
       silverman_bandwidth(torch.randn(100, d, device=device) * 10) >
       silverman_bandwidth(torch.randn(100, d, device=device) * 1))
    ok("Scala con n: più campioni → σ più piccolo",
       silverman_bandwidth(torch.randn(1000, d, device=device)) <
       silverman_bandwidth(torch.randn(10, d, device=device)))

    #Bandwidth Grid
    print("\n── bandwidth_grid ──")

    grid = bandwidth_grid(x, y, n_values=7)
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    sm = median_bandwidth(x, y)

    ok("Restituisce esattamente n_values=7 valori", len(grid) == 7)
    ok("Tutti i valori > 0", all(s > 0 for s in grid))
    ok("Valori in ordine crescente",
       all(grid[i] < grid[i + 1] for i in range(6)))
    ok("Scala geometrica (rapporti consecutivi costanti ±1%)",
       all(abs(r - ratios[0]) / ratios[0] < 0.01 for r in ratios))
    ok("σ_mediana è contenuto nella griglia",
       grid[0] <= sm <= grid[-1],
       f"σ_med={sm:.3f}, grid=[{grid[0]:.3f}…{grid[-1]:.3f}]")
    ok("n_values=1 restituisce [σ_mediana]",
       bandwidth_grid(x, y, n_values=1) == [median_bandwidth(x, y)])

    #Riepilogo test
    print(f"\n{'=' * 45}")
    print(f"  {passed}/{passed + failed} test passati — device: {device}")
    print(f"{'=' * 45}\n")
    return failed == 0

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)