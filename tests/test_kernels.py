"""
    Esecuzione:
        python tests\test_kernels.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from device import get_device, move_to_device
from kernels import GaussianKernel, LaplacianKernel, LinearKernel, MultiScaleGaussianKernel

def run():
    device = get_device()
    torch.manual_seed(0)
    passed = failed = 0

    def ok(name, cond, detail =""):
        nonlocal passed, failed
        if cond:
            print(f"  ✓  {name}")
            passed += 1
        else:
            print(f"  ✗  {name}" + (f"\n     → {detail}" if detail else ""))
            failed += 1

    #sfrutta dati sintentici: simula 30 brani con 16 feature l'uno
    x = torch.randn(30, 16)
    y = torch.randn(30, 16)
    x, y = move_to_device(x, y, device=device)

    print("\n── Gaussian Kernel ──")
    k = GaussianKernel(sigma=1.0)
    K_xy = k(x, y)
    K_xx = k(x, x) #prodotti scalari in funzioni kernel Gaussiani

    ok("Valori ∈ (0, 1]",
       K_xy.min().item() >= 0 and K_xy.max().item() <= 1.0 + 1e-6,
       f"range = [{K_xy.min():.4f}, {K_xy.max():.4f}]")

    ok("Diagonale K(x,x) = 1  [k(xᵢ, xᵢ) deve essere esattamente 1]",
       torch.allclose(K_xx.diagonal(), torch.ones(30, device=device)),
       f"diag_mean = {K_xx.diagonal().mean():.6f}")

    print(f"     max diff = {(K_xy - k(y, x).T).abs().max():.2e}")
    ok("Simmetria: K(x,y) = K(y,x)ᵀ",
       torch.allclose(K_xy, k(y, x).T, atol=1e-4))

    ok("σ grande → kernel piatto (K ≈ 1, tutti i punti 'simili')",
       GaussianKernel(sigma=1e4)(x, y).mean().item() > 0.999)

    ok("σ piccolo → kernel stretto (K ≈ 0, solo i punti identici sono simili)",
       GaussianKernel(sigma=1e-4)(x, y).mean().item() < 1e-3)

    ok("Device output = device input",
       str(K_xy.device).startswith(device.type))

    #Errori Attesi
    print("\n── Validazione input ──")
    if device.type != "cpu":
        try:
            GaussianKernel(sigma=1.0)(x, y.cpu())
            ok("Device mismatch rilevato", False, "avrebbe dovuto lanciare ValueError")
        except ValueError:
            ok("Device mismatch rilevato", True)
    else:
        print("  –  Device mismatch  (saltato: CPU-only, non ci sono due device distinti)")

    try:
        GaussianKernel(sigma=1.0)(x, torch.randn(30, 8, device=device))
        ok("Dimensioni incompatibili rilevate", False, "avrebbe dovuto lanciare ValueError")
    except ValueError:
        ok("Dimensioni incompatibili rilevate", True)

    try:
        GaussianKernel(sigma=1.0)(torch.randn(30, device=device), y)
        ok("Tensore 1D rifiutato", False)
    except ValueError:
        ok("Tensore 1D rifiutato", True)


    print("\n── LaplacianKernel ──")
    kl = LaplacianKernel(sigma=1.0)
    Kl = kl(x, y)
    ok("Output shape [n, m]", Kl.shape == (30, 30))
    ok("Valori ∈ (0, 1]",
       Kl.min().item() >= 0 and Kl.max().item() <= 1.0 + 1e-6)
    ok("Simmetria",
       torch.allclose(Kl, kl(y, x).T, atol=1e-5))

    print("\n── LinearKernel ──")
    kln = LinearKernel()
    Klin = kln(x, y)
    ok("Output shape [n, m]", Klin.shape == (30, 30))
    # Il kernel lineare può essere negativo — è normale
    ok("Coincide con prodotto matriciale manuale",
       torch.allclose(Klin, x @ y.T, atol=1e-5))

    print("\n── MultiScaleGaussianKernel ──")
    km = MultiScaleGaussianKernel(sigmas=[0.5, 1.0, 2.0])
    Km = km(x, y)
    ok("Output shape [n, m]", Km.shape == (30, 30))
    ok("Valori ∈ (0, 1]  (media di kernel ∈ (0,1])",
       Km.min().item() >= 0 and Km.max().item() <= 1.0 + 1e-6)

    # Verifica che sia effettivamente la media dei tre kernel
    K_ref = (GaussianKernel(0.5)(x, y) +
             GaussianKernel(1.0)(x, y) +
             GaussianKernel(2.0)(x, y)) / 3
    ok("È la media esatta dei kernel alle scale date",
       torch.allclose(Km, K_ref, atol=1e-5))

   #Riepilogo test
    print(f"\n{'=' * 40}")
    print(f"  {passed}/{passed + failed} test passati — device: {device}")
    print(f"{'=' * 40}\n")
    return failed == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)