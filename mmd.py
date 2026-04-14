#calcolo MMD e test statistico
#riferimento a Gretton "Kernel Two-Sample Test" (2012)

from dataclasses import dataclass
import torch
from sympy.abc import alpha
from torch import Tensor

from device import get_device, move_to_device
from kernels import BaseKernel, GaussianKernel
from bandwidth import median_bandwidth

#Dataclass che raccoglie tutti i risultati in un unico oggetto: MMD value, parametro σ, p-value etc
@dataclass
class MMDResult:
    mmd2: float #valore MMD osservato
    sigma: float #bandwidth σ usato dal kernel
    p_value: float | None = None
    reject_h0: bool | None = None #True se p_value < alpha. None se non calcolato
    alpha: float = 0.05 #livello di significatività (0.05 per default)
    n_perm: int = 0 #numero di permutazioni bootstrap eseguite
    null_distribution: Tensor | None = None #Tensor 1D con i valori MMD sotto H0
    device_used: str = "cpu" #device su cui è stato effettuato il calcolo

    def __str__(self) -> str:
        lines = [
            f"MMD² = {self.mmd2:.6f}"
            f"σ (kernel) = {self.sigma:.4f}",
            f"Device = {self.device_used}",
        ]

        if self.p_value is not None:
            pv = f"{self.p_value:.4f}" if self.p_value >= 0.001 else "< 0.001"
            #verdetto su ipotesi nulla
            verdict = "RIFIUTATA → distribuzioni diverse ✓" \
                        if self.reject_h0 else \
                        "NON rifiutata → distribuzioni simili"
            lines += [
                f"p-value    = {pv}  (n_perm={self.n_perm}, α={self.alpha})",
                f"H₀ (P=Q)   {verdict}",
            ]

        return "\n".join(lines)

#funzione interna utile per svolgere la matematica dell'MMD.
#Sfrutta la formula U-statistics da Gretton -> MMD² = 1/n(n-1) Σᵢ≠ⱼ k(xᵢ,xⱼ)  +  1/m(m-1) Σᵢ≠ⱼ k(yᵢ,yⱼ)  -  2/nm Σᵢⱼ k(xᵢ,yⱼ)

def _mmd2_from_kernels(
        K_xx: Tensor, #Kernel tra campioni reali
        K_yy: Tensor, #Kernel tra campioni generati
        K_xy: Tensor, #Kernel incrociato
) -> float:
    n = K_xx.shape[0]
    m = K_yy.shape[0] #dimensioni

    # .trace() somma la diagonale — la sottraiamo per escludere k(xᵢ,xᵢ)
    sum_xx = (K_xx.sum() - K_xx.trace()) / (n * (n - 1))
    sum_yy = (K_yy.sum() - K_yy.trace()) / (m * (m - 1))
    sum_xy = K_xy.sum() / (n * m)

    return (sum_xx + sum_yy - 2.0 * sum_xy).item()

#funzione di esecuzione rapida che calcola MMD tra x brani reali e y generati
def compute_mmd2(
        x: Tensor, #feature brani reali [n, d]
        y: Tensor, #feature brani generati [m, d]
        kernel: BaseKernel | None = None, #istanza Kernel, se None costruisce GaussianKernel
        sigma: float | None = None, #bandwidth automatico se Kernel non viene fornito
        device: torch.device | None = None,
) -> tuple[float, float]:

    if device is not None:
        x, y = move_to_device(x, y, device=device)

    #Se sigma non è fornito, viene calcolati dai dati tramite median heuristic
    if kernel is None: #kernel non fornito
        if sigma is None: #sigma di conseguenza non fornito
            sigma = median_bandwidth(x, y)
        kernel = GaussianKernel(sigma=sigma) #istanzia un kernel Gaussiano se non ne è stato fornito uno in input
    else:
        #Altrimenti recupera il sigma dal kernel fornito
        sigma = getattr(kernel, 'sigma', float('nan'))

    with torch.no_grad(): #disabilita il calcolo del gradiente, dispendioso e inutile se non si deve fare addestramento
        K_xx = kernel(x, x)
        K_yy = kernel(y, y)
        K_xy = kernel(x, y)

    return _mmd2_from_kernels(K_xx, K_yy, K_xy), float(sigma)

#svolge il test statistico completo di MMD osservato + p_value tramite test di permutazione
def bootstrap_test(
        x: Tensor, #feature brani reali [n, d]
        y: Tensor, #feature brani generati [m, d]
        kernel: BaseKernel | None = None, ##istanza Kernel, se None costruisce GaussianKernel
        sigma: float | None = None, #bandwidth esplicito
        n_perm: int = 1000, #numero permutazioni bootstrap. Più è alto più è lento, ma il p_value è più preciso
        alpha: float = 0.05, #livello di significatività
        device: torch.device | None = None,
        seed: int | None = None, #seed per riproducibilità dei risultati
) -> MMDResult:
    """
        1. Calcola MMD sui dati reali -> risulta il valore osservato
        2. Unisce X e Y in un pool, mescola i gruppi e ricalcola MMD
        3. Riassegna i gruppi e ricalcola MMD
        4. p_value = frazione di valori nulli >= valore osservato

        Ottimizzazioni GPU:
        - torch.randperm(device=device) → permutazione direttamente su CUDA
        - null_dist allocato su device → nessun trasferimento durante il loop
        - .cpu() solo alla fine → libera VRAM dopo il calcolo
        - torch.no_grad() → niente grafo computazionale, -30% memoria
    """

    if seed is not None:
        torch.manual_seed(seed)

    #migrazione device
    if device is None:
        device = x.device
    x, y = move_to_device(x, y, device=device)

    #Costruisce il kernel con sigma fissato (e costante tra le permutazioni)
    if kernel is None:
        if sigma is None:
            sigma = median_bandwidth(x, y)
        kernel = GaussianKernel(sigma=sigma)
    else:
        sigma = getattr(kernel, 'sigma', float('nan'))

    #imposto le dimensioni
    n = x.shape[0]
    m = y.shape[0]

    #1. MMD osservato sui dati reali
    with torch.no_grad():
        K_xx = kernel(x, x)
        K_yy = kernel(y, y)
        K_xy = kernel(x, y)
    mmd2_obs = _mmd2_from_kernels(K_xx, K_yy, K_xy)

    #2. Permutation Test
    xy_pool = torch.cat([x, y], dim=0)
    total = xy_pool.shape[0] #dimensione pool campioni

    #pre-alloca il buffer della distribuzione nulla sul device per evitare n_perm allocazioni separate all'interno del loop
    null_dist = torch.zeros(n_perm, device=device, dtype=x.dtype)

    with torch.no_grad(): #ricalcolo sul nuovo pool
        for i in range(n_perm):
            perm = torch.randperm(total, device=device) #istruzione che gira su CUDA se il device ne è dotato
            xy_shuf = xy_pool[perm]
            x_p, y_p = xy_shuf[:n], xy_shuf[n:]

            #ricalcola i kernel sulle nuove permutazioni
            K_xx_p = kernel(x_p, x_p)
            K_yy_p = kernel(y_p, y_p)
            K_xy_p = kernel(x_p, y_p)

            null_dist[i] = _mmd2_from_kernels(K_xx_p, K_yy_p, K_xy_p)

    #3. p_value
    p_value = (null_dist >= mmd2_obs).float().mean().item()
    reject = p_value < alpha #accetto o rifiuto l'ipotesi nulla H0

    return MMDResult(
        mmd2 = mmd2_obs,
        sigma = float(sigma),
        p_value = p_value,
        reject_h0 = reject, #verdetto sull'ipotesi
        alpha = alpha,
        n_perm = n_perm,
        null_distribution = null_dist.cpu(), #spostata su CPU per non lasciare Tensor grandi in VRAM
        device_used = str(device),
    )