"""
median_bandwidth() - da Gretton (2012)
silverman_bandwidth() - alternativa teorica utile per confronti
bandwidth_grid() - griglia per esperimenti di sensitività
"""

import warnings
import torch
from torch import Tensor

#Median Heuristic
#calcola tutte le distanze tra i campioni (‖zᵢ - zⱼ‖)
#prende la mediana e la usa come parametro σ
#la mediana è meno sensibile agli outlier (caso di brani con feature lontane dagli altri)
def median_bandwidth(
    x: Tensor,
    y: Tensor | None = None,
    max_samples: int = 2000, ) -> float:
    #σ = mediana( ‖zᵢ − zⱼ‖ )   per tutte le coppie i < j, dove Z = X ∪ Y

    #Costruisce Z = X ∪ Y sul device di x
    z = torch.cat([x,y], dim=0) if y is not None else x
    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z = z[idx]

    #calcola le distanze a coppie
    with torch.no_grad():
        D = torch.cdist(z, z, 2)

    n = D.shape[0]
    upper_mask = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=D.device),
        diagonal=1  # diagonal=1 esclude la diagonale principale
    )
    pairwise = D[upper_mask] #tensore 1D con tutte le distanze uniche

    if pairwise.numel() == 0:
        raise ValueError("Nessuna coppia di distanze disponibile."
                         "Servono almento 2 campioni.")

    sigma = pairwise.median().item() #convesrione in float

    #caso limite dati costanti o quasi identici
    if sigma < 1e-8:
        warnings.warn(
            f"Median heuristic ha restituito σ ≈ {sigma:.2e}. "
            "I campioni potrebbero essere quasi identici. "
            "Usando σ = 1.0 come fallback.",
            UserWarning,
            stacklevel=2
        )
        sigma = 1.0

    return float(sigma)

#Regola di Silverman
#sfrutta σ = ( 4/(d+2) )^(1/(d+4))  ×  n^(-1/(d+4))  ×  std_medio
#utile a dimostrare che le conclusioni sono robuste alla scelta del bandwidth le rende più credibili
#Tuttavia Silverman sovrastima o sottostima il parametro se le feature audio non sono Gaussiane

def silverman_bandwidth(x: Tensor) -> float:
    n, d = x.shape
    std_mean = x.std(dim=0).mean().item()
    sigma = ( (4.0 / (d + 2)) ** (1.0 / (d + 4)) * float(n) ** (-1.0 / (d + 4)) * std_mean )

    return max(float(sigma), 1e-8)

#Sensitivity Analysis con griglia di bandwidth
#utile a dimostrare che i risultati non dipendono in maniera critica dalla scelta del parametro

def bandwidth_grid(x: Tensor, y: Tensor, n_values: int = 10, scale_range: tuple[float, float] = (0.1,10.0)) -> list[float]:
    #Griglia di bandwith costruita attorno alla median heuristic
    #genera n_values valori di σ distribuiti in un intervallo [ σ_mediana × scale_range[0],  σ_mediana × scale_range[1] ]

    sigma_med = median_bandwidth(x, y)
    lo = sigma_med * scale_range[0]
    hi = sigma_med * scale_range[1]

    #costruisce la scala geometrica
    if n_values == 1:
        return [sigma_med]

    ratio = (hi / lo) ** (1.0 / (n_values - 1))
    return [float(lo * ratio ** i) for i in range(n_values)]
