from abc import ABC, abstractmethod
import torch
from torch import Tensor

class BaseKernel(ABC):

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        #calcola la matrice Kernel K[i,j] = k(xi,yj)
        #x : [n,d] - n campioni d-dimensionali
        #y : [m,d] - m campioni d-dimensionali
        #ritorna K : [n,m] - matrice di similarità

        self._validate(x,y) #verifica di compatibilità degli input
        return self._compute(x,y) #calcolo effettivo matrice

    @abstractmethod
    def _compute(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    @staticmethod
    def _validate(x: Tensor, y: Tensor) -> None:
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError(f"I tensori devono essere 2D [n, d]. "
                f"Ricevuti: x={tuple(x.shape)}, y={tuple(y.shape)}")

        if x.shape[1] != y.shape[1]:
            raise ValueError(f"Le dimensioni delle feature devono coincidere. "
                f"x ha d={x.shape[1]}, y ha d={y.shape[1]}")

        if x.device != y.device:
            raise ValueError(f"x e y devono stare sullo stesso device. "
                f"x è su {x.device}, y è su {y.device}. "
                f"Usa move_to_device() da device.py prima di chiamare il kernel.")


    @staticmethod
    def _sq_distances(x: Tensor, y: Tensor) -> Tensor:
        #Calcolo della distanza quadratica per tutte le coppie (i,j) (‖xᵢ - yⱼ‖²)
        #Formula usata: ‖x - y‖² = ‖x‖² + ‖y‖² - 2⟨x, y⟩ per minimizzare l'utilizzo di memoria
        #Usare (x - y)² richiederebbe una matrice [n,m,d] troppo dispendiosa

        x_sq = (x*x).sum(dim=1, keepdim=True) # [n, 1] , ‖xᵢ‖²
        y_sq = (y*y).sum(dim=1, keepdim=True).T # [1, m] , ‖yⱼ‖²
        xy = torch.mm(x, y.T) # [n, m] , ⟨xᵢ, yⱼ⟩

        return torch.clamp(x_sq + y_sq - 2 * xy, min=0.0) #clamp(min=0.0) evita valori negativi minuscoli dovuti all'arrotondamento del floating point

class GaussianKernel(BaseKernel):
    #Sfrutta k(x, y) = exp( -‖x - y‖² / (2σ²) )
    #vale idealmente 1 quando x = y e si avvicina a 0 quando x e y si discostano
    #il valore σ indica quanto rapidamente converge a 0
    #In generale MMD(P,D) = 0 solo se P = Q, garantito per qualsiasi σ>0

    def __init__(self, sigma: float | None = None): #sigma è il bandwidth > 0 del kernel calcolato automaticamente
        if sigma is not None and sigma <= 0:
            raise ValueError(f"sigma deve essere > 0, ricevuto: {sigma}")
        self.sigma = sigma

    def _compute(self, x: Tensor, y: Tensor) -> Tensor:
        if self.sigma is None:
            raise RuntimeError("sigma non è impostato. "
                "Crea il kernel con GaussianKernel(sigma=valore) "
                "oppure usa median_bandwidth() da bandwidth.py per calcolarlo.")
        D_sq = self._sq_distances(x, y)

        return torch.exp(-D_sq / (2.0 * self.sigma**2))

    def __repr__(self) -> str:
        s = f"{self.sigma:.4f}" if self.sigma is not None else "None (da impostare)"
        return f"GaussianKernel(sigma={s})"

class LaplacianKernel(BaseKernel):
    #Usa la formula k(x, y) = exp( -‖x - y‖ / σ ) e sfrutta la distanza L2 (non quadratica)
    #Utile per distribuzioni con outlier marcati o per distanze elevate tra le features reali e generate

    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError(f"sigma deve essere > 0, ricevuto: {sigma}")
        self.sigma = sigma

    def _compute(self, x: Tensor, y: Tensor) -> Tensor:
        #Funzione torch.cdist è utile la distanza non al quadrato (L29
        D = torch.cdist(x, y, p=2.0)
        return torch.exp(-D / self.sigma)

    def __repr__(self) -> str:
        return f"LaplacianKernel(sigma={self.sigma:.4f})"

class LinearKernel(BaseKernel):
    #Utilizza il semplice prodotto scalare k(x, y) = ⟨x, y⟩
    #Utile per capire se le differenze tra reale e generato sono catturabili linearmente

    def _compute(self , x: Tensor, y: Tensor) -> Tensor:
        return torch.mm(x, y.T) #funzione lineare

    def __repr__(self) -> str:
        return "LinearKernel()"

class MultiScaleGaussianKernel(BaseKernel):
    #Sfrutta k(x, y) = (1/S) Σₛ exp( -‖x - y‖² / (2σₛ²) ) e fa la media di S kernel Gaussiani con bandwith diversi
    #Metriche diverse catturano strutture locali o globali su scale diverse tra loro (vedi paper di Gretton)

    def __init__(self, sigmas: list[float] | None = None):
        if sigmas is None:
            sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        if any(s <=0 for s in sigmas):
            raise ValueError("Tutti i valori di sigma devono essere > 0")
        self.sigmas = sigmas

    def _compute(self, x: Tensor, y: Tensor) -> Tensor:
        D_sq = self._sq_distances(x, y)
        #Accumula su un tensore già sul device corretto (quello di D_sq)
        K = torch.zeros_like(D_sq)
        for s in self.sigmas:
            K = K + torch.exp(-D_sq / (2.0 * s ** 2))

        return K / len(self.sigmas)

    def __repr__(self) -> str:
        return f"MultiScaleGaussianKernel(sigmas={[round(s, 3) for s in self.sigmas]})"