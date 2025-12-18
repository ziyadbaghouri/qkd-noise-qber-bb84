import numpy as np
from qutip import Qobj
from src.states import BB84States

def measure_in_basis(rho: Qobj, basis_id: int, rng: np.random.Generator) -> int:
    P0, P1 = BB84States.proj[basis_id]
    p0 = (P0 * rho).tr().real

    if p0 < 0.0: p0 = 0.0
    if p0 > 1.0: p0 = 1.0

    return 0 if rng.random() < p0 else 1
