import numpy as np
from qutip import Qobj

from measurement import measure_in_basis
from states import prepare_density_matrix

def intercept_resend(rho_in: Qobj, rng: np.random.Generator) -> Qobj:
    eve_basis = int(rng.integers(0, 2))              # random basis choice
    eve_bit   = measure_in_basis(rho_in, eve_basis, rng)  # measurement collapses state
    rho_out   = prepare_density_matrix(eve_basis, eve_bit) # resend corresponding pure state
    return rho_out
