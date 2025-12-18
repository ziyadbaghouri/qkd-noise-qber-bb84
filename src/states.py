from __future__ import annotations
from dataclasses import dataclass
from qutip import basis, ket2dm, qeye

@dataclass(frozen=True)
class BB84States:
    # kets
    ket0 = basis(2, 0)
    ket1 = basis(2, 1)
    ket_plus  = (ket0 + ket1).unit()
    ket_minus = (ket0 - ket1).unit()

    # map (basis, bit) -> ket
    # basis: 0 -> '+' ; 1 -> 'x'
    state = {
        (0, 0): ket0,
        (0, 1): ket1,
        (1, 0): ket_plus,
        (1, 1): ket_minus,
    }

    # projectors for measurement in each basis: [P(bit=0), P(bit=1)]
    proj = {
        0: [ket0 * ket0.dag(), ket1 * ket1.dag()],
        1: [ket_plus * ket_plus.dag(), ket_minus * ket_minus.dag()],
    }

    I2 = qeye(2)

def prepare_density_matrix(basis_id: int, bit: int):
    """Return rho = |psi><psi| for the chosen (basis, bit)."""
    from qutip import ket2dm
    psi = BB84States.state[(basis_id, bit)]
    return ket2dm(psi)
