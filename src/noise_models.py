from qutip import Qobj, sigmax
from states import BB84States

def apply_bitflip_channel(rho: Qobj, p_noise: float) -> Qobj:
    if not (0.0 <= p_noise <= 1.0):
        raise ValueError("p_noise must be in [0, 1]")

    X = sigmax()
    return (1 - p_noise) * rho + p_noise * (X * rho * X)


def apply_depolarizing_channel(rho: Qobj, p_noise: float) -> Qobj:
    if not (0.0 <= p_noise <= 1.0):
        raise ValueError("p_noise must be in [0, 1]")

    return (1 - p_noise) * rho + p_noise * (BB84States.I2 / 2)
