import numpy as np
from typing import Tuple

def sift_keys(alice_bits: np.ndarray,bob_bits: np.ndarray, alice_bases: np.ndarray, bob_bases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not (alice_bits.shape == bob_bits.shape == alice_bases.shape == bob_bases.shape):
        raise ValueError("All input arrays must have the same shape")

    keep = (alice_bases == bob_bases)
    return alice_bits[keep], bob_bits[keep]

def qber(alice_sifted: np.ndarray, bob_sifted: np.ndarray) -> float:
    if alice_sifted.shape != bob_sifted.shape:
        raise ValueError("alice_sifted and bob_sifted must have the same shape")

    n = int(alice_sifted.size)
    if n == 0:
        return float("nan")
    return float(np.mean(alice_sifted != bob_sifted))

def count_errors(alice_sifted: np.ndarray, bob_sifted: np.ndarray) -> int:
    if alice_sifted.shape != bob_sifted.shape:
        raise ValueError("alice_sifted and bob_sifted must have the same shape")
    return int(np.sum(alice_sifted != bob_sifted))
