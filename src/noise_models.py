import numpy as np

### Simulate a channel where each transmitted bit flips with probability p_noise
def apply_bitflip(bits: np.ndarray, p_noise: float, rng: np.random.Generator) -> np.ndarray:
    bits = bits.astype(np.uint8, copy=True) 
    flips = rng.random(bits.shape[0]) < p_noise 
    bits[flips] ^= 1
    return bits

### Simulate a depolarizing channel where each state looses its information with probability p_noise
def depolarizing_mask(n: int, p_noise: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n) < p_noise
