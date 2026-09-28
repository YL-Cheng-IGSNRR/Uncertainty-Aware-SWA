"""One reproducible input-noise realization; no coefficient estimation."""
import numpy as np


def perturb_inputs(data, seed=20260929):
    """Copy a DataFrame; independently perturb bands, preserving reference LST.

    Gaussian standard deviations: BT 0.05 K, emissivity 0.01,
    WVC 10% of each original value. Clip as in the TD-All experiment.
    All algorithms must receive this same perturbed table.
    """
    rng = np.random.default_rng(seed)
    noisy = data.copy(deep=True)
    for column in ('T31_K', 'T32_K'):
        noisy[column] += rng.normal(0, .05, len(noisy))
    for column in ('emissivity31', 'emissivity32'):
        noisy[column] = np.clip(noisy[column] + rng.normal(0, .01, len(noisy)), 0, .999)
    noisy['wvc_g_cm2'] = np.clip(noisy.wvc_g_cm2 + rng.normal(0, .1 * noisy.wvc_g_cm2, len(noisy)), 0, 9.99)
    return noisy
