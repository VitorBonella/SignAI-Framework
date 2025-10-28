import numpy as np
import pywt
from vibdata.deep.signal.transforms import Transform

class Energy(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        coeffs = np.asarray(data["signal"])
        energy = np.sum(coeffs ** 2)
        return energy


class Entropy(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        coeffs = np.asarray(data["signal"])
        energy = coeffs ** 2
        p = energy / (np.sum(energy) + 1e-12)
        entropy = -np.sum(p * np.log2(p + 1e-12))
        return entropy


class RelativeEnergyRatio(Transform):
    """
    Expects data["signal"] to be a list or array of wavelet levels (e.g. [cA, cD1, cD2, ...]).
    Computes the relative energy ratio of each level w.r.t. total energy.
    Returns the ratio of the current level (if single-level input) or a mean if multi-level.
    """
    def __init__(self):
        super().__init__()

    def transform(self, data):
        coeffs = data["signal"]

        # handle multiple levels (list or tuple of arrays)
        if isinstance(coeffs, (list, tuple)):
            energies = np.array([np.sum(c ** 2) for c in coeffs])
            ratios = energies / (np.sum(energies) + 1e-12)
            return ratios.mean()
        else:
            # single level (no other context)
            energy = np.sum(coeffs ** 2)
            return energy  # relative only makes sense if multi-level


class LevelCorrelationCoefficients(Transform):
    """
    Computes correlation coefficients between adjacent wavelet levels.
    Expects data["signal"] to be a list or tuple of wavelet levels: [cA, cD1, cD2, ...].
    """
    def __init__(self):
        super().__init__()

    def transform(self, data):
        coeffs = data["signal"]
        if not isinstance(coeffs, (list, tuple)) or len(coeffs) < 2:
            return 0.0  # not enough levels to correlate

        # Align lengths by truncating to min length
        min_len = min(len(c) for c in coeffs)
        coeffs = [c[:min_len] for c in coeffs]

        corrs = []
        for i in range(len(coeffs) - 1):
            r = np.corrcoef(coeffs[i], coeffs[i + 1])[0, 1]
            corrs.append(r)
        return np.mean(corrs)