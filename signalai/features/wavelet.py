import numpy as np
from vibdata.deep.signal.transforms import Transform

class Energy(Transform):
    def transform(self, data):
        coeffs = np.asarray(data["signal"])
        energy = np.sum(coeffs ** 2)
        return energy

class Entropy(Transform):
    def transform(self, data):
        coeffs = np.asarray(data["signal"])
        energy = coeffs ** 2
        p = energy / (np.sum(energy) + 1e-12)
        entropy = -np.sum(p * np.log2(p + 1e-12))
        return entropy

class RelativeEnergyRatio(Transform):
    def transform(self, data):
        coeffs = data["signal"]
        if isinstance(coeffs, (list, tuple)):
            energies = np.array([np.sum(c ** 2) for c in coeffs])
            ratios = energies / (np.sum(energies) + 1e-12)
            return ratios.mean()
        else:
            return np.sum(np.asarray(coeffs) ** 2)

class LevelCorrelationCoefficients(Transform):
    def transform(self, data):
        coeffs = data["signal"]
        if not isinstance(coeffs, (list, tuple)) or len(coeffs) < 2:
            return 0.0
        min_len = min(len(c) for c in coeffs)
        coeffs = [c[:min_len] for c in coeffs]
        corrs = []
        for i in range(len(coeffs) - 1):
            r = np.corrcoef(coeffs[i], coeffs[i + 1])[0, 1]
            corrs.append(r)
        return np.mean(corrs)
