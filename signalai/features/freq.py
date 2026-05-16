import numpy as np
from scipy.stats import entropy
from vibdata.deep.signal.transforms import Transform

class SpectralCentroid(Transform):
    def transform(self, data):
        spectrum = np.abs(data["signal"])
        fs = data["metainfo"]["sample_rate"]
        freqs = np.linspace(0, fs / 2, len(spectrum))
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)
        return centroid

class SpectralBandwidth(Transform):
    def transform(self, data):
        spectrum = np.abs(data["signal"])
        fs = data["metainfo"]["sample_rate"]
        freqs = np.linspace(0, fs / 2, len(spectrum))
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-12))
        return bandwidth

class SpectralFlatness(Transform):
    def transform(self, data):
        spectrum = np.abs(data["signal"])
        geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-12)))
        arithmetic_mean = np.mean(spectrum)
        flatness = geometric_mean / (arithmetic_mean + 1e-12)
        return flatness

class DominantFrequency(Transform):
    def transform(self, data):
        spectrum = np.abs(data["signal"])
        fs = data["metainfo"]["sample_rate"]
        freqs = np.linspace(0, fs / 2, len(spectrum))
        dominant_freq = freqs[np.argmax(spectrum)]
        return dominant_freq

class SpectralRolloff(Transform):
    def __init__(self, roll_percent=0.85):
        super().__init__()
        self.roll_percent = roll_percent

    def transform(self, data):
        spectrum = np.abs(data["signal"])
        fs = data["metainfo"]["sample_rate"]
        freqs = np.linspace(0, fs / 2, len(spectrum))
        cumulative = np.cumsum(spectrum)
        threshold = self.roll_percent * cumulative[-1]
        rolloff_freq = freqs[np.where(cumulative >= threshold)[0][0]]
        return rolloff_freq

class SpectralEntropy(Transform):
    def transform(self, data):
        spectrum = np.abs(data["signal"])
        psd = spectrum ** 2
        psd_norm = psd / (np.sum(psd) + 1e-12)
        spec_entropy = entropy(psd_norm)
        return spec_entropy
