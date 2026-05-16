import numpy as np
import pywt
import librosa
from scipy.fft import rfft, rfftfreq
from scipy.signal import freqz
from vibdata.deep.signal.transforms import Transform, Mean
try:
    from PyEMD import EMD
except ImportError:
    EMD = None
try:
    from tftb.processing import smoothed_pseudo_wigner_ville
except ImportError:
    smoothed_pseudo_wigner_ville = None

# Note: We expect freq_features and wavelet_features to be available in the environment
# as they were in the original project. For the new package, we might want to integrate them
# or keep them as external dependencies if they are part of another package.
# Assuming they are local modules for now, we'll import them if they exist or 
# suggest moving them into signalai/features/ as well.

class WaveletCoeffsFeatures(Transform):
    def __init__(self, wavelet='db4', level=4, features=[Mean()]) -> None:
        super().__init__()
        self.level = level
        self.wavelet = wavelet
        self.features = features
        # These would normally be imported from a wavelet_features module
        # For now, we assume they are passed or available
        self.wavelet_multilevel_feat = [] 

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            coeffs = pywt.wavedec(sig, self.wavelet, level=self.level)

            new_data = []
            for c in coeffs:
                m_data = {'signal': c}
                for f in self.features:
                    new_data.append(f(m_data))

            m_data = {'signal': coeffs}
            for f in self.wavelet_multilevel_feat:
                new_data.append(f(m_data))
            ret.append(new_data)

        data["signal"] = ret
        return data

class PowerSpectralDensity(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        psd_list = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            sig = np.asarray(sig)
            sig_sample_rate = entry["sample_rate"]
            fft_vals = rfft(sig, norm="forward")

            if "original_sample_rate" in entry:
                freqs = rfftfreq(len(sig), d=1 / sig_sample_rate)
                bandwidth = entry["original_sample_rate"] / 2
                mask = freqs >= bandwidth
                fft_vals[mask] = 0.0

            psd = (np.abs(fft_vals) ** 2) / len(sig)
            psd_list.append(psd)

        data["signal"] = psd_list
        return data

class EMDCoeffsFeatures(Transform):
    def __init__(self, max_imf=None, features=[Mean()]) -> None:
        super().__init__()
        if EMD is None:
            raise ImportError("PyEMD is required for EMDCoeffsFeatures")
        self.max_imf = max_imf
        self.features = features

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            emd = EMD()
            imfs = emd(sig, max_imf=self.max_imf)
            
            new_data = []
            for imf in imfs:
                m_data = {'signal': imf}
                for f in self.features:
                    new_data.append(f(m_data))
            ret.append(new_data)

        data["signal"] = ret
        return data

class SpectralEnvelope(Transform):
    def __init__(self, n_lpc=16) -> None:
        super().__init__()
        self.n_lpc = n_lpc

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            sr = entry["sample_rate"] if "sample_rate" in entry else data["metainfo"]["sample_rate"]
            spectrum = np.abs(np.fft.rfft(sig))
            lpc = librosa.lpc(sig, order=self.n_lpc)
            w, h = freqz(1, a=lpc, worN=len(spectrum), fs=sr)
            envelope = np.abs(h)
            ret.append(envelope)

        data["signal"] = ret
        return data

class WignerVilleFeatures(Transform):
    def __init__(self, features=[Mean()], time_avg=True) -> None:
        super().__init__()
        if smoothed_pseudo_wigner_ville is None:
            raise ImportError("tftb is required for WignerVilleFeatures")
        self.features = features
        self.time_avg = time_avg

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            wvd = smoothed_pseudo_wigner_ville(sig, freq_bins=256) 
            if self.time_avg:
                energy = np.mean(np.abs(wvd), axis=1)
            else:
                energy = wvd.flatten()

            new_data = []
            m_data = {'signal': energy}
            for f in self.features:
                new_data.append(f(m_data))
            ret.append(new_data)

        data["signal"] = ret
        return data
