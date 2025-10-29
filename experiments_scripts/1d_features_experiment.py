import sys
import librosa
import numpy as np
import pywt
from sklearn.ensemble import RandomForestClassifier
import vibdata.raw as raw_datasets
from vibdata.deep.signal.transforms import (
    Kurtosis, RootMeanSquare, StandardDeviation, Mean, LogAttackTime,
    TemporalDecrease, TemporalCentroid, EffectiveDuration, ZeroCrossingRate,
    PeakValue, CrestFactor, Skewness, ClearanceFactor, ImpulseFactor,
    ShapeFactor, UpperBoundValueHistogram, LowerBoundValueHistogram,
    Variance, PeakToPeak, Transform, Sequential, SplitSampleRate,
    FeatureExtractor, FilterByValue, Aggregator, FFT
)
from vibdata.deep.DeepDataset import convertDataset
from vibdata.deep.signal.core import SignalSample
from signalAI.utils.group_dataset import GroupDataset
from signalAI.utils.fold_idx_generator import FoldIdxGeneratorUnbiased
from signalAI.experiments.features_1d import Features1DExperiment

from tftb.processing import smoothed_pseudo_wigner_ville
from scipy.signal import freqz
from scipy.signal.windows import hamming

import freq_features
import wavelet_features
from ims_resampler import ResamplerIMS

# ======================================
# Feature extraction setup
# ======================================
features_funcs_time = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(), LogAttackTime(),
    TemporalDecrease(), TemporalCentroid(), EffectiveDuration(), ZeroCrossingRate(),
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), UpperBoundValueHistogram(), LowerBoundValueHistogram(), Variance(), PeakToPeak()
]

features_funcs_freq = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(),
    CrestFactor(), Skewness(), Variance(), PeakValue(),
    freq_features.SpectralRolloff(), freq_features.SpectralFlatness(),
    freq_features.SpectralEntropy(), freq_features.SpectralCentroid(), freq_features.SpectralBandwidth(),
    freq_features.DominantFrequency()
]

features_funcs_wavelet = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(), 
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), Variance(), wavelet_features.Energy(),
    wavelet_features.Entropy(),
]

transforms_time = Sequential([
    SplitSampleRate(),
    FeatureExtractor(features=features_funcs_time),
])

transforms_frequency = Sequential([
    SplitSampleRate(),
    FFT(),
    FeatureExtractor(features=features_funcs_freq),
])

transforms_time_frequency = Sequential([
    SplitSampleRate(),
    Aggregator([
        FeatureExtractor(features=features_funcs_time),  # Time domain features
        Sequential([FFT(), FeatureExtractor(features=features_funcs_freq)])  # Frequency domain features
    ])
])

class WaveletCoeffsFeatures(Transform):
    def __init__(self, wavelet='db4', level=4, features = [Mean()]) -> None:
        super().__init__()
        self.level = level
        self.wavelet = wavelet
        self.features = features
        self.wavelet_multilevel_feat = [wavelet_features.LevelCorrelationCoefficients(),wavelet_features.RelativeEnergyRatio()]

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            coeffs = pywt.wavedec(sig, self.wavelet, level=self.level)

            new_data = []
            for c in coeffs:
                m_data = {}
                m_data['signal'] = c
                [new_data.append(f(m_data)) for f in self.features]

            m_data = {}
            m_data['signal'] = coeffs
            [new_data.append(f(m_data)) for f in self.wavelet_multilevel_feat]
            ret.append(new_data)


        data["signal"] = ret
        return data

transform_wavelet = Sequential([
    SplitSampleRate(),
    WaveletCoeffsFeatures(features = features_funcs_wavelet)
])

import numpy as np
from scipy.fft import rfft, rfftfreq
from vibdata.deep.signal.transforms import Transform  # Assuming same base

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

            # Compute FFT
            fft_vals = rfft(sig, norm="forward")

            # Low-pass filter (like your FFT class)
            if "original_sample_rate" in entry:
                freqs = rfftfreq(len(sig), d=1 / sig_sample_rate)
                bandwidth = entry["original_sample_rate"] / 2
                mask = freqs >= bandwidth
                fft_vals[mask] = 0.0

            # Compute Power Spectral Density
            # PSD = (|FFT|^2) / N, normalized by sampling frequency
            psd = (np.abs(fft_vals) ** 2) / len(sig)

            psd_list.append(psd)

        data["signal"] = psd_list
        return data


features_funcs_psd = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(),
    CrestFactor(), Skewness(), Variance(), PeakValue(), freq_features.DominantFrequency(),
    freq_features.SpectralCentroid(), freq_features.SpectralBandwidth()
]

transforms_psd = Sequential([
    SplitSampleRate(),
    PowerSpectralDensity(),
    FeatureExtractor(features=features_funcs_psd),
])


from PyEMD import EMD

class EMDCoeffsFeatures(Transform):
    """
    Decomposes a signal using Empirical Mode Decomposition (EMD)
    and extracts features from each Intrinsic Mode Function (IMF),
    as well as global multi-level EMD-based features.
    """

    def __init__(self, max_imf=None, features=[Mean()]) -> None:
        """
        Args:
            max_imf (int, optional): Maximum number of IMFs to extract. Default = None (auto).
            features (list): List of feature extractor instances (e.g., [Mean(), RMS(), Kurtosis()]).
        """
        super().__init__()
        self.max_imf = max_imf
        self.features = features

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            # --- EMD decomposition ---
            emd = EMD()
            imfs = emd(sig, max_imf=self.max_imf)
            
            new_data = []
            # --- Extract per-IMF features ---
            for imf in imfs:
                m_data = {'signal': imf}
                [new_data.append(f(m_data)) for f in self.features]

            # --- Extract multi-level/global EMD features ---
            ret.append(new_data)

        data["signal"] = ret
        return data

features_funcs_emd = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(),
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), Variance(), PeakToPeak(), wavelet_features.Energy(),
    wavelet_features.Entropy()
]

transform_emd = Sequential([
    SplitSampleRate(),
    EMDCoeffsFeatures(features=features_funcs_emd, max_imf=5)
])

class SpectralEnvelope(Transform):
    def __init__(self, n_lpc=16) -> None:
        """
        Args:
            n_lpc (int): Number of LPC coefficients for envelope estimation.
        """
        super().__init__()
        self.n_lpc = n_lpc

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            sr = entry["sample_rate"] if "sample_rate" in entry else data["metainfo"]["sample_rate"]

            # --- Compute magnitude spectrum ---
            spectrum = np.abs(np.fft.rfft(sig))

            # --- Estimate LPC coefficients and envelope ---
            lpc = librosa.lpc(sig, order=self.n_lpc)
            w, h = freqz(1, a=lpc, worN=len(spectrum), fs=sr)
            envelope = np.abs(h)

            ret.append(envelope)

        data["signal"] = ret
        return data


features_funcs_spectral = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(),
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), Variance(), PeakToPeak(), freq_features.SpectralCentroid(),
    freq_features.SpectralBandwidth()
]


transform_spectral_envelope = Sequential([
    SplitSampleRate(),
    SpectralEnvelope(n_lpc=16),
    FeatureExtractor(features=features_funcs_spectral)
])


class WignerVilleFeatures(Transform):
    """
    Extracts statistical and spectral features from the Wigner-Ville distribution (WVD) of the signal.
    """

    def __init__(self, features=[Mean()], time_avg=True) -> None:
        """
        Args:
            features (list): List of feature extractor instances applied to the WVD energy distribution.
            time_avg (bool): If True, averages WVD over time to get frequency-energy distribution.
        """
        super().__init__()
        self.features = features
        self.time_avg = time_avg

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            sr = entry["sample_rate"] if "sample_rate" in entry else data["metainfo"]["sample_rate"]

            # --- Compute Wigner–Ville distribution ---
            wvd = smoothed_pseudo_wigner_ville(sig, freq_bins=256) 
  
            # --- Average over time or flatten ---
            if self.time_avg:
                energy = np.mean(np.abs(wvd), axis=1)  # Frequency-energy profile
            else:
                energy = wvd.flatten()  # Full joint distribution flattened

            new_data = []
            # --- Statistical features on WVD energy distribution ---
            m_data = {'signal': energy}
            [new_data.append(f(m_data)) for f in self.features]


            ret.append(new_data)

        data["signal"] = ret
        return data

features_funcs_wigner = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(),
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), Variance(), PeakToPeak(), wavelet_features.Energy(),
    wavelet_features.Entropy()
]
 
transform_wigner_ville = Sequential([
    SplitSampleRate(),
    WignerVilleFeatures(features=features_funcs_wigner, time_avg=False)
])


transform_all_features = Sequential([
    SplitSampleRate(),
    Aggregator([
        FeatureExtractor(features=features_funcs_time),  # Time domain features
        Sequential([FFT(), FeatureExtractor(features=features_funcs_freq)]),  # Frequency domain features
        WaveletCoeffsFeatures(features = features_funcs_wavelet),  # Wavelet features
        EMDCoeffsFeatures(features=features_funcs_emd, max_imf=5),  # EMD features
        Sequential([PowerSpectralDensity(), FeatureExtractor(features=features_funcs_psd)]),  # PSD features
        Sequential([SpectralEnvelope(n_lpc=16), FeatureExtractor(features=features_funcs_spectral)]),  # Spectral Envelope features
        WignerVilleFeatures(features=features_funcs_wigner, time_avg=False)  # Wigner-Ville features
    ])
])

# ======================================
# Dataset grouping strategies
# ======================================
class GroupMultiRoundMFPT(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        return sample_metainfo["label"].astype(str) + " " + sample_metainfo["load"].astype(int).astype(str)


class GroupMultiRoundCWRULoad(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        return sample_metainfo["label"].astype(str) + " " + sample_metainfo["load"].astype(int).astype(str)

class GroupMultiRoundPU(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        condition_fields = ["radial_force_n", "rotation_hz", "load_nm"]
        condition_str = "_".join([sample_metainfo[field].astype(str) for field in condition_fields])
        return sample_metainfo["label"].astype(str) + " " + condition_str

class GroupIMS(GroupDataset):
    NUM_FOLDS = 3

    def __init__(self, dataset, custom_name: str = None) -> None:
        super().__init__(dataset, custom_name, shuffle=True)

        keys = dataset.get_labels()
        values = dataset.get_labels_name()

        self.labels_name = dict(zip(keys, values))
        # Compute how much of each class uniformed distributed should be assinged to each fold

        name_to_label = dict(zip(values, keys))

        metainfo = dataset.get_metainfo()
        defects_frequency = metainfo[metainfo.label != name_to_label["Normal"]].label.value_counts()
        # Create a dict with the amount of samples per fold
        self.defects_bins = {
            label: {"samples_per_fold": np.ceil(total / GroupIMS.NUM_FOLDS), "current_amount": 0}
            for label, total in defects_frequency.items()
        }

    def _get_group_divided(self, label: int):
        current_amount = self.defects_bins[label]["current_amount"]
        samples_per_fold = self.defects_bins[label]["samples_per_fold"]

        group = (current_amount // samples_per_fold) + 1
        self.defects_bins[label]["current_amount"] += 1
        return group-1

    def _assigne_group(self, sample: SignalSample) -> int:
        bearing = sample["metainfo"]["bearing"]
        test = sample["metainfo"]["test"]
        label = sample["metainfo"]["label"]
        label_str = self.labels_name[label]

        if test == 1 and bearing == 3:
            return 0 if label_str == "Normal" else self._get_group_divided(label)
        elif test == 1 and bearing == 4:
            return 1 if label_str == "Normal" else self._get_group_divided(label)
        elif test == 2 and bearing == 1:
            return 2 if label_str == "Normal" else self._get_group_divided(label)
        else:
            raise Exception(
                "Unexpected sample. The sample received is one of the conditions left out.\n"
                "The sample is of test: " + str(test) + " and bearing: " + str(bearing)
            )

class GroupUOC(GroupDataset):
    NUM_FOLDS = 5

    def __init__(self, dataset, custom_name: str = None) -> None:
        super().__init__(dataset, custom_name, shuffle=True)

        keys = dataset.get_labels()
        values = dataset.get_labels_name()

        self.labels_name = dict(zip(keys, values))

        self.labels_bins = {label: {fold: 0 for fold in range(1, GroupUOC.NUM_FOLDS + 1)} for label in keys}

    def _assigne_group(self, sample: SignalSample) -> int:
        severity = sample["metainfo"]["severity"]
        if severity != "-":
            return int(severity)-1
        else:
            label = sample["metainfo"]["label"]
            group = min(self.labels_bins[label], key=self.labels_bins[label].get)
            self.labels_bins[label][group] += 1
            return group-1
# ======================================
# Main experiment runner
# ======================================
def main(classifier_name, dataset_name, transform_name, transforms):
    print(f"\n=== Running experiment ===")
    print(f"Dataset: {dataset_name}")
    print(f"Classifier: {classifier_name}")
    print(f"Transform: {transform_name}")
    print("==========================")

    # ---- Dataset setup ----
    dataset_key = dataset_name.split("_")[0]
    raw_root_dir = f"../data/raw_data/{dataset_key}"
    deep_root_dir = f"../data/deep_data/{dataset_name}_{transform_name}"

    raw_dataset_fn = getattr(raw_datasets, dataset_key + "_raw")
    raw_dataset = raw_dataset_fn(raw_root_dir, download=True)
    print("Raw dataset loaded with length:", len(raw_dataset))

    # ---- Filtering ----
    if "CWRU" in dataset_name:
        if "48K" in dataset_name:
            filter = FilterByValue(on_field="sample_rate", values=48000)
        elif "12K" in dataset_name:
            filter = FilterByValue(on_field="sample_rate", values=12000)
        else:
            filter = None
    elif "MFPT" in dataset_name:
        filter = FilterByValue(on_field="sample_rate", values=48828)
    else:
        filter = None

    # ---- Convert dataset ----
    print("Converting dataset...")
    deep_dataset = convertDataset(raw_dataset, filter=filter, transforms=transforms,
                                  dir_path=deep_root_dir, batch_size=16)
    print("Dataset converted and has length:", len(deep_dataset))

    # ---- Fold generation ----
    print("Generating folds...")
    CLASS_DEF = None
    CONDITION_DEF = None
    if "MFPT" in dataset_name:
        CLASS_DEF = {23: "N", 25: "O", 24: "I"}
        CONDITION_DEF = {"0":"C1","25":"C2","50":"C3","100":"C4","150":"C5","200":"C6","250":"C7","300":"C8"}
        
        GroupClass = GroupMultiRoundMFPT
    elif "CWRU" in dataset_name:
        CLASS_DEF = {0: "N", 1: "O", 2: "I", 3: "R"}
        CONDITION_DEF = {"0": "0", "1": "1", "2": "2", "3": "3"}
        GroupClass = GroupMultiRoundCWRULoad
    elif "IMS" in dataset:
        #resample
        deep_dataset = ResamplerIMS().resample(deep_dataset)
        print("Dataset resampled and has length:", len(deep_dataset))
        GroupClass = GroupIMS
    elif "UOC" in dataset:
        GroupClass = GroupUOC
    else:
        CLASS_DEF = {26: "N", 27: "O", 28: "I", 29: "R"}
        CONDITION_DEF = {"1000_15.0_0.7": "0", "1000_25.0_0.1": "1", "1000_25.0_0.7": "2", "400_25.0_0.7": "3"}
        GroupClass = GroupMultiRoundPU

    if CLASS_DEF and CONDITION_DEF:
        folds = FoldIdxGeneratorUnbiased(
            deep_dataset,
            GroupClass,
            dataset_name=dataset_name + "_" + transform_name,
            multiround=True,
            class_def=CLASS_DEF,
            condition_def=CONDITION_DEF
        ).generate_folds()
    else:
        folds = FoldIdxGeneratorUnbiased(
            deep_dataset,
            GroupClass,
            dataset_name=dataset_name + "_" + transform_name,
        ).generate_folds()
    print("Folds generated.")

    # ---- Classifier setup ----
    if classifier_name == "svm":
        from sklearn.svm import SVC
        model = SVC(random_state=42)
        model_parameters_search_space = {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf", "poly"],
            "model__gamma": ["scale", "auto"]
        }
    elif classifier_name == "rf":
        model = RandomForestClassifier(random_state=42)
        model_parameters_search_space = {
            "model__n_estimators": [50, 100, 200],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [10, 25, 50],
            "model__min_samples_split": [2, 5, 10]
        }
    
    # ---- Experiment ----
    experiment = Features1DExperiment(
        name=f"Vibration_Analysis_{classifier_name.upper()}_{dataset_name}_{transform_name}",
        description="Feature extraction and classification on vibration datasets",
        feature_names=features_name,
        dataset=deep_dataset,
        data_fold_idxs=folds,
        n_inner_folds=4,
        model=model,
        model_parameters_search_space=model_parameters_search_space
    )

    experiment.run()


# ======================================
# Script entrypoint
# ======================================
if __name__ == "__main__":
    valid_classifiers = ["svm", "rf"]
    valid_datasets = ["MFPT", "CWRU_12K", "CWRU_48K","PU","IMS","UOC"]
    valid_transforms = ["time", "frequency", "time_and_frequency","wavelet","psd","emd","spectral_envelope","wigner_ville","all"]

    if len(sys.argv) < 4:
        raise ValueError("Usage: python script.py <classifier> <dataset> <transform>")

    classifier = sys.argv[1]
    dataset = sys.argv[2].upper()
    transform_name = sys.argv[3]

    if classifier not in valid_classifiers:
        raise ValueError(f"Classifier {classifier} not recognized. Valid options: {valid_classifiers}")

    if dataset not in valid_datasets:
        raise ValueError(f"Dataset {dataset} not recognized. Valid options: {valid_datasets}")

    if transform_name not in valid_transforms:
        raise ValueError(f"Transform {transform_name} not recognized. Valid options: {valid_transforms}")

    # ---- Select transform ----
    if transform_name == "time":
        transforms = transforms_time
        features_name = features_funcs_time
    elif transform_name == "frequency":
        transforms = transforms_frequency
        features_name = features_funcs_freq
    elif transform_name == "wavelet":
        transforms = transform_wavelet
        features_name = features_funcs_wavelet +  [wavelet_features.LevelCorrelationCoefficients(),wavelet_features.RelativeEnergyRatio()]
    elif transform_name == "psd":
        transforms = transforms_psd
        features_name = features_funcs_psd
    elif transform_name == "emd":
        transforms = transform_emd
        features_name = features_funcs_emd
    elif transform_name == "spectral_envelope":
        transforms = transform_spectral_envelope
        features_name = features_funcs_spectral
    elif transform_name == "wigner_ville":
        transforms = transform_wigner_ville
        features_name = features_funcs_wigner
    elif transform_name == "all":
        transforms = transform_all_features
        features_name = (
            features_funcs_time +
            features_funcs_freq +
            features_funcs_wavelet + [wavelet_features.LevelCorrelationCoefficients(),wavelet_features.RelativeEnergyRatio()] +
            features_funcs_emd +
            features_funcs_psd +
            features_funcs_spectral +
            features_funcs_wigner
        )
    else:
        transforms = transforms_time_frequency

    main(classifier, dataset, transform_name, transforms)
