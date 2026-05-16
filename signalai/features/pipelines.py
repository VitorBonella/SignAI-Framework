from vibdata.deep.signal.transforms import (
    Kurtosis, RootMeanSquare, StandardDeviation, Mean, LogAttackTime,
    TemporalDecrease, TemporalCentroid, EffectiveDuration, ZeroCrossingRate,
    PeakValue, CrestFactor, Skewness, ClearanceFactor, ImpulseFactor,
    ShapeFactor, UpperBoundValueHistogram, LowerBoundValueHistogram,
    Variance, PeakToPeak, Sequential, SplitSampleRate,
    FeatureExtractor, FFT, Aggregator
)

from .freq import (
    SpectralRolloff, SpectralFlatness, SpectralEntropy, 
    SpectralCentroid, SpectralBandwidth, DominantFrequency
)
from .wavelet import (
    Energy, Entropy, LevelCorrelationCoefficients, RelativeEnergyRatio
)
from .custom import (
    WaveletCoeffsFeatures, PowerSpectralDensity, 
    EMDCoeffsFeatures, SpectralEnvelope
)

# --- Feature lists ---
FEATURES_TIME = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(), LogAttackTime(),
    TemporalDecrease(), TemporalCentroid(), EffectiveDuration(), ZeroCrossingRate(),
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), UpperBoundValueHistogram(), LowerBoundValueHistogram(), Variance(), PeakToPeak()
]

FEATURES_FREQ = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(),
    CrestFactor(), Skewness(), Variance(), PeakValue(),
    SpectralRolloff(), SpectralFlatness(), SpectralEntropy(), 
    SpectralCentroid(), SpectralBandwidth(), DominantFrequency()
]

FEATURES_WAVELET = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(), 
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), Variance(), Energy(), Entropy(),
]

FEATURES_PSD = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(),
    CrestFactor(), Skewness(), Variance(), PeakValue(), 
    DominantFrequency(), SpectralCentroid(), SpectralBandwidth()
]

FEATURES_SPECTRAL = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(),
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), Variance(), PeakToPeak(), SpectralCentroid(), SpectralBandwidth()
]

# --- Pipelines ---
PIPELINE_TIME = Sequential([
    SplitSampleRate(),
    FeatureExtractor(features=FEATURES_TIME),
])

PIPELINE_FREQUENCY = Sequential([
    SplitSampleRate(),
    FFT(),
    FeatureExtractor(features=FEATURES_FREQ),
])

PIPELINE_TIME_FREQUENCY = Sequential([
    SplitSampleRate(),
    Aggregator([
        FeatureExtractor(features=FEATURES_TIME),
        Sequential([FFT(), FeatureExtractor(features=FEATURES_FREQ)])
    ])
])

PIPELINE_WAVELET = Sequential([
    SplitSampleRate(),
    WaveletCoeffsFeatures(
        features=FEATURES_WAVELET,
        # Manually adding multilevel features as they require specific handling
    )
])

# For WaveletCoeffsFeatures, we need to inject the multilevel features
PIPELINE_WAVELET.transforms[1].wavelet_multilevel_feat = [
    LevelCorrelationCoefficients(), RelativeEnergyRatio()
]

PIPELINE_PSD = Sequential([
    SplitSampleRate(),
    PowerSpectralDensity(),
    FeatureExtractor(features=FEATURES_PSD),
])

PIPELINE_SPECTRAL_ENVELOPE = Sequential([
    SplitSampleRate(),
    SpectralEnvelope(n_lpc=16),
    FeatureExtractor(features=FEATURES_SPECTRAL)
])

PIPELINE_ALL = Sequential([
    SplitSampleRate(),
    Aggregator([
        FeatureExtractor(features=FEATURES_TIME),
        Sequential([FFT(), FeatureExtractor(features=FEATURES_FREQ)]),
        PIPELINE_WAVELET.transforms[1], # Reusing Wavelet transform
        Sequential([PowerSpectralDensity(), FeatureExtractor(features=FEATURES_PSD)]),
        Sequential([SpectralEnvelope(n_lpc=16), FeatureExtractor(features=FEATURES_SPECTRAL)]),
    ])
])

def get_pipeline(name: str):
    pipelines = {
        "time": PIPELINE_TIME,
        "frequency": PIPELINE_FREQUENCY,
        "time_and_frequency": PIPELINE_TIME_FREQUENCY,
        "wavelet": PIPELINE_WAVELET,
        "psd": PIPELINE_PSD,
        "spectral_envelope": PIPELINE_SPECTRAL_ENVELOPE,
        "all": PIPELINE_ALL
    }
    if name not in pipelines:
        raise ValueError(f"Unknown pipeline: {name}. Available: {list(pipelines.keys())}")
    return pipelines[name]

def get_feature_names(name: str):
    # This logic replicates the naming convention in 1d_features_experiment.py
    if name == "time":
        return [type(f).__name__ for f in FEATURES_TIME]
    elif name == "frequency":
        return [type(f).__name__ for f in FEATURES_FREQ]
    elif name == "wavelet":
        names = []
        for f in FEATURES_WAVELET:
            for level in range(1, 6): # Assuming level 4 + cA = 5 components
                names.append(f"{type(f).__name__}_L{level}")
        return names + ["LevelCorrelationCoefficients", "RelativeEnergyRatio"]
    elif name == "psd":
        return [type(f).__name__ for f in FEATURES_PSD]
    elif name == "spectral_envelope":
        return [type(f).__name__ for f in FEATURES_SPECTRAL]
    elif name == "all":
        # Simplified 'all' naming
        t_names = [type(f).__name__ + "_time" for f in FEATURES_TIME]
        f_names = [type(f).__name__ + "_freq" for f in FEATURES_FREQ]
        w_names = []
        for f in FEATURES_WAVELET:
            for level in range(1, 6):
                w_names.append(f"{type(f).__name__}_wavelet_L{level}")
        w_names += ["LevelCorrelationCoefficients_wavelet", "RelativeEnergyRatio_wavelet"]
        p_names = [type(f).__name__ + "_psd" for f in FEATURES_PSD]
        s_names = [type(f).__name__ + "_spectral" for f in FEATURES_SPECTRAL]
        return t_names + f_names + w_names + p_names + s_names
    return []
