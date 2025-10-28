import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from vibdata.deep.DeepDataset import DeepDataset, resample_dataset

class ResamplerIMS():
    def resample(self, dataset: DeepDataset) -> DeepDataset:
        # Get metainfo from complete dataset
        metainfo = dataset.get_metainfo()

        # All the defects signals are kept
        defect_mask = metainfo.label != 6
        new_defects = metainfo[defect_mask]

        # Number of samples to be resampling of the class `Normal` (label = 6)
        # Is the same number of samples of the defects classes
        normals_number = new_defects.shape[0]

        # Resample the normals labels
        normals_mask = [
            (metainfo.test == 1) & (metainfo.bearing == 4) & (metainfo.label == 6),
            (metainfo.test == 1) & (metainfo.bearing == 3) & (metainfo.label == 6),
            (metainfo.test == 2) & (metainfo.bearing == 1) & (metainfo.label == 6),
        ]
        all_normals = pd.concat([metainfo[mask] for mask in normals_mask]).sort_index()
        new_normals = self._resample_normals(all_normals, normals_number)

        # Concat the results in order to compute the new results
        resampled_metainfo = pd.concat([new_normals, new_defects])
        # Get the indexes of the samples that will be kept and ensure that they are sorted
        new_indexes = resampled_metainfo.index.values
        new_indexes.sort()
        resampled_dataset = resample_dataset(dataset, new_indexes)
        return resampled_dataset

    @staticmethod
    def _resample_normals(normals: pd.DataFrame, normals_number: int) -> pd.DataFrame:
        # Compute column to indetify each condition
        normals["set"] = normals["test"] + normals["bearing"]

        def compute_new_frequency(set: int):
            old_freq = sum(normals["set"] == set) / normals.shape[0]
            return np.ceil(normals_number * old_freq).astype("int32")

        # Compute the new frequency for each condition
        sampling_strategy = {set: compute_new_frequency(set) for set in normals.set.unique()}

        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        y = normals.set.values.astype("int32")
        new_normals, y_resampled = rus.fit_resample(normals, y)
        new_normals.drop(columns=["set"], inplace=True)

        return new_normals