import os

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from vibdata.deep.DeepDataset import DeepDataset
from vibdata.deep.signal.core import SignalSample


class GroupDataset:
    def __init__(
        self, dataset: DeepDataset, custom_name: str = None, shuffle: bool = False, groups_dir: str = None
    ) -> None:
        self.dataset = dataset
        #self.config
        self.shuffle_before_iter = shuffle
        if groups_dir:
            self.groups_dir = groups_dir
        else:
            self.groups_dir = "../data/grouping"
        
        file_name = "groups_" + (custom_name if custom_name else "default_name")
        self.groups_file = os.path.join(self.groups_dir, file_name + ".npy")

    def groups(self) -> npt.NDArray[np.int_]:
        """
        Get the groups from all samples of the dataset. It tries to load from memory at `groups_dir` but if it
        doesnt exists it will compute the groups and save it in `groups_file`.

        Returns:
            npt.NDArray[np.int_]: groups of all dataset
        """
        if os.path.exists(self.groups_file):
            print(f"Loading group dataset from: {self.groups_file}")
            return np.load(self.groups_file,allow_pickle=True)
        else:
            groups = self._random_grouping() if self.shuffle_before_iter else self._sequential_grouping()
            os.makedirs(self.groups_dir, exist_ok=True)  # Ensure that the directory exists
            np.save(self.groups_file, groups)
            return groups

    def _sequential_grouping(self) -> npt.NDArray[np.int_]:
        """Generate the groups iterating sequentially over the dataset

        Returns:
            npt.NDArray[np.int_]: groups of each sample in dataset
        """
        mapped_samples = map(
            self._assigne_group,
            tqdm(self.dataset, total=len(self.dataset), unit="sample", desc="Grouping dataset: "),
        )
        groups = np.array(list(mapped_samples))
        return groups

    def _random_grouping(self) -> npt.NDArray[np.int_]:
        """Generate the groups randomly iterating over the dataset, is equivalent to make a shuffle
        in the dataset. Despite the shuffle, the groups are ordered back to the original order.

        This kind of grouping is needed for datasets where grouping are not predefined

        Returns:
            npt.NDArray[np.int_]: groups of each sample in dataset, in the original order
        """
        # Create the indexes shuffled
        rng = np.random.default_rng(self.config["seed"])  # Ensure thats the seed is correct
        indexs_shuffled = np.arange(len(self.dataset))
        rng.shuffle(indexs_shuffled)
        # Map the dataset ramdomly
        mapped_samples = list(
            map(
                lambda i: self._assigne_group(self.dataset[i]),
                tqdm(indexs_shuffled, total=len(self.dataset), unit="sample", desc="Grouping dataset: "),
            )
        )
        # Sort the output back to the dataset original order
        groups = np.array([value for _, value in sorted(zip(indexs_shuffled, mapped_samples))])
        return groups

    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        """
        Get a signal sample and based on the dataset criterion, assigne a group
        to the given sample

        Args:
            sample (SignalSample): sample to be assigned

        Returns:
            int: group id
        """
        pass