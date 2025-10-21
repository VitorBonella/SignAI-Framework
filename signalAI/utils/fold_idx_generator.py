from itertools import combinations, product
from time import time
from tqdm import tqdm
from vibdata.deep.DeepDataset import DeepDataset
from vibdata.deep.signal.core import SignalSample

from signalAI.utils.group_dataset import GroupDataset
import numpy as np

class FoldIdxGeneratorUnbiased:
    def __init__(self, dataset: DeepDataset,
                custom_group_dataset: GroupDataset,
                dataset_name: str = None,
                multiround: bool = False,
                class_def: dict = None,
                condition_def: dict = None,
                random_state: int = 42) -> None:
        self.dataset = dataset
        self.custom_group_dataset = custom_group_dataset
        self.random_state = random_state
        self.multiround = multiround
        self.dataset_name = dataset_name if dataset_name else "default_dataset"

        self.class_def = class_def if class_def else {}
        self.condition_def = condition_def if condition_def else {}

    def generate_folds(self):
        if self.multiround:
            return self.generate_folds_unbiased_multiround()
        else:
            return self.generate_folds_unbiased_singleround()

    def generate_folds_unbiased_singleround(self):    
        folds = self.custom_group_dataset(self.dataset, custom_name="CustomGroup"+self.dataset_name).groups()
        return folds
    
    def generate_folds_unbiased_multiround(self):
        self.dataset_name = self.dataset_name + "_multiround"
        labels = []
        for sample in self.dataset:
            labels.append(sample['metainfo']['label'])

        self.y = np.array(labels)

        folds = self.custom_group_dataset(self.dataset, custom_name="CustomGroup"+self.dataset_name).groups()

        MIN_TOTAL_SPLITS = 30
        n_splits = int(np.unique(folds).shape[0] / np.unique(self.y).shape[0])
        n_repeats = np.ceil(MIN_TOTAL_SPLITS / n_splits).astype(int)
        print("Per round splits: ", n_splits)
        print("Number of repeats: ", n_repeats)

        return self.compute_combinations(self.y, folds, n_splits, n_repeats, self.random_state)

    def compute_combinations(self, y, groups, n_splits, n_repeats, seed):
        initial_states = {label: np.unique(groups[y == label]).tolist() for label in np.unique(y)}

        def custom_sort_key(gp):
            condition = gp.split(" ")[-1]
            returned = (
                f"{int(condition):07}"
                if "_" not in condition
                else "_".join([f"{float(c):05.2f}" for c in condition.split("_")])
            )
            return returned

        def englobe_all_data(round_comb):
            matrix = np.array(round_comb)
            n_groups = matrix.shape[0]
            n_labels = matrix.shape[1]
            return all([np.unique(matrix[:, col]).size == n_groups for col in range(n_labels)])

        initial_states = {label: sorted(groups, key=custom_sort_key) for label, groups in initial_states.items()}
        folds = list(product(*initial_states.values()))
        print("Total combinations of folds:", len(folds))

        start = time()
        round_combs_generator = combinations(folds, r=n_splits)
        round_combinations = list(round_combs_generator)
        print("Total combinations between folds", len(round_combinations))
        end = time()
        print("Time to generate combinations: {:.2f} seconds".format(end - start))

        valid_combs = []
        #remove the first combination the unbiased one
        for i, comb in enumerate(round_combinations):
                if englobe_all_data(comb):
                    valid_combs.append(comb)
                    break

        round_combinations = round_combinations[i:]

        #shuffle the combinations
        rng = np.random.default_rng(seed)
        rng.shuffle(round_combinations)

        #select the first valid n_repeats combinations(englobe all data)
        
        def equal_folds(x1, x2):
            sample_x_folds = set(x1)
            sample_y_folds = set(x2)
            return len(sample_x_folds.intersection(sample_y_folds))

        iterator = tqdm(round_combinations)
        for comb in iterator:
            if englobe_all_data(comb):
                if not any([equal_folds(comb, valid) != 0 for valid in valid_combs]):
                    valid_combs.append(comb)
                if len(valid_combs) == n_repeats:
                    iterator.close()
                    break

        if self.class_def and self.condition_def:
            self.print_combinations(valid_combs, categorical_groups=False)
        if self.class_def:
            self.print_combinations(valid_combs, categorical_groups=True)

        fold_map = {}
        folds_multiround = []
        for comb in valid_combs:
            for fold_idx, fold in enumerate(comb):
                for gp in fold:
                    fold_map[gp] = fold_idx
            folds_multiround.append(np.array([fold_map[gp] for gp in groups]))

        return folds_multiround

    def print_combinations(self, combs, categorical_groups=False):
        '''
        if dataset_name == "PU":
            CLASS_DEF = {26: "N", 27: "O", 28: "I", 29: "R"}
            CONDITION_DEF = {"1000_15.0_0.7": "0", "1000_25.0_0.1": "1", "1000_25.0_0.7": "2", "400_25.0_0.7": "3"}
        elif dataset_name == "CWRU":
            CLASS_DEF = {0: "N", 1: "O", 2: "I", 3: "R"}
            CONDITION_DEF = {"0": "0", "1": "1", "2": "2", "3": "3"}
        elif dataset_name == "MFPT":
            CLASS_DEF = {23: "N", 25: "O", 24: "I"}

            class foo(dict):
                def __getitem__(self, key):
                    return str(key)

            CONDITION_DEF = foo()
        '''
        print("Total combs: ", len(combs))
        folds = set()
        for r, c in enumerate(combs):
            print("round: ", r)
            for f, fold_groups in enumerate(c):
                print("fold: ", f, end=" -> ")
                for gp in fold_groups:
                    if not categorical_groups:
                        gp = self.class_def[int(gp.split(" ")[0])] + " " + self.condition_def[gp.split(" ")[1]]
                    print(f"{gp}", end=", ")
                print(" ", end="")
                fold_repr = "-".join(list(map(str, fold_groups)))
                if fold_repr in folds:
                    print("REPEATED")
                else:
                    print(f"=> {len(folds)}")
                folds.add(fold_repr)
            print()

from sklearn.model_selection import StratifiedKFold

class FoldIdxGeneratorBiased:
    def __init__(self, dataset: DeepDataset,
                dataset_name: str = None,
                n_folds: int = 4,
                random_state: int = 42) -> None:
        self.dataset = dataset
        self.random_state = random_state
        self.dataset_name = dataset_name if dataset_name else "default_dataset"
        self.n_folds = n_folds

    def generate_folds(self):
        # Extract labels from the dataset
        labels = [item['metainfo']['label'] for item in self.dataset]
        
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Create array to store fold assignments
        fold_assignments = np.zeros(len(self.dataset), dtype=int)
        
        # Generate fold assignments
        for fold_idx, (_, test_idx) in enumerate(skf.split(range(len(self.dataset)), labels)):
            fold_assignments[test_idx] = fold_idx

        return fold_assignments
