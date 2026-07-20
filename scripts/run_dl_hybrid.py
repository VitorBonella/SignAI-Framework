import argparse
import os

import torch
from sklearn.preprocessing import LabelEncoder

import vibdata.raw as raw_datasets
from vibdata.deep.DeepDataset import convertDataset
from vibdata.deep.signal.transforms import FilterByValue

from signalai.data.grouping import get_dataset_grouping
from signalai.experiments.deep_learning import DeepLearningExperiment
from signalai.experiments.hybrid import run_hybrid_classification
from signalai.features.pipelines import get_pipeline
from signalai.models.autoencoder import DCAE1D
from signalai.sampling.generators import FoldIdxGeneratorUnbiased


def main():
    parser = argparse.ArgumentParser(
        description="Train DCAE then classify with extracted features (RF + SVM)."
    )
    parser.add_argument("dataset", help="Dataset name (e.g., MFPT, CWRU_12K)")
    parser.add_argument("transform", choices=["raw_time", "raw_freq"], help="Raw transform")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="results", help="Base output directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Dataset loading ---
    dataset_key = args.dataset.split("_")[0]
    raw_root_dir = f"./data/raw_data/{dataset_key}"
    deep_root_dir = f"./data/deep_data/{args.dataset}_{args.transform}"

    raw_dataset_fn = getattr(raw_datasets, f"{dataset_key.upper()}_raw")
    raw_dataset = raw_dataset_fn(raw_root_dir, download=True)

    filter_obj = None
    if "CWRU" in args.dataset:
        sr = 48000 if "48K" in args.dataset else 12000
        filter_obj = FilterByValue(on_field="sample_rate", values=sr)
    elif "MFPT" in args.dataset:
        filter_obj = FilterByValue(on_field="sample_rate", values=48828)

    pipeline = get_pipeline(args.transform)
    deep_dataset = convertDataset(
        raw_dataset, filter=filter_obj, transforms=pipeline,
        dir_path=deep_root_dir, batch_size=args.batch_size,
    )

    # --- Fold generation ---
    GroupClass, deep_dataset = get_dataset_grouping(args.dataset, deep_dataset)
    generator = FoldIdxGeneratorUnbiased(
        deep_dataset, GroupClass, dataset_name=f"{args.dataset}_{args.transform}"
    )
    folds = generator.generate_folds()

    # --- Train DCAE ---
    dcae_output_dir = os.path.join(args.output, args.dataset, args.transform, ".dcae_tmp")
    dcae_exp_name = f"DL_dcae_{args.dataset}_{args.transform}"
    print(f"\n=== Training DCAE: {dcae_exp_name} ===")

    dcae_experiment = DeepLearningExperiment(
        name=dcae_exp_name,
        description=f"DCAE reconstruction on {args.dataset}",
        dataset=deep_dataset,
        data_fold_idxs=folds,
        model=DCAE1D(),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=dcae_output_dir,
        is_reconstruction=True,
        lr=0.0001,
        start_time="",
    )
    dcae_experiment.run()

    X = dcae_experiment.X
    raw_labels = [sample["metainfo"]["label"] for sample in deep_dataset]
    le = LabelEncoder()
    y = le.fit_transform(raw_labels)

    # --- Hybrid classification ---
    run_hybrid_classification(
        X, y, folds, dcae_experiment.run_dir, args.output, args.dataset, args.transform, device
    )


if __name__ == "__main__":
    main()
