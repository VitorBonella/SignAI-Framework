import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from signalai.models.cnn import WDCNN, TICNN
from signalai.models.autoencoder import DCAE1D
import vibdata.raw as raw_datasets
from vibdata.deep.DeepDataset import convertDataset
from vibdata.deep.signal.transforms import FilterByValue

from signalai.data.grouping import get_dataset_grouping
from signalai.sampling.generators import FoldIdxGeneratorUnbiased
from signalai.features.pipelines import get_pipeline
from signalai.experiments.deep_learning import DeepLearningExperiment
from signalai.experiments.hybrid import run_hybrid_classification

def main():
    parser = argparse.ArgumentParser(description="Run Deep Learning vibration experiment.")
    parser.add_argument("model", choices=["wdcnn", "ticnn", "dcae"], help="Model architecture")
    parser.add_argument("dataset", help="Dataset name (e.g., MFPT, CWRU_12K)")
    parser.add_argument("transform", choices=["raw_time", "raw_freq"], help="Raw transform")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", default="results", help="Base output directory")

    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"DL_{args.model}_{args.dataset}_{args.transform}"
    output_dir = os.path.join(args.output, args.dataset, args.transform, args.model)

    print(f"=== Running DL experiment: {exp_name} ===")

    # --- Dataset Setup ---
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
        dir_path=deep_root_dir, batch_size=args.batch_size
    )
    
    # --- Fold Generation ---
    GroupClass, deep_dataset = get_dataset_grouping(args.dataset, deep_dataset)
    generator = FoldIdxGeneratorUnbiased(deep_dataset, GroupClass, dataset_name=args.dataset)
    folds = generator.generate_folds()

    # --- Model Selection ---
    n_classes = len(np.unique([s['metainfo']['label'] for s in deep_dataset]))
    
    if args.model == "wdcnn":
        model = WDCNN(out_channels=n_classes)
        is_reconstruction = False
    elif args.model == "ticnn":
        model = TICNN(out_channels=n_classes)
        is_reconstruction = False
    elif args.model == "dcae":
        model = DCAE1D()
        is_reconstruction = True

    # --- Run Experiment ---
    experiment = DeepLearningExperiment(
        name=exp_name,
        description=f"Deep Learning experiment {args.model} on {args.dataset}",
        dataset=deep_dataset,
        data_fold_idxs=folds,
        model=model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=output_dir,
        is_reconstruction=is_reconstruction,
        lr=0.0001,
        start_time=""
    )

    experiment.run()

    # After DCAE reconstruction training, also classify with extracted features
    if args.model == "dcae":
        raw_labels = [sample["metainfo"]["label"] for sample in deep_dataset]
        le = LabelEncoder()
        y = le.fit_transform(raw_labels)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_hybrid_classification(
            experiment.X, y, folds, experiment.run_dir,
            args.output, args.dataset, args.transform, timestamp, device
        )

if __name__ == "__main__":
    main()
