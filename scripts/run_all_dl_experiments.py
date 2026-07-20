import subprocess
import os

datasets = [ "CWRU_12K", "CWRU_48K", "UOC", "PU", "IMS","MFPT"]
models = ["wdcnn", "ticnn"]
#models = ["dcae"]
transforms = ["raw_time", "raw_freq"]

output_root = "results"

for ds in datasets:
    for model in models:
        for trans in transforms:
            print(f"\n>>> Running experiment: DS={ds}, Model={model}, Transform={trans}")

            if model == "dcae":
                # run_dl_hybrid trains DCAE and immediately classifies with RF + SVM
                cmd = [
                    "python", "scripts/run_dl_hybrid.py",
                    ds, trans,
                    "--epochs", "3",
                    "--output", output_root,
                ]
            else:
                cmd = [
                    "python", "scripts/run_deep_learning.py",
                    model, ds, trans,
                    "--epochs", "100",
                    "--output", output_root,
                ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running experiment for {ds} {model} {trans}: {e}")

print("\nDone with all DL experiments.")
