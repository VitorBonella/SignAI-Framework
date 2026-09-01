import subprocess
import time
import os
from pathlib import Path

# --- Configuration ---
DATASETS = [ "CWRU_12K", "CWRU_48K", "UOC", "PU", "IMS","MFPT"]
TRANSFORMS = ["time", "frequency", "time_and_frequency", "wavelet", "psd", "spectral_envelope", "all"]
#CLASSIFIERS = ["svm", "rf", "lgbm"]
CLASSIFIERS = ["lgbm"]
#SELECTORS = ["hybrid", "anova", "sfs", "mrmr"] # Only for the "all" transform
SELECTORS = ["anova", "mrmr"] 

BASE_RESULTS_DIR = "results"
PYTHON_EXECUTABLE = "python" # Or path to your .venv/bin/python
EXPERIMENT_SCRIPT = "scripts/run_classification.py"

def run_command(cmd):
    """Executes a command and returns the return code and duration."""
    # print(f"\n[EXEC] {' '.join(cmd)}")
    start_time = time.time()
    
    # Redirect stdout and stderr to DEVNULL to suppress terminal output
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    return process.returncode, duration

def main():
    total_start_time = time.time()
    
    # Calculate total number of experiments
    # For each dataset:
    #   For each classifier:
    #     For each transform (except 'all'): 1 experiment (no selector)
    #     For transform 'all': 1 experiment (no selector) + 3 experiments (with selectors)
    num_transforms_no_all = len([t for t in TRANSFORMS if t != "all"])
    num_experiments_per_ds_clf = num_transforms_no_all + 1 + len(SELECTORS)
    total_experiments = len(DATASETS) * len(CLASSIFIERS) * num_experiments_per_ds_clf
    
    current_exp_count = 0
    
    print("=" * 80)
    print(f"STARTING ALL EXPERIMENTS ({total_experiments} total)")
    print("=" * 80)

    for dataset in DATASETS:
        for classifier in CLASSIFIERS:
            for transform in TRANSFORMS:
                
                # Determine selectors to run for this transform
                selectors_to_run = [None]
                if transform == "all":
                    selectors_to_run.extend(SELECTORS)
                
                for selector in selectors_to_run:
                    current_exp_count += 1
                    
                    # 1. Determine Output Directory
                    # Pattern: results/dataset/feature_set/classifier
                    # For selection: results/dataset/all/classifier+feature_selector
                    if selector:
                        output_dir = os.path.join(BASE_RESULTS_DIR, dataset, transform, f"{classifier}+{selector}")
                    else:
                        output_dir = os.path.join(BASE_RESULTS_DIR, dataset, transform, classifier)
                    
                    os.makedirs(output_dir, exist_ok=True)
                    
                    print(f"\n[{current_exp_count}/{total_experiments}] "
                          f"DS: {dataset} | CLF: {classifier} | TF: {transform} | SEL: {selector or 'None'}")
                    print(f"Output: {output_dir}")
                    
                    # 2. Build Command
                    cmd = [PYTHON_EXECUTABLE, EXPERIMENT_SCRIPT, classifier, dataset, transform, "--output", output_dir, "--no-timestamp"]
                    if selector:
                        cmd.extend(["--selector", selector])
                    
                    # 3. Run
                    return_code, duration = run_command(cmd)
                    
                    # 4. Track progress
                    if return_code == 0:
                        print(f"✅ Success! Duration: {duration:.2f}s")
                    else:
                        print(f"❌ Failed with return code {return_code}. Duration: {duration:.2f}s")
                    
                    elapsed_time = time.time() - total_start_time
                    avg_time_per_exp = elapsed_time / current_exp_count
                    remaining_exps = total_experiments - current_exp_count
                    estimated_remaining_time = avg_time_per_exp * remaining_exps
                    
                    print(f"Elapsed: {elapsed_time/60:.2f}m | Estimated Remaining: {estimated_remaining_time/60:.2f}m")
                    print("-" * 40)

    total_duration = time.time() - total_start_time
    print("=" * 80)
    print(f"ALL EXPERIMENTS COMPLETED.")
    print(f"Total Duration: {total_duration/3600:.2f} hours")
    print("=" * 80)

if __name__ == "__main__":
    main()
