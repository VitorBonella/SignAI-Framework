#!/bin/bash
# Run all combinations of classifier, dataset, and transform
# and save each output into a txt file.

# Define lists
classifiers=("svm" "rf")
datasets=("MFPT" "CWRU_12K" "CWRU_48K" "PU" "IMS" "UOC")
transforms=("time" "frequency" "time_and_frequency" "wavelet" "psd" "emd" "spectral_envelope" "wigner_ville" "all")

# Create output folder
mkdir -p results

# Loop through all combinations
for clf in "${classifiers[@]}"; do
  for ds in "${datasets[@]}"; do
    for tf in "${transforms[@]}"; do
      echo "Running classifier=$clf, dataset=$ds, transform=$tf ..."
      output_file="results/${clf}_${ds}_${tf}.txt"

      # Run your Python script and save output
      python 1d_features_experiment.py "$clf" "$ds" "$tf" > "$output_file" 2>&1
    done
  done
done

echo "✅ All runs completed. Check the 'results/' directory for output files."