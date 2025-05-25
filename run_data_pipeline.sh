#!/bin/bash

echo "🔵 Data Pipeline Execution Started"
echo "📁 Working directory: $(pwd)"
echo "📦 Files:"
ls -alh

steps=(
  "mian_cli.py data fetch-commits --input-file repos.json --output-folder data/raw_data"
  "mian_cli.py data extract-raw-commit-messages --input-folder data/raw_data --output-file data/csv_data/raw_commit_messages.csv"
  "mian_cli.py data label-commits --input-file data/csv_data/raw_commit_messages.csv --output-file data/csv_data/labeled_commits.csv"
  "mian_cli.py data clean-commits --input-file data/csv_data/labeled_commits.csv --output-file data/csv_data/cleaned_commits.csv --nonconv-output data/csv_data/non_conventional_commits.csv"
  "mian_cli.py data visualize-cleaned-commits --input-file data/csv_data/cleaned_commits.csv --output-dir data/plots"
  "mian_cli.py data apply-mlsmote --input-file data/csv_data/cleaned_commits.csv --output-file data/csv_data/resampled_mlsmote.csv"
  "mian_cli.py data split-dataset --input-file data/csv_data/resampled_mlsmote.csv --train-output data/csv_data/train_re_sampled_mlsmote.csv --test-output data/csv_data/test_re_sampled_mlsmote.csv"
  "mian_cli.py data visualize-mlsmote-distribution --resampled-file data/csv_data/resampled_mlsmote.csv --output-image data/plots/resampled_label_distribution.png"
)

for step in "${steps[@]}"; do
  echo -e "\n🔸 Running: $step"
  python $step
done

echo -e "\n✅ All steps completed!"
