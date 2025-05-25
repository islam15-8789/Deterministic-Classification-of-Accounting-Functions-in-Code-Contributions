# 🔍 Classification of DEMPE Functions in Code Contributions

This project provides a **Docker-based Command-Line Interface (CLI)** to classify DEMPE business functions from GitHub commit messages using pretrained machine learning models.

---

## 🎯 Predict DEMPE Classes (Pretrained)

You don't need to install Python or train any models manually — everything runs seamlessly through Docker.

> ✅ Prerequisite: Make sure [Docker](https://docs.docker.com/desktop/) is installed on your system. Please follow the steps accordingy, Let's start:

### 📥 Step 1: Clone the Repository

```bash
git clone https://github.com/islam15-8789/Deterministic-Classification-of-Accounting-Functions-in-Code-Contributions.git
cd Deterministic-Classification-of-Accounting-Functions-in-Code-Contributions
```

---

### 🐳 Step 2: Build the Docker Image
> Note: Building the Docker image may take some time.
On a MacBook Air M2 (8GB RAM), the initial build took approx. 10 minutes (609s).

```bash
docker build -t dempe-classifier . --no-cache
```

---

### 🔍 Step 3: Run the Predictor

```bash
docker run -it --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py dempe predict-dempe"
```

> After running the Docker container, you’ll be prompted to choose a model and enter **commit message** e.g. 'feat: Menubar added', and the model will return the **predicted DEMPE function(s)** based on your input. To try a different model, simply exit and repeat Step 3.


##  (Optional) Run the Complete Data Pipeline

If you'd like to prepare and process your own dataset, run the full pipeline:

### 📝 Step 1: Create a `repos.json` File

Example:

```json
[
  {
    "repo_name": "https://github.com/OWNER/REPO",
    "owner": "OWNER",
    "token": "ghp_yourGitHubTokenHere"
  }
]
```

> 🔐 Create a [GitHub Personal Access Token](https://github.com/settings/tokens) with `repo` or `public_repo` access.

---

###  Step 2: Execute the Pipeline

```bash
# 🔹 Step 1: Fetch Commits
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  -v "$(pwd)/repos.json:/usr/src/app/repos.json" \
  dempe-classifier \
  -c "python main_cli.py data fetch-commits --input-file repos.json --output-folder data/raw_data"

# 🔹 Step 2: Extract Raw Commit Messages
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py data extract-raw-commit-messages --input-folder data/raw_data --output-file data/csv_data/raw_commit_messages.csv"

# 🔹 Step 3: Label Commit Messages
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py data label-commits --input-file data/csv_data/raw_commit_messages.csv --output-file data/csv_data/labeled_commits.csv"

# 🔹 Step 4: Clean Labeled Commit Messages
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py data clean-commits --input-file data/csv_data/labeled_commits.csv --output-file data/csv_data/cleaned_commits.csv --nonconv-output data/csv_data/non_conventional_commits.csv"

# 🔹 Step 5: Visualize Cleaned Data
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py data visualize-cleaned-commits --input-file data/csv_data/cleaned_commits.csv --output-dir data/plots"

# 🔹 Step 6: Apply MLSMOTE
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py data apply-mlsmote --input-file data/csv_data/cleaned_commits.csv --output-file data/csv_data/resampled_mlsmote.csv"

# 🔹 Step 7: Split Train/Test Dataset
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py data split-dataset --input-file data/csv_data/resampled_mlsmote.csv --train-output data/csv_data/train_re_sampled_mlsmote.csv --test-output data/csv_data/test_re_sampled_mlsmote.csv"

# 🔹 Step 8: Visualize Resampled Label Distribution
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py data visualize-mlsmote-distribution --resampled-file data/csv_data/resampled_mlsmote.csv --output-image data/plots/resampled_label_distribution.png"

```

This command performs **all data preparation steps**:

- 📥 **Fetching**: Clone raw commits from GitHub
- 🧠 **Extraction**: Extract commit messages
- 🏷️ **Labeling**: Assign DEMPE class labels using conventional commit prefixes
- 🧹 **Cleaning**: Normalize and filter messages
- 📊 **Visualization**: View class imbalance
- 🔁 **Oversampling**: Apply MLSMOTE to balance minority classes
- 🔡 **Vectorization**: Encode text with Sentence-BERT
- 🖼️ **Post-Oversampling Visualization**
- 🧪 **Splitting**: Train/test split with stratification

---

## 🤖 (Optional) Train Your Own Models

To retrain all models using the processed dataset:

```bash
# ✅ Train Logistic Regression (One-vs-Rest)
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py train train-one-vs-rest-ovr"

# ✅ Train Random Forest (One-vs-Rest)
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py train train-random-forest-ovr"

# ✅ Train Gradient Boosting Model (XGBoost / LightGBM)
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py train train-gbm-ovr"

# ✅ Train Neural Network (with Keras Tuner)
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py train train-nn"

# ✅ Train Classifier Chain Model (based on Logistic Regression)
docker run --rm \
  -v "$(pwd)/data:/usr/src/app/data" \
  dempe-classifier \
  -c "python main_cli.py train train-classifier-chain"
```

The following models will be trained and saved:

- ✅ Logistic Regression (One-vs-Rest)
- ✅ Random Forest
- ✅ XGBoost / LightGBM
- ✅ Neural Network (with Keras Tuner)
- ✅ Classifier Chain (with Logistic Regression base)

---

## 🧭 Explore CLI Commands

You can inspect all available commands with:

```bash
docker run --rm \
  dempe-classifier \
  -c "python main_cli.py" --help

docker run --rm \
  dempe-classifier \
  -c "python main_cli.py data" --help

docker run --rm \
  dempe-classifier \
  -c "python main_cli.py train" --help

docker run --rm \
  dempe-classifier \
  -c "python main_cli.py dempe" --help
```

---

##  Cleanup

Remove the Docker image when done:

```bash
docker rmi dempe-classifier
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

Any queries, please contact **arni.ai.islam@fau.de**
