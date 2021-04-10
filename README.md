# CZ4034 - Aspect Classification model

# Dependencies
```
transformers
sentencepiece
tqdm
loguru
PyYAML
```

# Step-by-step to train the model in the repo

Note: all paths (either in the jupyter notebook file, in the config file or as argument when executing commands) should be modified accordingly.

## 1. Preprocess train and test data

1. First download the datasets and place it in `data/` directory.

Splitting the dataset into train, test and evaluation set with 60% dataset responsible for training, 20% responsible for test and 20% for evaluation.

## 2. Train and evaluate
Execute
```
python train.py -c work_dirs/config.yaml
```
to train using baseline config. Training data will then be saved under `work_dirs/yyyymmdd_hhmmss/`.

## 3. Evaluate

To evaluate model performance on any fold (i.e., with groundtruth labels), execute
```
python evaluate.py \
  -c work_dirs/yyyymmdd_hhmmss/config.yaml \
  -m work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth \
  -d data/processed/splits/fold_6.json \
  -s work_dirs/yyyymmdd_hhmmss/test_results.csv
  --post-processing
```
The command above load config from `work_dirs/yyyymmdd_hhmmss/config.yaml`, load model from `work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth` (best model), load data from `data/` and save prediction results to `work_dirs/yyyymmdd_hhmmss/test_results.csv`. Accuracy will be printed out as well.

To generate prediction results for test set (i.e., without groundtruth labels), execute
```
python evaluate.py \
  -c work_dirs/yyyymmdd_hhmmss/config.yaml \
  -m work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth \
  -d data/processed/test_processed.json \
  -s work_dirs/yyyymmdd_hhmmss/test_results.csv
  --post-processing
  --confidence-threshold 0.51
```
Notes:
- `--post-processing`: If set to True, remove predictions where one span contains the other (keep the one with higher score).
- `--confidence-threshold`: This is set to 0.51 to account for model behavior: many predictions with exactly 0.5 confidence score are incorrect.
