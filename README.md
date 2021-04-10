# CZ4034 - Aspect Classification model (transfer - learning from the Bert-cased-model)

# Dependencies
```
transformers
sentencepiece
tqdm
loguru
pyYAML
pandas
torch
gdown
```
In order to install those required dependencies. Make sure that your local machine has installed latest Python and pip version. Then execute

```
pip install -r requirements.txt
```

# Step-by-step to train, evaluate and test the model

Note: all paths (either in the config file, python file or as argument when executing commands) should be modified accordingly.

## 1. Preprocess train and test data

1. First download the datasets and place it in `data/` directory.

Splitting the dataset into train, test and evaluation set with 60% dataset responsible for training, 20% responsible for test and 20% for evaluation.

## 2. Set up config.yaml

The ```config.yaml``` file contain required settings to train the model. 
- Most importantly, if training in your local machine, make sure to set the device to be ```cpu```. For this repo, our team train the model in Google Colab, hence, we set the device to be ```cuda``` 
- It is also possible to update other settings like epochs,  model_name_or_path (base-lined model used for training). 

## 2. Train and evaluate
Execute
```
python train.py -c work_dirs/config.yaml
```
to train using baseline config. Training model will then be saved under `work_dirs/yyyymmdd_hhmmss/`.

Notes: 

In order to download our pretrained model used in our search engine application, execute

```
python download_model.py
```

The checkpoint_best.pth will be stored inside the ```/work_dirs``` folder.

## 3. Evaluate

To evaluate model performance(i.e., with groundtruth labels), execute
```
python evaluate.py \
  -c work_dirs/yyyymmdd_hhmmss/config.yaml \
  -m work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth \
  -s work_dirs/yyyymmdd_hhmmss/test_results.csv
  --post-processing
```
The command above load config from `work_dirs/yyyymmdd_hhmmss/config.yaml`, load model from `work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth` (best model), load data from `data/` and save prediction results to `work_dirs/yyyymmdd_hhmmss/test_results.csv`. Accuracy will be printed out as well.

To generate prediction results for test set (i.e., without groundtruth labels), execute
```
python evaluate.py \
  -c work_dirs/yyyymmdd_hhmmss/config.yaml \
  -m work_dirs/yyyymmdd_hhmmss/checkpoint_best.pth \
  -s work_dirs/yyyymmdd_hhmmss/test_results.csv
  --post-processing
  --confidence-threshold 0.51
```
Notes:
- `--post-processing`: If set to True, remove predictions where one span contains the other (keep the one with higher score).
- `--confidence-threshold`: This is set to 0.51 to account for model behavior: many predictions with exactly 0.5 confidence score are incorrect.

## 4. Review aspect prediction
To predict an text with the model, call the ```predict_aspects()``` function from predict.py with passing the review text as a parameter. 
```config_path``` and ```load_from``` in ```predict.py``` will need to make a change accordingly w.r.t to your training model file and config file location.

