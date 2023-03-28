# Data selection for contrastive learning-based pretraining for cardiac arrhythmia detection


## DATASET

To run this code you need to download the dataset in : [Download dataset zip](https://drive.google.com/drive/folders/17OCTOtXYm5mW1qreXO9wWy_sx4AZ8khQ?usp=sharing)

Please make sure to modify the path in `import_data.py` to your respective local respository. 

## DATA SELECTION:

1. To run the CX-DaGAN-based signal selection on dataset `datasetd_05_2`

`python cxdagan_selection.py`

2. To generate the dataset file

`python generate_set_data_images.py`


## PRETRAINING, TRAINING AND MODEL EVALUATION:

To run contrastive learning-based pretraining, training and model evaluation on the model.

`python contrastive.py`

## TRAINING AND MODEL EVALUATION:

To run only training and model evaluation on the model.

run `python baseline.py`

