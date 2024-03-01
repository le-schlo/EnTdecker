# EnTdecker
This repository contains the code for the prediction of triplet energies and spin populations of organic molecules as described in this paper: TBA

The models can also be used in a web application: [entdecker.uni-muenster.de](http://entdecker.uni-muenster.de)

<p align="center">
  <img src="images/TOC.png" width="60%" />
</p>

## Installation
For installation run
```
git clone --recurse-submodules https://github.com/le-schlo/EnTdecker.git
# option --recurse-submodules is required in order to clone the respective submodules

cd EnTdecker
pip install -r requirements.txt

cd EasyChemML
pip install ./
```


## Data

The datasets can be found in `Data/EnT_data.csv` and `Data/EnT_SD_data.csv`, for the triplet energy
prediction and the spin population prediction, respectively. The triplet energy values in `Data/EnT_data.csv` are given in kcal/mol. `Data/EnT_SD_data.csv` contains the input and target strings, to which the spin population for each heavy atom is inserted. Those files can be directly used for training the models. The dataset for finetuning the model can be found in `Data/Retrain_Disulfide.csv` for the triplet energy prediction and in `Data/Retrain_SD_Disulfide.csv` for the spin population prediction. `Data/Pretrain.csv` contains the data
for pretraining the sequence-to-sequence model on the canonicalisation task.

## DataProcessing

The folder `DataProcessing` contains helper functions and scripts for obtaining the splits for the triplet
energy prediction (`RandomSplitter.py`, `MurckoSplitter.py`, `SimilaritySplitter.py`, `OutOfSample.py`) as
well as the data processing for the spin population prediction. The file `SpinExtraction.py` contains a
function that returns the input for the sequence-to-sequence model given a smiles and the
corresponding spin population. The file `TargetRearrangement.py` contains a function that returns
rearranged SMILES including the binned spin population.

## Train models
Sample scripts to train the models can be found in the [Models](Models) directory and the respective sub-directories.

## Use pretrained models
Pretrained models can be downloaded from [Zenodo](https://zenodo.org/uploads/10391170).

Sample scripts to use these for obtaining predictions can be found under [Models/triplet_energy/chemprop/eval.py](Models/triplet_energy/chemprop/eval.py) and [Models/spin_population/eval.py](Models/spin_population/eval.py).
