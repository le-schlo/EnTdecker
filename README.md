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
## Train models
Sample scripts to train the models can be found in the [Models](Models) directory and the respective sub-directories.

## Use pretrained models
Pretrained models can be downloaded from [Zenodo](https://zenodo.org/uploads/10391170).

Sample scripts to use these for obtaining predictions can be found under [Models/triplet_energy/chemprop/eval.py](Models/triplet_energy/chemprop/eval.py) and [Models/spin_population/eval.py](Models/spin_population/eval.py).
