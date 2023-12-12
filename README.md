# EnTdecker_public
This repository contains the code for the prediction of triplet energies and spin populations of organic molecules as described in this paper: TBA

The models can also be used in a web application: entdecker.uni-muenster.de

#Installation
For installation run
'''
git clone --recurse-submodules -j8 git://github.com/le-schlo/entdecker_public.git
# option --recurse-submodules is required in order to clone the respective submodules
cd entdecker_public
pip install -r requirements.txt
cd EasyChemML
pip install ./
'''
