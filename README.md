# Imputation-of-Missing-Values

### Abstract

Real datasets are rarely complete, and missing values have to be removed or estimated during preprocessing before the data can be used for classification, association rules or clustering. Neither of the easy options is satisfying: discarding every affected record throws away information that may be scarce to begin with, while filling the gaps with a constant such as zero distorts the very distribution the later stages depend on. In [this paper](https://www.sciencedirect.com/science/article/pii/S0020025513000789), the authors utilize a fuzzy c-means clustering hybrid approach that combines support vector regression and a genetic algorithm. In this method, the fuzzy clustering parameters, cluster size and weighting factor are optimized and the missing values are estimated. The proposed novel hybrid method yields sufficient and sensible imputation performance results, compared with those of fuzzy c-means genetic algorithm imputation, support vector regression genetic algorithm imputation and zero imputation.

This repository is a Python implementation of that method, together with six datasets to try it on (Glass, Hamberman, Iris, Musk, Wine and Yeast). A chosen dataset is min-max normalized and then damaged on purpose, blanking one attribute in a configurable fraction of its rows, so the original values stay available to score the estimates against. Two estimators are run over the incomplete data: a support vector regression model, trained on the complete rows to predict an affected attribute from the remaining ones, and a fuzzy c-means estimator, which clusters the complete rows and rebuilds a missing entry as the membership-weighted combination of the cluster centers, measuring distances over the observed attributes only. Rather than fixing the two fuzzy clustering parameters by hand, a genetic algorithm searches over the cluster size and the weighting factor, scoring each candidate pair by how far the fuzzy c-means estimates it produces diverge from the support vector regression ones, and the search is repeated until the two estimators agree closely enough. The estimated values are then printed next to the true ones, along with the RMSE.

#### To use this work on your researches or projects you need:
* Python 3.7.0
* Python packages:
	* numpy
	* pandas
	* scikit-learn
	* scikit-fuzzy

##

#### To install Python:
_First, check if you already have it installed or not_.
~~~~
python3 --version
~~~~
_If you don't have python 3 in your computer you can use the code below_:
~~~~
sudo apt-get update
sudo apt-get install python3
~~~~
##

#### To install packages via pip install:
~~~~
sudo pip3 install numpy scikit_fuzzy pandas scikit_learn
~~~~
_If you haven't installed pip, you can use the codes below in your terminal_:
~~~~
sudo apt-get update
sudo apt install python3-pip
~~~~
_You should check and update your pip_:
~~~~
pip3 install --upgrade pip
~~~~
##
