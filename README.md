# Imputation-of-Missing-Values

### Abstract

Missing values in datasets should be extracted from the datasets or should be estimated before they are used for classification, association rules or clustering in the preprocessing stage of data mining. In [this paper](https://www.sciencedirect.com/science/article/pii/S0020025513000789), authors utilize a fuzzy c-means clustering hybrid approach that combines support vector regression and a genetic algorithm. In this method, the fuzzy clustering parameters, cluster size and weighting factor are optimized and missing values are estimated. The proposed novel hybrid method yields sufficient and sensible imputation performance results. The results are compared with those of fuzzy c-means genetic algorithm imputation, support vector regression genetic algorithm imputation and zero imputation. This project is an implementation of this method.

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
