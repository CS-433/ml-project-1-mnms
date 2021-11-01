# Project 1 -  The Higgs Boson

Names : Berezantev Mihaela, Sinsoillier Mike Junior, Du Couédic De Kergoualer Sophie Zhuo Ran

## Background
This repository contains the code to produce a model dedicated to classify the datapoints from the CERN particule accelerator data, in order to identify the Higgs Boson.

This repository contains:
* [data](data/) : contains the dataset for both the training and the testing.
* [report.pdf](report.pdf) : the report explaining the complete process of this project
* [scripts](scripts/): all the executable code, in particular
  * [project1.ipynb](scripts/project1.ipynb): a notebook with various procedures to generate predictive models
  * [data_exploration.ipynb](scripts/data_exploration.ipynb): a notebook illustrating our procedure to preprocess the data
  * [preprocessing.py](scripts/preprocessing.py): the implementation of the functions actually used to preprocess the data
  * [implementations.py](scripts/implementations.py): implementation of the different machine learning methods for the prediction
  * [cross_validation.py](scripts/cross_validation.py): contains several functions to help tunning the hyper-parameters, as well as validating the error and the accuracy
  * [proj1_helpers.py](scripts/proj1_helpers.py): some helper functions to load and store the data

## Prequisites
Make sure you have the following installed on your developement machine:
* python3 - [Download & Install python](https://www.python.org/downloads/). All the implementation are coded in python3.
* [numpy](https://numpy.org/) : for an easy manipulations of the data arrays. You can install it via pip.
```bash
pip install numpy
```

## Usage
To run and produce the predictions on the test data, get into the [scripts/](scripts/) folder and run `run.py`
```bash
cd scripts/
python3 run.py
```
This will produce a file `submission.csv` in the [data](data/) folder. The mean and standard deviation of accuracy using 4-fold is computed as well. The produced file can be submitted on [aircrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) for an accuracy score of 0.819 and F1 score of 0.721.
