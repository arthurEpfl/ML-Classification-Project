# EPFL-ML-Project-1

## Overview

Welcome to the EPFL-ML-Project-1 repository. In this project, a large dataset based on interviews of more than 300 000 patients in the US was studied. The interviews were conducted in order to predict the possibility of heart attacks. The goal of the project was to apply machine learning concepts and algorithms to the dataset in order to be able to predict the risk of an individual to develop a cardiovascular disease in the future. This README provides an overview of the project structure and how to get started.

## Getting Started

To begin working with this project, follow these instructions:

1. Clone this repository to your local machine.

2. Place the dataset files in the following directory:

./data/dataset_to_release/
- x_train.csv
- y_train.csv
- x_test.csv 

## Usage
run.py: This script contains the code used for the best submission to AIcrowd.com. Execute this script to run the main part of your project. The submission file with name best_submission_rlr will be created then and can be submitted to AIcrowd.

config_reg_logistic_regression.yml: This YAML file holds the hyperparameters used for training the logistic regression model. You can customize the hyperparameters here.

decision_tree_run.py and random_forest_run.py with respective config_decision_tree.yml and config_random_forest.yml: These scripts contain codes used for the best submissions obtained using decision tree and random forest models, respectively. Use them to explore alternative models and their performance. 




