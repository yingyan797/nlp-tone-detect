# Detecting condescending and patronizing language in text

## Introduction
This program aims to provide a solution to the "Don't Patronize Me" contest on natural language classification. Baseline models like simple RoBERTa classifier achieves good performance to some extent, but there exists limitations, especially in the classifier part. This program builds on top of RoBERTa embedding + attention components and applies Graph Convolutional Network (GCN) layers in addition to the linear classifier, in order to better utilize language features and improve F1 score.

## RoBERTa-GCN Algorithm, architecture, and dataset
* Main model components (see ./pipiline.py): RoBERTa Embedding + RoBERTa Self-Attention + 2-layered GCN + Linear classifier [RoBERTa-GCN]
* Training dataset (./dataset/dontpatraonizeme_pcl.tsv): English sentences with an integer "patronizing" score from 0 to 4, and some other information like country code and paragraph ID. Score 0/1 means "No patronizing (negative)", and 2/3/4 means "Patronizing (positive)"
* Test dataset (./dataset/task4_test.tsv): English sentences with no labels. 
* Train-dev split: The training set is splitted to 2 groups according to the spcidically designed indices.

## Program output
The program outputs predicted labels 0 (negative) or 1 (positive) for all sentences on the test dataset and dev dataset. Check ./predictions/dev.txt and ./predictions/test.txt for results.

## Data analytics
Analysis of the training dataset includes text length distribution, country code distribution, label distribution by class, and RoBERTa embedding visualization, etcs. Check ./images/ for plots.

## RoBERTa-GCN training results and baseline 
Hyperparameters, epochs, train/validation loss, accuracy, and F1 score are recorded in ./training_record_best.csv, for the main RoBERTa-GCN model this program implements. Some baseline models are also provided for comparison. See ./baseline/ for information.
