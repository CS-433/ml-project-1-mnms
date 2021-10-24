# -*- coding: utf-8 -*-
"""Functions used for the preprocessing of the data."""
import numpy as np


def replace_undefined_values_with_mean(tx):
    """Replace the -999 values of each feature column by the mean of its elements"""
    
    cols = tx.shape[1]
    
    for col in range(cols):
        vect = tx[:, col]
        #delete -999 values before calculating the mean
        vect = np.delete(vect, np.where(vect == -999))
        #replace -999 values by mean
        tx[:,col][tx[:,col] == -999] = np.mean(vect)
    
    return tx


def remove_skewness(tx):
    """Remove skewness by doing a log transform of the skewed features"""
    skewed_features = [0,1,2,3,5,8,9,10,13,16,19,21,23,26,29]
    
    for skewed_feature in skewed_features:
        #apply log(1+x) function
        tx[:, skewed_feature] = np.log(1 + tx[:, skewed_feature])
        
    return tx


def drop_phi_features(tx):
    """Remove features that contain phi"""
    phi_columns = [15, 18, 20, 25, 28]
    cols = np.arange(tx.shape[1])
    not_phi_cols = [col for col in cols if col not in phi_columns]

    return tx[:, not_phi_cols]


def add_bias_term(input_data):
    """Add column of 1s which corresponds to the bias term"""
    
    num_samples = input_data.shape[0]
    tx = np.c_[np.ones(num_samples), input_data]
    
    return tx


def standardize(tx):
    """Standardize the data set."""

    cols = tx.shape[1]
    for col in range(cols):
        mean = np.mean(tx[:,col])
        std = np.std(tx[:,col])
        if std == 0:
            tx[:,col] = (tx[:,col] - mean)
        else:
            tx[:,col] = (tx[:,col] - mean) / std
    
    return tx


#TODO do all your needed preprocessing functions inside this function
def preprocess_data(input_data):
    """Apply all preprocessing functions to the input data """

    preprocessed_data = replace_undefined_values_with_mean(input_data)
    preprocessed_data = remove_skewness(preprocessed_data)
    preprocessed_data = drop_phi_features(preprocessed_data)
    preprocessed_data = add_bias_term(preprocessed_data)
    preprocessed_data = standardize(preprocessed_data)

    return preprocessed_data
