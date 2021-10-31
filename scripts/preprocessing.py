# -*- coding: utf-8 -*-
"""Functions used for the preprocessing of the data."""
import numpy as np


def remove_feature_when_too_much_missing_values(tx):
    """Remove a feature when the fraction of missing values is beyond 0.5"""
    cols = tx.shape[1]
    tot_elems = tx.shape[0]
    
    to_keep = []
    for col in range(cols):
        vect = tx[:, col]
        frac = np.count_nonzero(vect == -999)/tot_elems
        if frac < 0.5:
            to_keep.append(col)
        
    return tx[:, to_keep]


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


def replace_undefined_values_with_median(tx):
    """Replace the -999 values of each feature column by the mean of its elements"""
    
    cols = tx.shape[1]
    
    for col in range(cols):
        vect = tx[:, col]
        #delete -999 values before calculating the mean
        vect = np.delete(vect, np.where(vect == -999))
        #replace -999 values by mean
        tx[:,col][tx[:,col] == -999] = np.median(vect)
    
    return tx


def remove_skewness(tx, removed_miss_cols = False):
    """Remove skewness by doing a log transform of the heavy tailes features"""
    heavy_tailed_features = [0,1,2,3,5,8,9,10,13,16,19,21,23,26,29]
    
    if removed_miss_cols :
        heavy_tailed_features = [0,1,2,3,5,6,7,9,12,15,17,19,22]
    
    for skewed_feature in heavy_tailed_features:
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


# do all your needed preprocessing functions inside this function
def preprocess_data(input_data):
    """Apply all preprocessing functions to the input data """
    preprocessed_data = input_data.copy()
    #preprocessed_data = remove_feature_when_too_much_missing_values(preprocessed_data)
    preprocessed_data = replace_undefined_values_with_median(preprocessed_data)
    preprocessed_data = remove_skewness(preprocessed_data)
    preprocessed_data = drop_phi_features(preprocessed_data)
    preprocessed_data = standardize(preprocessed_data)
    preprocessed_data = add_bias_term(preprocessed_data)

    return preprocessed_data


def change_11_to_01_categories(labels):
    return np.where(labels == -1, 0, 1)