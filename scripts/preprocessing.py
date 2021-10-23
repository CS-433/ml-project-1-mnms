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


def standardize(tx):
    """Standardize the data set."""
    
    centered_data = tx - np.mean(tx, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data


def add_bias_term(input_data):
    """Add column of 1s which corresponds to the bias term"""
    
    num_samples = input_data.shape[0]
    tx = np.c_[np.ones(num_samples), input_data]
    
    return tx

#TODO do all your needed preprocessing functions inside this function
def preprocess_data(input_data):
    """Apply all preprocessing functions to the input data """
    clean_input_data = replace_undefined_values_with_mean(input_data)
    clean_std_input_data = standardize(clean_input_data)
    preprocessed_data = add_bias_term(clean_std_input_data)
    
    return preprocessed_data