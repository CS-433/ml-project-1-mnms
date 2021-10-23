# -*- coding: utf-8 -*-
"""Functions used for cross-validation."""
from implementations import *

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    # get k'th subgroup in test, others in train
    #k'th subset is reserved for the testing
    test_indices = k_indices[k]
    #the rest can be used for the training
    index_mask = np.arange(len(k_indices)) != k
    #all subsets, but the k'th
    train_indices = k_indices[index_mask]
    #flatten the lists of lists
    train_indices = train_indices.reshape(-1)
    
    #get the corresponding inputs and outputs
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # form data with polynomial degree
    extended_feature_matrix_train = build_poly(x_train, degree)
    extended_feature_matrix_test = build_poly(x_test, degree)

    # ridge regression
    w, _ = ridge_regression(y_train, extended_feature_matrix_train, lambda_)
    
    # calculate the loss (RMSE) for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_train, extended_feature_matrix_train, w))
    loss_te = np.sqrt(2 * compute_loss(y_test, extended_feature_matrix_test, w))
    
    return loss_tr, loss_te


def select_best_degree_and_lambda(y, x, degrees, lambdas, k_fold, seed = 1):
    """return best lambda and best degree for ridge regression by following the next steps. 
    For each degree, do k-fold cross validation and select the best lambda which corresponds to the lowest test error. 
    Keep track of the best lambda and its associated RMSE. 
    Select the best degree by choosing the degree corresponding to the lowest RMSE.
    This lowest RMSE also has a lambda associated to it. """
    best_lambdas = []
    best_rmses = []
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    for degree in degrees:
        rmse_te = []
        for lambda_ in lambdas:
            rmse_te_lambda_ = []
            for k in range(k_fold):
                _, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
                rmse_te_lambda_.append(loss_te)
            rmse_te.append(np.mean(rmse_te_lambda_))
            
        index_best_lambda_degree = np.argmin(rmse_te)
        best_lambdas.append(lambdas[index_best_lambda_degree])
        best_rmses.append(rmse_te[index_best_lambda_degree])
        
    index_best_degree = np.argmin(best_rmses)
    #the best rmse has an associated lambda value
    best_lambda = best_lambdas[index_best_degree]
    best_degree = degrees[index_best_degree]
    
    return best_lambda, best_degree