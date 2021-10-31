# -*- coding: utf-8 -*-
"""Functions used for cross-validation."""
from implementations import *
from preprocessing import change_11_to_01_categories
from proj1_helpers import predict_labels

def accuracy(y, tx, w):
    """compute the accuracy of the model"""
    predictions =  predict_labels(w, tx)
    return (y == predictions).mean()


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if (degree == 1):
        return x
    
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


def cross_validation(y, x, k_indices, k, degree, model, lambda_=0, max_iters=0, gamma=0):
    """return the loss of the model."""
    
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
    extended_feature_matrix_train = x_train
    extended_feature_matrix_test = x_test
    if degree != 1: 
        extended_feature_matrix_train = build_poly(x_train, degree)
        extended_feature_matrix_test = build_poly(x_test, degree)
    
    # --- apply model ---
    w_init = initial_w = np.zeros((extended_feature_matrix_train.shape[1]))
    if model == 'least_squares_GD':
        w, _ = least_squares_GD(y_train, extended_feature_matrix_train, w_init, max_iters, gamma)
        
    elif model == 'least_squares_SGD':
        w, _ = least_squares_SGD(y_train, extended_feature_matrix_train, w_init, max_iters, gamma)
        
    elif model == 'least_squares':
        w, _ = least_squares(y_train, extended_feature_matrix_train)
        
    elif model == 'ridge_regression':
        w, _ = ridge_regression(y_train, extended_feature_matrix_train, lambda_)
        
    elif model == 'logistic_regression':
        y_01 = change_11_to_01_categories(y_train)
        w, _ = logistic_regression(y_01, extended_feature_matrix_train, w_init, max_iters, gamma)
    
    elif model == 'reg_logistic_regression':
        y_01 = change_11_to_01_categories(y_train)
        w, _ = reg_logistic_regression(y_01, extended_feature_matrix_train, lambda_, w_init, max_iters, gamma)
    
    # calculate the loss (RMSE) for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_train, extended_feature_matrix_train, w))
    loss_te = np.sqrt(2 * compute_loss(y_test, extended_feature_matrix_test, w))
    
    # calculate the accuracy for test data
    predictions =  predict_labels(w, extended_feature_matrix_train)
    acc = (y_train == predictions).mean()
    
    return loss_tr, loss_te, acc


def select_best_hyperparams_least_squares_GD(y, x, k_fold, degrees, max_iters, gammas, seed=1):
    """ return best degree, max_iters and gamma for least squares GD """
    degree, lambda_, miters, gamma, rmse = select_best_hyperparams_generic(y, x, k_fold, 'least_squares_GD', degrees, [0], max_iters, gammas, seed)
    
    return degree, miters, gamma, rmse


def select_best_hyperparams_least_squares_SGD(y, x, k_fold, degrees, max_iters, gammas, seed=1):
    """ return best degree, max_iters and gamma for least squares SGD """
    degree, lambda_, miters, gamma, rmse = select_best_hyperparams_generic(y, x, k_fold, 'least_squares_SGD', degrees, [0], max_iters, gammas, seed)
    
    return degree, miters, gamma, rmse


def select_best_hyperparams_least_squares(y, x, k_fold, degrees, seed=1):
    """ return best degree, max_iters and gamma for least squares """
    degree, lambda_, miters, gamma, rmse = select_best_hyperparams_generic(y, x, k_fold, 'least_squares', degrees, [0], [0], [0], seed)
    
    return degree, rmse


def select_best_hyperparams_ridge_regression(y, x, k_fold, degrees, lambdas_, seed=1):
    """ return best degree and lambda for ridge regression """
    degree, lambda_, miters, gamma, rmse = select_best_hyperparams_generic(y, x, k_fold, 'ridge_regression', degrees, lambdas_, [0], [0], seed)
    
    return degree, lambda_, rmse

       
def select_best_hyperparams_logistic_regression(y, x, k_fold, degrees, max_iters, gammas, seed=1):
    """ return best degree, max_iters and gamma for logistic regression """
    degree, lambda_, miters, gamma, rmse = select_best_hyperparams_generic(y, x, k_fold, 'logistic_regression', degrees, [0], max_iters, gammas, seed)
    
    return degree, miters, gamma, rmse


def select_best_hyperparams_reg_logistic_regression(y, x, k_fold, degrees, lambdas_, max_iters, gammas, seed=1):
    """ return best degree, max_iters and gamma for regularized logistic regression """
    degree, lambda_, miters, gamma, rmse = select_best_hyperparams_generic(y, x, k_fold, 'reg_logistic_regression', degrees, lambdas_, max_iters, gammas, seed)
    
    return degree, lambda_, miters, gamma, rmse


def select_best_hyperparams_generic(y, x, k_fold, model, degrees, lambdas, max_iters, gammas, seed):
    """ return best hyperparameters and best degree for a given model using the following procedure:
        1. For each degree, do k-fold cross validation and select the best hyperparameters (by testing all the combinations) which corresponds to the lowest test error.       
        2. Keep track of the best hyperparameters and their associated RMSE. 
        3. Select the best degree by choosing the degree corresponding to the lowest RMSE.
        This lowest RMSE also has all the hyperparameters associated to it.
    """
    
    # note: depending on the model some hyperparameters will be ignored (ie. have only one iteration)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    rmse_te_degrees = []
    degrees_lambda = []
    degrees_mitter = []
    degrees_gamma = []
    for degree in degrees:
        
        rmse_te_lambdas = []
        lambdas_mitter= []
        lambdas_gamma = []
        for lambda_ in lambdas:
            
            rmse_te_miters = []
            miters_gamma = []
            for miter in max_iters:
                
                rmse_te_gammas = [] # |gammas|
                for gamma in gammas:
                    
                    rmse_te_gamma_k = [] # |k|
                    for k in range(k_fold):
                        _, loss_te, _ = cross_validation(y, x, k_indices, k, degree, model, lambda_, miter, gamma)
                        rmse_te_gamma_k.append(loss_te)
                    
                    # append gamma's rmse
                    rmse_te_gammas.append(np.mean(rmse_te_gamma_k))
                
                # append max_iters's rmse and sub-params
                id_best_gamma = np.argmin(rmse_te_gammas)
                miters_gamma.append(gammas[id_best_gamma]) #gamma
                rmse_te_miters.append(rmse_te_gammas[id_best_gamma])#rmse
            
            # append lambda's rmse and sub-params
            id_best_miter = np.argmin(rmse_te_miters)
            lambdas_mitter.append(max_iters[id_best_miter]) #mitter
            lambdas_gamma.append(miters_gamma[id_best_miter]) #gamma
            rmse_te_lambdas.append(rmse_te_miters[id_best_miter]) #rmse
            
        # append degree's rmse and sub-params
        id_best_lambda = np.argmin(rmse_te_lambdas)
        degrees_lambda.append(lambdas[id_best_lambda]) #lambda
        degrees_mitter.append(lambdas_mitter[id_best_lambda]) #mitter
        degrees_gamma.append(lambdas_gamma[id_best_lambda]) #gamma
        rmse_te_degrees.append(rmse_te_lambdas[id_best_lambda]) #rmse             
                                 
    id_best_degree = np.argmin(rmse_te_degrees)
    best_degree = degrees[id_best_degree] # best degree
    best_lambda = degrees_lambda[id_best_degree] # best lambda
    best_max_iteration = degrees_mitter[id_best_degree] # best max_iterations
    best_gamma = degrees_gamma[id_best_degree] # best gamma
    best_rmse = rmse_te_degrees[id_best_degree] # best rmse
                                 
    return best_degree, best_lambda, best_max_iteration, best_gamma, best_rmse


def select_best_hyperparams_ridge_regression2(y, x, k_fold, degrees, lambdas_, seed = 1):
    """return best hyperparameters and best degree for ridge regression by following the next steps:
    1. For each degree, do k-fold cross validation and select the best lambda which corresponds to the lowest test error. 
    2. Keep track of the best lambda and its associated RMSE. 
    3. Select the best degree by choosing the degree corresponding to the lowest RMSE.
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
                _, loss_te, _ = cross_validation(y, x, k_indices, k, degree, 'ridge_regression', lambda_)
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


def accuracy_variance(y, x, k_fold, degree, model, lambda_=0, max_iters=0, gamma=0, seed=1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    rmse_tr = []
    rmse_te = []
    acc = []
    for k in range(k_fold):
        loss_tr, loss_te, acc_te = cross_validation(y, x, k_indices, k, degree, model, lambda_, max_iters, gamma)
        
        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)
        acc.append(acc_te)
        
    return rmse_tr, rmse_te, acc
        