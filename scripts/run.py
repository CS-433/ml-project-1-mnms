import numpy as np
from proj1_helpers import *
from preprocessing import *
from implementations import *
from cross_validation import *

    # Run this file to produce the model with the best prediction's accuracy

def main():

    DATA_TRAIN_PATH = '../data/train.csv'
    y, train_input_data, ids = load_csv_data(DATA_TRAIN_PATH)

    DATA_TEST_PATH = '../data/test.csv'
    y_test, test_input_data, ids_test = load_csv_data(DATA_TEST_PATH)

    tX = preprocess_data(train_input_data)
    tX_test = preprocess_data(test_input_data)
    
    # --- cross-validation to find best hyper-parameters : uncomment to run (beware, it runs for more than 30 minutes)
    # degrees = np.arange(1,12)
    # lambdas = np.logspace(-4, -1, 30)
    # degree, lambda_, _, _ = select_best_hyperparams_ridge_regression(y, tX, k_fold, degrees, lambdas, seed=6)
    
    # obtained using cross-validation
    degree = 9
    lambda_ = 0.00041753189365604

    extended_feature_matrix_train = build_poly(tX, degree)
    extended_feature_matrix_test = build_poly(tX_test, degree)

    weights, loss = ridge_regression(y, extended_feature_matrix_train, lambda_)

    print("compute model accuracy using cross-validation...")
    _, _, acc_GD = error_accuracy_validation(y, tX, 4, degree, 'ridge_regression', lambda_, 0, 0, seed=5)
    print("accuracy mean : ", np.mean(acc_GD))
    print("accuracy std : ", np.std(acc_GD))

    OUTPUT_PATH = '../data/submission.csv'
    y_pred = predict_labels(weights, extended_feature_matrix_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


if __name__ == '__main__':
    main()
