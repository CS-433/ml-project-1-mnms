# -*- coding: utf-8 -*-
"""Additional helper functions."""
import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using MSE."""
    e = y - tx @ w
    return 0.5*np.mean(e**2)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y - tx @ w
    return -1/N * tx.T @ e


def compute_gradient_SGD(y, tx, w):
    """Compute the gradient for the SGD."""
    e = y - tx @ w
    return - tx.T * e


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using the gradient descent algorithm."""

    w = initial_w

    # repeat gradient descent max_iters times
    # the last value of w is the most optimized one
    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)

        # update w by gradient
        w = w - gamma*gradient

    # compute the loss of the optimized w
    loss = compute_loss(y, tx, w)

    return w, loss


# TODO do not forget to remove
def least_squares_GD_complete(y, tx, initial_w, max_iters, gamma):
    """Linear regression using the gradient descent algorithm."""

    w = initial_w
    
    ws = [initial_w]
    losses = []

    # repeat gradient descent max_iters times
    # the last value of w is the most optimized one
    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = w - gamma*gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)

    # compute the loss of the optimized w
    loss = compute_loss(y, tx, w)

    return w, loss, ws, losses


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using the stochastic gradient descent algorithm."""

    w = initial_w
    # repeat stochastic gradient descent max_iters times
    # the last value of w is the most optimized one
    for _ in range(max_iters):
        #choose random item
        rand_id = np.random.choice(len(y))
        random_xn = tx[rand_id]
        random_yn = y[rand_id]
        
        gradient = compute_gradient_SGD(random_yn, random_xn, w)

        # update w by gradient
        w = w - gamma*gradient

    # compute the loss of the optimized w
    loss = compute_loss(y, tx, w)

    return w, loss


# TODO remove
def least_squares_SGD_complete(y, tx, initial_w, max_iters, gamma):
    """Linear regression using the stochastic gradient descent algorithm."""

    w = initial_w
    
    ws = [initial_w]
    losses = []
    
    # repeat stochastic gradient descent max_iters times
    # the last value of w is the most optimized one
    for _ in range(max_iters):
        rand_id = np.random.choice(len(y))
        random_xn = tx[rand_id]
        random_yn = y[rand_id]

        gradient = compute_gradient_SGD(random_yn, random_xn, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = w - gamma * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)

    # compute the loss of the optimized w
    loss = compute_loss(y, tx, w)

    return w, loss, ws, losses


def least_squares(y, tx):
    """Calculate the least squares solution using normal equations"""

    b = tx.T @ y
    A = tx.T @ tx
    # solve the linear system to find w
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""

    lambda_prime = 2 * len(y) * lambda_

    # the identity matrix should be DxD
    A = tx.T @ tx + lambda_prime * np.identity(tx.shape[1])
    b = tx.T @ y
    # solve the linear system to find w
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)

    return w, loss


def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1 / (1 + np.exp(-t))


def calculate_loss_logistic_regression(y, tx, w):
    """compute the loss for logistic regression: negative log likelihood."""
    prediction = sigmoid(tx @ w)
    left_term = y.T @ np.log(prediction).reshape(-1)
    right_term = (1 - y).T @ np.log(1 - prediction).reshape(-1)
    return - (left_term + right_term)


def calculate_gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss for logistic regression."""
    prediction = sigmoid(tx @ w)
    return tx.T @ (prediction - y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """logistic regression using gradient descent."""
    w = initial_w

    # repeat gradient descent max_iters times
    # the last value of w is the most optimized one
    for _ in range(max_iters):
        gradient = calculate_gradient_logistic_regression(y, tx, w)

        # update w by gradient
        w = w - gamma * gradient

    # compute the loss of the optimized w
    loss = calculate_loss_logistic_regression(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """regularized logistic regression using gradient descent"""
    w = initial_w

    # repeat gradient descent max_iters times
    # the last value of w is the most optimized one
    for _ in range(max_iters):
        #add the regularization term
        gradient = calculate_gradient_logistic_regression(y, tx, w) + 2 * lambda_ * w

        # update w by gradient
        w = w - gamma * gradient
        
    #compute norm of w
    w_norm = np.linalg.norm(w)
    # compute the loss of the optimized w, add the regularization term
    loss = calculate_loss_logistic_regression(y, tx, w) + lambda_ * w_norm * w_norm

    return w, loss