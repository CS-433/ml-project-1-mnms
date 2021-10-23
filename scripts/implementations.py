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
        w = w - gamma*gradient
        
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
