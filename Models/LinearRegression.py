import numpy as np

def linear_regression(X, y, learning_rate=0.01, num_iterations=1000):
    m = len(y)  # Number of training examples
    n = X.shape[1]  # Number of features

    theta = np.zeros((n, 1))
    bias = 0

    for iteration in range(num_iterations):
        y_pred = np.dot(X, theta) + bias

        d_theta = (1/m) * np.dot(X.T, (y_pred - y))
        d_bias = (1/m) * np.sum(y_pred - y)

        theta -= learning_rate * d_theta
        bias -= learning_rate * d_bias

    return theta, bias