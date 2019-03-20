import numpy as np


def update_weights(delta, X, y, learning_rate=0.01, iterations=10000):
    n = len(y)
    m = np.shape(X)[0]
    w = np.zeros(m)     # X is m*n, w is m*1, b and y is n*1
    b = np.zeros(n)

    # Calculate the partial derivative
    for i in range(iterations):
        prediction = np.dot(w, X) + b
        grad0 = cal_grad_w(delta, X, prediction, y)
        grad1 = cal_grad_b(delta, prediction, y)

        # update the weights
        w = w - learning_rate * grad0
        b = b - learning_rate * grad1

    return w, b


def cal_grad_w(delta, X, prediction, y):
    # using the Huber loss
    dist = np.linalg.norm(y - prediction)
    direction = np.divide((y - prediction), np.linalg.norm(y - prediction))
    cost_gd = np.sum(np.where(dist <= delta, np.dot(np.transpose(dist), X), delta * np.dot(direction, X)))

    return cost_gd


def cal_grad_b(delta, prediction, y):
    dist = np.linalg.norm(y - prediction)
    direction = np.divide((y - prediction), np.linalg.norm(y - prediction))
    cost_gd = np.sum(np.where(dist <= delta, dist, delta * direction))

    return cost_gd
