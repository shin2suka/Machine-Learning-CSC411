'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import timeit

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, len(train_data[0])))
    # Compute means
    for k in range(10):
        numerator = np.zeros((1, len(train_data[0])))
        denominator = 0
        for i in range(len(train_labels)):
            if train_labels[i] == k:
                numerator += np.reshape(train_data[i], (1, len(train_data[0])))
                denominator += 1
        mean = np.divide(numerator, denominator)
        means[k] = mean

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, len(train_data[0]), len(train_data[0])))
    means = compute_mean_mles(train_data, train_labels)
    # Compute covariances
    for k in range(10):
        numerator = np.zeros((len(train_data[0]), len(train_data[0])))
        denominator = 0
        for i in range(len(train_labels)):
            if train_labels[i] == k:
                numerator += np.dot(np.reshape(train_data[i], (len(train_data[0]), 1)) - np.reshape(means[k], (len(train_data[0]), 1)),
                                    np.transpose(np.reshape(train_data[i], (len(train_data[0]), 1)) - np.reshape(means[k], (len(train_data[0]), 1))))
                denominator += 1
        covariance = np.divide(numerator, denominator) + 0.01 * np.identity(len(train_data[0]))
        covariances[k] = covariance

    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    gll = np.zeros((len(digits), 10))
    for i in range(len(digits)):
        gll_k = np.zeros(10)
        for k in range(10):
            cov_inverse = np.linalg.inv(covariances[k])
            det = np.linalg.det(covariances[k])
            last_term = np.dot(np.dot(np.transpose(digits[i] - means[k]), cov_inverse), digits[i] - means[k])
            logp = (-len(digits[0])/2) * np.log(2*np.pi) - 1/2 * np.log(det) - 1/2 * last_term
            gll_k[k] = logp
        gll[i] = gll_k

    return gll

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gll = generative_likelihood(digits, means, covariances)
    px = np.dot(0.1 * np.sum(np.exp(gll), axis=1).reshape(len(digits), 1), np.ones(10).reshape(1, 10))
    py = np.divide(np.ones((len(digits), 10)), 10)
    cll = gll + np.log(py) - np.log(px)

    return cll


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    sum_ = 0
    for i in range(len(labels)):
        sum_ += cond_likelihood[i][int(labels[i])]

    return sum_ / len(labels)


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    max_label = np.argmax(cond_likelihood, axis=1)

    return max_label


def main():
    start = timeit.default_timer()
    print("Let's start!")
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # Evaluation
    train_predicted_label = classify_data(train_data, means, covariances)
    print("Please wait...")
    test_predicted_label = classify_data(test_data, means, covariances)
    train_acll = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print("Finishing...")
    test_acll = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    train_accuracy = np.sum(train_predicted_label == train_labels) / len(train_labels)
    test_accuracy = np.sum(test_predicted_label == test_labels) / len(test_labels)
    print('avg conditional log-likelihood (train set): ' + str(train_acll))
    print('avg conditional log-likelihood (test set): ' + str(test_acll))
    print('-' * 80)
    print('Train Accuracy: ' + str(train_accuracy * 100) + '%')
    print('Test Accuracy: ' + str(test_accuracy * 100) + '%')

    max_eigenv = ()
    for k in range(len(covariances)):
        w, v = np.linalg.eig(covariances[k])
        max_w = w.argmax()
        max_eigenv += (v[:,max_w].reshape((8, 8)),)

    max_eigenv = np.concatenate(max_eigenv, axis=1)
    plt.imshow(max_eigenv, cmap='gray')
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    plt.show()


if __name__ == '__main__':
    main()

