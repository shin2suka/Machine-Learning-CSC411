# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist



#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    dist = l2(test_datum.reshape(1, test_datum.shape[0]), x_train)
    numerator = np.exp(np.divide(-dist, 2 * tau**2))

    denominator = np.exp(logsumexp(np.divide(-dist, 2 * tau**2)))
    A = np.diag(np.divide(numerator, denominator)[0,:])

    w = np.linalg.solve(np.dot(np.dot(np.transpose(x_train), A), x_train) + lam * np.identity(x_train.shape[1]), np.dot(np.dot(np.transpose(x_train), A), y_train))
    y_hat = np.dot(test_datum.transpose(), w)

    return y_hat



def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    train_size = int((1 - val_frac) * x.shape[0])
    # shuffle the whole data set simountinously
    s = np.arange(x.shape[0])
    np.random.seed(0)
    np.random.shuffle(s)
    x_ = x[s]
    y_ = y[s]
    x_train, x_val = x_[:train_size], x_[train_size:]
    y_train, y_val = y_[:train_size], y_[train_size:]

    train_loss = np.zeros(len(taus))
    val_loss = np.zeros(len(taus))
    for n, tau in enumerate(taus):
        y_list = []
        for i in range(len(x_val)):
            y_hat = LRLS(x_val[i], x_train, y_train, tau)
            loss = 0.5 * (y_val[i] - y_hat) ** 2
            y_list.append(loss)
        val_avg_loss = sum(y_list) / len(y_list)
        val_loss[n] = val_avg_loss

    for n, tau in enumerate(taus):
        y_list = []
        for i in range(len(x_train)):
            y_hat = LRLS(x_train[i], x_train, y_train, tau)
            loss = 0.5 * (y_train[i] - y_hat) ** 2
            y_list.append(loss)
        train_avg_loss = sum(y_list) / len(y_list)
        train_loss[n] = train_avg_loss

    return train_loss, val_loss


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)
    print("Done!")
    plt.show()
