import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_cost(X,y,theta):
    """
    Cost fuction

    :param X: np.array
    :param y: np.array
    :param theta: np.array
    :return: float
    """

    m = len(y)
    J = 1/(2*m) * np.sum((X@theta - y)**2)
    return J


def gradient_descent(X,y,theta,alpha,num_iters):
    '''
    Gradient descent function

    :param X: np.array
    :param y: np.array
    :param theta: np.array
    :param alpha: float
    :param num_iters: int
    :return: np.array
    '''
    m = len(y)
    hist = []

    for i in range(num_iters):

        theta -= alpha*(1/m)*((X@theta - y).T@X).T
        hist.append(compute_cost(X,y,theta))


    return theta,hist

def normalization(X):
    '''

    :param X: np.array
    :return: np.array
    '''
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    X_norm = (X-mean)/std

    return X_norm,mean,std


