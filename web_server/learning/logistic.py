from math import e
from array import array

import numpy as np

def log_reg(x, theta):
    """
    logistic regression hypothesis function
    """
    return np.exp(np.dot(x,theta))/(1.+np.exp(np.dot(x,theta)))

def log_reg_classify_acc(x, y, theta):
    return np.mean(np.asarray(np.asarray(log_reg(x, theta) >= 1/2, np.int) == y, np.int))

def log_reg_sgd(x, y, weight=None, lr=0.1, max_iter=100, threshold = 0.1, debug=True):
    """
    Stochastic gradient descent for two class (0,1) logistic regression
    with static learning rate
    """
    #initialize algorithm state
    m, n = x.shape
    if weight is None:
        weight = np.ones(m)
    theta = np.random.random(n)
    error = 0
    for t in range(max_iter):
        #shuffle indices prior to searching
        #for each training example
        for i in range(0, len(y), 10):
            last = len(y) if i+10 >= len(y) else i+10
            theta += lr * np.dot(weight[i:last] * x[i:last].T, (y[i:last] - log_reg(x[i:last], theta))) / np.sum(weight[i:last])
        if t % 10 == 0:
            error = 1 - log_reg_classify_acc(x, y, theta)
            if error < threshold:
                break
        # for i in range(m):
        #     #update weights
        #     theta = theta + lr * (y[i]-log_reg(x[i],theta))*x[i]
    #         #compute the error
    #         if debug: err.append(sum([(y[i]-h(x[i],theta))**2 for i in range(m)]))
    # if debug: return theta,err
    return theta, 1 - error

def my_logistic_regression(X_train, Y_train, X_test, Y_test, example_weights=None):
    theta, train_acc = log_reg_sgd(X_train, Y_train, example_weights, max_iter=100)
    test_acc = log_reg_classify_acc(X_test, Y_test, theta)
    return (test_acc, train_acc)

def log_reg_regularized_sgd(x,y,a,l=0.1,max_iter=100,debug=True):
    """
    Stochastic gradient descent for logistic regression with regularization
    and static learning rate
    """
    if debug: err = array('f',[])
    #initialize algorithm state
    m,n = x.shape
    theta = np.random.random(n)
    z = np.arange(m)
    for t in range(max_iter):
        #shuffle indices prior to searching
        z = np.random.permutation(z)
        #for each training example
        for i in z:
            #update weights
            theta = theta + a*(y[i]-h(x[i],theta))*x[i] - l*2.*a*theta
            #compute the error, current incorrect
            if debug:err.append(sum([(y[i]-h(x[i],theta))**2 for i in range(m)]))
    if debug: return theta,err
    return theta
