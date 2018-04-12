import numpy as np
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm

def classifier_acc(Y_pred, Y_true):
    return np.mean((Y_pred == Y_true).astype(np.int))


def regression_mse(Y_pred, Y_true, weight=None):
    if weight is None:
        weight = np.ones(len(Y_pred))
    return np.dot(weight, (Y_pred - Y_true) ** 2) / np.sum(weight)

def least_squares(X_train, Y_train, X_test, Y_test, example_weights=None):
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train, example_weights)
    train_mse = regression_mse(model.predict(X_train), Y_train) #, weight=example_weights)
    test_mse = regression_mse(model.predict(X_test), Y_test)
    return (test_mse, train_mse)

def least_squares_ridge(X_train, Y_train, X_test, Y_test, example_weights=None):
    model = linear_model.Ridge(alpha=1, fit_intercept=True)
    model.fit(X_train, Y_train, example_weights)
    train_mse = regression_mse(model.predict(X_train), Y_train) #, weight=example_weights)
    test_mse = regression_mse(model.predict(X_test), Y_test)
    return (test_mse, train_mse)


def logistic_regression(X_train, Y_train, X_test, Y_test, example_weights=None):
    # theta, train_acc = log_reg_sgd(X_train, Y_train, example_weights)
    # test_acc = log_reg_classify_acc(X_test, Y_test, theta)
    model = linear_model.LogisticRegression(C=100000, fit_intercept=True)
    model.fit(X_train, Y_train, example_weights)
    test_acc = model.score(X_test, Y_test)
    train_acc = model.score(X_train, Y_train, example_weights)
    return (test_acc, train_acc)


def svc(X_train, Y_train, X_test, Y_test, example_weights=None):
    model = svm.SVC(kernel='linear', C=1)
    model.fit(X_train, Y_train, example_weights)
    test_acc = model.score(X_test, Y_test)
    train_acc = model.score(X_train, Y_train) #, example_weights)
    return (test_acc, train_acc)


# def rf(X_train, Y_train, X_test, Y_test, example_weights=None):
#     model = RandomForestClassifier()
#     model.fit(X_train, Y_train, example_weights)
#     pred = (model.predict(X_test) > 0).astype(np.int)
#     acc = classifier_acc(pred, Y_test)
#     return (None, None, acc)
