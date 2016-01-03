import numpy as np
from sklearn import datasets

def normalize(X, norm='l2', axis=1):
    """scales individual samples to have unit norm."""
    if axis == 0:
        X = X.T

    if norm == 'l1':
        norms = np.abs(X).sum(axis=1)
    else:
        norms = np.sqrt((X * X).sum(axis=1))
    new = X / norms[:, np.newaxis]

    if axis == 0:
        new = new.T

    return new

def scale(X, axis=0):
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)

def train_test_split(*arrays, test_size=None, train_size=None):
    length = len(arrays[0])
    p = np.random.permutation(length)

    if type(test_size) == int:
        index = length - test_size
    elif type(test_size) == float:
        index = length - int(length * test_size)
    else:
        if type(train_size) == int:
            index = train_size
        elif type(train_size) == float:
            index = int(length * train_size)
        else:
            index = length - int(length * 0.25)
    return [b for a in arrays for b in (a[p][:index], a[p][index:])]


def main():
    iris = datasets.load_iris()
    # a = np.array([[1,2,3],[3,4,5]])
    # print(normalize(a, axis=0))
    # X = np.array([[ 1., -1.,  2.],
    #               [ 2.,  0.,  0.],
    #               [ 0.,  1., -1.]])
    # print(scale(X))
    # X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=.75)



if __name__ == '__main__': main()