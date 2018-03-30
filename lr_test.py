# -*- coding: utf-8 -*-
import numpy as np
import time


def createTrainDataSet():
    trainDataMat = [[1, 1, 4], 
                    [1, 2, 3], 
                    [1, -2, 3], 
                    [1, -2, 2], 
                    [1, 0, 1], 
                    [1, 1, 2]]
    trainShares = [1, 1, 1, 0, 0, 0]
    return trainDataMat, trainShares

def createTestDataSet():
    testDataMat = [[1, 1, 1], 
                   [1, 2, 0], 
                   [1, 2, 4], 
                   [1, 1, 3]]
    return testDataMat


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradAscent(trainingData, trainingLabel, iter, learning_rate):
    td = np.array(trainingData)
    tl = np.array(trainingLabel)
    w = np.ones(len(td[0]))
    for it in range(iter):
        for i in range(len(td)):
            h = sigmoid(np.dot(td[i], w))
            w = w + learning_rate * (tl[i] - h) * td[i]
    return w
 

def gradAscent2(trainingData, trainingLabel, iter, learning_rate):
    td = np.mat(trainingData)
    tl = np.mat(trainingLabel).transpose()
    w = np.mat(np.ones(td.shape[1])).transpose()
    for it in range(iter):
        h = sigmoid(td * w)
        w = w + learning_rate * td.transpose()* (tl - h)
    return w


def stochasticGradAscent(trainingData, trainingLabel, iter, learning_rate):
    td = np.array(trainingData)
    tl = np.array(trainingLabel)
    w = np.ones(len(td[0]))
    for it in range(iter):
        np.random.shuffle(td)
        i = 0
        h = sigmoid(np.dot(td[i], w))
        w = w + learning_rate * (tl[i] - h) * td[i]
    return w


def miniBatchGradAscent(trainingData, trainingLabel, iter, learning_rate, batch_size):
    td = np.mat(trainingData)
    tl = np.mat(trainingLabel).transpose()
    w = np.mat(np.ones(td.shape[1])).transpose()
    for it in range(iter):
        #r = [(lower, upper) for lower in xrange(0, len(td), batch_size)]
        np.random.shuffle(td)
        batch_td, batch_tl = td[0: batch_size], tl[0: batch_size]
        batch_h = sigmoid(batch_td * w)
        w = w + learning_rate * batch_td.transpose()* (batch_tl - batch_h)
    return w


def pred(w, testData):
    td = np.array(testData)
    res = np.zeros(len(td))
    for i in range(len(td)):
        res[i] = 1.0 if sigmoid(np.dot(w.T, td[i])) >= 0.5 else 0.0
    return res


def do_main():
    trainDataSet, trainShares = createTrainDataSet()
    testDataSet = createTestDataSet()

    print '\nbgd'
    w = gradAscent(trainDataSet, trainShares, 200, 0.01)
    print w

    print '\nbgd2'
    w = gradAscent2(trainDataSet, trainShares, 200, 0.01)
    print w

    print '\nsgd'
    w = stochasticGradAscent(trainDataSet, trainShares, 1000, 0.01)
    print w

    print '\nmini-batch sgd'
    w = miniBatchGradAscent(trainDataSet, trainShares, 500, 0.01, len(trainDataSet)/2)
    print w

    predData = pred(w, testDataSet)
    print predData


if __name__ == '__main__':
    start = time.clock()
    do_main()
    end = time.clock()
    print('finish all in %s' % str(end - start))
