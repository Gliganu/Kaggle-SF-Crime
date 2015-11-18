import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import DataReader as dataReader
import OHEFeatureExtractor as oheFeatExtr
import RegularFeatureExtractor as regularFeatExtr
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import Validator as validator
import ClassifierSelector as classifierSelector

import time
import matplotlib.pyplot as plt

def showTrainAndTestErrorPlot(yTrainValues,yTestValues,trainDataSize):

    plt.plot(trainDataSize, yTrainValues )
    plt.plot(trainDataSize, yTestValues)

    plt.xlabel('Number of training examples')
    plt.ylabel('Error')

    plt.legend(['Train Error', 'Test Error'], loc='upper left')
    plt.show()


def testDataOverTime():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    trainDataSizes = range(100000,200000,20000)

    yTestValues =[]
    yTrainValues = []

    initialTrainData, initialTestData = dataReader.getTargetAndTrainDataWithLabels(250000)

    for trainDataSize in trainDataSizes:
        print "Train data size is {}".format(trainDataSize)

        trainData = initialTrainData.sample(trainDataSize)
        testData = initialTestData.sample(1000)

        trainData, testData = oheFeatExtr.convertTargetFeatureToNumeric(trainData, testData)

        xTrain, yTrain, xTest, yTest = regularFeatExtr.getRegularFeatures(trainData, testData)

        print "Finished vectorizing after: {}".format(time.time() - startTime)

        classifier = classifierSelector.trainRandomForest(xTrain, yTrain)

        print("Predicting...")
        yTestPred = classifier.predict(xTest)
        yTrainPred = classifier.predict(xTrain)

        yTestAccuracy = np.sum(yTestPred == yTest) * 1. / len(yTest)
        yTrainAccuracy = np.sum(yTrainPred == yTrain) * 1. / len(yTrain)

        yTestValues.append(yTestAccuracy)
        yTrainValues.append(yTrainAccuracy)


    showTrainAndTestErrorPlot(yTestValues, yTrainValues,trainDataSizes)


    # outputResult(yPred)
    print("Total run time:{}".format(time.time() - allAlgorithmStartTime))


testDataOverTime()
