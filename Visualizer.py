import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import DataReader as dataReader
import RegularFeatureExtractor as regularFeatExtr
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import Validator as validator
import ClassifierSelector as classifierSelector
import MainScript as mainScript
import Utils as utils
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import numpy as np

import time
import matplotlib.pyplot as plt

def showTrainAndTestErrorPlot(yTrainValues,yTestValues,trainDataSize):

    plt.plot(trainDataSize, yTrainValues,'o-' )
    plt.plot(trainDataSize, yTestValues,'x-')

    plt.xlabel('Number of training examples')
    plt.ylabel('Error')

    plt.legend(['Train Error', 'Test Error'], loc='upper left')
    plt.show()


def testDataOverTime():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    # trainDataSize = 100000
    testDataSize = 3000


    yTestValues = []
    yTrainValues = []

    trainDataSizes = np.linspace(100000,200000,4,dtype=int)

    for trainDataSize in trainDataSizes:

        print("Currently at {}".format(trainDataSize))

        classifier,xTrain,yTrain = utils.trainClassifierOnTrainingDataReturnAll(trainDataSize)

        # Cut the data on which the predicition will be made to be the same lenght as test
        xTrain = xTrain[0:testDataSize]
        yTrain = yTrain[0:testDataSize]

        mockTrainData = dataReader.getTrainData(testDataSize)

        mockTrainData = mockTrainData.append(dataReader.getSuffixDataFrame())

        xTest,yTest = mainScript.constructTestData(mockTrainData)

        print("Predicting...")
        yTestPred = classifier.predict(xTest)
        yTrainPred = classifier.predict(xTrain)

        yTestAccuracy =  accuracy_score(yTestPred,yTest)
        yTrainAccuracy = accuracy_score(yTrainPred,yTrain)

        yTestValues.append(yTestAccuracy)
        yTrainValues.append(yTrainAccuracy)


    showTrainAndTestErrorPlot(yTestValues, yTrainValues,trainDataSizes)


    print("Total run time:{}".format(time.time() - allAlgorithmStartTime))
#

# if __name__ == '__main__':
#     testDataOverTime()
#

