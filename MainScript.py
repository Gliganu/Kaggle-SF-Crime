import time

import ClassifierSelector as classifierSelector
import DataReader as dataReader
import RegularFeatureExtractor as regularFeatExtr
import Validator as validator
import Utils as utils
import Visualizer as visualizer
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def predictForSubmission():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    numberOfTrainingExamples = -1
    classifier = trainClassifierOnTrainingData(numberOfTrainingExamples)

    print "Beginning to load test data..."

    partitionNumber = utils.numberOfPartitions
    for index in range(partitionNumber):

        miniTestData = dataReader.getSerializedMiniTestData(index)

        xTest,yTest = constructTestData(miniTestData)

        print "Predicting..."
        yPred = classifier.predict_proba(xTest)

        dataReader.writePredToCsv(yPred,index)

    print "Post processing..."
    dataReader.postProcessCsv()
    print("Total run time:{}".format(time.time() - allAlgorithmStartTime))




def trainClassifierOnTrainingData(trainData=None, numberOfTrainingExamples = -1, margins=None):

    if trainData is None:
        trainData = dataReader.getTrainData(numberOfTrainingExamples,margins)

    # feature engineering
    trainData =  regularFeatExtr.convertTargetFeatureToNumeric(trainData)
    xTrain, yTrain = regularFeatExtr.getRegularFeatures(trainData, True)


     # classifier training
    classifier = classifierSelector.trainClassifier(xTrain, yTrain)

    return classifier

def constructTestData(testData):

    testData =  regularFeatExtr.convertTargetFeatureToNumeric(testData)
    xTest, yTest = regularFeatExtr.getRegularFeatures(testData, False)

    return xTest,yTest


def constructTrainingData(trainDataSize):

    #training data
    trainData = dataReader.getTrainData(trainDataSize)
    trainData = trainData.append(dataReader.getSuffixDataFrame())

    # feature engineering
    trainData =  regularFeatExtr.convertTargetFeatureToNumeric(trainData)
    xTrain, yTrain = regularFeatExtr.getRegularFeatures(trainData, True)


    return xTrain,yTrain

def testGeneralPerformanceUsingCrossValidationScore():
    # train 28k and test = 7k
    # trainDataSize = 35000
    trainDataSize = 150000

    classifier = classifierSelector.constructGradientBoostingClassifier()
    # classifier = classifierSelector.constructRandomForestClassifier()
    # classifier = SVC(verbose=1)

    xTrain,yTrain = constructTrainingData(trainDataSize)

    cv = StratifiedShuffleSplit(yTrain,n_iter=1,train_size=50000,test_size=100000)

    cv_scores = cross_val_score(classifier, xTrain, yTrain, cv=cv, n_jobs=-1,scoring="log_loss",verbose=1)

    scoreMean = cv_scores.mean()

    print "Mean score is {}".format(scoreMean)


def testParameterPerformance():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    # define sizes
    trainDataSize = 10000
    testDataSize = 100000

    trainData,testData = utils.getDifferentTrainAndTestData(trainDataSize,testDataSize)

    #in order to assure that we have members form each class present
    testData = testData.append(dataReader.getSuffixDataFrame())

    classifier = trainClassifierOnTrainingData(trainData=trainData)

    xTest,yTest = constructTestData(testData)

    yPred = classifier.predict(xTest)

    validator.performValidation(yPred, yTest)


    print("Total run time:{} s".format((time.time() - allAlgorithmStartTime)))



