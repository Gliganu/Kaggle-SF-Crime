import time

import ClassifierSelector as classifierSelector
import DataReader as dataReader
import OHEFeatureExtractor as oheFeatExtr
import RegularFeatureExtractor as regularFeatExtr
import Validator as validator
import Utils as utils

def predictForSubmission():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    classifier = trainClassifierOnTrainingData(50000)

    print "Beginning to load test data..."

    partitionNumber = utils.numberOfPartitions
    for index in range(partitionNumber):

        miniTestData = dataReader.getSerializedMiniTestData(index)

        xTest,yTest = constructTestData(miniTestData)

        print "Predicting..."
        yPred = classifier.predict(xTest)

        dataReader.writePredToCsv(yPred,index)

    print("Total run time:{}".format(time.time() - allAlgorithmStartTime))




def trainClassifierOnTrainingData(numberOfTrainingExamples):

    trainData = dataReader.getTrainData(numberOfTrainingExamples)

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

def predictForValidation():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    trainDataSize = 5000
    miniBatchDataSize = 1000

    classifier = trainClassifierOnTrainingData(trainDataSize)


    print "Beginning to load test data..."

    for index in range(1):

        mockTrainData = dataReader.getTrainData(miniBatchDataSize)

        mockTrainData = mockTrainData.append(dataReader.getSuffixDataFrame())

        xTest,yTest = constructTestData(mockTrainData)

        yPred = classifier.predict(xTest)

        # validator.performValidation(yPred, yTest)

        dataReader.writePredToCsv(yPred,index)

    print("Total run time:{}".format(time.time() - allAlgorithmStartTime))


predictForValidation()






