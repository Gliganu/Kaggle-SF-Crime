import time

import ClassifierSelector as classifierSelector
import DataReader as dataReader
import RegularFeatureExtractor as regularFeatExtr
import Validator as validator
import Utils as utils

def predictForSubmission():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    classifier = trainClassifierOnTrainingData()

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




def trainClassifierOnTrainingData(numberOfTrainingExamples = -1):

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

    trainDataSize = 100000
    miniBatchDataSize = 3000

    classifier = trainClassifierOnTrainingData(trainDataSize)


    print "Beginning to load test data..."

    for index in range(1):

        mockTrainData = dataReader.getTrainData(miniBatchDataSize)

        mockTrainData = mockTrainData.append(dataReader.getSuffixDataFrame())

        xTest,yTest = constructTestData(mockTrainData)

        yPred = classifier.predict(xTest)

        validator.performValidation(yPred, yTest)

        # dataReader.writePredToCsv(yPred,index)

    print("Total run time:{}".format(time.time() - allAlgorithmStartTime))


# if __name__ == '__main__':
#     predictForValidation()
    # predictForSubmission()
#
#



