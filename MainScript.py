import time

import ClassifierSelector as classifierSelector
import DataReader as dataReader
import RegularFeatureExtractor as regularFeatExtr
import Validator as validator
import Utils as utils
import Visualizer as visualizer

def predictForSubmission():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    numberOfTrainingExamples = 200000
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

    trainDataSize = 10000
    miniBatchDataSize = 5000

    classifier = trainClassifierOnTrainingData(trainDataSize)

    print "Beginning to load test data..."

    mockTrainData = dataReader.getTrainData(miniBatchDataSize)

    mockTrainData = mockTrainData.append(dataReader.getSuffixDataFrame())

    xTest,yTest = constructTestData(mockTrainData)

    yPred = classifier.predict(xTest)

    validator.performValidation(yPred, yTest)

    totalMinutes = (time.time() - allAlgorithmStartTime)/60
    hours = totalMinutes/60
    minutes = totalMinutes - 60*hours

    print("Total run time:{} h {} s".format(hours,minutes))


# if __name__ == '__main__':
    # predictForValidation()
    # predictForSubmission()
#
#



