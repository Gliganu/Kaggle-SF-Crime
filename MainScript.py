import time

import ClassifierSelector as classifierSelector
import DataReader as dataReader
import OHEFeatureExtractor as oheFeatExtr
import RegularFeatureExtractor as regularFeatExtr
import Validator as validator
import Utils as utils

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



def predict():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    classifier = trainClassifierOnTrainingData(10000)

    print "Beginning to load test data..."

    # partitionNumber = utils.numberOfPartitions
    for index in range(3):

        # miniTestData = dataReader.getSerializedMiniTestData(index)
        miniTestData = dataReader.getRandomMiniTestData()

        xTest,yTest = constructTestData(miniTestData)

        print "Predicting..."
        yPred = classifier.predict(xTest)

        validator.performValidation(yPred, yTest)

        # dataReader.writePredToCsv(yPred,index)

    print("Total run time:{}".format(time.time() - allAlgorithmStartTime))


predict()






