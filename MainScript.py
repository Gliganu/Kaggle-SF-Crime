import time

import ClassifierSelector as classifierSelector
import DataReader as dataReader
import OHEFeatureExtractor as oheFeatExtr
import RegularFeatureExtractor as regularFeatExtr
import Validator as validator


def predict():
    startTime = time.time()
    allAlgorithmStartTime = startTime

    # trainData, testData = dataReader.getTargetAndTrainData()
    trainData, testData = dataReader.getTargetAndTrainDataWithLabels(100,100)

    # feature engineering
    trainData, testData = oheFeatExtr.convertTargetFeatureToNumeric(trainData, testData)

    # xTrain, yTrain, xTest, yTest = oheFeatExtr.getOHEFeatures(trainData, testData)
    xTrain, yTrain, xTest, yTest = regularFeatExtr.getRegularFeatures(trainData, testData)

    print "Finished vectorizing after: {}".format(time.time() - startTime)

    # train model
    classifier = classifierSelector.getClassifier(xTrain, yTrain)

    print("Predicting...")
    yPred = classifier.predict_log_proba(xTest)

    # validator.performValidation(yPred, yTest)
    #
    dataReader.writePredToCsv(yPred)

    print("Total run time:{}".format(time.time() - allAlgorithmStartTime))


predict()






