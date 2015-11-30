import time

import ClassifierSelector as classifierSelector
import DataReader as dataReader
import RegularFeatureExtractor as regularFeatExtr
import Validator as validator
import Utils as utils
import Visualizer as visualizer
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold


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




def trainClassifierOnTrainingData(numberOfTrainingExamples = -1, margins=None):

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

def predictForValidation():
    # train 28k and test = 7k
    trainDataSize = 35000

    classifier = classifierSelector.constructGradientBoostingClassifier()

    xTrain,yTrain = constructTrainingData(trainDataSize)

    cv = StratifiedKFold(yTrain,n_folds=5)

    cv_scores = cross_val_score(classifier, xTrain, yTrain, cv=cv, n_jobs=-1,scoring="f1_weighted",verbose=1)

    scoreMean = cv_scores.mean()

    print "Mean score is {}".format(scoreMean)


if __name__ == '__main__':
    predictForValidation()
    # predictForSubmission()
#
#



