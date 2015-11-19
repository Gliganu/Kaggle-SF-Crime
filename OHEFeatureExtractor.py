import time
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
import DataReader as dataReader


def convertTargetFeatureToNumeric(data):
    categoryDictionary = dataReader.getCategoryDictionaries()
    data = data.replace(categoryDictionary.keys(), range(len(categoryDictionary.keys())))

    return data


def handleNumericFeaturesForOHE(trainData, testData):
    numeric_cols = ['X', 'Y']
    trainDataNumeric = trainData[numeric_cols].as_matrix()
    testDataNumeric = testData[numeric_cols].as_matrix()

    return trainDataNumeric, testDataNumeric


def performFeatureEngineeringforOHE(trainData, testData):
    print("Performing feature engineering...")

    # drop certain columns
    trainData = trainData.drop(['Dates', 'X', 'Y', 'Descript', 'Address', 'Resolution'],
                               1)  # descript must be droped because there is no descript in test

    if ('Id' in testData.columns.values):
        testData = testData.drop(['Dates', 'X', 'Y', 'Address', 'Id'], 1)
    else:
        testData = testData.drop(['Dates', 'X', 'Y', 'Address'], 1)

    print("Train data features: {}".format(trainData.columns.values))
    print("Test data features: {}".format(testData.columns.values))

    return trainData, testData


def vectorizeDataForOHE(xTrain, xTest):
    print("Vectorizing xTrain...")

    startTime = time.time()
    vectorizer = DV(sparse=False)
    xTrain = vectorizer.fit_transform(xTrain)

    print "\t Fit+Transform on Xtrain took: {}".format(time.time() - startTime)
    startTime = time.time()

    print("\t Vectorizing xTest...")
    xTest = vectorizer.transform(xTest)

    print "\t Transform on xTest  took:: {}".format(time.time() - startTime)

    return xTrain, xTest


def getOHEFeatures(trainData, testData):
    startTime = time.time()

    trainDataNumeric, testDataNumberic = handleNumericFeaturesForOHE(trainData, testData)
    trainData, testData = performFeatureEngineeringforOHE(trainData, testData)

    print "Finished feature engineering after: {}".format(time.time() - startTime)
    startTime = time.time()

    # splitting data into X and Y
    yTrain = trainData.Category
    trainData = trainData.drop(['Category'], 1)

    # depends whether out "testData" is actually training data
    try:
        yTest = testData.Category
        testData = testData.drop(['Category'], 1)
    except AttributeError:
        yTest = []

    xTrain = trainData.T.to_dict().values()
    xTest = testData.T.to_dict().values()

    xTrain, xTest = vectorizeDataForOHE(xTrain, xTest)

    # re-attach the numeric columns
    xTrain = np.hstack((trainDataNumeric, xTrain))
    xTest = np.hstack((testDataNumberic, xTest))

    return xTrain, yTrain, xTest, yTest
