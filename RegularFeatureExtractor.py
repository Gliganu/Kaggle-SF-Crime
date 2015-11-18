import time
import numpy as np

def replaceDiscreteFeaturesWithNumericalOnes(trainData, testData):
    discreteColumns = ['DayOfWeek', 'PdDistrict']

    for column in discreteColumns:

        entryDictionary = {}
        id = 1

        for index, entry in enumerate(trainData.get(column)):

            if entry not in entryDictionary:
                entryDictionary[entry] = id
                id += 1

            trainData.get(column).values[index] = entryDictionary.get(entry)


        for index, entry in enumerate(testData.get(column)):

            if entry not in entryDictionary:
                entryDictionary[entry] = id
                id += 1

            testData.get(column).values[index] = entryDictionary.get(entry)



    return trainData, testData


def performRegularFeatureEngineering(trainData, testData):
    print("Performing feature engineering...")

    # drop certain columns
    trainData = trainData.drop(['Descript', 'Address', 'Resolution'],
                               1)  # descript must be droped because there is no descript in test


    # depends if out testData is actually training data
    if 'Id' in testData.columns.values:
        testData = testData.drop(['Address', 'Id'], 1)
    else:
        testData = testData.drop(['Address','Descript','Resolution'], 1)

    # get only the year from date
    trainData.Dates = [date.split("-")[0] for date in trainData.Dates]
    testData.Dates = [date.split("-")[0] for date in testData.Dates]

    trainData, testData = replaceDiscreteFeaturesWithNumericalOnes(trainData, testData)

    return trainData, testData


def getRegularFeatures(trainData, testData):
    startTime = time.time()

    trainData, testData = performRegularFeatureEngineering(trainData, testData)

    print "Finished feature engineering after: {}".format(time.time() - startTime)

    # splitting data into X and Y
    yTrain = trainData.Category
    trainData = trainData.drop(['Category'], 1)
    xTrain = trainData.values

    try:
        yTest = testData.Category
        testData = testData.drop(['Category'], 1)
    except AttributeError:
        yTest = []

    xTest = testData.values

    print("Train data features: {}".format(trainData.columns.values))
    print("Test data features: {}".format(testData.columns.values))

    return xTrain, yTrain, xTest, yTest
