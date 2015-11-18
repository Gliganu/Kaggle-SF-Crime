import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
import time


def getTrainAndTestData(numberOfTrainEx):
    print("Reading data...")

    numberOfTestEx = numberOfTrainEx / 3

    inputFileName = 'data\\train.csv'

    trainData = pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True, nrows=numberOfTrainEx)

    testData = pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True, skiprows=numberOfTrainEx * 10,
                           nrows=numberOfTestEx)

    testData.columns = trainData.columns

    return trainData, testData


def getWholeTrainingData():
    inputFileName = 'data\\train.csv'
    data = pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True)

    return data


def getWholeTestData():
    # inputFileName = 'C:\Users\GligaBogdan\Desktop\Machine Learning\Kaggle\SF Crime\\sampleSubmission.csv'
    inputFileName = 'data\\test.csv'

    data = pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True)

    return data


def writeToCsv(data):
    outputFileName = 'data\\out.csv'

    data.to_csv(outputFileName)


def getCategoryDictionaries():
    categories = "ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS";

    categories = categories.split(",")

    indexes = range(len(categories));

    categToIndexDictionary = {}

    for index, category in zip(indexes, categories):
        categToIndexDictionary[category] = index

    return categToIndexDictionary


def writePredToCsv(yPred):
    print("Writing to csv...")

    categToIndexDictionary = getCategoryDictionaries()

    numberOfEntries = len(yPred)
    numberOfColumns = len(categToIndexDictionary.keys())

    resultMatrix = np.zeros((numberOfEntries, numberOfColumns))

    for index, probabilities in enumerate(yPred):
        resultMatrix[index,:] = probabilities

    data = pd.DataFrame(resultMatrix, range(numberOfEntries), columns=categToIndexDictionary.keys())

    outputFileName = 'data\\out.csv'

    os.remove(outputFileName)
    data.to_csv(outputFileName)


def constructSerializedTrainingDataFrame():
    print("Reading......")
    # dataTrain,_ = getTrainAndTestData(10)
    dataTrain = getWholeTrainingData()

    print("Outputting.....")
    outputFileName = "data\\frames\\trainingDataFrame.pkl"
    joblib.dump(dataTrain, outputFileName)


def constructSerializedTestDataFrame():
    print("Reading......")
    # dataTrain,_ = getTrainAndTestData(10)
    dataTest = getWholeTestData()

    print("Outputting.....")
    outputFileName = "data\\frames\\testDataFrame.pkl"
    joblib.dump(dataTest, outputFileName)


def getSerializedTrainingData():
    print("Getting training data...")
    inputFileName = "data\\frames\\trainingDataFrame.pkl"
    dataRead = joblib.load(inputFileName)

    return dataRead


def getSerializedTestData():
    print("Getting test data...")

    inputFileName = "data\\frames\\testDataFrame.pkl"
    dataRead = joblib.load(inputFileName)

    return dataRead


def getTargetAndTrainData(trainDataSize = -1, testDataSize = -1):

    trainData = getSerializedTrainingData()
    testData = getSerializedTestData()

    if trainDataSize != -1:
        trainData = trainData.sample(trainDataSize)

    if testDataSize != -1:
        testData = testData.sample(testDataSize)


    print "TrainData size is: {}".format(trainData.shape)
    print "TestData size is: {}".format(testData.shape)

    return trainData, testData


def getTargetAndTrainDataWithLabels(trainDataSize = -1, testDataSize = -1):

    trainData = getSerializedTrainingData()
    testData = getSerializedTrainingData()

    if trainDataSize != -1:
        trainData = trainData.sample(trainDataSize)

    if testDataSize != -1:
        testData = testData.sample(testDataSize)


    #df_rest = df.loc[~df.index.isin(df_0.7.index)] // ia tot ce nu ii in aialalta ( to split it ) si la aia iii dai df_0.7 = df.sample(frac=0.7)

    print "TrainData size is: {}".format(trainData.shape)
    print "TestData size is: {}".format(testData.shape)

    return trainData, testData
