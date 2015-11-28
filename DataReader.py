import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
import random
import Utils as utils
import RegularFeatureExtractor as featureExtractor

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
    inputFileName = 'data\\test.csv'

    data = pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True)

    return data

def getSuffixDataFrame():
    inputFileName = 'data\\suffixDataFrameForTesting.csv'

    data = pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True)

    return data


def writeToCsv(data,):
    outputFileName = 'data\\out.csv'

    data.to_csv(outputFileName)

def getSerializedMiniTestData(index):
    inputFileName = 'data\\miniTestData\\miniDataFrame'+str(index)+'.csv'

    print "\n Mini test batch:"+str(index)
    return pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True)

def getRandomMiniTestData():

    randomIndex = random.randint(0,utils.numberOfPartitions)

    return getSerializedMiniTestData(randomIndex)



def getCategoryDictionaries():
    categories = "ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS"

    categories = categories.split(",")

    indexes = range(len(categories))

    categToIndexDictionary = {}

    for index, category in zip(indexes, categories):
        categToIndexDictionary[category] = index

    return categToIndexDictionary


def writePredToCsv(yPred,miniTestDataIndex):
    print("Writing to csv...")

    categToIndexDictionary = getCategoryDictionaries()

    numberOfEntries = len(yPred)
    numberOfColumns = len(categToIndexDictionary.keys())

    resultMatrix = np.zeros((numberOfEntries, numberOfColumns))

    for index, predictions in enumerate(yPred):
        resultMatrix[index,:] = predictions[1]

    data = pd.DataFrame(resultMatrix, range(numberOfEntries), columns=categToIndexDictionary.keys())

    outputFileName = 'data\\out.csv'
    if miniTestDataIndex == 0:

        if os.path.isfile(outputFileName):
            os.remove(outputFileName)

        data.to_csv(outputFileName)
    else:
        with open(outputFileName, 'a') as f:
            data.to_csv(f, header=False)


def constructSerializedTrainingDataFrame():
    print("Reading......")
    dataTrain = getWholeTrainingData()

    print("Outputting.....")
    outputFileName = "data\\frames\\trainingDataFrame.pkl"
    joblib.dump(dataTrain, outputFileName)


def constructSerializedTestDataFrame():
    print("Reading......")
    dataTest = getWholeTestData()

    print("Outputting.....")
    outputFileName = "data\\frames\\testDataFrame.pkl"
    joblib.dump(dataTest, outputFileName)


def getSerializedTrainingData():
    inputFileName = "data\\frames\\trainingDataFrame.pkl"
    dataRead = joblib.load(inputFileName)

    return dataRead


def getSerializedTestData():
    print("Getting test data...")

    inputFileName = "data\\frames\\testDataFrame.pkl"
    dataRead = joblib.load(inputFileName)

    return dataRead



def getTrainData(trainDataSize = -1):

    print("Getting training data: "+str(trainDataSize))

    trainData = getSerializedTrainingData()

    if trainDataSize != -1:
        trainData = trainData.sample(trainDataSize)

    print "TrainData size is: {}".format(trainData.shape)

    return trainData

def getTestData(testDataSize = -1, withLabels = True):

    if withLabels:
        testData = getSerializedTrainingData()
    else:
        testData = getSerializedTestData()

    if testDataSize != -1:
        testData = testData.sample(testDataSize)

    print "TestData size is: {}".format(testData.shape)

    return testData


def postProcessCsv():
    # this is needed to add remove old mini-batch-id and add global id
    outputFileName = "data\\out.csv"

    data = pd.read_csv(outputFileName, quotechar='"', skipinitialspace=True)
    data = data.drop(data.columns[0],1)

    # removing the bad indexed version
    os.remove(outputFileName)

    data.to_csv(outputFileName,index_label="Id")



