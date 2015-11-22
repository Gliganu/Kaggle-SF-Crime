
import numpy as np
import DataReader as dataReader
import RegularFeatureExtractor as featureExtractor
import ClassifierSelector as classifierSelector
import glob
import pandas as pd

numberOfPartitions = 80

def splitTestDataIntoChunks():

    testData = dataReader.getWholeTestData()

    miniDataFrames = np.array_split(testData, numberOfPartitions)

    for i in range(numberOfPartitions):
        outputFileName = 'data\\miniTestData\\miniDataFrame'+str(i)+'.csv'
        miniDataFrames[i].to_csv(outputFileName,index=False)

def trainClassifierOnTrainingDataReturnAll(numberOfTrainingExamples = -1):

    trainData = dataReader.getTrainData(numberOfTrainingExamples)

    # feature engineering
    trainData =  featureExtractor.convertTargetFeatureToNumeric(trainData)
    xTrain, yTrain = featureExtractor.getRegularFeatures(trainData, True)


     # classifier training
    classifier = classifierSelector.trainClassifier(xTrain, yTrain)

    return classifier, xTrain, yTrain


def getDataFrame(inputFileName):
    return pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True)

def createEnsembleResult():
    fileRegex = "data\\submissions\\*.csv"
    submissionPaths = glob.glob(fileRegex)

    numberOfSubmission = len(submissionPaths)

    dataFrames = [getDataFrame(submissionPath) for submissionPath in submissionPaths]

    # add all the results together
    multipliedDataFrame = pd.DataFrame(reduce(lambda df1, df2: df1.values+df2.values, dataFrames))

    # divide by the number of submissions
    ensembleDataFrame = pd.DataFrame(multipliedDataFrame.values/numberOfSubmission)

    outputFileName = "data\\ensemble\\ensemble.csv"
    ensembleDataFrame.to_csv(outputFileName,index_label="Id")

    return ensembleDataFrame



createEnsembleResult()
