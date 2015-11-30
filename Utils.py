
import numpy as np
import DataReader as dataReader
import RegularFeatureExtractor as featureExtractor
import ClassifierSelector as classifierSelector
import glob as glob
import pandas as pd
import os as os

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


def getDataFrameValues(inputFileName):
    return pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True)

# def multiplyValues(df1,df2):
#     return df1.values + df2.values


def createEnsembleResult():
    fileRegex = "data\\submissions\\*.csv"
    submissionPaths = glob.glob(fileRegex)

    numberOfSubmission = len(submissionPaths)

    targetColumns =  "Id,KIDNAPPING,WEAPON LAWS,SECONDARY CODES,WARRANTS,LOITERING,EMBEZZLEMENT,SUICIDE,DRIVING UNDER THE INFLUENCE,VEHICLE THEFT,ROBBERY,BURGLARY,STOLEN PROPERTY,PORNOGRAPHY/OBSCENE MAT,SUSPICIOUS OCC,ARSON,BRIBERY,FORGERY/COUNTERFEITING,BAD CHECKS,DRUNKENNESS,GAMBLING,OTHER OFFENSES,RECOVERED VEHICLE,FRAUD,FAMILY OFFENSES,DRUG/NARCOTIC,SEX OFFENSES NON FORCIBLE,LARCENY/THEFT,VANDALISM,MISSING PERSON,LIQUOR LAWS,TRESPASS,TREA,SEX OFFENSES FORCIBLE,EXTORTION,ASSAULT,RUNAWAY,NON-CRIMINAL,DISORDERLY CONDUCT,PROSTITUTION"
    targetColumns = targetColumns.split(",")


    print "Creating dataframes..."
    dataFramesValues = [getDataFrameValues(submissionPath) for submissionPath in submissionPaths]

    print "Multiplying dataframes..."
    # add all the results together
    multipliedDataFrame = pd.DataFrame(reduce(lambda df1, df2: df1+df2, dataFramesValues))
    # multipliedDataFrame = pd.DataFrame(reduce(multiplyValues, dataFrames))

    # divide by the number of submissions
    ensembleDataFrame = pd.DataFrame(multipliedDataFrame.values/numberOfSubmission, columns=targetColumns)

    ensembleDataFrame=ensembleDataFrame.drop(['Id'],1)

    print "Outputting..."
    outputFileName = "data\\ensemble\\ensemble.csv"

    if os.path.isfile(outputFileName):
        os.remove(outputFileName)

    ensembleDataFrame.to_csv(outputFileName,index_label="Id")

    return ensembleDataFrame



class InitialClassifierAdapter(object):
    def __init__(self, est):
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict_proba(X)[:, 1][:, np.newaxis]
