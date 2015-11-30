import os as os
import pandas as pd
import Utils as utils
import glob as glob
import DataReader as dataReader
import time as time
import MainScript as mainScript
import numpy as np
from joblib import Parallel, delayed


def getDataFrameValues(inputFileName):
    return pd.read_csv(inputFileName, quotechar='"', skipinitialspace=True)


def createEnsembleBasedOnExitingPredictions(fileRegex=None):

    if fileRegex is None:
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

def constructPredictionWithOutput(classifier,classifierIndex,xTest, testBatchIndex):

    print "Predicting with classifier {}".format(classifierIndex)

    yPred = classifier.predict_proba(xTest)

    print "Writing to csv..."
    outputFileName="data\\ensembleTraining\\out"+str(classifierIndex)+".csv"
    dataReader.writePredToCsv(yPred,testBatchIndex,outputFileName=outputFileName)



def createEnsembleBasedODifferentTrainingSets():


    # constructing the limits
    margins = np.linspace(0,878000,5,dtype=int)

    marginTuples=[]
    for i in range(len(margins)-1):
        marginTuples.append((margins[i],margins[i+1]))


    # training classifiers
    allClassifiers = Parallel(n_jobs=-1)(delayed(mainScript.trainClassifierOnTrainingData)(margins=marginTuple) for marginTuple in marginTuples)

    # Predicting on batch test data
    partitionNumber = utils.numberOfPartitions
    for batchIndex in range(partitionNumber):

        print "Predicting batch {}".format(batchIndex)
        miniTestData = dataReader.getSerializedMiniTestData(batchIndex)

        xTest,yTest = mainScript.constructTestData(miniTestData)

        for classifierIndex,currentClassifier in enumerate(allClassifiers):
            constructPredictionWithOutput(currentClassifier,classifierIndex,xTest,batchIndex)


    # post process
    print "Post processing everything..."
    outputFileNames = ["data\\ensembleTraining\\out"+str(index)+".csv" for index in range(len(allClassifiers))]

    for outputFileName in outputFileNames:
        dataReader.postProcessCsv(outputFileName=outputFileName)



    #Merging everything together
    print "Merging all solutions...."
    fileRegex = "data\\ensembleTraining\\*.csv"
    createEnsembleBasedOnExitingPredictions(fileRegex=fileRegex)
