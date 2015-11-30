
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


# def multiplyValues(df1,df2):
#     return df1.values + df2.values





class InitialClassifierAdapter(object):
    def __init__(self, est):
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict_proba(X)[:, 1][:, np.newaxis]
