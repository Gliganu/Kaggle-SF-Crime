
import numpy as np
import DataReader as dataReader
import RegularFeatureExtractor as featureExtractor
import ClassifierSelector as classifierSelector
import glob as glob
import pandas as pd
import os as os
import random as random

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



def getDifferentTrainAndTestData(trainDataSize, testDataSize):

    data = dataReader.getWholeTrainingData()

    if trainDataSize+testDataSize > data.shape[0]: # request more rows than the DF has
        print "Getting different train & test data with possible duplicates"
        trainData = data.sample(trainDataSize)
        testData = data.sample(testDataSize)
    else:
        print "Getting totally different train & test data"
        indexes = np.arange(data.shape[0]) #0->873k
        random.shuffle(indexes) # works in-place

        trainData = data.ix[indexes[0:trainDataSize]]
        testData = data.ix[indexes[trainDataSize+1:trainDataSize+1+testDataSize]]


    return trainData,testData




class InitialClassifierAdapter(object):
    def __init__(self, est):
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict_proba(X)[:, 1][:, np.newaxis]

