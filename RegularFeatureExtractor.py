import time
import numpy as np
import DataReader as dataReader
import datetime
import pandas as pd

def replaceDiscreteFeaturesWithNumericalOnes(data,discreteColumns):


    for column in discreteColumns:

        entryDictionary = {}
        id = 1

        for index, entry in enumerate(data.get(column)):

            if entry not in entryDictionary:
                entryDictionary[entry] = id
                id += 1

            data.get(column).values[index] = entryDictionary.get(entry)


    return data


def performDateFeatureEngineering(data):

    # Add Year,Month,Day Columns
    data['Year'] = data['Dates'].map(lambda entryDate: entryDate.split("-")[0] )
    data['Month'] = data['Dates'].map(lambda entryDate: entryDate.split("-")[1] )
    data['Day'] = data['Dates'].map(lambda entryDate: entryDate.split("-")[2].split(" ")[0] )
    data['Hour'] = data['Dates'].map(lambda entryDate: entryDate.split(" ")[1].split(":")[0] )

    # Delete old "Dates" column
    data = data.drop('Dates',1)

    return data


def getStreetName(fullAddress):
    addressWords = fullAddress.split(" ")
    return addressWords[len(addressWords)-2]

def performAddressFeatureEngineering(data):
    data['Address']=  data['Address'].map(getStreetName)

    # data = data.drop(['Address'],1)

    return data

def addTimeFeature(data):

    data['Time'] = data['Dates']
    data['Time'] =   pd.to_datetime(data.Time).map(lambda entry: (entry-datetime.datetime(1970,1,1)).total_seconds())

    return data

# TODO FIX THIS so that isTrainData is no longer needed

def performPdDistrictOHEFeatureEngineering(data):
    data = pd.concat([data,pd.get_dummies(data['PdDistrict'], prefix='Police_District')], axis=1)
    data = pd.concat([data,pd.get_dummies(data['DayOfWeek'], prefix='Day')], axis=1)
    data = data.drop(['PdDistrict','DayOfWeek'],1)

    return data


def performRegularFeatureEngineering(data, isTrainData):
    print("Performing feature engineering...")


    # trainData
    if isTrainData:
        data = data.drop(['Descript','Resolution'], 1)

    # testData
    else:
          # depends if out testData is actually training data
        if 'Id' in data.columns.values:
            data = data.drop(['Id'], 1)
        else:
            data = data.drop(['Descript','Resolution'], 1)


    # ( trainData & testData)
    # data = addTimeFeature(data)
    data = performDateFeatureEngineering(data)
    data = performAddressFeatureEngineering(data)
    # data = performPdDistrictOHEFeatureEngineering(data)

    # discreteColumns = ['Address']
    discreteColumns = ['Address', 'DayOfWeek', 'PdDistrict']

    # map using integer dictionary ( trainData & testData)
    data = replaceDiscreteFeaturesWithNumericalOnes(data,discreteColumns)

    return data

def convertTargetFeatureToNumeric(data):
    categoryDictionary = dataReader.getCategoryDictionaries()
    data = data.replace(categoryDictionary.keys(), range(len(categoryDictionary.keys())))

    return data



def getRegularFeatures(data, isTrainData):

    data = performRegularFeatureEngineering(data, isTrainData)

    # splitting data into X and Y
    if 'Category' in data.columns.values:
        yData =  data.Category
        data = data.drop(['Category'], 1)
    else:
        yData = []

    xData = data.values


    dataReader.serializeObject(data.columns.values,"data\\misc\\columns.csv")

    print "Features used {}".format(data.columns.values)
    return xData,yData
