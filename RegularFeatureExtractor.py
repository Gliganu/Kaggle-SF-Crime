import time
import numpy as np
import DataReader as dataReader

def replaceDiscreteFeaturesWithNumericalOnes(data):
    discreteColumns = ['DayOfWeek', 'PdDistrict']

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

def performRegularFeatureEngineering(data, isTrainData):
    print("Performing feature engineering...")


    # trainData
    if isTrainData:
        data = data.drop(['Descript', 'Address', 'Resolution'], 1)

    # testData
    else:
          # depends if out testData is actually training data
        if 'Id' in data.columns.values:
            data = data.drop(['Address', 'Id'], 1)
        else:
            data = data.drop(['Address','Descript','Resolution'], 1)

    # ( trainData & testData)
    data = performDateFeatureEngineering(data)


    # map using integer dictionary ( trainData & testData)
    data = replaceDiscreteFeaturesWithNumericalOnes(data)

    return data

def convertTargetFeatureToNumeric(data):
    categoryDictionary = dataReader.getCategoryDictionaries()
    data = data.replace(categoryDictionary.keys(), range(len(categoryDictionary.keys())))

    return data



def getRegularFeatures(data, isTrainData):

    data = performRegularFeatureEngineering(data, isTrainData)

    # splitting data into X and Y
    # trainData
    if isTrainData:
        yData =  data.Category
        data = data.drop(['Category'], 1)
        xData = data.values

    #testData
    else:
        try:
            yData = data.Category
            data = data.drop(['Category'], 1)
        except AttributeError:
            yData = []

        xData = data.values


    print "Features used {}".format(data.columns.values)
    return xData,yData
