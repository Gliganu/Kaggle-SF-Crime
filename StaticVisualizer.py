import DataReader as dataReader
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import RegularFeatureExtractor as featureExtractor

def plotCrimeCategoryVsNumberOfOccurences():

    trainData = dataReader.getSerializedTrainingData()


    people = "ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS".split(",")

    y_pos = np.arange(len(people))
    performance = trainData.groupby('Category').count().Address.values

    plt.barh(y_pos, performance, align='center', alpha=0.4)
    plt.yticks(y_pos, people)
    plt.xlabel('Number of occurrences')
    plt.xlabel('Categories of crimes')
    plt.title('Crime distribution')

    plt.show()


def pltCrimeCategoryVsYear():

    trainData = dataReader.getSerializedTrainingData()
    trainData = featureExtractor.performDateFeatureEngineering(trainData)

    years = np.linspace(2003,2015,13)

    y_pos = np.arange(len(years))

    performance = trainData.groupby('Year').count().Address.values

    plt.plot(y_pos, performance, alpha=0.4)
    plt.xticks(y_pos, years)
    plt.xlabel('Year')
    plt.ylabel('Number of occurrences')
    plt.title('Crime distribution / year')

    plt.show()


def pltCrimeCategoryVsDayOfWeek():
    trainData = dataReader.getSerializedTrainingData()
    trainData = featureExtractor.performDateFeatureEngineering(trainData)

    years = "SUNDAY,MONDAY,TUESDAY,WEDNESDAY,THURSDAY,FRIDAY,SATURDAY".split(",")

    y_pos = np.arange(len(years))

    performance = trainData.groupby('DayOfWeek').count().Address.values

    plt.bar(y_pos, performance,align='center', alpha=0.4)
    plt.xticks(y_pos, years)
    plt.xlabel('Day of Week')
    plt.ylabel('Number of occurrences')
    plt.title('Crime distribution / year')

    plt.show()


def scatterPlotBasedOnGeolocationData():
    trainData = dataReader.getTrainData(1000)

    trainData = featureExtractor.convertTargetFeatureToNumeric(trainData)
    xData = trainData.X
    yData = trainData.Y

    crimeCategory = trainData.Category

    plt.scatter(xData,yData,c=crimeCategory)
    plt.colorbar()
    plt.show()


