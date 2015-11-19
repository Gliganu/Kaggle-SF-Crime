
import numpy as np
import DataReader as dataReader

numberOfPartitions = 80

def splitTestDataIntoChunks():

    testData = dataReader.getWholeTestData()

    miniDataFrames = np.array_split(testData, numberOfPartitions)

    for i in range(numberOfPartitions):
        outputFileName = 'data\\miniTestData\\miniDataFrame'+str(i)+'.csv'
        miniDataFrames[i].to_csv(outputFileName,index=False)

