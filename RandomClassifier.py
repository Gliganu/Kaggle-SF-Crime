import DataReader as dr
from random import randint
import gensim


def predict():
    data = dr.getData()

    values = data.values

    valueSize = values[0].size

    for value in values:
        randomIndex = randint(1,valueSize-1)
        value[38] = 0 # reset it to 0
        value[randomIndex] = 1
        # print(value)

    dr.writeToCsv(data)

predict()
