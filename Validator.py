from sklearn import metrics
import DataReader as dataReader
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

def performValidation(yPred, yTest):

    dictionary = dataReader.getCategoryDictionaries()
    print(metrics.classification_report(yPred, yTest, target_names=dictionary.keys()))

    # print ("Accuracy:", accuracy_score(yPred,yTest))
    # print ("Recall:", recall_score(yPred,yTest))
    # print ("Precision:", precision_score(yPred,yTest))
    # print ("F1:", f1_score(yPred,yTest))

    #todo show the confusion matrix here ( take from Olivier Grisel)
    #todo plot ROC curve( take from Olivier Grisel)

