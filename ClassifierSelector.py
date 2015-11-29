import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import ClassifierSelector as classifierSelector
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np

def trainSVC(xTrain,yTrain,probability):
    classifier = SVC(probability=probability, kernel='linear', cache_size=5000)
    #todo Olivier: gamma = 0.001 , C = 100
     #todo change cache? no_of_jobs?

    classifier.fit(xTrain, yTrain)

    return classifier

def trainKNeighbors(xTrain, yTrain):
    classifier = KNeighborsClassifier()

    classifier.fit(xTrain, yTrain)

    return classifier

def trainLogisticRegression(xTrain, yTrain):
    classifier = LogisticRegression()

    classifier.fit(xTrain, yTrain)


    return classifier

def trainSGDClassifier(xTrain, yTrain):
    classifier = SGDClassifier()

    #todo change cache? no_of_jobs?
    classifier.fit(xTrain, yTrain)


    return classifier



def trainRandomForest(xTrain, yTrain):

    # 10000/3000 =>  {'n_estimators': 90 'max_features': 0.8, 'max_depth': 9}
    classifier = RandomForestClassifier(n_estimators=90,max_features=0.8, max_depth=9, n_jobs=-1)

    classifier.fit(xTrain, yTrain)

    return classifier

def trainGradientBoostingClassifier(xTrain, yTrain):

    # n_estimators=80, max_depth= 7 !! These were the lowest params from test in GridSearch. Check if lower is possible!
    classifier = GradientBoostingClassifier(n_estimators=60,max_depth=3,min_samples_leaf=1,learning_rate=0.1,max_features=None, verbose=1)

    classifier.fit(xTrain, yTrain)

    return classifier

def trainUsingGridSearch(classifier, paramGrid, xTrain, yTrain):

    classifier = GridSearchCV(classifier, param_grid=paramGrid, cv=3, n_jobs=-1, verbose=1)

    classifier.fit(xTrain, yTrain)

    print("Best choice is: {}".format(classifier.best_params_))

    return classifier

def trainClassifier(xTrain,yTrain):

    print "Training classifier..."

    # classifier = trainKNeighbors(xTrain, yTrain)
    # classifier = trainSGDClassifier(xTrain, yTrain)
    # classifier  = trainSVC(xTrain,yTrain,False)
    # classifier = trainRandomForest(xTrain,yTrain)
    classifier = trainGradientBoostingClassifier(xTrain,yTrain)
    # classifier = trainLogisticRegression(xTrain,yTrain)
    # classifier = trainGradientBoostingClassifier(xTrain,yTrain)


    return classifier