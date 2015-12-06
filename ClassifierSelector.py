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
from sklearn.cross_validation import StratifiedKFold

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
    classifier = constructRandomForestClassifier()
    classifier.fit(xTrain, yTrain)

    return classifier



def constructRandomForestClassifier():

    # 10000/3000 =>  {'n_estimators': 90 'max_features': 0.8, 'max_depth': 9}
    classifier = RandomForestClassifier(n_estimators=90,max_features=0.8, max_depth=9, n_jobs=-1)

    return classifier


def constructGradientBoostingClassifier():

    # n_estimators = 120, learning_rate = 0.07
    # max_features= 0.5, max_depth= 6
    # subsample = 0.9
    classifier = GradientBoostingClassifier(n_estimators=120,max_depth=6,min_samples_leaf=1,learning_rate=0.07,max_features=0.5, verbose=1)
    # classifier = GradientBoostingClassifier(verbose=1)


    return classifier

def trainGradientBoostingClassifier(xTrain, yTrain):

    classifier = constructGradientBoostingClassifier()

    paramGrid = {
        "learning_rate":[0.1,0.13,0.16],
    }

    classifier = trainUsingGridSearch(classifier,paramGrid,xTrain,yTrain)
    # classifier.fit(xTrain, yTrain)

    return classifier

def trainUsingGridSearch(classifier, paramGrid, xTrain, yTrain):

    cv = StratifiedKFold(yTrain,n_folds=3)

    classifier = GridSearchCV(classifier, scoring="f1_weighted", param_grid=paramGrid, cv=cv, n_jobs=-1, verbose=1)

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