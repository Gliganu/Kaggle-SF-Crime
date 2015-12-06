import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve

import DataReader as dataReader
import RegularFeatureExtractor as featureExtractor
import ClassifierSelector as classifierSelector
from sklearn import cross_validation

def plot_learning_curve(estimator, X, y,train_sizes):

    n_jobs = -1

    # cv=3
    # cv = cross_validation.ShuffleSplit(len(X), n_iter=1, test_size=0.1)

    plt.figure()

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,scoring="f1_weighted", cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,verbose=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()

def plot_validation_curve(classifier,xTrain,yTrain,paramName,paramRange):

    train_scores, test_scores = validation_curve(
        classifier, xTrain, yTrain, param_name=paramName, param_range=paramRange,
        cv=3, scoring="f1_weighted", n_jobs=-1,verbose=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(paramName)
    plt.ylabel("Score")

    plt.plot(paramRange, train_scores_mean, 'o-', label="Training score", color="r")
    plt.fill_between(paramRange, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")

    plt.plot(paramRange, test_scores_mean, 'o-', label="Cross-validation score",
                 color="g")
    plt.fill_between(paramRange, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()




def calculateValidationCurve():
    classifier = classifierSelector.constructGradientBoostingClassifier()

    numberOfTrainData = 50000

    trainData = dataReader.getTrainData(numberOfTrainData)

    # feature engineering
    trainData =  featureExtractor.convertTargetFeatureToNumeric(trainData)
    xTrain, yTrain = featureExtractor.getRegularFeatures(trainData, True)

    paramRange = [0.1,0.13,0.16]

    plot_validation_curve(classifier,xTrain,yTrain,"learning_rate",paramRange)


def calculateLearningCurve():
    classifier = classifierSelector.constructGradientBoostingClassifier()
    trainData = dataReader.getTrainData()

    # feature engineering
    trainData =  featureExtractor.convertTargetFeatureToNumeric(trainData)
    xTrain, yTrain = featureExtractor.getRegularFeatures(trainData, True)

    trainSizes =  np.linspace(100000,200000,2,dtype=int)

    plot_learning_curve(classifier,xTrain,yTrain,trainSizes)


def plotFeatureImportance(classifier):

    fx_imp = pd.Series(classifier.feature_importances_)
    fx_imp /= fx_imp.max()  # normalize
    fx_imp.sort()
    fx_imp.plot(kind='barh',figsize=(11,7))
