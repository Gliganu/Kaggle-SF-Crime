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
    cv = cross_validation.ShuffleSplit(len(X), n_iter=1, test_size=0.3)

    plt.figure()

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,scoring="log_loss", cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,verbose=1)
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
        cv=3, scoring="log_loss", n_jobs=-1,verbose=1)

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

    trainSizes =  np.linspace(100000,500000,5,dtype=int)

    plot_learning_curve(classifier,xTrain,yTrain,trainSizes)


def plotFeatureImportance(classifier):

    featureNames = dataReader.deserializeObject("data\\misc\\columns.csv")

    feature_importance = classifier.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos,featureNames[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
