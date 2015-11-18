import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import ClassifierSelector as classifierSelector
from sklearn.grid_search import GridSearchCV



def trainSVC(xTrain,yTrain):
    classifier = SVC()

    # FOR 10 000 -> 9 neighbors
    # print("Tuning parameter for training...")
    # tuning the hyper-parameters
    # n_neighbors = np.arange(1, 10,2)
    # classifier = GridSearchCV(classifier, param_grid={'n_neighbors': n_neighbors}, cv=5)

    print("Training...")
    classifier.fit(xTrain, yTrain)

    # print("Best choice is: {}".format(classifier.best_params_))
    return classifier

def trainKNeighbors(xTrain, yTrain):
    classifier = KNeighborsClassifier()
    # classifier = SVC()

    # FOR 10 000 -> 9 neighbors
    # print("Tuning parameter for training...")
    # tuning the hyper-parameters
    # n_neighbors = np.arange(1, 10,2)
    # classifier = GridSearchCV(classifier, param_grid={'n_neighbors': n_neighbors}, cv=5)

    print("Training...")
    classifier.fit(xTrain, yTrain)

    # print("Best choice is: {}".format(classifier.best_params_))


    return classifier

def trainLogisticRegression(xTrain, yTrain):
    classifier = LogisticRegression()
    # classifier = SVC()

    # FOR 10 000 -> 9 neighbors
    # print("Tuning parameter for training...")
    # tuning the hyper-parameters
    # n_neighbors = np.arange(1, 10,2)
    # classifier = GridSearchCV(classifier, param_grid={'n_neighbors': n_neighbors}, cv=5)

    print("Training...")
    classifier.fit(xTrain, yTrain)

    # print("Best choice is: {}".format(classifier.best_params_))


    return classifier


def trainRandomForest(xTrain, yTrain):
    verbose = 1
    n_jobs = 1

    rf = RandomForestClassifier(90) # best classif on 10 000

    # n_trees = range(50, 150, 20)
    # rf = GridSearchCV(estimator=rf, param_grid={'n_estimators': n_trees}, cv=5)

    print("Training...")
    startTime = time.time()

    rf.fit(xTrain, yTrain)

    print("Training took: {}".format(time.time() - startTime))
    # print("Best parameters: {}".format(rf.best_params_))
    return rf


def getClassifier(xTrain,yTrain):
      # classifier = trainKNeighbors(xTrain, yTrain)
    # classifier = trainSVC(xTrain, yTrain)
    classifier = trainRandomForest(xTrain,yTrain)

    return classifier