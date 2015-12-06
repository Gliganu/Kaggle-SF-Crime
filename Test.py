import Utils as utils
import os
import Visualizer as visualizer
import EnsembleWorker as worker
import DataReader as dataReader
import MainScript as mainScript

if __name__ == '__main__':
    # visualizer.calculateLearningCurve()
    visualizer.calculateValidationCurve()

    # worker.createEnsembleBasedODifferentTrainingSets()
    # worker.createEnsembleBasedODifferentTrainingSets()

    # mainScript.testGeneralPerformanceUsingCrossValidationScore()
    # mainScript.testParameterPerformance()
    # mainScript.predictForSubmission()

