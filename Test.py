import Utils as utils
import os
import Visualizer as visualizer
import EnsembleWorker as worker
import DataReader as dataReader

if __name__ == '__main__':
    # visualizer.calculateLearningCurve()
    worker.createEnsembleBasedODifferentTrainingSets()
    # data = dataReader.getTrainData(-1,(3,5))