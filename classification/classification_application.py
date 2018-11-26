from __future__ import division
import numpy as np
import pandas
import math
import json
import glob
from six.moves import cPickle as pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import Sequential

from ..general.ensemble_functions import *

class Classifier():
    def __init__(self, name, features, inputPipeLoad=True):
        self.ensemble = []
        self.weights = None
        self.inputPipe = None
        self.compileArgs = None
        self.inputFeatures = features
        self.ensemble, self.weights, self.compileArgs, self.inputPipe, _ = load_ensemble(name, inputPipeLoad=inputPipeLoad)
        
    def predict(self, inData):
        return ensemble_predict(self.inputPipe.transform(inData[self.inputFeatures].values.astype('float64')),
            self.ensemble, self.weights, n=len(self.ensemble))[:,0]