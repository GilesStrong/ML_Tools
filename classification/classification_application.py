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
    def __init__(self, name, features, load_input_pipe=True):
        self.ensemble = []
        self.weights = None
        self.input_pipe = None
        self.compile_args = None
        self.input_features = features
        self.ensemble, self.weights, self.compile_args, self.input_pipe, _ = load_ensemble(name, load_input_pipe=load_input_pipe)
        
    def predict(self, inData):
        return ensemble_predict(self.input_pipe.transform(inData[self.input_features].values.astype('float64')),
            self.ensemble, self.weights, n=len(self.ensemble))[:,0]
            