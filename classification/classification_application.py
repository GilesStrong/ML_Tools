from __future__ import division

from ..general.ensemble_functions import ensemble_predict, load_ensemble


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
                                self.ensemble, self.weights, n=len(self.ensemble))[:, 0]
            