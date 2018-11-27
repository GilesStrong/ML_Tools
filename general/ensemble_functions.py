from __future__ import division
import numpy as np
import pandas as pd
import math
import json
import os
import types
from six.moves import cPickle as pickle
import glob

from keras.models import Sequential, Model, model_from_json, load_model

from rep.estimators import XGBoostClassifier

from sklearn.metrics import roc_auc_score

def ensemble_predict(in_data, ensemble, weights, output_pipe=None, n_out=1, n=-1): #Loop though each classifier and predict data class
    if isinstance(in_data, np.ndarray):
        pred = np.zeros((len(in_data), n_out)) #Purely continuous
    else:
        pred = np.zeros((len(in_data[0]), n_out)) # Contains categorical
        
    if n == -1:
        n = len(ensemble)+1
    ensemble = ensemble[0:n] #Use only specified number of classifiers
    weights = weights[0:n]
    weights = weights/weights.sum() #Renormalise weights
    for i, model in enumerate(ensemble):
        if isinstance(model, Sequential) or isinstance(model, Model):
            tempPred =  model.predict(in_data, verbose=0)
        elif isinstance(model, XGBoostClassifier):
            tempPred = model.predict_proba(in_data)[:,1][:, np.newaxis] #Works for one output, might need to be fixed for multiclass
        else:
            print ("MVA not currently supported")
            return None
        if not isinstance(output_pipe, type(None)):
            tempPred = output_pipe.inverse_transform(tempPred)
        pred += weights[i] * tempPred
    return pred

def load_trained_model(cycle, compileArgs, mva='NN', loadMode='model', location='train_weights/train_'): 
    cycle = int(cycle)
    model = None
    if mva == 'NN':
        if loadMode == 'model':
            model = load_model(location + str(cycle) + '.h5')
        elif loadMode == 'weights':
            model = model_from_json(open(location + str(cycle) + '.json').read())
            model.load_weights(location + str(cycle) + '.h5')
            model.compile(**compileArgs)
        else:
            print ("No other loading currently supported")
    else:
        with open(location + str(cycle) + '.pkl', 'r') as fin:   
            model = pickle.load(fin)
    return model

def get_weights(value, metric, weighting='reciprocal'):
    if weighting == 'reciprocal':
        return 1/value
    if weighting == 'uniform':
        return 1
    else:
        print ("No other weighting currently supported")
    return None

def assemble_ensemble(results, size, metric, compileArgs=None, weighting='reciprocal', mva='NN', loadMode='model', location='train_weights/train_'):
    ensemble = []
    weights = []
    print ("Choosing ensemble by", metric)
    dtype = [('cycle', int), ('result', float)]
    values = np.sort(np.array([(i, result[metric]) for i, result in enumerate(results)], dtype=dtype),
                     order=['result'])
    for i in range(min([size, len(results)])):
        ensemble.append(load_trained_model(values[i]['cycle'], compileArgs, mva, loadMode, location))
        weights.append(get_weights(values[i]['result'], metric, weighting))
        print ("Model", i, "is", values[i]['cycle'], "with", metric, "=", values[i]['result'])
    weights = np.array(weights)
    weights = weights/weights.sum() #normalise weights
    return ensemble, weights

def save_ensemble(name, ensemble, weights, compileArgs=None, overwrite=False, inputPipe=None, output_pipe=None, saveMode='model'): #Todo add saving of input feature names
    if (len(glob.glob(name + "*.json")) or len(glob.glob(name + "*.h5")) or len(glob.glob(name + "*.pkl"))) and not overwrite:
        print ("Ensemble already exists with that name, call with overwrite=True to force save")
    else:
        os.system("rm " + name + "*.json")
        os.system("rm " + name + "*.h5")
        os.system("rm " + name + "*.pkl")
        saveCompileArgs = False
        for i, model in enumerate(ensemble):
            if isinstance(model, Sequential) or isinstance(model, Model):
                saveCompileArgs = True
                if saveMode == 'weights':
                    json_string = model.to_json()
                    open(name + '_' + str(i) + '.json', 'w').write(json_string)
                    model.save_weights(name + '_' + str(i) + '.h5')
                elif saveMode == 'model':
                    model.save(name + '_' + str(i) + '.h5')
                else:
                    print ("No other saving currently supported")
                    return None
            elif isinstance(model, XGBoostClassifier):
                with open(name + '_' + str(i) + '.pkl', 'wb') as fout:
                    pickle.dump(model, fout)
            else:
                print ("MVA not currently supported")
                return None
        if saveCompileArgs:
            with open(name + '_compile.json', 'w') as fout:
                json.dump(compileArgs, fout)
        with open(name + '_weights.pkl', 'wb') as fout:
            pickle.dump(weights, fout)
        if inputPipe != None:
            with open(name + '_inputPipe.pkl', 'wb') as fout:
                pickle.dump(inputPipe, fout)
        if output_pipe != None:
            with open(name + '_output_pipe.pkl', 'wb') as fout:
                pickle.dump(output_pipe, fout)

def load_ensemble(name, ensembleSize=10, inputPipeLoad=False, output_pipeLoad=False, loadMode='model'): #Todo add loading of input feature names
    ensemble = []
    weights = None
    inputPipe = None
    output_pipe = None
    compileArgs = None
    try:
        with open(name + '_compile.json', 'r') as fin:
            compileArgs = json.load(fin)
    except:
        pass
    for i in range(ensembleSize):
        if len(glob.glob(name + "_" + str(i) + '.pkl')): #BDT
            with open(name + '_' + str(i) + '.pkl', 'rb') as fin:   
                model = pickle.load(fin)    
        else: #NN
            if loadMode == 'weights':
                model = model_from_json(open(name + '_' + str(i) + '.json').read())
                model.load_weights(name + "_" + str(i) + '.h5')
            elif loadMode == 'model':
                model = load_model(name + "_" + str(i) + '.h5')
        ensemble.append(model)
    with open(name + '_weights.pkl', 'rb') as fin:
        weights = pickle.load(fin)
    if inputPipeLoad:
        with open(name + '_inputPipe.pkl', 'rb') as fin:
            inputPipe = pickle.load(fin)
    if output_pipeLoad:
        with open(name + '_output_pipe.pkl', 'rb') as fin:
            output_pipe = pickle.load(fin)
    return ensemble, weights, compileArgs, inputPipe, output_pipe

def test_ensemble_auc(X, y, ensemble, weights, size=10):
    for i in range(size):
        pred = ensemble_predict(X, ensemble, weights, n=i+1)
        auc = roc_auc_score(y, pred)
        print ('Ensemble with {} classifiers, AUC = {:2f}'.format(i+1, auc))