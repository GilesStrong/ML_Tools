from __future__ import division
import numpy as np
import pandas
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
            tmp_pred =  model.predict(in_data, verbose=0)
        elif isinstance(model, XGBoostClassifier):
            tmp_pred = model.predict_proba(in_data)[:,1][:, np.newaxis] #Works for one output, might need to be fixed for multiclass
        else:
            print ("MVA not currently supported")
            return None
        if not isinstance(output_pipe, type(None)):
            tmp_pred = output_pipe.inverse_transform(tmp_pred)
        pred += weights[i] * tmp_pred
    return pred

def load_trained_model(cycle, compile_args, mva='NN', load_mode='model', location='train_weights/train_'): 
    cycle = int(cycle)
    model = None
    if mva == 'NN':
        if load_mode == 'model':
            model = load_model(location + str(cycle) + '.h5')
        elif load_mode == 'weights':
            model = model_from_json(open(location + str(cycle) + '.json').read())
            model.load_weights(location + str(cycle) + '.h5')
            model.compile(**compile_args)
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

def assemble_ensemble(results, size, metric, compile_args=None, weighting='reciprocal', mva='NN', load_mode='model', location='train_weights/train_'):
    ensemble = []
    weights = []
    print ("Choosing ensemble by", metric)
    dtype = [('cycle', int), ('result', float)]
    values = np.sort(np.array([(i, result[metric]) for i, result in enumerate(results)], dtype=dtype),
                     order=['result'])
    for i in range(min([size, len(results)])):
        ensemble.append(load_trained_model(values[i]['cycle'], compile_args, mva, load_mode, location))
        weights.append(get_weights(values[i]['result'], metric, weighting))
        print ("Model", i, "is", values[i]['cycle'], "with", metric, "=", values[i]['result'])
    weights = np.array(weights)
    weights = weights/weights.sum() #normalise weights
    return ensemble, weights

def save_ensemble(name, ensemble, weights, compile_args=None, overwrite=False, input_pipe=None, output_pipe=None, save_mode='model'): #Todo add saving of input feature names
    if (len(glob.glob(name + "*.json")) or len(glob.glob(name + "*.h5")) or len(glob.glob(name + "*.pkl"))) and not overwrite:
        print ("Ensemble already exists with that name, call with overwrite=True to force save")
    else:
        os.system("rm " + name + "*.json")
        os.system("rm " + name + "*.h5")
        os.system("rm " + name + "*.pkl")
        save_compile_args = False
        for i, model in enumerate(ensemble):
            if isinstance(model, Sequential) or isinstance(model, Model):
                save_compile_args = True
                if save_mode == 'weights':
                    json_string = model.to_json()
                    open(name + '_' + str(i) + '.json', 'w').write(json_string)
                    model.save_weights(name + '_' + str(i) + '.h5')
                elif save_mode == 'model':
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
        if save_compile_args:
            with open(name + '_compile.json', 'w') as fout:
                json.dump(compile_args, fout)
        with open(name + '_weights.pkl', 'wb') as fout:
            pickle.dump(weights, fout)
        if input_pipe != None:
            with open(name + '_input_pipe.pkl', 'wb') as fout:
                pickle.dump(input_pipe, fout)
        if output_pipe != None:
            with open(name + '_output_pipe.pkl', 'wb') as fout:
                pickle.dump(output_pipe, fout)

def load_ensemble(name, ensemble_size=10, load_input_pipe=False, load_output_pipe=False, load_mode='model'): #Todo add loading of input feature names
    ensemble = []
    weights = None
    input_pipe = None
    output_pipe = None
    compile_args = None
    try:
        with open(name + '_compile.json', 'r') as fin:
            compile_args = json.load(fin)
    except:
        pass
    for i in range(ensemble_size):
        if len(glob.glob(name + "_" + str(i) + '.pkl')): #BDT
            with open(name + '_' + str(i) + '.pkl', 'rb') as fin:   
                model = pickle.load(fin)    
        else: #NN
            if load_mode == 'weights':
                model = model_from_json(open(name + '_' + str(i) + '.json').read())
                model.load_weights(name + "_" + str(i) + '.h5')
            elif load_mode == 'model':
                model = load_model(name + "_" + str(i) + '.h5')
        ensemble.append(model)
    with open(name + '_weights.pkl', 'rb') as fin:
        weights = pickle.load(fin)
    if load_input_pipe:
        with open(name + '_input_pipe.pkl', 'rb') as fin:
            input_pipe = pickle.load(fin)
    if load_output_pipe:
        with open(name + '_output_pipe.pkl', 'rb') as fin:
            output_pipe = pickle.load(fin)
    return ensemble, weights, compile_args, input_pipe, output_pipe

def test_ensemble_auc(X, y, ensemble, weights, size=10):
    for i in range(size):
        pred = ensemble_predict(X, ensemble, weights, n=i+1)
        auc = roc_auc_score(y, pred)
        print ('Ensemble with {} classifiers, AUC = {:2f}'.format(i+1, auc))