from __future__ import division
import numpy as np
import json
import os
from six.moves import cPickle as pickle
import glob
import warnings
from fastprogress import progress_bar, master_bar
from pathlib import Path
import timeit
from typing import Dict, Union, Any, List, Optional
from sklearn.pipeline import Pipeline
from keras.models import Model, model_from_json, load_model
import h5py

from .activations import Swish
from .fold_yielder import FoldYielder


class Ensemble():
    def __init__(self, input_pipe:Pipeline=None, output_pipe:Pipeline=None):
        self.input_pipe,self.output_pipe = input_pipe,output_pipe
        self.models = []
        self.weights = []
        self.size = 0
        
    def add_input_pipe(self, pipe:Pipeline) -> None:
        self.input_pipe = pipe

    def add_output_pipe(self, pipe:Pipeline) -> None:
        self.output_pipe = pipe
    
    @staticmethod
    def load_trained_model(model_id, compile_args:Dict[str,Any]=None,
                           load_mode:str='model', name:str='train_weights/train_') -> Model: 
        if load_mode == 'model':
            model = load_model(f'{name}{model_id}.h5', custom_objects={'Swish': Swish}, compile=False)
        elif load_mode == 'weights':
            model = model_from_json(open(f'{name}{model_id}.json')).read()
            model.load_weights(f'{name}{model_id}.h5')
            if compile_args is not None: model.compile(**compile_args)
        else:
            raise ValueError("No other loading currently supported")
        return model
    
    @staticmethod
    def _get_weights(value:float, metric:str, weighting='reciprocal') -> float:
        if 'metric'.lower() == 'ams': value = 1/value
        if   weighting == 'reciprocal': return 1/value
        elif weighting == 'uniform':    return 1
        else: raise ValueError("No other loading currently supported")

    def build_ensemble(self, results:List[Dict[str,float]], size:int,
                       metric:str='loss', weighting:str='reciprocal',
                       snapshot_args:Dict[str,Any]={}, compile_args:Dict[str,Any]=None,
                       load_mode:str='model', location:Path=Path('train_weights'), verbose:bool=True) -> None:

        cycle_losses     = None if 'cycle_losses'     not in snapshot_args else snapshot_args['cycle_losses']
        n_cycles         = None if 'n_cycles'         not in snapshot_args else snapshot_args['n_cycles']
        load_cycles_only = None if 'load_cycles_only' not in snapshot_args else snapshot_args['load_cycles_only']
        patience         = 2    if 'patience'         not in snapshot_args else snapshot_args['patience']
        weighting_pwr    = 0    if 'weighting_pwr'    not in snapshot_args else snapshot_args['weighting_pwr']    
    
        if (cycle_losses is not None and n_cycles is None) or (cycle_losses is None and n_cycles is not None):
            warnings.warn("Warning: cycle ensembles requested, but not enough information passed")
        if cycle_losses is not None and n_cycles is not None and metric is not 'loss':
            warnings.warn("Warning: Setting ensemble metric to loss")
            metric = 'loss'
        if cycle_losses is not None and n_cycles is not None and weighting is not 'uniform':
            warnings.warn("Warning: Setting model weighting to uniform")
            weighting = 'uniform'
    
        self.models = []
        weights = []
    
        if verbose: print("Choosing ensemble by", metric)
        dtype = [('model', int), ('result', float)]
        values = np.sort(np.array([(i, result[metric]) for i, result in enumerate(results)], dtype=dtype), order=['result'])
    
        for i in progress_bar(range(min([size, len(results)]))):
            if not (load_cycles_only and n_cycles):
                self.models.append(self.load_trained_model(values[i]['model'], compile_args, load_mode, location/'train_'))
                weights.append(self._get_weights(values[i]['result'], metric, weighting))

                if verbose: print("Model", i, "is", values[i]['model'], "with", metric, "=", values[i]['result'])

            if n_cycles:
                end_cycle = len(cycle_losses[values[i]['model']]) - patience
                if load_cycles_only:
                    end_cycle += 1

                for n, c in enumerate(range(end_cycle, max(0, end_cycle - n_cycles), -1)):
                    self.models.append(self.load_trained_model(c, compile_args, load_mode, location/f'{values[i]["model"]}_cycle_'))
                    weights.append((n + 1)**weighting_pwr)

                    if verbose: print("Model", i, "cycle", c, "has", metric, "=", cycle_losses[values[i]['model']][c], 'and weight', weights[-1])
        
        weights = np.array(weights)
        self.weights = weights/weights.sum()  # normalise weights
        self.size = len(self.models)
        self.n_out = self.models[0].layers[-1].output_shape[1]
        self.compile_args = compile_args
        self.results = results
    
    @staticmethod
    def save_fold_pred(pred:np.ndarray, fold:int, datafile:h5py.File, pred_name:str='pred') -> None:
        try: datafile.create_dataset(f'{fold}/{pred_name}', shape=pred.shape, dtype='float32')
        except RuntimeError: pass

        datafile[f'{fold}/{pred_name}'][...] = pred
        
    def predict_array(self, in_data:Union[List[np.ndarray], np.ndarray], n_models:Optional[int]=None,
                      parent_bar:Optional[master_bar]=None) -> np.ndarray:
        if isinstance(in_data, np.ndarray):
            pred = np.zeros((len(in_data), self.n_out))  # Purely continuous
        else:
            pred = np.zeros((len(in_data[0]), self.n_out))  # Contains categorical
        
        n_models = len(self.models)+1 if n_models is None else n_models
        ensemble = self.models[:n_models]
        weights = self.weights[:n_models]
        weights = weights/weights.sum()  # Renormalise weights
        
        for i, m in enumerate(progress_bar(ensemble, parent=parent_bar)):
            tmp_pred = m.predict(in_data, verbose=0)
            if self.output_pipe is not None:
                tmp_pred = self.output_pipe.inverse_transform(tmp_pred)
            pred += weights[i] * tmp_pred
            
        return pred
    
    def fold_predict(self, fold_yielder:FoldYielder, n_models:Optional[int]=None, pred_name:str='pred') -> None:
        n_models = self.models if n_models is None else n_models
        
        times = []
        mb = master_bar(range(len(fold_yielder.source)))
        for fold_id in mb:
            fold_tmr = timeit.default_timer()

            if not fold_yielder.test_time_aug:
                fold = fold_yielder.get_fold(fold_id)['inputs']
                pred = self.predict_array(fold, n_models, mb)

            else:
                tmpPred = []
                for aug in range(fold_yielder.aug_mult):  # Multithread this?
                    fold = fold_yielder.get_test_fold(fold_id, aug)['inputs']
                    tmpPred.append(self.predict_array(fold, n_models, mb))
                pred = np.mean(tmpPred, axis=0)

            times.append((timeit.default_timer()-fold_tmr)/len(fold))

            if self.n_out > 1:
                self.save_fold_pred(pred, f'fold_{fold_id}', fold_yielder.source, pred_name=pred_name)
            else:
                self.save_fold_pred(pred[:, 0], f'fold_{fold_id}', fold_yielder.source, pred_name=pred_name)
        print(f'Mean time per event = {np.mean(times):.4E}±{np.std(times, ddof=1)/np.sqrt(len(times)):.4E}')

    def predict(self, in_data:Union[np.ndarray, FoldYielder, List[np.ndarray]], n_models:Optional[int]=None,
                pred_name:str='pred') -> Union[None, np.ndarray]:
        if not isinstance(in_data, FoldYielder): return self.predict_array(in_data, n_models)
        self.fold_predict(in_data, n_models, pred_name)
    
    def save(self, name:str, feats:List[str]=None, overwrite:bool=False, save_mode:str='model') -> None:
        if (len(glob.glob(f"{name}*.json")) or len(glob.glob(f"{name}*.h5")) or len(glob.glob(f"{name}*.pkl"))) and not overwrite:
            raise FileExistsError("Ensemble already exists with that name, call with overwrite=True to force save")
        else:
            os.makedirs(name, exist_ok=True)
            os.system(f"rm {name}*.json {name}*.h5 {name}*.pkl")
            
            for i, model in enumerate(progress_bar(self.models)):
                if save_mode == 'weights':
                    open(f'{name}_{i}.json', 'w').write(model.to_json())
                    model.save_weights(f'{name}_{i}.h5')
                elif save_mode == 'model':
                    model.save(f'{name}_{i}.h5', include_optimizer=False)
                else:
                    raise ValueError("No other saving currently supported")
                    
            with open(f'{name}_weights.pkl', 'wb') as fout: pickle.dump(self.weights, fout)
            with open(f'{name}_results.pkl', 'wb') as fout: pickle.dump(self.results, fout)
            if self.compile_args is not None: 
                with open(f'{name}_compile.json', 'w')     as fout: json.dump(self.compile_args, fout)
            if self.input_pipe   is not None: 
                with open(f'{name}_input_pipe.pkl', 'wb')  as fout: pickle.dump(self.input_pipe, fout)
            if self.output_pipe  is not None: 
                with open(f'{name}_output_pipe.pkl', 'wb') as fout: pickle.dump(self.output_pipe, fout)
            if feats             is not None: 
                with open(f'{name}_feats.pkl', 'wb')       as fout: pickle.dump(feats, fout)
                    
    def load(self, name:str, load_mode:str='model') -> None:
        self.models = []
        
        for n in progress_bar(glob.glob(f'{name}_*.h5')):
            if load_mode == 'weights':
                m = model_from_json(open(f'{n[:n.rfind(".")]}.json').read())
                m.load_weights(n)
            elif load_mode == 'model':
                m = load_model(n, custom_objects={'Swish': Swish}, compile=False)
            self.models.append(m)
            self.n_out = self.models[0].layers[-1].output_shape[1]
            
        with     open(f'{name}_weights.pkl', 'rb')     as fin: self.weights      = pickle.load(fin)
        try: 
            with open(f'{name}_compile.json', 'r')     as fin: self.compile_args = json.load(fin)
        except FileNotFoundError: pass
        try: 
            with open(f'{name}_input_pipe.pkl', 'rb')  as fin: self.input_pipe   = pickle.load(fin)
        except FileNotFoundError: pass
        try: 
            with open(f'{name}_output_pipe.pkl', 'rb') as fin: self.output_pipe  = pickle.load(fin)
        except FileNotFoundError: pass
        try: 
            with open(f'{name}_feats.pkl', 'rb')       as fin: self.feats        = pickle.load(fin)
        except FileNotFoundError: pass
