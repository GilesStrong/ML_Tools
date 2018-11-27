from __future__ import division

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from keras import utils

from six.moves import cPickle as pickle
import timeit
import types
import numpy as np
import os
from random import shuffle
import sys
from pathlib import Path

from .misc_functions import uncert_round
from ..plotting_and_evaluation.plotters import *
from .ensemble_functions import *
from .callbacks import *
from .metrics import *
from .fold_yielder import FoldYielder

'''
Todo:
- Make it possible to anneal Momentum without anealing LR
- Change classifier/regressor to class? Could use static methods to still provide flxibility for prototyping
- Add docstrings and stuff
- Pass abitrary metrics
'''

def get_train_folds(n, n_splits):
    train = [x for x in range(n_splits) if x != n]
    #shuffle(train)
    return train

def fold_lr_find(fold_yielder,
                model_gen, model_gen_params,
                train_params, train_on_weights=True,
                lrBounds=[1e-5, 10], verbose=False, n_folds=-1):

    start = timeit.default_timer()
    binary = None
    
    if not isinstance(fold_yielder, FoldYielder):
        print ("HDF5 as input is depreciated, converting to FoldYielder")
        fold_yielder = FoldYielder(fold_yielder)
    
    if n_folds < 1:
        indeces = range(fold_yielder.n_folds)
    else:
        indeces = range(min(n_folds, fold_yielder.n_folds))
    
    lr_finders = []
    for index in indeces:
        model = None
        model = model_gen(**model_gen_params)
        model.reset_states #Just checking
    
        trainfold = fold_yielder.get_fold(index) #Load fold
        nSteps = math.ceil(len(trainfold['targets'])/train_params['batch_size'])
        if verbose: print ("Using {} steps".format(nSteps))   

        lrFinder = LRFinder(nSteps=nSteps, lrBounds=lrBounds, verbose=verbose)

        if 'class' in model_gen_params['mode'].lower():
            if binary == None: #Check classification mode
                binary = True
                nClasses = len(np.unique(trainfold['targets']))
                if nClasses > 2:
                    print (nClasses, "classes found, running in multiclass mode\n")
                    trainfold['targets'] = utils.to_categorical(trainfold['targets'], num_classes=nClasses)
                    binary = False
                else:
                    print (nClasses, "classes found, running in binary mode\n")

        if train_on_weights:
            model.fit(trainfold['inputs'], trainfold['targets'],
                      sample_weight=trainfold['weights'],
                      callbacks = [lrFinder], **train_params) #Train for one epoch

        else:
            model.fit(trainfold['inputs'], trainfold['targets'],
                      class_weight = 'auto',callbacks = [lrFinder], **train_params) #Train for one epoch
        
        lr_finders.append(lrFinder)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    if n_folds != 1:
        get_lr_finder_mean_plot(lr_finders, loss='loss', cut=-10)
    else:
        lr_finders[0].plot_lr()    
        lr_finders[0].plot(n_skip=10)
    print("______________________________________\n")
        
    return lr_finders

def save_fold_pred(foldPred, fold, datafile, pred_name='pred'):
    try:
        datafile.create_dataset(fold + "/" + pred_name, shape=foldPred.shape, dtype='float32')
    except RuntimeError:
        pass
    
    pred = datafile[fold + "/" + pred_name]
    pred[...] = foldPred
        
def fold_ensemble_predict(ensemble, weights, fold_yielder, pred_name='pred', n_out=1, output_pipe=None, ensemble_size=None, n_folds=-1, verbose=False):
    if isinstance(ensemble_size, type(None)):
        ensemble_size = len(ensemble)

    if not isinstance(fold_yielder, FoldYielder):
        print ("Passing HDF5 as input is depreciated, converting to FoldYielder")
        fold_yielder = FoldYielder(fold_yielder)

    if n_folds < 0:
        n_folds = len(fold_yielder.source)

    for fold_id in range(n_folds):
        if verbose:
            print ('Predicting fold {} out of {}'.format(fold_id+1, n_folds))
            start = timeit.default_timer()

        if not fold_yielder.test_time_aug:
            fold = fold_yielder.get_fold(fold_id)['inputs']
            foldPred = ensemble_predict(fold, ensemble, weights, n=ensemble_size, n_out=n_out, output_pipe=output_pipe)

        else:
            tmpPred = []
            for aug in range(fold_yielder.aug_mult): #Multithread this?
                fold = fold_yielder.get_test_fold(fold_id, aug)['inputs']
                tmpPred.append(ensemble_predict(fold, ensemble, weights, n=ensemble_size, n_out=n_out, output_pipe=output_pipe))
            foldPred = np.mean(tmpPred, axis=0)

        if verbose: 
            print ("Prediction took {}s per sample\n".format((timeit.default_timer() - start)/len(fold)))

        if n_out > 1:
            save_fold_pred(foldPred, 'fold_' + str(fold_id), fold_yielder.source, pred_name=pred_name)
        else:
            save_fold_pred(foldPred[:,0], 'fold_' + str(fold_id), fold_yielder.source, pred_name=pred_name)
        
def get_feature(feature, datafile, n_folds=-1, ravel=True, set_fold=-1):
    if set_fold < 0:
        data = []
        for i, fold in enumerate(datafile):
            if i >= n_folds and n_folds > 0:
                break
            data.append(np.array(datafile[fold + '/' + feature]))
            
        data = np.concatenate(data)
    else:
         data = np.array(datafile['fold_' + str(set_fold) + '/' + feature])
    if ravel:
        return data.ravel()
    return data

def fold_train_model(fold_yielder, n_models,
                      model_gen, model_gen_params, train_params, use_callbacks={},
                      train_on_weights=True, patience=10, max_epochs=10000, 
                      plots=['history'], ams_args={'n_total':0, 'br':0, 'deltaB':0}, saveloc=Path('train_weights/'),
                      logoutput=False, verbose=False):
    
    os.makedirs(f"mkdir {saveloc}", exist_ok=True)
    os.system(f"rm {saveloc}/*.h5")
    os.system(f"rm {saveloc}/*.json")
    os.system(f"rm {saveloc}/*.pkl")
    os.system(f"rm {saveloc}/*.png")
    os.system(f"rm {saveloc}/*.log")
    
    if logoutput:
        old_stdout = sys.stdout
        log_file = open(saveloc/'training_log.log', 'w')
        sys.stdout = log_file

    start = timeit.default_timer()
    results = []
    histories = []
    if 'class' in model_gen_params['mode'].lower():
        binary = None
    else:
        binary = False

    if not isinstance(fold_yielder, FoldYielder):
        print ("HDF5 as input is depreciated, converting to FoldYielder")
        fold_yielder = FoldYielder(fold_yielder)

    if len(use_callbacks):
        nb = math.ceil(len(fold_yielder.source['fold_0/targets'])/train_params['batch_size'])

    if train_on_weights: print ("Training using weights")

    n_folds = fold_yielder.n_folds

    for model_num, test_id in enumerate([4]):#enumerate(np.random.choice(range(n_folds), size=n_models, replace=False)):
        model_start = timeit.default_timer()
        print("Training model", model_num+1, "/", n_models, "\tTesting on fold", test_id)
        os.system(f"rm {saveloc}best.h5")
        best = -1
        best_lr = -1
        redux_decay_active = False
        tmp_patience = patience
        epochCounter = 0
        sub_epoch = 0
        stop = False
        lossHistory = {'val_loss':[], 'swa_val_loss':[]}
        train_ids = get_train_folds(test_id, n_folds) #Get fold indeces for training and testing for current fold
        print(train_ids)

        model = None
        model = model_gen(**model_gen_params)
        model.reset_states #Just checking
        
        testfold = fold_yielder.get_fold(test_id) #Load testing fold

        callbacks = []
        cycling = False
        redux_decay = False
        lr_cycler = None
        mom_cycler = None
        swa_start = -1

        if 'OneCycle' in use_callbacks:
            print("Using 1-cycle")
            lr_cycler = OneCycle(**{'nb':nb, **use_callbacks['OneCycle']})
            callbacks.append(lr_cycler)

        else:
            if 'LinearCLR' in use_callbacks:
                print("Using linear LR cycle")
                cycling = True
                lr_cycler = LinearCLR(**{'nb':nb, **use_callbacks['LinearCLR']})
                callbacks.append(lr_cycler)

            elif 'CosAnnealLR' in use_callbacks:
                print("Using cosine LR annealing")
                cycling = True
                redux_decay = use_callbacks['CosAnnealLR']['redux_decay']
                lr_cycler = CosAnnealLR(nb, use_callbacks['CosAnnealLR']['cycle_mult'], use_callbacks['CosAnnealLR']['reverse'])
                callbacks.append(lr_cycler)

            if 'LinearCMom' in use_callbacks:
                print("Using linear momentum cycle")
                cycling = True
                mom_cycler = LinearCMom(**{'nb':nb, **use_callbacks['LinearCMom']})
                callbacks.append(mom_cycler)

            elif 'CosAnnealMom' in use_callbacks:
                print("Using cosine momentum annealing")
                cycling = True
                mom_cycler = CosAnnealMomentum(**{'nb':nb, **use_callbacks['CosAnnealMom']})
                callbacks.append(mom_cycler)
  
            if 'SWA' in use_callbacks:
                swa_start = use_callbacks['SWA']['start']
                if cycling:
                    swa = SWA(swa_start, testfold, model_gen(**model_gen_params), verbose,
                              use_callbacks['SWA']['renewal'], lr_cycler, train_on_weights=train_on_weights, replace=use_callbacks['SWA']['replace'])
                else:
                    swa = SWA(swa_start, testfold, model_gen(**model_gen_params), verbose,
                              use_callbacks['SWA']['renewal'], train_on_weights=train_on_weights, replace=use_callbacks['SWA']['replace'])
                callbacks.append(swa)
        use_swa = False

        for epoch in range(max_epochs):
            for train_id in train_ids: #Loop through training folds
                trainfold = fold_yielder.get_fold(train_id) #Load fold data
                sub_epoch += 1
                print(train_id)
                
                if binary == None: #First run, check classification mode
                    binary = True
                    nClasses = len(np.unique(trainfold['targets']))
                    if nClasses > 2:
                        print (nClasses, "classes found, running in multiclass mode\n")
                        trainfold['targets'] = utils.to_categorical(trainfold['targets'], num_classes=nClasses)
                        binary = False
                    else:
                        print (nClasses, "classes found, running in binary mode\n")

                if train_on_weights:
                    model.fit(trainfold['inputs'], trainfold['targets'],
                              sample_weight=trainfold['weights'],
                              callbacks=callbacks, **train_params) #Train for one epoch

                    if swa_start >= 0 and swa.active:
                        losses = swa.get_losses()
                        print('{} swa loss {}, default loss {}'.format(sub_epoch, losses['swa'], losses['base']))
                        if losses['swa'] < losses['base']:
                            loss = losses['swa']
                            use_swa = True
                        else:
                            loss = losses['base']
                            use_swa = False
                        
                    else:
                        loss = model.evaluate(testfold['inputs'], testfold['targets'], sample_weight=testfold['weights'], verbose=0)
                    
                else:
                    model.fit(trainfold['inputs'], trainfold['targets'],
                              class_weight='auto',
                              callbacks=callbacks, **train_params) #Train for one epoch
                    
                    if swa_start >= 0 and swa.active:
                        losses = swa.get_losses()
                        print('{} swa loss {}, default loss {}'.format(sub_epoch, losses['swa'], losses['base']))
                        if losses['swa'] < losses['base']:
                            loss = losses['swa']
                            use_swa = True
                        else:
                            loss = losses['base']
                            use_swa = False
                    else:
                        loss = model.evaluate(testfold['inputs'], testfold['targets'], verbose=0)
                
                if swa_start >= 0 and swa.active and cycling:
                    print ("{} SWA loss:", sub_epoch, loss)
                
                if swa_start >= 0:
                    if swa.active:
                        lossHistory['swa_val_loss'].append(losses['swa'])
                        lossHistory['val_loss'].append(losses['base'])
                    else:
                        lossHistory['swa_val_loss'].append(loss)
                        lossHistory['val_loss'].append(loss)
                else:
                    lossHistory['val_loss'].append(loss)        

                if loss <= best or best < 0: #Save best
                    best = loss
                    if cycling:
                        if lr_cycler.lrs[-1] > 0:
                            best_lr = lr_cycler.lrs[-1]
                        else:
                            best_lr = lr_cycler.lrs[-2]
                    epochCounter = 0
                    if swa_start >= 0 and swa.active and use_swa:
                        swa.test_model.save_weights(saveloc/"best.h5")
                    else:
                        model.save_weights(saveloc/"best.h5")
                    if redux_decay_active:
                        lr_cycler.lrs.append(float(K.get_value(model.optimizer.lr)))
                    if verbose:
                        print ('{} New best found: {}'.format(sub_epoch, best))
                elif cycling and not redux_decay_active:
                    if lr_cycler.cycle_end:
                        epochCounter += 1
                else:
                    epochCounter += 1
                    if redux_decay_active:
                        lr = 0.8*float(K.get_value(model.optimizer.lr))
                        lr_cycler.lrs.append(lr)
                        K.set_value(model.optimizer.lr, lr)

                if epochCounter >= tmp_patience: #Early stopping
                    if cycling and redux_decay and not redux_decay_active:
                        print ('CosineAnneal stalling after {} epochs, entering redux decay at LR={}'.format(sub_epoch, best_lr))
                        model.load_weights(saveloc/"best.h5")
                        lr_cycler.lrs.append(best_lr)
                        K.set_value(model.optimizer.lr, best_lr)
                        tmp_patience = 10
                        epochCounter = 0
                        callbacks = []
                        redux_decay_active = True
                    else:
                        if verbose:
                            print ('Early stopping after {} epochs'.format(sub_epoch))
                        stop = True
                        break
            
            if stop:
                break

        model.load_weights(saveloc/"best.h5")

        histories.append({})
        histories[-1]['val_loss'] = lossHistory['val_loss']
        if swa_start >= 0:
            histories[-1]['swa_val_loss'] = lossHistory['swa_val_loss']
        
        results.append({})
        results[-1]['loss'] = best
        if binary:
            testfold = fold_yielder.get_fold(test_id) #Load testing fold
            prediction = model.predict(testfold['inputs'], verbose=0)
            if not isinstance(testfold['weights'], type(None)):
                results[-1]['wAUC'] = 1-roc_auc_score(testfold['targets'],
                                                      prediction,
                                                      sample_weight=testfold['weights'])
            results[-1]['AUC'] = 1-roc_auc_score(testfold['targets'], prediction)

            if ams_args['n_total'] > 0:
                 results[-1]['AMS'], results[-1]['cut'] = ams_scan_quick(fold_yielder.get_fold_df(test_id, preds=prediction, weight_name='orig_weights'), wFactor=ams_args['n_total']/len(prediction), br=ams_args['br'], deltaB=ams_args['deltaB'])
        print ("Score is:", results[-1])

        if 'lr' in plots: lr_cycler.plot()
        if 'mom' in plots: mom_cycler.plot()

        print("Fold took {:.3f}s\n".format(timeit.default_timer() - model_start))

        model.save(str(saveloc/('train_' + str(model_num) + '.h5')))
        with open(saveloc/'resultsFile.pkl', 'wb') as fout: #Save results
            pickle.dump(results, fout)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    if 'history' in plots: plot_training_history(histories, save=saveloc/'loss_history.png')
    for score in results[0]:
        mean = uncert_round(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print ("Mean", score, "= {} +- {}".format(mean[0], mean[1]))
    print("______________________________________\n")
                      
    if logoutput:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories
