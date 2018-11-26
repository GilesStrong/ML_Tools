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
- Change callbacks for nicer interface e.g. pass arguments in dictionary
- Make it possible to annealMomentum without anealing LR
- Change classifier/regressor to class? Could use static methods to still provide flxibility for prototyping
- Tidy code and move to PEP 8
- Add docstrings and stuff
'''

def get_batch(index, datafile):
    print ("Depreciated, use to moving a FoldYielder class")
    index = str(index)
    weights = None
    if 'fold_' + index + '/weights' in datafile:
        weights = np.array(datafile['fold_' + index + '/weights'])
    return {'inputs':np.array(datafile['fold_' + index + '/inputs']),
            'targets':np.array(datafile['fold_' + index + '/targets']),
            'weights':weights}

def get_folds(n, n_splits):
    train = [x for x in range(n_splits) if x != n]
    shuffle(train)
    test = n
    return train, test

def fold_lr_find(fold_yielder,
                model_gen, model_gen_params,
                train_params, train_on_weights=True,
                lrBounds=[1e-5, 10], verbose=False, nFolds=-1):

    start = timeit.default_timer()
    binary = None
    
    if not isinstance(fold_yielder, FoldYielder):
        print ("HDF5 as input is depreciated, converting to FoldYielder")
        fold_yielder = FoldYielder(fold_yielder)
    
    if nFolds < 1:
        indeces = range(fold_yielder.nFolds)
    else:
        indeces = range(nFolds)
    
    lrFinders = []
    for index in indeces:
        model = None
        model = model_gen(**model_gen_params)
        model.reset_states #Just checking
    
        trainbatch = fold_yielder.get_batch(np.random.choice(range(fold_yielder.nFolds))) #Load fold
        nSteps = math.ceil(len(trainbatch['targets'])/train_params['batch_size'])
        if verbose: print ("Using {} steps".format(nSteps))   

        lrFinder = LRFinder(nSteps=nSteps, lrBounds=lrBounds, verbose=verbose)

        if 'class' in model_gen_params['mode'].lower():
            if binary == None: #Check classification mode
                binary = True
                nClasses = len(np.unique(trainbatch['targets']))
                if nClasses > 2:
                    print (nClasses, "classes found, running in multiclass mode\n")
                    trainbatch['targets'] = utils.to_categorical(trainbatch['targets'], num_classes=nClasses)
                    binary = False
                else:
                    print (nClasses, "classes found, running in binary mode\n")

        if train_on_weights:
            model.fit(trainbatch['inputs'], trainbatch['targets'],
                      sample_weight=trainbatch['weights'],
                      callbacks = [lrFinder], **train_params) #Train for one epoch

        else:
            model.fit(trainbatch['inputs'], trainbatch['targets'],
                      class_weight = 'auto',callbacks = [lrFinder], **train_params) #Train for one epoch
        
        lrFinders.append(lrFinder)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    if nFolds != 1:
        get_lr_finder_mean_plot(lrFinders, loss='loss', cut=-10)
    else:
        lrFinders[0].plot_lr()    
        lrFinders[0].plot(n_skip=10)
    print("______________________________________\n")
        
    return lrFinders

def save_fold_pred(batchPred, fold, datafile, predName='pred'):
    try:
        datafile.create_dataset(fold + "/" + predName, shape=batchPred.shape, dtype='float32')
    except RuntimeError:
        pass
    
    pred = datafile[fold + "/" + predName]
    pred[...] = batchPred
        
def fold_ensemble_predict(ensemble, weights, fold_yielder, predName='pred', nOut=1, outputPipe=None, ensembleSize=None, nFolds=-1, verbose=False):
    if isinstance(ensembleSize, type(None)):
        ensembleSize = len(ensemble)

    if not isinstance(fold_yielder, FoldYielder):
        print ("Passing HDF5 as input is depreciated, converting to FoldYielder")
        fold_yielder = FoldYielder(fold_yielder)

    if nFolds < 0:
        nFolds = len(fold_yielder.source)

    for fold in range(nFolds):
        if verbose:
            print ('Predicting batch {} out of {}'.format(fold+1, nFolds))
            start = timeit.default_timer()

        if not fold_yielder.testTimeAug:
            batch = fold_yielder.get_batch(fold)['inputs']
            batchPred = ensemble_predict(batch, ensemble, weights, n=ensembleSize, nOut=nOut, outputPipe=outputPipe)

        else:
            tmpPred = []
            for aug in range(fold_yielder.augMult): #Multithread this?
                batch = fold_yielder.getTestBatch(fold, aug)['inputs']
                tmpPred.append(ensemble_predict(batch, ensemble, weights, n=ensembleSize, nOut=nOut, outputPipe=outputPipe))
            batchPred = np.mean(tmpPred, axis=0)

        if verbose: 
            print ("Prediction took {}s per sample\n".format((timeit.default_timer() - start)/len(batch)))

        if nOut > 1:
            save_fold_pred(batchPred, 'fold_' + str(fold), fold_yielder.source, predName=predName)
        else:
            save_fold_pred(batchPred[:,0], 'fold_' + str(fold), fold_yielder.source, predName=predName)
        
def get_feature(feature, datafile, nFolds=-1, ravel=True, setFold=-1):
    if setFold < 0:
        data = []
        for i, fold in enumerate(datafile):
            if i >= nFolds and nFolds > 0:
                break
            data.append(np.array(datafile[fold + '/' + feature]))
            
        data = np.concatenate(data)
    else:
         data = np.array(datafile['fold_' + str(setFold) + '/' + feature])
    if ravel:
        return data.ravel()
    return data

def fold_train_model(fold_yielder, n_splits,
                      model_gen, model_gen_params, train_params, use_callbacks=[],
                      train_on_weights=True, patience=10, max_epochs=10000, 
                      plots=[], ams_size=0, saveloc=Path('train_weights/'),
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

    for fold in range(n_splits):
        foldStart = timeit.default_timer()
        print ("Running fold", fold+1, "/", n_splits)
        os.system("rm {saveloc}best.h5")
        best = -1
        bestLR = -1
        redux_decay_active = False
        tmp_patience = patience
        epochCounter = 0
        sub_epoch = 0
        stop = False
        lossHistory = {'val_loss':[], 'swa_val_loss':[]}
        trainID, testID = get_folds(fold, n_splits) #Get fold indeces for training and testing for current fold

        model = None
        model = model_gen(**model_gen_params)
        model.reset_states #Just checking
        
        testbatch = fold_yielder.get_batch(testID) #Load testing fold

        callbacks = []
        cycling = False
        redux_decay = False
        lr_cycler = None
        mom_cycler = None

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
                lr_cycler = CosAnnealLR(**{'nb':nb, **use_callbacks['CosAnnealLR']})
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
                    swa = SWA(swa_start, testbatch, model_gen(**model_gen_params), verbose,
                              use_callbacks['SWA']['renewal'], lr_cycler, train_on_weights=train_on_weights, replace=use_callbacks['SWA']['replace'])
                else:
                    swa = SWA(swa_start, testbatch, model_gen(**model_gen_params), verbose,
                              use_callbacks['SWA']['renewal'], train_on_weights=train_on_weights, replace=use_callbacks['SWA']['replace'])
                callbacks.append(swa)
        use_swa = False

        for epoch in range(max_epochs):
            for n in trainID: #Loop through training folds
                trainbatch = fold_yielder.get_batch(n) #Load fold data
                sub_epoch += 1
                
                if binary == None: #First run, check classification mode
                    binary = True
                    nClasses = len(np.unique(trainbatch['targets']))
                    if nClasses > 2:
                        print (nClasses, "classes found, running in multiclass mode\n")
                        trainbatch['targets'] = utils.to_categorical(trainbatch['targets'], num_classes=nClasses)
                        binary = False
                    else:
                        print (nClasses, "classes found, running in binary mode\n")

                if train_on_weights:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
                              sample_weight=trainbatch['weights'],
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
                        loss = model.evaluate(testbatch['inputs'], testbatch['targets'], sample_weight=testbatch['weights'], verbose=0)
                    
                else:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
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
                        loss = model.evaluate(testbatch['inputs'], testbatch['targets'], verbose=0)
                
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
                            bestLR = lr_cycler.lrs[-1]
                        else:
                            bestLR = lr_cycler.lrs[-2]
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
                        print ('CosineAnneal stalling after {} epochs, entering redux decay at LR={}'.format(sub_epoch, bestLR))
                        model.load_weights(saveloc/"best.h5")
                        lr_cycler.lrs.append(bestLR)
                        K.set_value(model.optimizer.lr, bestLR)
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
            testbatch = fold_yielder.get_batch(testID) #Load testing fold
            prediction = model.predict(testbatch['inputs'], verbose=0)
            if not isinstance(testbatch['weights'], type(None)):
                results[-1]['wAUC'] = 1-roc_auc_score(testbatch['targets'],
                                                      prediction,
                                                      sample_weight=testbatch['weights'])
            results[-1]['AUC'] = 1-roc_auc_score(testbatch['targets'], prediction)

            if ams_size:
                 results[-1]['AMS'], results[-1]['cut'] = ams_scan_quick(fold_yielder.getBatchDF(testID, preds=prediction, weightName='orig_weights'), wFactor=ams_size/len(prediction))
        print ("Score is:", results[-1])

        if 'lr' in plots: lr_cycler.plot()
        if 'mom' in plots: mom_cycler.plot()

        print("Fold took {:.3f}s\n".format(timeit.default_timer() - foldStart))

        model.save(saveloc +  'train_' + str(fold) + '.h5')
        with open(saveloc +  'resultsFile.pkl', 'wb') as fout: #Save results
            pickle.dump(results, fout)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    if 'history' in plots: plot_training_history(histories, save=saveloc + 'loss_history.png')
    for score in results[0]:
        mean = uncert_round(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print ("Mean", score, "= {} +- {}".format(mean[0], mean[1]))
    print("______________________________________\n")
                      
    if logoutput:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories
