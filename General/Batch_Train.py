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

from .Misc_Functions import uncertRound
from ..plotting_and_evaluation.Plotters import *
from .Ensemble_Functions import *
from .Callbacks import *
from .Metrics import *
from .BatchYielder import BatchYielder

'''
Todo:
- Change callbacks for nicer interface e.g. pass arguments in dictionary
- Make it possible to annealMomentum without anealing LR
- Change classifier/regressor to class? Could use static methods to still provide flxibility for prototyping
- Tidy code and move to PEP 8
- Add docstrings and stuff
'''

def getBatch(index, datafile):
    print ("Depreciated, use to moving a BatchYielder class")
    index = str(index)
    weights = None
    if 'fold_' + index + '/weights' in datafile:
        weights = np.array(datafile['fold_' + index + '/weights'])
    return {'inputs':np.array(datafile['fold_' + index + '/inputs']),
            'targets':np.array(datafile['fold_' + index + '/targets']),
            'weights':weights}

def getFolds(n, nSplits):
    train = [x for x in range(nSplits) if x != n]
    #shuffle(train)
    return train

def batchLRFind(batchYielder,
                modelGen, modelGenParams,
                trainParams, trainOnWeights=True,
                lrBounds=[1e-5, 10], verbose=False, nFolds=-1):

    start = timeit.default_timer()
    binary = None
    
    if not isinstance(batchYielder, BatchYielder):
        print ("HDF5 as input is depreciated, converting to BatchYielder")
        batchYielder = BatchYielder(batchYielder)
    
    if nFolds < 1:
        indeces = range(batchYielder.nFolds)
    else:
        indeces = range(nFolds)
    
    lrFinders = []
    for index in indeces:
        model = None
        model = modelGen(**modelGenParams)
        model.reset_states #Just checking
    
        trainbatch = batchYielder.getBatch(np.random.choice(range(batchYielder.nFolds))) #Load fold
        nSteps = math.ceil(len(trainbatch['targets'])/trainParams['batch_size'])
        if verbose: print ("Using {} steps".format(nSteps))   

        lrFinder = LRFinder(nSteps=nSteps, lrBounds=lrBounds, verbose=verbose)

        if 'class' in modelGenParams['mode'].lower():
            if binary == None: #Check classification mode
                binary = True
                nClasses = len(np.unique(trainbatch['targets']))
                if nClasses > 2:
                    print (nClasses, "classes found, running in multiclass mode\n")
                    trainbatch['targets'] = utils.to_categorical(trainbatch['targets'], num_classes=nClasses)
                    binary = False
                else:
                    print (nClasses, "classes found, running in binary mode\n")

        if trainOnWeights:
            model.fit(trainbatch['inputs'], trainbatch['targets'],
                      sample_weight=trainbatch['weights'],
                      callbacks = [lrFinder], **trainParams) #Train for one epoch

        else:
            model.fit(trainbatch['inputs'], trainbatch['targets'],
                      class_weight = 'auto',callbacks = [lrFinder], **trainParams) #Train for one epoch
        
        lrFinders.append(lrFinder)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    if nFolds != 1:
        getLRFinderMeanPlot(lrFinders, loss='loss', cut=-10)
    else:
        lrFinders[0].plot_lr()    
        lrFinders[0].plot(n_skip=10)
    print("______________________________________\n")
        
    return lrFinders

def saveBatchPred(batchPred, fold, datafile, predName='pred'):
    try:
        datafile.create_dataset(fold + "/" + predName, shape=batchPred.shape, dtype='float32')
    except RuntimeError:
        pass
    
    pred = datafile[fold + "/" + predName]
    pred[...] = batchPred
        
def batchEnsemblePredict(ensemble, weights, batchYielder, predName='pred', nOut=1, outputPipe=None, ensembleSize=None, nFolds=-1, verbose=False):
    if isinstance(ensembleSize, type(None)):
        ensembleSize = len(ensemble)

    if not isinstance(batchYielder, BatchYielder):
        print ("Passing HDF5 as input is depreciated, converting to BatchYielder")
        batchYielder = BatchYielder(batchYielder)

    if nFolds < 0:
        nFolds = len(batchYielder.source)

    for fold in range(nFolds):
        if verbose:
            print ('Predicting batch {} out of {}'.format(fold+1, nFolds))
            start = timeit.default_timer()

        if not batchYielder.testTimeAug:
            batch = batchYielder.getBatch(fold)['inputs']
            batchPred = ensemblePredict(batch, ensemble, weights, n=ensembleSize, nOut=nOut, outputPipe=outputPipe)

        else:
            tmpPred = []
            for aug in range(batchYielder.augMult): #Multithread this?
                batch = batchYielder.getTestBatch(fold, aug)['inputs']
                tmpPred.append(ensemblePredict(batch, ensemble, weights, n=ensembleSize, nOut=nOut, outputPipe=outputPipe))
            batchPred = np.mean(tmpPred, axis=0)

        if verbose: 
            print ("Prediction took {}s per sample\n".format((timeit.default_timer() - start)/len(batch)))

        if nOut > 1:
            saveBatchPred(batchPred, 'fold_' + str(fold), batchYielder.source, predName=predName)
        else:
            saveBatchPred(batchPred[:,0], 'fold_' + str(fold), batchYielder.source, predName=predName)
        
def getFeature(feature, datafile, nFolds=-1, ravel=True, setFold=-1):
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

def batchTrainClassifier(batchYielder, nSplits, modelGen, modelGenParams, trainParams,
                         cosAnnealMult=0, reverseAnneal=False, plotLR=False, reduxDecay=False,
                         annealMomentum=False, reverseAnnealMomentum=False, plotMomentum=False,
                         oneCycle=False, ratio=0.25, reverse=False, lrScale=10, momScale=0.1, plotOneCycle=False, scale=30, mode='sgd',
                         swaStart=-1, swaRenewal=-1, sgdReplacement=False,
                         trainOnWeights=True,
                         saveLoc='train_weights/', patience=10, maxEpochs=10000,
                         verbose=False, logoutput=False, amsSize=0, plot=True):
    print("Depreciated, move to using batchTrainModel")

def batchTrainModel(batchYielder, n_models,
                    modelGen, modelGenParams, trainParams, 
                    use_callbacks={},
                    trainOnWeights=True, patience=10, maxEpochs=10000,
                    plots=['history'], ams_args={'n_total':0, 'br':0, 'deltaB':0},
                    saveloc=Path('train_weights'), verbose=False, logoutput=False):
    
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
    if 'class' in modelGenParams['mode'].lower():
        binary = None
    else:
        binary = False

    if not isinstance(batchYielder, BatchYielder):
        print ("HDF5 as input is depreciated, converting to BatchYielder")
        batchYielder = BatchYielder(batchYielder)

    if trainOnWeights: print ("Training using weights")
    
    n_folds = batchYielder.nFolds
    nb = math.ceil(len(batchYielder.source['fold_0/targets'])/trainParams['batch_size'])

    for model_num, test_id in enumerate([4]):#enumerate(np.random.choice(range(n_folds), size=n_models, replace=False)):
        model_start = timeit.default_timer()
        print ("Running fold", model_num+1, "/", n_models)
        os.system(f"rm {saveloc}/best.h5")
        best = -1
        bestLR = -1
        reduxDecayActive = False
        tmpPatience = patience
        epochCounter = 0
        subEpoch = 0
        stop = False
        lossHistory = {'val_loss':[], 'swa_val_loss':[]}
        trainID = getFolds(test_id, n_folds) #Get fold indeces for training and testing for current fold

        model = None
        model = modelGen(**modelGenParams)
        model.reset_states #Just checking
        
        testbatch = batchYielder.getBatch(test_id) #Load testing fold

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
                lr_cycler = CosAnneal(nb, use_callbacks['CosAnnealLR']['cycle_mult'], use_callbacks['CosAnnealLR']['reverse'])
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
                    swa = SWA(swa_start, testbatch, modelGen(**modelGenParams), verbose,
                              use_callbacks['SWA']['renewal'], lr_cycler, trainOnWeights=trainOnWeights, sgdReplacement=use_callbacks['SWA']['replace'])
                else:
                    swa = SWA(swa_start, testbatch, modelGen(**modelGenParams), verbose,
                              use_callbacks['SWA']['renewal'], trainOnWeights=trainOnWeights, sgdReplacement=use_callbacks['SWA']['replace'])
                callbacks.append(swa)
        use_swa = False

        print(trainID)
        for epoch in range(maxEpochs):
            
            for n in trainID: #Loop through training folds
                trainbatch = batchYielder.getBatch(n) #Load fold data
                subEpoch += 1
                print(n)
                if binary == None: #First run, check classification mode
                    binary = True
                    nClasses = len(np.unique(trainbatch['targets']))
                    if nClasses > 2:
                        print (nClasses, "classes found, running in multiclass mode\n")
                        trainbatch['targets'] = utils.to_categorical(trainbatch['targets'], num_classes=nClasses)
                        binary = False
                    else:
                        print (nClasses, "classes found, running in binary mode\n")

                if trainOnWeights:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
                              sample_weight=trainbatch['weights'],
                              callbacks = callbacks, **trainParams) #Train for one epoch

                    if swa_start >= 0 and swa.active:
                        losses = swa.get_losses()
                        print('{} swa loss {}, default loss {}'.format(subEpoch, losses['swa'], losses['base']))
                        if losses['swa'] < losses['base']:
                            loss = losses['swa']
                            useSWA = True
                        else:
                            loss = losses['base']
                            useSWA = False
                        
                    else:
                        loss = model.evaluate(testbatch['inputs'], testbatch['targets'], sample_weight=testbatch['weights'], verbose=0)
                    
                else:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
                              class_weight = 'auto',
                              callbacks = callbacks, **trainParams) #Train for one epoch
                    
                    if swa_start >= 0 and swa.active:
                        losses = swa.get_losses()
                        print('{} swa loss {}, default loss {}'.format(subEpoch, losses['swa'], losses['base']))
                        if losses['swa'] < losses['base']:
                            loss = losses['swa']
                            useSWA = True
                        else:
                            loss = losses['base']
                            useSWA = False
                    else:
                        loss = model.evaluate(testbatch['inputs'], testbatch['targets'], verbose=0)
                
                if swa_start >= 0 and swa.active and cycling and use_callbacks['CosAnnealLR']['cycle_mult'] > 1:
                    print ("{} SWA loss:", subEpoch, loss)
                
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
                    if swa_start >= 0 and swa.active and useSWA:
                        swa.test_model.save_weights(saveloc/"best.h5")
                    else:
                        model.save_weights(saveloc/"best.h5")
                    if reduxDecayActive:
                        lr_cycler.lrs.append(float(K.get_value(model.optimizer.lr)))
                    if verbose:
                        print ('{} New best found: {}'.format(subEpoch, best))
                elif cycling and not reduxDecayActive:
                    if lr_cycler.cycle_end:
                        epochCounter += 1
                else:
                    epochCounter += 1
                    if reduxDecayActive:
                        lr = 0.8*float(K.get_value(model.optimizer.lr))
                        lr_cycler.lrs.append(lr)
                        K.set_value(model.optimizer.lr, lr)

                if epochCounter >= tmpPatience: #Early stopping
                    if cycling and redux_decay and not reduxDecayActive:
                        print ('CosineAnneal stalling after {} epochs, entering redux decay at LR={}'.format(subEpoch, bestLR))
                        model.load_weights(saveloc/"best.h5")
                        lr_cycler.lrs.append(bestLR)
                        K.set_value(model.optimizer.lr, bestLR)
                        tmpPatience = 10
                        epochCounter = 0
                        callbacks = []
                        reduxDecayActive = True
                    else:
                        if verbose:
                            print ('Early stopping after {} epochs'.format(subEpoch))
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
            testbatch = batchYielder.getBatch(test_id) #Load testing fold
            prediction = model.predict(testbatch['inputs'], verbose=0)
            if not isinstance(testbatch['weights'], type(None)):
                results[-1]['wAUC'] = 1-roc_auc_score(testbatch['targets'],
                                                      prediction,
                                                      sample_weight=testbatch['weights'])
            results[-1]['AUC'] = 1-roc_auc_score(testbatch['targets'], prediction)

            if ams_args['n_total'] > 0:
                 results[-1]['AMS'], results[-1]['cut'] = amsScanQuick(batchYielder.getTestDF(test_id, preds=prediction, weightName='orig_weights'), wFactor=ams_args['n_total']/len(prediction), br=ams_args['br'], deltaB=ams_args['deltaB'])
        
        print ("Score is:", results[-1])

        if 'lr' in plots: lr_cycler.plot_lr()
        if 'mom' in plots: mom_cycler.plot_momentum()

        print("Fold took {:.3f}s\n".format(timeit.default_timer() - model_start))

        model.save(str(saveloc/('train_' + str(model_num) + '.h5')))
        with open(saveloc/'resultsFile.pkl', 'wb') as fout: #Save results
            pickle.dump(results, fout)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    if 'history' in plots: plotTrainingHistory(histories, save=saveloc/'loss_history.png')
    for score in results[0]:
        mean = uncertRound(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print ("Mean", score, "= {} +- {}".format(mean[0], mean[1]))
    print("______________________________________\n")
                      
    if logoutput:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories
