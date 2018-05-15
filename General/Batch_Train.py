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

from ML_Tools.General.Misc_Functions import uncertRound
from ML_Tools.Plotting_And_Evaluation.Plotters import plotTrainingHistory
from ML_Tools.General.Ensemble_Functions import *
from ML_Tools.General.Callbacks import *
from ML_Tools.General.BatchYielder import BatchYielder

'''
Todo:
- Change callbacks for nicer interface e.g. pass arguments in dictionary
- Include getFeature in BatchYielder
- Make LR finder run over all batches and show combined results
- Update regressor to use more callbacks and BatchYielder
- Combine classifier and regressor methods
- Change classifier/regressor to class? Could use static methods to still provide flxibility for prototyping
- Tidy code and move to PEP 8
- Add docstrings and stuff
- Add method to BatchYielder to import other data into correct format, e.g. csv
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
    shuffle(train)
    test = n
    return train, test

def batchLRFind(batchYielder,
                modelGen, modelGenParams,
                trainParams, trainOnWeights=True,
                lrBounds=[1e-5, 10], verbose=False):

    start = timeit.default_timer()
    binary = None
    
    foldStart = timeit.default_timer()

    model = None
    model = modelGen(**modelGenParams)
    model.reset_states #Just checking
    
    if not isinstance(batchYielder, BatchYielder):
        print ("HDF5 as input is depreciated, converting to BatchYielder")
        batchYielder = BatchYielder(batchYielder)

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
                  class_weight = 'auto', sample_weight=trainbatch['weights'],
                  callbacks = [lrFinder], **trainParams) #Train for one epoch

    else:
        model.fit(trainbatch['inputs'], trainbatch['targets'],
                  class_weight = 'auto',callbacks = [lrFinder], **trainParams) #Train for one epoch

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    lrFinder.plot_lr()    
    lrFinder.plot(n_skip=10)
    print("______________________________________\n")
        
    return lrFinder

def batchTrainRegressor(data, nSplits,
                        modelGen, modelGenParams,
                        trainParams, cosAnnealMult=0, trainOnWeights=True, getBatch=getBatch,
                        extraMetrics=None, monitorData=None,
                        saveLoc='train_weights/', patience=10, maxEpochs=10000, verbose=False, logoutput=False):
    
    os.system("mkdir " + saveLoc)
    os.system("rm " + saveLoc + "*.h5")
    os.system("rm " + saveLoc + "*.json")
    os.system("rm " + saveLoc + "*.pkl")
    os.system("rm " + saveLoc + "*.png")
    os.system("rm " + saveLoc + "*.log")
    
    if logoutput:
        old_stdout = sys.stdout
        log_file = open(saveLoc + 'training_log.log', 'w')
        sys.stdout = log_file

    start = timeit.default_timer()
    results = []
    histories = []
    
    if cosAnnealMult: print ("Using cosine annealing")

    monitor = False
    if not isinstance(monitorData, type(None)):
        monitorInputs = monitorData['inputs']
        monitorTargets = monitorData['targets']
        monitor = True
        print ("Using a monitor sample to judge convergence")

    for fold in range(nSplits):
        foldStart = timeit.default_timer()
        print ("Running fold", fold+1, "/", nSplits)
        os.system("rm " + saveLoc + "best.h5")
        best = -1
        epochCounter = 0
        subEpoch = 0
        stop = False
        lossHistory = []
        monitorHistory = []
        trainID, testID = getFolds(fold, nSplits) #Get fold indeces for training and testing for current fold
        testbatch = getBatch(testID, data) #Load testing fold

        model = None
        model = modelGen(**modelGenParams)
        model.reset_states #Just checking

        callbacks = []
        if cosAnnealMult:
            cosAnneal = CosAnneal(math.ceil(len(data['fold_0/targets'])/trainParams['batch_size']), cosAnnealMult)
            callbacks.append(cosAnneal)

        for epoch in range(maxEpochs):
            epochStart = timeit.default_timer()

            for n in trainID: #Loop through training folds
                trainbatch = getBatch(n, data) #Load fold data
                subEpoch += 1

                if trainOnWeights:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
                              sample_weight=trainbatch['weights'],
                              callbacks=callbacks, **trainParams) #Train for one epoch

                    loss = model.evaluate(testbatch['inputs'], testbatch['targets'], sample_weight=testbatch['weights'], verbose=0)
                else:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
                              callbacks=callbacks, **trainParams) #Train for one epoch
                    
                    loss = model.evaluate(testbatch['inputs'], testbatch['targets'], verbose=0)

                lossHistory.append(loss)

                monLoss = loss
                if monitor:
                    monLoss = model.evaluate(monitorInputs, monitorTargets, verbose=0)
                    monitorHistory.append(monLoss)

                if monLoss <= best or best < 0: #Save best
                    best = monLoss
                    epochCounter = 0
                    model.save_weights(saveLoc + "best.h5")
                    if verbose:
                        print ('{} New best found: {}'.format(subEpoch, best))
                elif cosAnnealMult:
                    if cosAnneal.cycle_end:
                        epochCounter += 1
                else:
                    epochCounter += 1

                if epochCounter >= patience: #Early stopping
                    if verbose:
                        print ('Early stopping after {} epochs'.format(subEpoch))
                    stop = True
                    break
            
            if stop:
                break

        model.load_weights(saveLoc +  "best.h5")

        histories.append({})
        histories[-1]['val_loss'] = lossHistory
        histories[-1]['mon_loss'] = monitorHistory
        
        results.append({})
        results[-1]['loss'] = best
        
        if not isinstance(extraMetrics, type(None)):
            metrics = extraMetrics(model.predict(testbatch['inputs'], verbose=0), testbatch['targets'], testbatch['weights'])
            for metric in metrics:
                results[-1][metric] = metrics[metric]

        print ("Score is:", results[-1])

        print("Fold took {:.3f}s\n".format(timeit.default_timer() - foldStart))

        model.save(saveLoc +  'train_' + str(fold) + '.h5')
        with open(saveLoc +  'resultsFile.pkl', 'wb') as fout: #Save results
            pickle.dump(results, fout)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    plotTrainingHistory(histories, save=saveLoc + 'loss_history.png')
    for score in results[0]:
        mean = uncertRound(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print ("Mean", score, "= {} +- {}".format(mean[0], mean[1]))
    print("______________________________________\n")
                      
    if logoutput:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories

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
                         oneCycle=False, ratio=0.25, reverse=False, lrScale=10, momScale=10, plotOneCycle=False, scale=30, mode='sgd',
                         swaStart=-1,
                         trainOnWeights=True,
                         saveLoc='train_weights/', patience=10, maxEpochs=10000,
                         verbose=False, logoutput=False):
    
    os.system("mkdir " + saveLoc)
    os.system("rm " + saveLoc + "*.h5")
    os.system("rm " + saveLoc + "*.json")
    os.system("rm " + saveLoc + "*.pkl")
    os.system("rm " + saveLoc + "*.png")
    os.system("rm " + saveLoc + "*.log")
    
    if logoutput:
        old_stdout = sys.stdout
        log_file = open(saveLoc + 'training_log.log', 'w')
        sys.stdout = log_file

    start = timeit.default_timer()
    results = []
    histories = []
    binary = None

    if not isinstance(batchYielder, BatchYielder):
        print ("HDF5 as input is depreciated, converting to BatchYielder")
        batchYielder = BatchYielder(batchYielder)

    if cosAnnealMult: print ("Using cosine annealing")
    if trainOnWeights: print ("Training using weights")

    for fold in range(nSplits):
        foldStart = timeit.default_timer()
        print ("Running fold", fold+1, "/", nSplits)
        os.system("rm " + saveLoc + "best.h5")
        best = -1
        bestLR = -1
        reduxDecayActive = False
        tmpPatience = patience
        epochCounter = 0
        subEpoch = 0
        stop = False
        lossHistory = []
        trainID, testID = getFolds(fold, nSplits) #Get fold indeces for training and testing for current fold

        model = None
        model = modelGen(**modelGenParams)
        model.reset_states #Just checking

        callbacks = []
        if cosAnnealMult:
            cosAnneal = CosAnneal(math.ceil(len(batchYielder.source['fold_0/targets'])/trainParams['batch_size']), cosAnnealMult, reverseAnneal)
            callbacks.append(cosAnneal)
        
        if annealMomentum:
            cosAnnealMomentum = CosAnnealMomentum(math.ceil(len(batchYielder.source['fold_0/targets'])/trainParams['batch_size']), cosAnnealMult, reverseAnnealMomentum)
            callbacks.append(cosAnnealMomentum)    

        if oneCycle:
            oneCycle = OneCycle(math.ceil(len(batchYielder.source['fold_0/targets'])/trainParams['batch_size']), ratio=ratio, reverse=reverse, lrScale=lrScale, momScale=momScale, scale=scale, mode=mode)
            callbacks.append(oneCycle)  
        
        if swaStart >= 0:
            if cosAnnealMult:
                swa = SWA(swaStart, cosAnneal)
            else:
                swa = SWA(swaStart)
            swaModel = modelGen(**modelGenParams)
            callbacks.append(swa)

        for epoch in range(maxEpochs):
            for n in trainID: #Loop through training folds
                trainbatch = batchYielder.getBatch(n) #Load fold data
                subEpoch += 1
                
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
                              class_weight = 'auto', sample_weight=trainbatch['weights'],
                              callbacks = callbacks, **trainParams) #Train for one epoch

                    testbatch = batchYielder.getBatch(testID) #Load testing fold
                    if swaStart >= 0 and swa.active:
                        swaModel.set_weights(swa.swa_model)
                        loss = swaModel.evaluate(testbatch['inputs'], testbatch['targets'], sample_weight=testbatch['weights'], verbose=0)
                    else:
                        loss = model.evaluate(testbatch['inputs'], testbatch['targets'], sample_weight=testbatch['weights'], verbose=0)
                    
                else:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
                              class_weight = 'auto',
                              callbacks = callbacks, **trainParams) #Train for one epoch
                    
                    testbatch = batchYielder.getBatch(testID) #Load testing fold
                    if swaStart >= 0 and swa.active:
                        swaModel.set_weights(swa.swa_model)
                        loss = swaModel.evaluate(testbatch['inputs'], testbatch['targets'], verbose=0)
                    else:
                        loss = model.evaluate(testbatch['inputs'], testbatch['targets'], verbose=0)
                
                if swaStart >= 0 and swa.active and cosAnnealMult > 1:
                    print ("SWA loss:", loss)

                lossHistory.append(loss)

                if loss <= best or best < 0: #Save best
                    best = loss
                    if cosAnnealMult:
                        if cosAnneal.lrs[-1] > 0:
                            bestLR = cosAnneal.lrs[-1]
                        else:
                            bestLR = cosAnneal.lrs[-2]
                    epochCounter = 0
                    if swaStart >= 0 and swa.active:
                        swaModel.save_weights(saveLoc + "best.h5")
                    else:
                        model.save_weights(saveLoc + "best.h5")
                    if reduxDecayActive:
                        cosAnneal.lrs.append(float(K.get_value(model.optimizer.lr)))
                    if verbose:
                        print ('{} New best found: {}'.format(subEpoch, best))
                elif cosAnnealMult and not reduxDecayActive:
                    if cosAnneal.cycle_end:
                        epochCounter += 1
                else:
                    epochCounter += 1
                    if reduxDecayActive:
                        lr = 0.8*float(K.get_value(model.optimizer.lr))
                        cosAnneal.lrs.append(lr)
                        K.set_value(model.optimizer.lr, lr)

                if epochCounter >= tmpPatience: #Early stopping
                    if cosAnnealMult and reduxDecay and not reduxDecayActive:
                        print ('CosineAnneal stalling after {} epochs, entering redux decay at LR={}'.format(subEpoch, bestLR))
                        model.load_weights(saveLoc +  "best.h5")
                        cosAnneal.lrs.append(bestLR)
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

        model.load_weights(saveLoc +  "best.h5")

        histories.append({})
        histories[-1]['val_loss'] = lossHistory
        
        results.append({})
        results[-1]['loss'] = best
        if binary:
            testbatch = batchYielder.getBatch(testID) #Load testing fold
            if not isinstance(testbatch['weights'], type(None)):
                results[-1]['wAUC'] = 1-roc_auc_score(testbatch['targets'],
                                                     model.predict(testbatch['inputs'], verbose=0),
                                                     sample_weight=testbatch['weights'])
            results[-1]['AUC'] = 1-roc_auc_score(testbatch['targets'],
                                                 model.predict(testbatch['inputs'], verbose=0))
        print ("Score is:", results[-1])

        if plotLR: cosAnneal.plot_lr()
        if plotMomentum: cosAnnealMomentum.plot_momentum()
        if plotOneCycle: oneCycle.plot()

        print("Fold took {:.3f}s\n".format(timeit.default_timer() - foldStart))

        model.save(saveLoc +  'train_' + str(fold) + '.h5')
        with open(saveLoc +  'resultsFile.pkl', 'wb') as fout: #Save results
            pickle.dump(results, fout)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    plotTrainingHistory(histories, save=saveLoc + 'loss_history.png')
    for score in results[0]:
        mean = uncertRound(np.mean([x[score] for x in results]), np.std([x[score] for x in results])/np.sqrt(len(results)))
        print ("Mean", score, "= {} +- {}".format(mean[0], mean[1]))
    print("______________________________________\n")
                      
    if logoutput:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories