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

'''
Todo:
- Make batchtrain output mean loss
- Change callbacks for nicer interface e.g. pass arguments in dictionary
- Include getFeature in BatchYielder
- Move BatchYielder to separate file
- Make LR finder run over all batches and show combined results
- Update regressor to use more callbacks and BatchYielder
- Combine classifier and regressor methods
- Change classifier/regressor to class? Could use static methods to still provide flxibility for prototyping
- Tidy code and move to PEP 8
- Add docstrings and stuff
'''

class BatchYielder():
    def __init__(self, datafile=None):
        self.augmented = False
        self.augMult = 0
        self.trainTimeAug = False
        self.testTimeAug = False
        if not isinstance(datafile, types.NoneType):
            self.addSource(datafile)

    def addSource(self, datafile):
        self.source = datafile
        self.nFolds = len(self.source)

    def getBatch(self, index, datafile=None):
        if isinstance(datafile, types.NoneType):
            datafile = self.source

        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])
        return {'inputs':np.array(datafile['fold_' + index + '/inputs']),
                'targets':targets,
                'weights':weights}

class ReflectBatch(BatchYielder):
    def __init__(self, header, datafile=None, trainTimeAug=True, testTimeAug=True):
        self.header = header
        self.augmented = True
        self.augMult = 8
        self.trainTimeAug = trainTimeAug
        self.testTimeAug = testTimeAug
        if not isinstance(datafile, types.NoneType):
            self.addSource(datafile)
        
    def getBatch(self, index, datafile=None):
        if isinstance(datafile, types.NoneType):
            datafile = self.source
            
        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])

        inputs = pandas.DataFrame(np.array(datafile['fold_' + index + '/inputs']), columns=self.header)
        for coord in ['_px','_py','_pz']:
            inputs['aug' + coord] = np.random.randint(0, 2, size=len(inputs))
            for feat in [x for x in inputs.columns if coord in x and x != 'aug' + coord]:
                inputs.loc[inputs['aug' + coord] == 1, feat] = -inputs.loc[inputs['aug' + coord] == 1, feat]

        return {'inputs':inputs[self.header].values,
                'targets':targets,
                'weights':weights}
    
    def getTestBatch(self, index, augIndex, datafile=None):
        if augIndex >= self.augMult:
            print "Invalid augmentation index passed", augIndex
            return -1
        
        if isinstance(datafile, types.NoneType):
            datafile = self.source
            
        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])

        augMode = '{0:03b}'.format(augIndex) #Get binary rep
        inputs = pandas.DataFrame(np.array(datafile['fold_' + index + '/inputs']), columns=self.header)
        coords = ['_px','_py','_pz']
        for coordIndex, active in enumerate(augMode):
            if active == '1':
                for feat in [x for x in inputs.columns if coords[coordIndex] in x]:
                    inputs.loc[:, feat] = -inputs.loc[:, feat]

        return {'inputs':inputs[self.header].values,
                'targets':targets,
                'weights':weights}

class RotationBatch(BatchYielder):
    def __init__(self, header, datafile=None, augMult=8, trainTimeAug=True, testTimeAug=True):
        self.header = header
        self.augmented = True
        self.augMult = augMult
        self.trainTimeAug = trainTimeAug
        self.testTimeAug = testTimeAug
        if not isinstance(datafile, types.NoneType):
            self.addSource(datafile)
    
    @staticmethod
    def rotate(inData):
        vectors = [x[:-3] for x in inData.columns if '_px' in x]
        for vector in vectors:
            inData.loc[:, vector + '_px'] = inData.loc[:, vector + '_px']*np.cos(inData.loc[:, 'aug_angle'])-inData.loc[:, vector + '_py']*np.sin(inData.loc[:, 'aug_angle'])
            inData.loc[:, vector + '_py'] = inData.loc[:, vector + '_py']*np.cos(inData.loc[:, 'aug_angle'])+inData.loc[:, vector + '_px']*np.sin(inData.loc[:, 'aug_angle'])
                
    def getBatch(self, index, datafile=None):
        if isinstance(datafile, types.NoneType):
            datafile = self.source
            
        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])

        inputs = pandas.DataFrame(np.array(datafile['fold_' + index + '/inputs']), columns=self.header)
        inputs['aug_angle'] = 2*np.pi*np.random.random(size=len(inputs))
        self.rotate(inputs)
        
        return {'inputs':inputs[self.header].values,
                'targets':targets,
                'weights':weights}
    
    def getTestBatch(self, index, augIndex, datafile=None):
        if augIndex >= self.augMult:
            print "Invalid augmentation index passed", augIndex
            return -1
        
        if isinstance(datafile, types.NoneType):
            datafile = self.source
            
        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])
            
        inputs = pandas.DataFrame(np.array(datafile['fold_' + index + '/inputs']), columns=self.header)
        inputs['aug_angle'] = np.linspace(0, 2*np.pi, self.augMult+1)[augIndex]
        self.rotate(inputs)

        return {'inputs':inputs[self.header].values,
                'targets':targets,
                'weights':weights}

def getBatch(index, datafile):
    print "Depreciated, use to moving a BatchYielder class"
    index = str(index)
    weights = None
    if 'fold_' + index + '/weights' in datafile:
        weights = np.array(datafile['fold_' + index + '/weights'])
    return {'inputs':np.array(datafile['fold_' + index + '/inputs']),
            'targets':np.array(datafile['fold_' + index + '/targets']),
            'weights':weights}

def getFolds(n, nSplits):
    train = [x for x in xrange(nSplits) if x != n]
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
        print "HDF5 as input is depreciated, converting to BatchYielder"
        batchYielder = BatchYielder(batchYielder)

    trainbatch = batchYielder.getBatch(np.random.choice(range(batchYielder.nFolds))) #Load fold
    nSteps = math.ceil(len(trainbatch['targets'])/trainParams['batch_size'])
    if verbose: print "Using {} steps".format(nSteps)   
        
    lrFinder = LRFinder(nSteps=nSteps, lrBounds=lrBounds, verbose=verbose)

    if 'class' in modelGenParams['mode'].lower():
        if binary == None: #Check classification mode
            binary = True
            nClasses = len(np.unique(trainbatch['targets']))
            if nClasses > 2:
                print nClasses, "classes found, running in multiclass mode\n"
                trainbatch['targets'] = utils.to_categorical(trainbatch['targets'], num_classes=nClasses)
                binary = False
            else:
                print nClasses, "classes found, running in binary mode\n"

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
    
    if cosAnnealMult: print "Using cosine annealing"

    monitor = False
    if not isinstance(monitorData, types.NoneType):
        monitorInputs = monitorData['inputs']
        monitorTargets = monitorData['targets']
        monitor = True
        print "Using a monitor sample to judge convergence"

    for fold in xrange(nSplits):
        foldStart = timeit.default_timer()
        print "Running fold", fold+1, "/", nSplits
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

        for epoch in xrange(maxEpochs):
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
                        print '{} New best found: {}'.format(subEpoch, best)
                elif cosAnnealMult:
                    if cosAnneal.cycle_end:
                        epochCounter += 1
                else:
                    epochCounter += 1

                if epochCounter >= patience: #Early stopping
                    if verbose:
                        print 'Early stopping after {} epochs'.format(subEpoch)
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
        
        if not isinstance(extraMetrics, types.NoneType):
            metrics = extraMetrics(model.predict(testbatch['inputs'], verbose=0), testbatch['targets'], testbatch['weights'])
            for metric in metrics:
                results[-1][metric] = metrics[metric]

        print "Score is:", results[-1]

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
    print "Mean", score, "= {} +- {}".format(mean[0], mean[1])
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
    if isinstance(ensembleSize, types.NoneType):
        ensembleSize = len(ensemble)

    if not isinstance(batchYielder, BatchYielder):
        print "Passing HDF5 as input is depreciated, converting to BatchYielder"
        batchYielder = BatchYielder(batchYielder)

    if nFolds < 0:
        nFolds = len(batchYielder.source)

    for fold in range(nFolds):
        if verbose:
            print 'Predicting batch {} out of {}'.format(fold+1, nFolds)
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
            print "Prediction took {}s per sample\n".format((timeit.default_timer() - start)/len(batch))

        if nOut > 1:
            saveBatchPred(batchPred, 'fold_' + str(fold), batchYielder.source, predName=predName)
        else:
            saveBatchPred(batchPred[:,0], 'fold_' + str(fold), batchYielder.source, predName=predName)
        
def getFeature(feature, datafile, nFolds=-1, ravel=True):
    data = []
    for i, fold in enumerate(datafile):
        if i >= nFolds and nFolds > 0:
            break
        data.append(np.array(datafile[fold + '/' + feature]))
        
    data = np.concatenate(data)
    if ravel:
        return data.ravel()
    return data

def batchTrainClassifier(batchYielder, nSplits, modelGen, modelGenParams, trainParams,
                         cosAnnealMult=0, reverseAnneal=False, plotLR=False, reduxDecay=False,
                         annealMomentum=False, reverseAnnealMomentum=False, plotMomentum=False,
                         oneCycle=False, ratio=0.25, reverse=False, lrScale=10, momScale=10, plotOneCycle=False, scale=30, mode='sgd',
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
        print "HDF5 as input is depreciated, converting to BatchYielder"
        batchYielder = BatchYielder(batchYielder)

    if cosAnnealMult: print "Using cosine annealing"
    if trainOnWeights: print "Training using weights"

    for fold in xrange(nSplits):
        foldStart = timeit.default_timer()
        print "Running fold", fold+1, "/", nSplits
        os.system("rm " + saveLoc + "best.h5")
        best = -1
        bestLR = -1
        reduxDecayActive = False
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

        for epoch in xrange(maxEpochs):
            for n in trainID: #Loop through training folds
                trainbatch = batchYielder.getBatch(n) #Load fold data
                subEpoch += 1
                
                if binary == None: #First run, check classification mode
                    binary = True
                    nClasses = len(np.unique(trainbatch['targets']))
                    if nClasses > 2:
                        print nClasses, "classes found, running in multiclass mode\n"
                        trainbatch['targets'] = utils.to_categorical(trainbatch['targets'], num_classes=nClasses)
                        binary = False
                    else:
                        print nClasses, "classes found, running in binary mode\n"

                if trainOnWeights:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
                              class_weight = 'auto', sample_weight=trainbatch['weights'],
                              callbacks = callbacks, **trainParams) #Train for one epoch

                    testbatch = batchYielder.getBatch(testID) #Load testing fold
                    loss = model.evaluate(testbatch['inputs'], testbatch['targets'], sample_weight=testbatch['weights'], verbose=0)
                else:
                    model.fit(trainbatch['inputs'], trainbatch['targets'],
                              class_weight = 'auto',
                              callbacks = callbacks, **trainParams) #Train for one epoch
                    
                    testbatch = batchYielder.getBatch(testID) #Load testing fold
                    loss = model.evaluate(testbatch['inputs'], testbatch['targets'], verbose=0)
        
                lossHistory.append(loss)

                if loss <= best or best < 0: #Save best
                    best = loss
                    if cosAnnealMult:
                        if cosAnneal.lr > 0:
                            bestLR = cosAnneal.lr
                        else:
                            bestLR = cosAnneal.lrs[-1]
                    epochCounter = 0
                    model.save_weights(saveLoc + "best.h5")
                    if verbose:
                        print '{} New best found: {}'.format(subEpoch, best)
                elif cosAnnealMult:
                    if cosAnneal.cycle_end:
                        epochCounter += 1
                else:
                    epochCounter += 1
                    if reduxDecayActive:
                        K.set_value(model.optimizer.lr, 0.5*model.optimizer.lr)

                if epochCounter >= patience: #Early stopping
                    if cosAnnealMult and reduxDecay and not reduxDecayActive:
                        print 'CosineAnneal stalling after {} epochs, entering redux decay at LR={}'.format(subEpoch, bestLR)
                        model.load_weights(saveLoc +  "best.h5")
                        K.set_value(model.optimizer.lr, bestLR)
                        cosAnnealMult = 0
                        patience = 10
                        epochCounter = 0
                        callbacks = []
                        reduxDecayActive = True
                    else:
                        if verbose:
                            print 'Early stopping after {} epochs'.format(subEpoch)
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
            if not isinstance(testbatch['weights'], types.NoneType):
                results[-1]['wAUC'] = 1-roc_auc_score(testbatch['targets'],
                                                     model.predict(testbatch['inputs'], verbose=0),
                                                     sample_weight=testbatch['weights'])
            results[-1]['AUC'] = 1-roc_auc_score(testbatch['targets'],
                                                 model.predict(testbatch['inputs'], verbose=0))
        print "Score is:", results[-1]

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
    print "Mean", score, "= {} +- {}".format(mean[0], mean[1])
    print("______________________________________\n")
                      
    if logoutput:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories