from __future__ import division

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import backend as K
from keras import utils

from six.moves import cPickle as pickle
import timeit
import types
import numpy as np
import os
from random import shuffle

from ML_Tools.General.Misc_Functions import uncertRound
from ML_Tools.Plotting_And_Evaluation.Plotters import plotTrainingHistory
from ML_Tools.General.Ensemble_Functions import *

def getBatch(index, datafile):
    index = str(index)
    if 'fold_' + index + '/weights' in datafile:
        weights = np.array(datafile['fold_' + index + '/weights'])
    else:
        weights = None
    return {'inputs':np.array(datafile['fold_' + index + '/inputs']),
            'targets':np.array(datafile['fold_' + index + '/targets']),
            'weights':weights}

def getFolds(n, nSplits):
    train = [x for x in xrange(nSplits) if x != n]
    shuffle(train)
    test = n
    return train, test

def batchTrainClassifier(data, nSplits, modelGen, modelGenParams, trainParams, getBatch=getBatch,
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
    binary = None

    for fold in xrange(nSplits):
        foldStart = timeit.default_timer()
        print "Running fold", fold+1, "/", nSplits
        os.system("rm " + saveLoc + "best.h5")
        best = -1
        epochCounter = 0
        subEpoch = 0
        stop = False
        lossHistory = []
        trainID, testID = getFolds(fold, nSplits) #Get fold indeces for training and testing for current fold
        testbatch = getBatch(testID, data) #Load testing fold

        model = None
        model = modelGen(**modelGenParams)
        model.reset_states #Just checking

        for epoch in xrange(maxEpochs):
            epochStart = timeit.default_timer()

            for n in trainID: #Loop through training folds
                trainbatch = getBatch(n, data) #Load fold data
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

                model.fit(trainbatch['inputs'], trainbatch['targets'],
                          class_weight = 'auto', sample_weight=trainbatch['weights'],
                          **trainParams) #Train for one epoch

                
                loss = model.evaluate(testbatch['inputs'], testbatch['targets'], sample_weight=testbatch['weights'], verbose=0)
                lossHistory.append(loss)

                if loss <= best or best < 0: #Save best
                    best = loss
                    epochCounter = 0
                    model.save_weights(saveLoc + "best.h5")
                    if verbose:
                        print '{} New best found: {}'.format(subEpoch, best)
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
        
        results.append({})
        results[-1]['loss'] = best
        if binary:
            results[-1]['AUC'] = 1-roc_auc_score(testbatch['targets'],
                                                 model.predict(testbatch['inputs'], verbose=0),
                                                 sample_weight=testbatch['weights'])
        print "Score is:", results[-1]

        print("Fold took {:.3f}s\n".format(timeit.default_timer() - foldStart))

        model.save(saveLoc +  'train_' + str(fold) + '.h5')
        with open(saveLoc +  'resultsFile.pkl', 'wb') as fout: #Save results
            pickle.dump(results, fout)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    plotTrainingHistory(histories, save=saveLoc + 'loss_history.png')

    meanLoss = uncertRound(np.mean([x['loss'] for x in results]), np.std([x['loss'] for x in results])/np.sqrt(len(results)))
    print "Mean loss = {} +- {}".format(meanLoss[0], meanLoss[1])
    if binary:
        meanAUC = uncertRound(np.mean([x['AUC'] for x in results]), np.std([x['AUC'] for x in results])/np.sqrt(len(results)))
        print "Mean AUC = {} +- {}".format(meanAUC[0], meanAUC[1])
    print("______________________________________\n")
                      
    if logoutput:
        sys.stdout = old_stdout
        log_file.close()
    return results, histories

def saveBatchPred(batchPred, fold, datafile):
    try:
        datafile.create_dataset(fold + "/pred_class", shape=batchPred.shape, dtype='float32')
    except RuntimeError:
        pass
    
    pred = datafile[fold + "/pred_class"]
    pred[...] = batchPred
        
def batchEnsemblePredict(ensemble, weights, datafile, ensembleSize=None, verbose=False):
    if isinstance(ensembleSize, types.NoneType):
        ensembleSize = len(ensemble)

    for i, fold in enumerate(datafile): #Todo make it work out number of folds

        if verbose:
            print 'Predicting batch {} out of {}'.format(i+1, len(datafile))
            start = timeit.default_timer()

        batch = np.array(datafile[fold + '/inputs'])
        batchPred = ensemblePredict(batch, ensemble, weights, n=ensembleSize)[:,0]

        if verbose: 
            print "Prediction took {}s per sample\n".format((timeit.default_timer() - start)/len(batch))

        saveBatchPred(batchPred, fold, datafile)
        
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