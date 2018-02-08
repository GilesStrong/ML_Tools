from __future__ import division

from sklearn.metrics import roc_auc_score, roc_curve

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

import numpy as np
import pandas
import types

from ML_Tools.General.Misc_Functions import uncertRound
from ML_Tools.Plotting_And_Evaluation.Bootstrap import * 

def plotFeat(inData, feat, cuts=None, labels=None, params={}):
    loop = False
    if not isinstance(cuts, types.NoneType):
        if isinstance(cuts, types.ListType):
            loop = True
            if not isinstance(cuts, types.ListType):
                print "{} plots requested, but not labels passed".format(len(cuts))
                return -1
            elif len(cuts) != len(labels):
                print "{} plots requested, but {} labels passed".format(len(cuts), len(labels))
                return -1

    plt.figure(figsize=(16, 8))
    if loop:
        for i in range(len(cuts)):
            if isinstance(params, types.ListType):
                tempParams = params[i]
            else:
                tempParams = params
            sns.distplot(inData[cuts[i]][feat], label=labels[i], **tempParams)
    else:
        sns.distplot(inData[feat], **params)
    if loop:
        plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.ylabel("Density", fontsize=24, color='black')
    plt.xlabel(feat, fontsize=24, color='black')
    plt.show()

def rocPlot(inData=None, curves=None, predName='pred_class', targetName='gen_target', weightName=None, labels=None, aucs=None, bootstrap=False, log=False, baseline=True, params=[{}]):
    buildCurves = True
    if isinstance(inData, types.NoneType) == isinstance(curves, types.NoneType):
        print "Must pass either targets and preds, or curves"
        return -1
    if not isinstance(curves, types.NoneType):
        buildCurves = False

    if buildCurves:
        curves = {}
        if bootstrap:
            aucArgs = []
            for i in range(len(inData)):
                aucArgs.append({'labels':inData[i][targetName], 'preds':inData[i][predName], 'name':labels[i], 'indeces':inData[i].index.tolist()})
                if not isinstance(weightName, types.NoneType):
                    aucArgs[-1]['weights'] = inData[i][weightName]
            aucs = mpRun(aucArgs, rocauc)
            meanScores = {}
            for i in labels:
                meanScores[i] = (np.mean(aucs[i]), np.std(aucs[i]))
                print str(i)+ ' ROC AUC, Mean = {} +- {}'.format(meanScores[i][0], meanScores[i][1])
        else:
            meanScores = {}
            for i in range(len(inData)):
                if isinstance(weightName, types.NoneType):
                    meanScores[labels[i]] = roc_auc_score(inData[i][targetName].values, inData[i][predName])
                else:
                    meanScores[labels[i]] = roc_auc_score(inData[i][targetName].values, inData[i][predName], sample_weight=inData[i][weightName])
                print str(i) + ' ROC AUC: {}'.format(meanScores[labels[i]])
        for i in range(len(inData)):
            if isinstance(weightName, types.NoneType):
                curves[labels[i]] = roc_curve(inData[i][targetName].values, inData[i][predName].values)[:2]
            else:
                curves[labels[i]] = roc_curve(inData[i][targetName].values, inData[i][predName].values, sample_weight=inData[i][weightName].values)[:2]

    plt.figure(figsize=[8, 8])
    for i in range(len(curves)):
        if buildCurves:
            if bootstrap:
                meanScore = uncertRound(*meanScores[labels[i]])
                plt.plot(*curves[labels[i]], label=labels[i] + r', AUC$={}\pm{}$'.format(meanScore[0], meanScore[1]), **params[i])
            else:
                plt.plot(*curves[labels[i]], label=labels[i] + r', AUC$={:.5f}$'.format(meanScores[labels[i]]), **params[i])
        else:
            plt.plot(*curves[i], label=labels[i], **params[i])
    
    if baseline:
        plt.plot([0, 1], [0, 1], 'k--', label='No discrimination')
    plt.xlabel('Background acceptance', fontsize=24, color='black')
    plt.ylabel('Signal acceptance', fontsize=24, color='black')
    if len(labels):
        plt.legend(loc='best', fontsize=16)
    if log:
        plt.xscale('log', nonposx='clip')
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.show()

def getClassPredPlot(inData, labels=['Background', 'Signal'], predName='pred_class', weightName=None,
                     lim=(0,1), logy=True, params={'hist' : True, 'kde' : False, 'norm_hist' : True}):
    plt.figure(figsize=(16, 8))
    for i in range(len(inData)):
        hist_kws = {}
        if not isinstance(weightName, types.NoneType):
            hist_kws['weights'] = inData[i][weightName]
        sns.distplot(inData[i][predName], label=labels[i], hist_kws=hist_kws, **params)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Class prediction", fontsize=24, color='black')
    plt.xlim(lim)
    plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=24, color='black')
    if logy:
        plt.yscale('log', nonposy='clip')
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.show() 

def _getSamples(inData, sampleName, weightName):
    samples = set(inData[sampleName])
    weights=[np.sum(inData[inData[sampleName] == sample][weightName]) for sample in samples]
    return [x[0] for x in np.array(sorted(zip(samples, weights), key=lambda x: x[1]))] #Todo improve sorting

def getSamplePredPlot(inData, 
                      targetName='gen_target', sampleName='gen_sample', predName='pred_class', weightName='gen_weight',
                      lim=(0,1), nBins = 35, logy=True, pallet='nipy_spectral', desat=0.8,
                      hist_params={'normed': True, 'alpha': 1, 'stacked':True, 'rwidth':1.0,}):
    
    hist_params['bins'] = nBins
    hist_params['range'] = lim
    
    plt.figure(figsize=(16, 8))
    
    sig = (inData[targetName] == 1)
    bkg = (inData[targetName] == 0)
    
    with sns.color_palette(pallet, len(set(inData[sampleName])), desat=desat):
        
        samples = _getSamples(inData[bkg], sampleName, weightName)
        plt.hist([inData[inData[sampleName] == sample][predName] for sample in samples],
                 weights=[inData[inData[sampleName] == sample][weightName] for sample in samples],
                 label=samples, **hist_params)

        samples = _getSamples(inData[sig], sampleName, weightName)
        for sample in samples:
            plt.hist(inData[inData[sampleName] == sample][predName],
                     weights=inData[inData[sampleName] == sample][weightName],
                     label='Signal ' + sample, histtype='step', linewidth='3', **hist_params)

        plt.legend(loc='best', fontsize=16)
        plt.xlabel("Class prediction", fontsize=24, color='black')
        plt.xlim(lim)
        if hist_params['normed']:
            plt.ylabel(r"$\frac{1}{\mathcal{A}\sigma} \frac{d\left(\mathcal{A}\sigma\right)}{dp}\ [pb]$", fontsize=24, color='black')
        else:
            plt.ylabel(r"$\frac{d\left(\mathcal{A}\sigma\right)}{dp}\ [pb]$", fontsize=24, color='black')
        if logy:
            plt.yscale('log', nonposy='clip')
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.show()      

def plotHistory(histories):
    print "Depreciated, move to plotTrainingHistory"
    plotTrainingHistory(histories)
    
def plotTrainingHistory(histories, save=False):
    plt.figure(figsize=(16,8))
    for i, history in enumerate(histories):
        if i == 0:
            try:
                plt.plot(history['loss'], color='g', label='Training')
            except:
                pass
            try:
                plt.plot(history['val_loss'], color='b', label='Testing')
            except:
                pass
            try:
                plt.plot(history['mon_loss'], color='r', label='Monitoring')
            except:
                pass
        else:
            try:
                plt.plot(history['loss'], color='g')
            except:
                pass
            try:
                plt.plot(history['val_loss'], color='b')
            except:
                pass
            try:
                plt.plot(history['mon_loss'], color='r')
            except:
                pass
    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Epoch", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()

    if save:
        plt.savefig(save)