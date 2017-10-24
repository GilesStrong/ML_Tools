from __future__ import division

from sklearn.metrics import roc_auc_score, roc_curve

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

import numpy as np
import pandas
import types

import sys
sys.path.append('../General')
from Misc_Functions import uncertRound
from Bootstrap import * 

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

def rocPlot(inData=None, curves=None, predName='pred_class', targetName='gen_target', labels=None, aucs=None, bootstrap=False, log=False, baseline=True, params=[{}]):
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
            aucs = mpRun(aucArgs, rocauc)
            meanScores = {}
            for i in labels:
                meanScores[i] = (np.mean(aucs[i]), np.std(aucs[i]))
                print str(i)+ ' ROC AUC, Mean = {} +- {}'.format(meanScores[i][0], meanScores[i][1])
        else:
            meanScores = {}
            for i in range(len(inData)):
                meanScores[labels[i]] = roc_auc_score(inData[i][targetName].values, inData[i][predName])
                print str(i) + ' ROC AUC: {}'.format(meanScores[labels[i]])
        for i in range(len(inData)):
            curves[labels[i]] = roc_curve(inData[i][targetName].values, inData[i][predName].values)[:2]

    plt.figure(figsize=[8, 8])
    for i in range(len(curves)):
        if buildCurves:
            if bootstrap:
                meanScore = uncertRound(*meanScores[labels[i]])
                plt.plot(*curves[labels[i]], label=labels[i] + r', $auc={}\pm{}$'.format(meanScore[0], meanScore[1]), **params[i])
            else:
                plt.plot(*curves[labels[i]], label=labels[i] + r', $auc={:.5f}$'.format(meanScores[labels[i]]), **params[i])
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

def getClassPredPlot(inData, labels=['Background', 'Signal'], predName='pred_class',
                     lim=(0,1), logy=True, params={'hist' : True, 'kde' : False, 'norm_hist' : True}):
    plt.figure(figsize=(16, 8))
    for i in range(len(inData)):
        sns.distplot(inData[i][predName], label=labels[i], **params)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Class prediction", fontsize=24, color='black')
    plt.xlim(lim)
    plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=24, color='black')
    if logy:
        plt.yscale('log', nonposy='clip')
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.show()       

def getWeightedClassPredPlot(inData, weights, labels=['Background', 'Signal'], predName='pred_class',
                     lim=(0,1), logy=True, nBins=50):
    hist_params = {'normed': False, 'bins': nBins, 'alpha': 0.4}
    plt.figure(figsize=(16, 8))
    for i in range(len(inData)):
        weight = np.empty_like(inData[i][predName])
        weight.fill(weights[i]*nBins/len(inData[i]))
        plt.hist(inData[i][predName], range=lim, label=labels[i], weights=weight, **hist_params)
    if logy:
        plt.yscale('log', nonposy='clip')
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Class prediction", fontsize=24, color='black')
    plt.ylabel( r"$\frac{d\left(\mathcal{A}\sigma\right)}{dp}\ [pb]$", fontsize=24, color='black')
    plt.show()

def getStackedClassPredPlot(inData, weights, labels=['Signal', 'Background'], predName='pred_class',
                     lim=(0,1), logy=True, nBins=50, colours = ['b', 'g']):
    hist_params = {'normed': False, 'bins': nBins, 'alpha': 0.4, 'stacked':True}
    plt.figure(figsize=(16, 8))
    setWeights = []
    for i in range(len(inData)):
        weight = np.empty_like(inData[i][predName])
        weight.fill(weights[i]*nBins/len(inData[i]))
        setWeights.append(weight)
    plt.hist([inData[i][predName] for i in range(len(inData))], range=lim,
             label=labels, weights=setWeights, color=colours, **hist_params)
    if logy:
        plt.yscale('log', nonposy='clip')
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Class prediction", fontsize=24, color='black')
    plt.ylabel( r"$\frac{d\left(\mathcal{A}\sigma\right)}{dp}\ [pb]$", fontsize=24, color='black')
    plt.show()

def plotHistory(histories):
    print "Depreciated, move to plotTrainingHistory"
    plotTrainingHistory(histories)
    
def plotTrainingHistory(histories):
    plt.figure(figsize=(16,8))
    for i, history in enumerate(histories):
        if i == 0:
            plt.plot(history['loss'], color='g', label='Training')
            plt.plot(history['val_loss'], color='b', label='Testing')
        else:
            plt.plot(history['loss'], color='g')
            plt.plot(history['val_loss'], color='b')
    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Epoch", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')