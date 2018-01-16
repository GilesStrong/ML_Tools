from __future__ import division

import numpy as np
import pandas
import types

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

def rankClassifierFeatures(data, trainFeatures, weights=None, target='gen_target', datatype='float32'):
    inputPipe, outputPipe = getPreProcPipes(normIn=True)
    inputPipe.fit(data[trainFeatures].values.astype(datatype))
    X = inputPipe.transform(data[trainFeatures].values.astype(datatype))
    y = data[target].values.astype('int')
    
    if weights != None:
        sig = data[data[target] == 1].index
        bkg = data[data[target] == 0].index
        w = data[weights].values.astype(datatype)
        w[sig] = w[sig]/np.sum(w[sig])
        w[bkg] = w[bkg]/np.sum(w[bkg])

    importantFeatures = []
    featureImportance = {}
    for i in trainFeatures:
        featureImportance[i] = 0

    kf = StratifiedKFold(n_splits=10, shuffle=True)
    folds = kf.split(X, y)
    for i, (train, test) in enumerate(folds):
        print "Running fold", i+1, "/10"
        
        xgbClass = XGBClassifier()
        if weights != None:
            xgbClass.fit(X[train], y[train], sample_weight=w[train])
            print 'ROC AUC: {:.5f}'.format(roc_auc_score(y[test], xgbClass.predict_proba(X[test])[:,1]), sample_weight=w[test])
        else:
            xgbClass.fit(X[train], y[train])
            print 'ROC AUC: {:.5f}'.format(roc_auc_score(y[test], xgbClass.predict_proba(X[test])[:,1]))
        
        
        indices = xgbClass.feature_importances_.argsort()
        N = len(indices)
        for n in range(N):
            if xgbClass.feature_importances_[indices[N-n-1]] > 0:
                importantFeatures.append(trainFeatures[indices[N-n-1]])
            featureImportance[trainFeatures[indices[N-n-1]]] += xgbClass.feature_importances_[indices[N-n-1]]

    names = np.array(list(set(importantFeatures)))
    scores = np.array([featureImportance[i]/10 for i in names])
    importance = np.array(sorted(zip(names, scores), key=lambda x: x[1], reverse=True))

    print len(importantFeatures), "important features identified"
    print 'Feature\tImportance'
    print '---------------------'
    for i in importance:
        print i[0], '\t', i[1]

    return [x[0] for x in importance], [x[1] for x in importance]

def getCorrMat(data0, data1 = None):
    corr = data0.corr()
    
    if not isinstance(data1, types.NoneType): #Plot difference in correlations
        corr -= data1.corr()

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})