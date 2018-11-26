from __future__ import division
import pandas
import numpy as np
import math
import multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

from ml_tools.general.misc_functions import uncert_round

def calc_ams(s, b, br=0, deltaB=0):
    """ Approximate Median Significance defined as:
        ams = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """

    if b == 0:
        print("No Background present, returning ams=-1")
        return -1
    
    if not deltaB:
        radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)

    else:
        sigmaB2 = np.square(deltaB*b)
        radicand = 2*(((s+b)*np.log((s+b)*(b+sigmaB2)/((b**2)+((s+b)*sigmaB2))))-
                      (((b**2)/sigmaB2)*np.log(1+((sigmaB2*s)/(b*(b+sigmaB2))))))

    if radicand < 0:
        print('radicand is negative. Exiting')
        return -1
    else:
        return math.sqrt(radicand)

def ams_scan_quick(inData, wFactor=1, br=0, deltaB=0):
    '''Determine optimum ams and cut,
    wFactor used rescale weights to get comparable amss
    sufferes from float precison'''
    amsMax = 0
    threshold = 0.0
    inData = inData.sort_values(by=['pred_class'])
    s = np.sum(inData.loc[(inData['gen_target'] == 1), 'gen_weight'])
    b = np.sum(inData.loc[(inData['gen_target'] == 0), 'gen_weight'])

    for i, cut in enumerate(inData['pred_class']):
        ams = calc_ams(max(0, s*wFactor), max(0, b*wFactor), br, deltaB)
        
        if ams > amsMax:
            amsMax = ams
            threshold = cut
        if inData['gen_target'].values[i]:
            s -= inData['gen_weight'].values[i]
        else:
            b -= inData['gen_weight'].values[i]
            
    return amsMax, threshold

def ams_scan_slow(inData, wFactor=1, br=0, systB=0, useStat=False, start=0.9, minEvents=10):
    '''Determine optimum ams and cut,
    wFactor used rescale weights to get comparable amss
    slower than quick, but doesn't suffer from float precision'''
    amsMax = 0
    threshold = 0.0
    signal = inData[inData['gen_target'] == 1]
    bkg = inData[inData['gen_target'] == 0]
    
    systB2 = np.square(systB)
    for i, cut in enumerate(inData.loc[inData.pred_class >= start, 'pred_class'].values):
        bkgPass = bkg.loc[(bkg.pred_class >= cut), 'gen_weight']
        nBkg = len(bkgPass)
        if nBkg < minEvents: continue

        s = np.sum(signal.loc[(signal.pred_class >= cut), 'gen_weight'])
        b = np.sum(bkgPass)
        if useStat:
            deltaB = np.sqrt(systB2+(1/nBkg))
        else:
            deltaB = systB
        ams = calc_ams(s*wFactor, b*wFactor, br, deltaB)
        
        if ams > amsMax:
            amsMax = ams
            threshold = cut
            
    return amsMax, threshold

def mp_ams(data, i, wFactor, br, out_q):
    ams, cut = ams_scan_quick(data, wFactor, br)
    out_q.put({str(i) + '_ams':ams, str(i) + '_cut':cut})

def mp_skfold_ams(data, i, size, nFolds, br, out_q):
    kf = StratifiedKFold(n_splits=nFolds, shuffle=True)
    folds = kf.split(data, data['gen_target'])
    uids = range(i*nFolds,(i+1)*nFolds)
    outdict = {}

    for j, (_, fold) in enumerate(folds):
        ams, cut = ams_scan_quick(data.iloc[fold], size/len(fold), br)
        if ams > 0:
            outdict[str(uids[j]) + '_ams'] = ams
            outdict[str(uids[j]) + '_cuts'] = cut
    out_q.put(outdict)

def bootstrap_mean_ams(data, wFactor=1, N=512, br=0):
    procs = []
    out_q = mp.Queue()
    for i in range(N):
        indeces = np.random.choice(data.index, len(data), replace=True)
        p = mp.Process(target=mp_ams, args=(data.iloc[indeces], i, wFactor, br, out_q))
        procs.append(p)
        p.start() 
    resultdict = {}
    for i in range(N):
        resultdict.update(out_q.get()) 
    for p in procs:
        p.join()  
        
    amss = np.array([resultdict[x] for x in resultdict if 'ams' in x])
    cuts = np.array([resultdict[x] for x in resultdict if 'cut' in x])

    meanams = uncert_round(np.mean(amss), np.std(amss))
    meanCut = uncert_round(np.mean(cuts), np.std(cuts))

    ams = calc_ams(wFactor*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 1), 'gen_weight']),
              wFactor*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 0), 'gen_weight']))
    
    print('\nMean ams={}+-{}, at mean cut of {}+-{}'.format(meanams[0], meanams[1], meanCut[0], meanCut[1]))
    print('Exact mean cut {}, corresponds to ams of {}'.format(np.mean(cuts), ams))
    return (meanams[0], meanCut[0])

def bootstrap_skfold_mean_ams(data, size=1, N=10, nFolds=500, br=0):
    print("Warning, this method might not be trustworthy: cut decreases with nFolds")
    procs = []
    out_q = mp.Queue()
    for i in range(N):
        p = mp.Process(target=mp_skfold_ams, args=(data, i, size, nFolds, br, out_q))
        procs.append(p)
        p.start() 
    resultdict = {}
    for i in range(N):
        resultdict.update(out_q.get()) 
    for p in procs:
        p.join()  
        
    amss = np.array([resultdict[x] for x in resultdict if 'ams' in x])
    cuts = np.array([resultdict[x] for x in resultdict if 'cut' in x])

    meanams = uncert_round(np.mean(amss), np.std(amss)/np.sqrt(N*nFolds))
    meanCut = uncert_round(np.mean(cuts), np.std(cuts)/np.sqrt(N*nFolds))

    scale = size/len(data)
    ams = calc_ams(scale*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 1), 'gen_weight']),
              scale*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 0), 'gen_weight']))
    
    print('\nMean ams={}+-{}, at mean cut of {}+-{}'.format(meanams[0], meanams[1], meanCut[0], meanCut[1]))
    print('Exact mean cut {}, corresponds to ams of {}'.format(np.mean(cuts), ams))
    return (meanams[0], meanCut[0])
