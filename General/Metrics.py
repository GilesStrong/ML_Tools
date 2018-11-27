from __future__ import division
import pandas
import numpy as np
import math
import multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

from .misc_functions import uncert_round

def calc_ams(s, b, br=0, delta_b=0):
    """ Approximate Median Significance defined as:
        calc_ams = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """

    if b == 0:
        return -1
    
    if not delta_b:
        radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)

    else:
        sigmaB2 = np.square(delta_b*b)
        radicand = 2*(((s+b)*np.log((s+b)*(b+sigmaB2)/((b**2)+((s+b)*sigmaB2))))-
                      (((b**2)/sigmaB2)*np.log(1+((sigmaB2*s)/(b*(b+sigmaB2))))))

    if radicand < 0:
        return -1
    else:
        return math.sqrt(radicand)

def ams_scan_quick(in_data, w_factor=1, br=0, delta_b=0):
    '''Determine optimum calc_ams and cut,
    w_factor used rescale weights to get comparable calc_amss
    sufferes from float precison'''
    max_ams = 0
    threshold = 0.0
    in_data = in_data.sort_values(by=['pred_class'])
    s = np.sum(in_data.loc[(in_data['gen_target'] == 1), 'gen_weight'])
    b = np.sum(in_data.loc[(in_data['gen_target'] == 0), 'gen_weight'])

    for i, cut in enumerate(in_data['pred_class']):
        ams = calc_ams(max(0, s*w_factor), max(0, b*w_factor), br, delta_b)
        
        if ams > max_ams:
            max_ams = ams
            threshold = cut
        if in_data['gen_target'].values[i]:
            s -= in_data['gen_weight'].values[i]
        else:
            b -= in_data['gen_weight'].values[i]
            
    return max_ams, threshold

def ams_scan_slow(in_data, w_factor=1, br=0, syst_b=0, use_stat_unc=False, start=0.9, min_events=10):
    '''Determine optimum calc_ams and cut,
    w_factor used rescale weights to get comparable calc_amss
    slower than quick, but doesn't suffer from float precision'''
    max_ams = 0
    threshold = 0.0
    signal = in_data[in_data['gen_target'] == 1]
    bkg = in_data[in_data['gen_target'] == 0]
    
    syst_b2 = np.square(syst_b)
    for i, cut in enumerate(in_data.loc[in_data.pred_class >= start, 'pred_class'].values):
        bkg_pass = bkg.loc[(bkg.pred_class >= cut), 'gen_weight']
        n_bkg = len(bkg_pass)
        if n_bkg < min_events: continue

        s = np.sum(signal.loc[(signal.pred_class >= cut), 'gen_weight'])
        b = np.sum(bkg_pass)
        if use_stat_unc:
            delta_b = np.sqrt(syst_b2+(1/n_bkg))
        else:
            delta_b = syst_b
        ams = calc_ams(s*w_factor, b*w_factor, br, delta_b)
        
        if ams > max_ams:
            max_ams = ams
            threshold = cut
            
    return max_ams, threshold

def mp_calc_ams(data, i, w_factor, br, out_q):
    ams, cut = ams_scan_quick(data, w_factor, br)
    out_q.put({str(i) + '_ams':ams, str(i) + '_cut':cut})

def mp_sk_fold_calc_ams(data, i, size, nFolds, br, out_q):
    kf = StratifiedKFold(n_splits=nFolds, shuffle=True)
    folds = kf.split(data, data['gen_target'])
    uids = range(i*nFolds,(i+1)*nFolds)
    out_dict = {}

    for j, (_, fold) in enumerate(folds):
        ams, cut = ams_scan_quick(data.iloc[fold], size/len(fold), br)
        if ams > 0:
            out_dict[str(uids[j]) + '_ams'] = ams
            out_dict[str(uids[j]) + '_cuts'] = cut
    out_q.put(out_dict)

def bootstrap_mean_calc_ams(data, w_factor=1, N=512, br=0):
    procs = []
    out_q = mp.Queue()
    for i in range(N):
        indeces = np.random.choice(data.index, len(data), replace=True)
        p = mp.Process(target=mp_calc_ams, args=(data.iloc[indeces], i, w_factor, br, out_q))
        procs.append(p)
        p.start() 
    result_dict = {}
    for i in range(N):
        result_dict.update(out_q.get()) 
    for p in procs:
        p.join()  
        
    amss = np.array([result_dict[x] for x in result_dict if 'ams' in x])
    cuts = np.array([result_dict[x] for x in result_dict if 'cut' in x])

    mean_ams = uncert_round(np.mean(amss), np.std(amss))
    mean_cut = uncert_round(np.mean(cuts), np.std(cuts))

    ams = calc_ams(w_factor*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 1), 'gen_weight']),
              w_factor*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 0), 'gen_weight']))
    
    print('\nMean calc_ams={}+-{}, at mean cut of {}+-{}'.format(mean_ams[0], mean_ams[1], mean_cut[0], mean_cut[1]))
    print('Exact mean cut {}, corresponds to calc_ams of {}'.format(np.mean(cuts), ams))
    return (mean_ams[0], mean_cut[0])

def bootstrap_sk_fold_mean_calc_ams(data, size=1, N=10, nFolds=500, br=0):
    print("Warning, this method might not be trustworthy: cut decreases with nFolds")
    procs = []
    out_q = mp.Queue()
    for i in range(N):
        p = mp.Process(target=mp_sk_fold_calc_ams, args=(data, i, size, nFolds, br, out_q))
        procs.append(p)
        p.start() 
    result_dict = {}
    for i in range(N):
        result_dict.update(out_q.get()) 
    for p in procs:
        p.join()  
        
    amss = np.array([result_dict[x] for x in result_dict if 'ams' in x])
    cuts = np.array([result_dict[x] for x in result_dict if 'cut' in x])

    mean_ams = uncert_round(np.mean(amss), np.std(amss)/np.sqrt(N*nFolds))
    mean_cut = uncert_round(np.mean(cuts), np.std(cuts)/np.sqrt(N*nFolds))

    scale = size/len(data)
    ams = calc_ams(scale*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 1), 'gen_weight']),
              scale*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 0), 'gen_weight']))
    
    print('\nMean calc_ams={}+-{}, at mean cut of {}+-{}'.format(mean_ams[0], mean_ams[1], mean_cut[0], mean_cut[1]))
    print('Exact mean cut {}, corresponds to calc_ams of {}'.format(np.mean(cuts), ams))
    return (mean_ams[0], mean_cut[0])
