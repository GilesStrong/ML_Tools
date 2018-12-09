from __future__ import division

from .misc_functions import uncert_round

import numpy as np
import pandas as pd
import math
import multiprocessing as mp
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")


def calc_ams(s, b, br=0, delta_b=0):
    """ Approximate Median Significance defined as:
        calc_ams = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """

    if b == 0:
        return -1
    
    if not delta_b:
        radicand = 2 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)

    else:
        sigmaB2 = np.square(delta_b * b)
        radicand = 2 * (((s + b) * np.log((s + b) * (b + sigmaB2) / ((b ** 2) + ((s + b) * sigmaB2)))) - (((b ** 2) / sigmaB2) * np.log(1 + ((sigmaB2 * s) / (b * (b + sigmaB2))))))

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
        ams = calc_ams(max(0, s * w_factor), max(0, b * w_factor), br, delta_b)
        
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
        if n_bkg < min_events:
            continue

        s = np.sum(signal.loc[(signal.pred_class >= cut), 'gen_weight'])
        b = np.sum(bkg_pass)
        if use_stat_unc:
            delta_b = np.sqrt(syst_b2 + (1 / n_bkg))
        else:
            delta_b = syst_b
        ams = calc_ams(s * w_factor, b * w_factor, br, delta_b)
        
        if ams > max_ams:
            max_ams = ams
            threshold = cut
            
    return max_ams, threshold


def mp_calc_ams(data, i, w_factor, br, delta_b, out_q):
    ams, cut = ams_scan_quick(data, w_factor=w_factor, br=br, delta_b=delta_b)
    out_q.put({str(i) + '_ams': ams, str(i) + '_cut': cut})


def bootstrap_mean_calc_ams(data, w_factor=1, N=512, br=0, delta_b=0):
    procs = []
    out_q = mp.Queue()
    for i in range(N):
        indeces = np.random.choice(data.index, len(data), replace=True)
        p = mp.Process(target=mp_calc_ams, args=(data.iloc[indeces], i, w_factor, br, delta_b, out_q))
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

    ams = calc_ams(w_factor * np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 1), 'gen_weight']),
                   w_factor * np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 0), 'gen_weight']),
                   br=br, delta_b=delta_b)
    
    print('\nMean calc_ams={}+-{}, at mean cut of {}+-{}'.format(mean_ams[0], mean_ams[1], mean_cut[0], mean_cut[1]))
    print('Exact mean cut {}, corresponds to calc_ams of {}'.format(np.mean(cuts), ams))
    return (mean_ams[0], mean_cut[0], np.mean(cuts))


def kde_optimise_cut(in_data: pd.DataFrame, top_perc=0.02, min_pred=0.9,
                     w_factor=1.0, br=0.0, delta_b=0.0):
    '''Use a KDE to find a fluctaution resiliant cut which should generalise better'''

    sig = (in_data.gen_target == 1)
    bkg = (in_data.gen_target == 0)
    if 'ams' not in in_data.columns:
        in_data['ams'] = -1
        in_data.loc[in_data.pred_class >= min_pred, 'ams'] = in_data[in_data.pred_class >= min_pred].apply(lambda row:
                                                                                                           calc_ams(w_factor * np.sum(in_data.loc[(in_data.pred_class >= row.pred_class) & sig, 'gen_weight']),
                                                                                                                    w_factor * np.sum(in_data.loc[(in_data.pred_class >= row.pred_class) & bkg, 'gen_weight']),
                                                                                                                    br=br, delta_b=delta_b), axis=1)
        
    in_data.sort_values(by='ams', ascending=False, inplace=True)
    cuts = in_data['pred_class'].values[0:int(top_perc * len(in_data))]
    
    kde = sm.nonparametric.KDEUnivariate(cuts.astype('float64'))
    kde.fit()
    
    points = np.array([(x, kde.evaluate(x)) for x in np.linspace(cuts.min(), cuts.max(), 1000)])
    cut = points[np.argmax(points[:, 1])][0]
    ams = calc_ams(w_factor * np.sum(in_data.loc[(in_data.pred_class >= cut) & sig, 'gen_weight']),
                   w_factor * np.sum(in_data.loc[(in_data.pred_class >= cut) & bkg, 'gen_weight']),
                   br=br, delta_b=delta_b)
    
    print('Best cut at', cut, 'corresponds to AMS of', ams)
    print('Maximum AMS for data is', in_data.iloc[0]['ams'], 'at cut of', in_data[.iloc[0]['pred_class'])
    sns.distplot(cuts)
    
    return cut
