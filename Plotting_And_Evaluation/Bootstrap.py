import multiprocessing as mp
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

def bootstrap(args, out_q):
    out_dict = {}
    mean = []
    std = []
    c68 = []
    boot = []
    if 'name' not in args: args['name'] = ''
    if 'n'    not in args: args['n']    = 100
    if 'kde'  not in args: args['kde']  = False
    if 'mean' not in args: args['mean'] = False
    if 'std'  not in args: args['std']  = False  
    if 'c68'  not in args: args['c68']  = False  
    for i in range(args['n']):
        points = np.random.choice(args['data'], len(args['data']), replace=True)
        if args['kde']:
            kde = sm.nonparametric.KDEUnivariate(points)
            kde.fit()
            boot.append([kde.evaluate(x) for x in args['x']])
        if args['mean']:
            mean.append(points.mean())
        if args['std']:
            std.append(points.std())
        if args['c68']:
            c68.append(np.percentile(np.abs(points), 68.2))
    if args['kde']:  out_dict[args['name'] + '_kde']  = boot
    if args['mean']: out_dict[args['name'] + '_mean'] = mean
    if args['std']:  out_dict[args['name'] + '_std']  = std
    if args['c68']:  out_dict[args['name'] + '_c68']  = c68
    out_q.put(out_dict)

def roc_auc(args, out_q):
    out_dict = {}
    boot = []
    if 'name' not in args: args['name'] = ''
    if 'n'    not in args: args['n'] = 100
    if 'weights' in args: 
        for i in range(args['n']):
            points = np.random.choice(args['indeces'], len(args['indeces']), replace=True)
            boot.append(roc_auc_score(args['labels'].loc[points].values, 
                                      args['preds'].loc[points].values,
                                      sample_weight=args['weights'].loc[points].values))
    else:
        for i in range(args['n']):
            points = np.random.choice(args['indeces'], len(args['indeces']), replace=True)
            boot.append(roc_auc_score(args['labels'].loc[points].values, 
                                      args['preds'].loc[points].values))
    out_dict[args['name']] = boot
    out_q.put(out_dict)

def mp_run(args, target=bootstrap):
    procs = []
    out_q = mp.Queue()
    for i in range(len(args)):
        p = mp.Process(target=target, args=(args[i], out_q))
        procs.append(p)
        p.start() 
    result_dict = {}
    for i in range(len(args)):
        result_dict.update(out_q.get()) 
    for p in procs:
        p.join()  
    return result_dict
    