from __future__ import division

from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np

from ..general.misc_functions import uncert_round
from .bootstrap import roc_auc, mp_run

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


def plot_feat(in_data, feat, cuts=None, labels=None, plot_bulk=True, weight_name=None, n_samples=100000, params={}, moments=False):
    loop = False
    if not isinstance(cuts, type(None)):
        if isinstance(cuts, list):
            loop = True
            if not isinstance(cuts, list):
                print("{} plots requested, but not labels passed".format(len(cuts)))
                return -1
            elif len(cuts) != len(labels):
                print("{} plots requested, but {} labels passed".format(len(cuts), len(labels)))
                return -1
    
    plt.figure(figsize=(16, 8))
    if loop:
        for i in range(len(cuts)):
            if isinstance(params, list):
                tmp_params = params[i]
            else:
                tmp_params = params

            if plot_bulk:  # Ignore tails for indicative plotting
                feat_range = np.percentile(in_data[feat], [1, 99])
                # feat_range = np.percentile(in_data.loc[cuts[i], feat], [1,99])
                if feat_range[0] == feat_range[1]: break
                
                cut = (cuts[i])
                cut = cut & (in_data[cut][feat] > feat_range[0]) & (in_data[cut][feat] < feat_range[1])
                if isinstance(weight_name, type(None)):
                    plot_data = in_data.loc[cut, feat]
                else:
                    plot_data = np.random.choice(in_data.loc[cut, feat], n_samples, p=in_data.loc[cut, weight_name] / np.sum(in_data.loc[cut, weight_name]))
                    
            else:
                if isinstance(weight_name, type(None)):
                    plot_data = in_data.loc[cuts[i], feat]
                else:
                    plot_data = np.random.choice(in_data.loc[cuts[i], feat], n_samples, p=in_data.loc[cuts[i], weight_name] / np.sum(in_data.loc[cuts[i], weight_name]))
            
            label = labels[i]
            if moments:
                label += r', $\bar{x}=$' + str(np.mean(plot_data)) + r', $\sigma_x=$' + str(np.std(plot_data))

            sns.distplot(plot_data, label=labels[i], **tmp_params)
    else:
        if plot_bulk:  # Ignore tails for indicative plotting
            feat_range = np.percentile(in_data[feat], [1, 99])
            if feat_range[0] == feat_range[1]: return -1
            
            cut = (in_data[feat] > feat_range[0]) & (in_data[feat] < feat_range[1])
             
            if isinstance(weight_name, type(None)):
                plot_data = in_data.loc[cut, feat]
            else:
                plot_data = np.random.choice(in_data.loc[cut, feat], n_samples, p=in_data.loc[cut, weight_name] / np.sum(in_data.loc[cut, weight_name]))     
                
        else:
            if isinstance(weight_name, type(None)):
                plot_data = in_data[feat]
            else:
                plot_data = np.random.choice(in_data[feat], n_samples, p=in_data[weight_name] / np.sum(in_data[weight_name]))
        
        label = ''
        if moments:
            label += r', $\bar{x}=$' + str(np.mean(plot_data)) + r', $\sigma_x=$' + str(np.std(plot_data))
        sns.distplot(plot_data, label=label, **params)

    if loop or moments:
        plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.ylabel("Density", fontsize=24, color='black')
    plt.xlabel(feat, fontsize=24, color='black')
    plt.show()


def roc_plot(in_data=None, curves=None, pred_name='pred_class', target_name='gen_target', weight_name=None, labels=None, aucs=None, bootstrap=False, log=False, baseline=True, params=[{}]):
    build_curves = True
    if isinstance(in_data, type(None)) == isinstance(curves, type(None)):
        print("Must pass either targets and preds, or curves")
        return -1
    if not isinstance(curves, type(None)):
        build_curves = False

    if build_curves:
        curves = {}
        if bootstrap:
            auc_args = []
            for i in range(len(in_data)):
                auc_args.append({'labels': in_data[i][target_name], 'preds': in_data[i][pred_name], 'name': labels[i], 'indeces': in_data[i].index.tolist()})
                if not isinstance(weight_name, type(None)):
                    auc_args[-1]['weights'] = in_data[i][weight_name]
            aucs = mp_run(auc_args, roc_auc)
            mean_scores = {}
            for i in labels:
                mean_scores[i] = (np.mean(aucs[i]), np.std(aucs[i]))
                print(str(i) + ' ROC AUC, Mean = {} +- {}'.format(mean_scores[i][0], mean_scores[i][1]))
        else:
            mean_scores = {}
            for i in range(len(in_data)):
                if isinstance(weight_name, type(None)):
                    mean_scores[labels[i]] = roc_auc_score(in_data[i][target_name].values, in_data[i][pred_name])
                else:
                    mean_scores[labels[i]] = roc_auc_score(in_data[i][target_name].values, in_data[i][pred_name], sample_weight=in_data[i][weight_name])
                print(str(i) + ' ROC AUC: {}'.format(mean_scores[labels[i]]))
        for i in range(len(in_data)):
            if isinstance(weight_name, type(None)):
                curves[labels[i]] = roc_curve(in_data[i][target_name].values, in_data[i][pred_name].values)[:2]
            else:
                curves[labels[i]] = roc_curve(in_data[i][target_name].values, in_data[i][pred_name].values, sample_weight=in_data[i][weight_name].values)[:2]

    plt.figure(figsize=[8, 8])
    for i in range(len(curves)):
        if build_curves:
            if bootstrap:
                mean_score = uncert_round(*mean_scores[labels[i]])
                plt.plot(*curves[labels[i]], label=labels[i] + r', AUC$={}\pm{}$'.format(mean_score[0], mean_score[1]), **params[i])
            else:
                plt.plot(*curves[labels[i]], label=labels[i] + r', AUC$={:.5f}$'.format(mean_scores[labels[i]]), **params[i])
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
        plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.show()


def get_class_pred_plot(in_data, labels=['Background', 'Signal'], pred_name='pred_class', weight_name=None,
                        lim=(0, 1), logy=True, params={'hist': True, 'kde': False, 'norm_hist': True}):
    plt.figure(figsize=(16, 8))
    for i in range(len(in_data)):
        hist_kws = {}
        if not isinstance(weight_name, type(None)):
            hist_kws['weights'] = in_data[i][weight_name]
        sns.distplot(in_data[i][pred_name], label=labels[i], hist_kws=hist_kws, **params)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Class prediction", fontsize=24, color='black')
    plt.xlim(lim)
    plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=24, color='black')
    if logy:
        plt.yscale('log', nonposy='clip')
        plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.show() 


def _get_samples(in_data, sample_name, weight_name):
    samples = set(in_data[sample_name])
    weights = [np.sum(in_data[in_data[sample_name] == sample][weight_name]) for sample in samples]
    return [x[0] for x in np.array(sorted(zip(samples, weights), key=lambda x: x[1]))]  # Todo improve sorting


def get_sample_pred_plot(in_data, 
                         target_name='gen_target', sample_name='gen_sample', pred_name='pred_class', weight_name='gen_weight',
                         lim=(0, 1), nBins=35, logy=True, pallet='magma', desat=1,
                         hist_params={'normed': True, 'alpha': 1, 'stacked': True, 'rwidth': 1.0}):
    
    hist_params['bins'] = nBins
    hist_params['range'] = lim
    
    plt.figure(figsize=(16, 8))
    
    sig = (in_data[target_name] == 1)
    bkg = (in_data[target_name] == 0)
    
    with sns.color_palette(pallet, len(set(in_data[sample_name])), desat=desat):
        
        samples = _get_samples(in_data[bkg], sample_name, weight_name)
        plt.hist([in_data[in_data[sample_name] == sample][pred_name] for sample in samples],
                 weights=[in_data[in_data[sample_name] == sample][weight_name] for sample in samples],
                 label=samples, **hist_params)

        samples = _get_samples(in_data[sig], sample_name, weight_name)
        for sample in samples:
            plt.hist(in_data[in_data[sample_name] == sample][pred_name],
                     weights=in_data[in_data[sample_name] == sample][weight_name],
                     label='Signal ' + sample, histtype='step', linewidth='3', **hist_params)

        plt.legend(loc='best', fontsize=16)
        plt.xlabel("Class prediction", fontsize=24, color='black')
        plt.xlim(lim)
        if hist_params['normed']:
            plt.ylabel(r"$\frac{1}{\mathcal{A}\sigma} \frac{d\left(\mathcal{A}\sigma\right)}{dp}$", fontsize=24, color='black')
        else:
            plt.ylabel(r"$\frac{d\left(\mathcal{A}\sigma\right)}{dp}\ [pb]$", fontsize=24, color='black')
        if logy:
            plt.yscale('log', nonposy='clip')
            plt.grid(True, which="both")
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.show()      
    

def plot_training_history(histories, save=False):
    plt.figure(figsize=(16, 8))
    for i, history in enumerate(histories):
        if i == 0:
            try:
                plt.plot(history['loss'], color='g', label='Training')
            except Exception:
                pass
            try:
                plt.plot(history['val_loss'], color='b', label='Validation')
            except Exception:
                pass
            try:
                plt.plot(history['mon_loss'], color='r', label='Monitoring')
            except Exception:
                pass
            try:
                plt.plot(history['swa_val_loss'], color='purple', label='SWA Validation')
            except Exception:
                pass
        else:
            try:
                plt.plot(history['loss'], color='g')
            except Exception:
                pass
            try:
                plt.plot(history['val_loss'], color='b')
            except Exception:
                pass
            try:
                plt.plot(history['mon_loss'], color='r')
            except Exception:
                pass
            try:
                plt.plot(history['swa_val_loss'], color='purple')
            except Exception:
                pass

    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Epoch", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()

    if save:
        plt.savefig(save)


def get_model_history_comparison_plot(histories, names, cv=False, log_y=False):
    '''Compare validation loss evolution for several models
    cv=True expects history for CV training and plots mean and 68% CI bands'''
    plt.figure(figsize=(16, 8))
    
    for i, (history, name) in enumerate(zip(histories, names)):
        if cv:
            sns.tsplot([history[x]['val_loss'] for x in range(len(history))], condition=name, color=sns.color_palette()[i])
        else:
            plt.plot(history['val_loss'], label=name)

    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Epoch", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    if log_y:
        plt.yscale('log')
        plt.grid(True, which="both")
    plt.show()


def get_lr_finder_comparison_plot(lr_finders, names, log_x=True, log_y=True, loss='loss', cut=-10):
    '''Compare loss evolultion against learning rate for several LRFinder callbacks'''
    plt.figure(figsize=(16, 8))
    
    for lr_finder, name in zip(lr_finders, names):
        plt.plot(lr_finder.history['lr'][:cut], lr_finder.history[loss][:cut], label=name)

    plt.legend(loc='best', fontsize=16)
    if log_x:
        plt.xscale('log')
        plt.grid(True, which="both")
    if log_y:
        plt.yscale('log')
        plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()


def get_lr_finder_mean_plot(lr_finders, loss='loss', cut=-10):
    '''Get mean loss evolultion against learning rate for several LRFinder callbacks'''
    plt.figure(figsize=(16, 8))
    min_len = np.min([len(lr_finders[x].history[loss][:cut]) for x in range(len(lr_finders))])
    
    sns.tsplot([lr_finders[x].history[loss][:min_len] for x in range(len(lr_finders))],
               time=lr_finders[0].history['lr'][:min_len], ci='sd')

    plt.legend(loc='best', fontsize=16)
    plt.xscale('log')
    plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()


def get_monitor_comparison_plot(monitors, names, x_axis='iter', y_axis='Loss', lr_log_x=True, log_y=True):
    '''Compare validation loss.accuracy evolution for several models on a per iteration/learning-rate/momentum basis'''
    plt.figure(figsize=(16, 8))
    for monitor, name in zip(monitors, names):
        if isinstance(monitor.history['val_loss'][0], list):
            if y_axis == 'Loss':
                y = np.array(monitor.history['val_loss'])[:, 0]
            else:
                y = np.array(monitor.history['val_loss'])[:, 1]
        else:
            y = monitor.history['val_loss']
                
        if x_axis == 'iter':
            plt.plot(range(len(monitor.history['val_loss'])), y, label=name)
        elif x_axis == 'mom':
            plt.plot(monitor.history['mom'], y, label=name)
        else:
            plt.plot(monitor.history['lr'], y, label=name)

    plt.legend(loc='best', fontsize=16)
    if lr_log_x: plt.xscale('log')
    if log_y: plt.yscale('log')
    plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    if x_axis == 'iter':
        plt.xlabel("Iteration", fontsize=24, color='black')
    elif x_axis == 'mom':
        plt.xlabel("Momentum", fontsize=24, color='black')
    else:
        plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel(y_axis, fontsize=24, color='black')
    plt.show()
    