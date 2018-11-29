from __future__ import division

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import utils

from six.moves import cPickle as pickle
import timeit
import numpy as np
import os
from pathlib import Path

from .callbacks import LossHistory
from .misc_functions import uncert_round
from ..plotting_and_evaluation.plotters import plot_training_history


def train_classifier(X, y, n_splits, model_gen, model_gen_params, train_params,
                     class_weights='auto', sample_weights=None,
                     saveloc=Path('train_weights'), patience=10):
    start = timeit.default_timer()
    results = []
    histories = []
    os.system("mkdir " + saveloc)
    os.system("rm " + saveloc + "*.h5")
    os.system("rm " + saveloc + "*.json")
    os.system("rm " + saveloc + "*.pkl")

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    folds = kf.split(X, y)

    binary = True
    n_classes = len(np.unique(y))
    if n_classes > 2:
        print(n_classes, "classes found, running in multiclass mode\n")
        y = utils.to_categorical(y, num_classes=n_classes)
        binary = False
        model_gen_params['nOut'] = n_classes
    else:
        print(n_classes, "classes found, running in binary mode\n")

    for i, (train, test) in enumerate(folds):
        print("Running fold", i + 1, "/", n_splits)
        os.system("rm " + saveloc + "best.h5")
        fold_start = timeit.default_timer()

        model = None
        model = model_gen(**model_gen_params)
        model.reset_states   # Just checking

        loss_history = LossHistory((X[train], y[train]))
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
        save_best = ModelCheckpoint(saveloc / "best.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        
        weights = None
        if not isinstance(sample_weights, type(None)):
            weights = sample_weights[train]
        
        model.fit(X[train], y[train],
                  validation_data=(X[test], y[test]),
                  callbacks=[early_stop, save_best, loss_history],
                  class_weight=class_weights, sample_weight=weights,
                  **train_params)
        histories.append(loss_history.losses)
        model.load_weights(saveloc / "best.h5")

        results.append({})
        results[-1]['loss'] = model.evaluate(X[test], y[test], verbose=0)
        if binary: results[-1]['AUC'] = 1 - roc_auc_score(y[test], model.predict(X[test], verbose=0), sample_weight=weights)
        print("Score is:", results[-1])

        print("Fold took {:.3f}s\n".format(timeit.default_timer() - fold_start))

        model.save(saveloc / 'train_' + str(i) + '.h5')
        with open(saveloc / 'resultsFile.pkl', 'wb') as fout:  # Save results
            pickle.dump(results, fout)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    plot_training_history(histories, saveloc + 'history.png')

    mean_loss = uncert_round(np.mean([x['loss'] for x in results]), np.std([x['loss'] for x in results]) / np.sqrt(len(results)))
    print("Mean loss = {} +- {}".format(mean_loss[0], mean_loss[1]))
    if binary:
        mean_auc = uncert_round(np.mean([x['AUC'] for x in results]), np.std([x['AUC'] for x in results]) / np.sqrt(len(results)))
        print("Mean AUC = {} +- {}".format(mean_auc[0], mean_auc[1]))
    print("______________________________________\n")

    return results, histories
