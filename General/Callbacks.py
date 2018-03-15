from __future__ import division

from keras.callbacks import Callback
from keras import backend as K

import math
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

class LossHistory(Callback):
    def __init__(self, trData):
        self.trainingData = trData

    def on_train_begin(self, logs={}):
        self.losses = {}
        self.losses['loss'] = []
        self.losses['val_loss'] = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses['loss'].append(self.model.evaluate(self.trainingData[0], self.trainingData[1], verbose=0))
        self.losses['val_loss'].append(logs.get('val_loss'))

class LRFinder(Callback):
    '''Adapted from fastai version'''

    def __init__(self, nSteps, lrBounds=[1e-7, 10], verbose=0):
        super(LRFinder, self).__init__()
        self.verbose = verbose
        
        self.lrBounds=lrBounds
        ratio = self.lrBounds[1]/self.lrBounds[0]
        self.lr_mult = ratio**(1/nSteps)
        
    def on_train_begin(self, logs={}):
        self.best=1e9
        K.set_value(self.model.optimizer.lr, self.lrBounds[0])
        self.history = {}
        self.history['loss'] = []
        self.history['lr'] = []
        
    def calc_lr(self, lr, batch):
        return self.lrBounds[0]*(self.lr_mult**batch)
    
    def plot(self, n_skip=10, n_max=-5):
        plt.figure(figsize=(16,8))
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.history['lr'][n_skip:n_max], self.history['loss'][n_skip:n_max])
        plt.xscale('log')
        plt.show()
        
    def plot_lr(self):
        plt.figure(figsize=(4,4))
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(range(len(self.history['lr'])), self.history['lr'])
        plt.show()

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        self.history['loss'].append(logs.get('loss'))
        self.history['lr'].append(float(K.get_value(self.model.optimizer.lr)))
        
        lr = self.calc_lr(float(K.get_value(self.model.optimizer.lr)), batch+1)
            
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('Batch %05d: LearningRateFinder increasing learning '
                  'rate to %s.' % (batch + 1, lr))
        
        if math.isnan(loss) or loss>self.best*10:
            if self.verbose > 0:
                print('Ending training early due to loss increase')
                self.model.stop_training = True
        if (loss<self.best and batch>10): self.best=loss

class CosAnneal(Callback):
    '''Adapted from fastai version'''
    def __init__(self, nb, cycle_mult=1):
        super(CosAnneal, self).__init__()
        self.nb = nb
        self.cycle_mult = cycle_mult
        self.cycle_iter = 0
        self.cycle_count = 0
        self.lrs = []
        self.lr = -1

    def on_train_begin(self, logs={}):
        if self.lr == -1:
            self.lr = float(K.get_value(self.model.optimizer.lr))
        
    def plot_lr(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(range(len(self.lrs)), self.lrs)
        plt.show()
        
    def calc_lr(self, batch):
        '''if batch < self.nb/20:
            self.cycle_iter += 1
            return self.lr/100.'''

        cos_out = np.cos(np.pi*(self.cycle_iter)/self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            self.cycle_count += 1
        return self.lr / 2 * cos_out

    def on_batch_end(self, batch, logs={}):
        lr = self.calc_lr(batch)
        self.lrs.append(lr)
        K.set_value(self.model.optimizer.lr, lr)