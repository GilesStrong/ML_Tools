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
    
    def plot(self, n_skip=0, n_max=-1):
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
    def __init__(self, nb, cycle_mult=1, reverse=False):
        super(CosAnneal, self).__init__()
        self.nb = nb
        self.cycle_mult = cycle_mult
        self.cycle_iter = 0
        self.cycle_count = 0
        self.lrs = []
        self.lr = -1
        self.reverse = reverse
        self.cycle_end = False

    def on_train_begin(self, logs={}):
        if self.lr == -1:
            self.lr = float(K.get_value(self.model.optimizer.lr))
        self.cycle_end = False
        
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
            self.cycle_end = True
        if self.reverse:
            return self.lr-(self.lr / 2 * cos_out)
        else:
            return self.lr / 2 * cos_out

    def on_batch_end(self, batch, logs={}):
        lr = self.calc_lr(batch)
        self.lrs.append(lr)
        K.set_value(self.model.optimizer.lr, lr)

class CosAnnealMomentum(Callback):
    def __init__(self, nb, cycle_mult=1, reverse=False):
        super(CosAnnealMomentum, self).__init__()
        self.nb = nb
        self.cycle_mult = cycle_mult
        self.cycle_iter = 0
        self.cycle_count = 0
        self.moms = []
        self.momentum = -1
        self.reverse = reverse
        self.cycle_end = False

    def on_train_begin(self, logs={}):
        if self.momentum == -1:
            self.momentum = float(K.get_value(self.model.optimizer.momentum))
        self.cycle_end = False
        
    def plot_momentum(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("iterations")
        plt.ylabel("momentum")
        plt.plot(range(len(self.moms)), self.moms)
        plt.show()
        
    def calc_momentum(self, batch):
        cos_out = np.cos(np.pi*(self.cycle_iter)/self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            self.cycle_count += 1
            self.cycle_end = True
        if self.reverse:
            return self.momentum-(self.momentum / 10 * cos_out)
        else:
            return (self.momentum-(self.momentum / 5))+self.momentum / 10 * cos_out

    def on_batch_end(self, batch, logs={}):
        momentum = self.calc_momentum(batch)
        self.moms.append(momentum)
        K.set_value(self.model.optimizer.momentum, momentum)

class OneCycle(Callback):
    def __init__(self, nb, ratio=0.25, reverse=False, lrScale=10, momScale=0.1):
        '''nb=number of minibatches per epoch, ratio=fraction of epoch spent in first stage,
           lrScale=number used to divide initial LR to get minimum LR,
           momScale=number to subtract from initial momentum to get minimum momentum'''
        super(OneCycle, self).__init__()
        self.nb = nb
        self.ratio = ratio
        self.nSteps = (math.ceil(self.nb*self.ratio), math.floor((1-self.ratio)*self.nb))
        self.cycle_iter = 0
        self.cycle_count = 0
        self.lrs = []
        self.moms = []
        self.momentum = -1
        self.lr = -1
        self.reverse = reverse
        self.cycle_end = False
        self.lrScale = lrScale
        self.momScale = momScale
        self.momStep1 = -self.momScale/float(self.nSteps[0])
        self.momStep2 = self.momScale/float(self.nSteps[1])

    def on_train_begin(self, logs={}):
        if self.momentum == -1:
            self.momMax = float(K.get_value(self.model.optimizer.momentum))
            self.momMin = self.momMax-self.momScale
            if self.reverse: 
                self.momentum=self.momMin
                self.momStep1 *= -1
                self.momStep2 *= -1
                K.set_value(self.model.optimizer.momentum, self.momentum)
            else:
                self.momentum = self.momMax

            self.momStep = self.momStep1

        if self.lr == -1:
            self.lrMax = float(K.get_value(self.model.optimizer.lr))
            self.lrMin = self.lrMax/self.lrScale
            self.lrStep1 = (self.lrMax-self.lrMin)/self.nSteps[0]
            self.lrStep2 = -(self.lrMax-self.lrMin)/self.nSteps[1]
            if self.reverse:
                self.lrStep1 *= -1
                self.lrStep2 *= -1
                self.lr = self.lrMax
            else:
                self.lr = self.lrMin
                K.set_value(self.model.optimizer.lr, self.lr)

            self.lrStep = self.lrStep1

        self.moms.append(self.momentum)
        self.lrs.append(self.lr)
        self.cycle_end = False
        
    def plot(self):
        fig, axs = plt.subplots(2,1,figsize=(16,4))
        for ax in axs:
            ax.set_xlabel("Iterations")
        axs[0].set_ylabel("Learning Rate")
        axs[1].set_ylabel("Momentum")
        axs[0].plot(range(len(self.lrs)), self.lrs)
        axs[1].plot(range(len(self.moms)), self.moms)
        plt.show()
        
    def calc(self, batch):
        if self.cycle_iter == self.nSteps[0]+1:
            self.lrStep = self.lrStep2
            self.momStep = self.momStep2

        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            self.cycle_count += 1
            self.cycle_end = True
            if self.reverse:
                self.lr = self.lrMax
                self.momentum = self.momMin
            else:
                self.lr = self.lrMin
                self.momentum = self.momMax
            self.lrStep = self.lrStep1
            self.momStep = self.momStep1

        else:
            self.momentum += self.momStep
            self.lr += self.lrStep

        self.moms.append(self.momentum)
        self.lrs.append(self.lr)

    def on_batch_end(self, batch, logs={}):
        self.cycle_iter += 1
        self.calc(batch)
        K.set_value(self.model.optimizer.momentum, self.momentum)
        K.set_value(self.model.optimizer.lr, self.lr)