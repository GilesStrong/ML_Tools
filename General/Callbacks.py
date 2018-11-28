from __future__ import division

from keras.callbacks import Callback
from keras import backend as K

import math
import numpy as np
import types

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

class ValidationMonitor(Callback):
    '''Callback to monitor validation performance and optimiser settings after every minibatch
    For short training diagnosis; slow to run, do not use for full training'''
    def __init__(self, val_data=None, val_batch_size=None, mode='sgd'):
        super(ValidationMonitor, self).__init__()
        self.val_data = val_data
        self.val_batch_size = val_batch_size
        self.mode = mode

    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.history['lr'] = []
        self.history['mom'] = []
        self.history['acc'] = []

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('acc'))
        
        if not isinstance(self.val_data, type(None)):
            mb_mask = np.zeros(len(self.val_data['y']), dtype=int)
            mb_mask[:self.val_batch_size] = 1
            np.random.shuffle(mb_mask)
            mb_mask = mb_mask.astype(bool)
            self.history['val_loss'].append(self.model.evaluate(x=self.val_data['x'][mb_mask],
                                                                y=self.val_data['y'][mb_mask],
                                                                sample_weight=self.val_data['sample_weight'][mb_mask], 
                                                                verbose=0))
                
        self.history['lr'].append(float(K.get_value(self.model.optimizer.lr)))

        if self.mode == 'sgd':
            self.history['mom'].append(float(K.get_value(self.model.optimizer.momentum)))
        elif self.mode == 'adam':
            self.history['mom'].append(float(K.get_value(self.model.optimizer.beta_1)))

class LossHistory(Callback):
    def __init__(self, train_data):
        self.train_data = train_data

    def on_train_begin(self, logs={}):
        self.losses = {}
        self.losses['loss'] = []
        self.losses['val_loss'] = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses['loss'].append(self.model.evaluate(self.train_data[0], self.train_data[1], verbose=0))
        self.losses['val_loss'].append(logs.get('val_loss'))

class LRFinder(Callback):
    '''Learning rate finder callback 
    - adapted from fastai version to work in Keras and to optionally run over validation data'''

    def __init__(self, n_steps, val_data=None, val_batch_size=None, lr_bounds=[1e-7, 10], verbose=0):
        super(LRFinder, self).__init__()
        self.verbose = verbose
        self.lr_bounds=lr_bounds
        ratio = self.lr_bounds[1]/self.lr_bounds[0]
        self.lr_mult = ratio**(1/n_steps)
        self.val_data = val_data
        self.val_batch_size = val_batch_size
        
    def on_train_begin(self, logs={}):
        self.best=1e9
        self.iter = 0
        K.set_value(self.model.optimizer.lr, self.lr_bounds[0])
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.history['lr'] = []
        
    def calc_lr(self, lr, batch):
        return self.lr_bounds[0]*(self.lr_mult**batch)
    
    def plot(self, n_skip=0, n_max=-1, yLim=None):
        plt.figure(figsize=(16,8))
        plt.plot(self.history['lr'][n_skip:n_max], self.history['loss'][n_skip:n_max], label='Training loss', color='g')
        if not isinstance(self.val_data, type(None)):
            plt.plot(self.history['lr'][n_skip:n_max], self.history['val_loss'][n_skip:n_max], label='Validation loss', color='b')
        
        if np.log10(self.lr_bounds[1])-np.log10(self.lr_bounds[0]) >= 3:
            plt.xscale('log')
        plt.ylim(yLim)
        plt.grid(True, which="both")
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.ylabel("Loss", fontsize=24, color='black')
        plt.xlabel("Learning rate", fontsize=24, color='black')
        plt.show()
        
    def plot_lr(self):
        plt.figure(figsize=(4,4))
        plt.xlabel("Iterations", fontsize=24, color='black')
        plt.ylabel("Learning rate", fontsize=24, color='black')
        plt.plot(range(len(self.history['lr'])), self.history['lr'])
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.show()
    
    def plot_genError(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("Iterations", fontsize=24, color='black')
        plt.ylabel("Generalisation Error", fontsize=24, color='black')
        plt.plot(range(len(self.history['lr'])), np.array(self.history['val_loss'])-np.array(self.history['loss']))
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.show()

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        self.history['loss'].append(logs.get('loss'))
        
        if not isinstance(self.val_data, type(None)):
            mb_mask = np.zeros(len(self.val_data['y']), dtype=int)
            mb_mask[:self.val_batch_size] = 1
            np.random.shuffle(mb_mask)
            mb_mask = mb_mask.astype(bool)
            self.history['val_loss'].append(self.model.evaluate(x=self.val_data['x'][mb_mask],
                                                                y=self.val_data['y'][mb_mask],
                                                                sample_weight=self.val_data['sample_weight'][mb_mask], 
                                                                verbose=0))
                
        self.history['lr'].append(float(K.get_value(self.model.optimizer.lr)))
        
        self.iter += 1
        lr = self.calc_lr(float(K.get_value(self.model.optimizer.lr)), self.iter)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('Batch %05d: LearningRateFinder increasing learning '
                  'rate to %s.' % (self.iter, lr))
        
        if math.isnan(loss) or loss>self.best*10:
            if self.verbose > 0:
                print('Ending training early due to loss increase')
            self.model.stop_training = True
        if (loss<self.best and self.iter>10): self.best=loss

class LinearCLR(Callback):
    '''Cyclical learning rate callback with linear interpolation'''
    def __init__(self, nb, max_lr, min_lr, scale=2, reverse=False):
        super(LinearCLR, self).__init__()
        self.nb = nb*scale
        self.cycle_iter = 0
        self.cycle_count = 0
        self.lrs = []
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.reverse = reverse
        self.cycle_end = False

    def on_train_begin(self, logs={}):
        self.cycle_end = False

    def on_train_end(self, logs={}):
        if self.plot_lr:
            self.plot_lr()
        
    def plot_lr(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("iterations", fontsize=24, color='black')
        plt.ylabel("learning rate", fontsize=24, color='black')
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.plot(range(len(self.lrs)), self.lrs)
        plt.show()
        
    def calc_lr(self, batch):
        cycle = math.floor(1+(self.cycle_iter/(2*self.nb)))
        x = np.abs((self.cycle_iter/self.nb)-(2*cycle)+1)
        lr = self.min_lr+((self.max_lr-self.min_lr)*np.max([0, 1-x]))

        self.cycle_iter += 1
        if self.reverse:
            return self.max_lr-(lr-self.min_lr)
        else:
            return lr

    def on_batch_end(self, batch, logs={}):
        lr = self.calc_lr(batch)
        self.lrs.append(lr)
        K.set_value(self.model.optimizer.lr, lr)

class CosAnnealLR(Callback):
    '''Adapted from fastai version'''
    def __init__(self, nb, cycle_mult=1, reverse=False):
        super(CosAnnealLR, self).__init__()
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
        plt.xlabel("iterations", fontsize=24, color='black')
        plt.ylabel("learning rate", fontsize=24, color='black')
        plt.plot(range(len(self.lrs)), self.lrs)
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
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

class LinearCMom(Callback):
    '''Cyclical momentum callback with linear interpolation'''
    def __init__(self, nb, max_mom, min_mom, scale=2, reverse=False, mode='sgd'):
        super(LinearCMom, self).__init__()
        self.nb = nb*scale
        self.cycle_iter = 0
        self.cycle_count = 0
        self.moms = []
        self.max_mom = max_mom
        self.min_mom = min_mom
        self.reverse = reverse
        self.cycle_end = False
        self.mode = mode

    def on_train_begin(self, logs={}):
        self.cycle_end = False

    def on_train_end(self, logs={}):
        if self.plot_mom:
            self.plot_mom()
        
    def plot_mom(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("iterations", fontsize=24, color='black')
        plt.ylabel("momentum", fontsize=24, color='black')
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.plot(range(len(self.moms)), self.moms)
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.show()
        
    def calc_mom(self, batch):
        cycle = math.floor(1+(self.cycle_iter/(2*self.nb)))
        x = np.abs((self.cycle_iter/self.nb)-(2*cycle)+1)
        mom = self.min_mom+((self.max_mom-self.min_mom)*np.max([0, 1-x]))

        self.cycle_iter += 1
        if not self.reverse:
            return self.max_mom-(mom-self.min_mom)
        else:
            return mom

    def on_batch_end(self, batch, logs={}):
        mom = self.calc_mom(batch)
        self.moms.append(mom)
        if self.mode == 'sgd':
            K.set_value(self.model.optimizer.momentum, mom)
        elif self.mode == 'adam':
            K.set_value(self.model.optimizer.beta_1, mom)

class CosAnnealMom(Callback):
    def __init__(self, nb, cycle_mult=1, reverse=False):
        super(CosAnnealMom, self).__init__()
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
        plt.xlabel("iterations", fontsize=24, color='black')
        plt.ylabel("momentum", fontsize=24, color='black')
        plt.plot(range(len(self.moms)), self.moms)
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
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
    def __init__(self, nb, scale=30, ratio=0.5, reverse=False, lr_scale=10, mom_scale=0.1, mode='sgd'):
        '''nb=number of minibatches per epoch, ratio=fraction of epoch spent in first stage,
           lr_scale=number used to divide initial LR to get minimum LR,
           mom_scale=number to subtract from initial momentum to get minimum momentum'''
        super(OneCycle, self).__init__()
        self.nb = nb*scale
        self.ratio = ratio
        self.n_steps = (math.ceil(self.nb*self.ratio), math.floor((1-self.ratio)*self.nb))
        self.cycle_iter = 0
        self.cycle_count = 0
        self.lrs = []
        self.moms = []
        self.momentum = -1
        self.lr = -1
        self.reverse = reverse
        self.cycle_end = False
        self.lr_scale = lr_scale
        self.mom_scale = mom_scale
        self.mom_step_1 = -self.mom_scale/float(self.n_steps[0])
        self.mom_step_2 = self.mom_scale/float(self.n_steps[1])
        self.mode = mode.lower()

    def on_train_begin(self, logs={}):
        if self.momentum == -1:
            if self.mode == 'sgd':
                self.mom_max = float(K.get_value(self.model.optimizer.momentum))
            elif self.mode == 'adam':
                self.mom_max = float(K.get_value(self.model.optimizer.beta_1))
            self.mom_min = self.mom_max-self.mom_scale
            if self.reverse: 
                self.momentum = self.mom_min
                self.mom_step_1 *= -1
                self.mom_step_2 *= -1
                if self.mode == 'sgd':
                    K.set_value(self.model.optimizer.momentum, self.momentum)
                elif self.mode == 'adam':
                    K.set_value(self.model.optimizer.beta_1, self.momentum)
            else:
                self.momentum = self.mom_max

            self.mom_step = self.mom_step_1

        if self.lr == -1:
            self.lr_max = float(K.get_value(self.model.optimizer.lr))
            self.lr_min = self.lr_max/self.lr_scale
            self.lr_step_1 = (self.lr_max-self.lr_min)/self.n_steps[0]
            self.lr_step_2 = -(self.lr_max-self.lr_min)/self.n_steps[1]
            if self.reverse:
                self.lr_step_1 *= -1
                self.lr_step_2 *= -1
                self.lr = self.lr_max
            else:
                self.lr = self.lr_min
                K.set_value(self.model.optimizer.lr, self.lr)

            self.lr_step = self.lr_step_1

        self.moms.append(self.momentum)
        self.lrs.append(self.lr)
        self.cycle_end = False

    def plot_mom(self):
        self.plot_lr()

    def plot_lr(self):
        fig, axs = plt.subplots(2,1,figsize=(16,4))
        for ax in axs:
            ax.set_xlabel("Iterations", fontsize=24, color='black')
        axs[0].set_ylabel("Learning Rate", fontsize=24, color='black')
        axs[1].set_ylabel("Momentum", fontsize=24, color='black')
        axs[0].plot(range(len(self.lrs)), self.lrs)
        axs[1].plot(range(len(self.moms)), self.moms)
        plt.show()
        
    def calc(self, batch):
        if self.cycle_iter == self.n_steps[0]+1:
            self.lr_step = self.lr_step_2
            self.mom_step = self.mom_step_2

        if self.cycle_iter>=self.nb:
            self.lr/2

        else:
            self.momentum += self.mom_step
            self.lr += self.lr_step

        self.moms.append(self.momentum)
        self.lrs.append(self.lr)

    def on_batch_end(self, batch, logs={}):
        self.cycle_iter += 1
        self.calc(batch)
        if self.mode == 'sgd':
            K.set_value(self.model.optimizer.momentum, self.momentum)
        elif self.mode == 'adam':
            K.set_value(self.model.optimizer.beta_1, self.momentum)
        K.set_value(self.model.optimizer.lr, self.lr)

class SWA(Callback):
    '''Modified from fastai version'''
    def __init__(self, start, test_fold, test_model, verbose=False, renewal=-1,
                 lr_callback=None, train_on_weights=False, sgd_replacement=False):
        super(SWA, self).__init__()
        self.swa_model = None
        self.swa_model_new = None
        self.start = start
        self.epoch = -1
        self.swa_n = -1
        self.renewal = renewal
        self.n_since_renewal = -1
        self.losses = {'swa':None, 'base':None}
        self.active = False
        self.test_fold = test_fold
        self.weighted = train_on_weights
        self.lr_callback = lr_callback
        self.test_model = test_model
        self.verbose = verbose
        self.sgd_replacement = sgd_replacement
        
    def on_train_begin(self, logs={}):
        if isinstance(self.swa_model, type(None)):
            self.swa_model = self.model.get_weights()
            self.swa_model_new = self.model.get_weights()
            self.epoch = 0
            self.swa_n = 0
            self.n_since_renewal = 0
            self.first_completed= False
            self.cylcle_since_replacement = 1
            
    def on_epoch_begin(self, metrics, logs={}):
        self.losses = {'swa':None, 'base':None}

    def on_epoch_end(self, metrics, logs={}):
        if (self.epoch + 1) >= self.start and (isinstance(self.lr_callback, type(None)) or self.lr_callback.cycle_end):
            if self.swa_n == 0 and not self.active:
                print ("SWA beginning")
                self.active = True
            elif not isinstance(self.lr_callback, type(None)) and self.lr_callback.cycle_mult > 1:
                print ("Updating average")
                self.active = True
            self.update_average_model()
            self.swa_n += 1
            
            if self.swa_n > self.renewal:
                self.first_completed = True
                self.n_since_renewal += 1
                if self.n_since_renewal > self.cylcle_since_replacement*self.renewal and self.renewal > 0:
                    self.compareAverages()
            
        if isinstance(self.lr_callback, type(None)) or self.lr_callback.cycle_end:
            self.epoch += 1

        if self.active and not (isinstance(self.lr_callback, type(None)) or self.lr_callback.cycle_end or self.lr_callback.cycle_mult == 1):
            self.active = False
            
    def update_average_model(self):
        # update running average of parameters
        print("model is {} epochs old".format(self.swa_n))
        for model_param, swa_param in zip(self.model.get_weights(), self.swa_model):
            swa_param *= self.swa_n
            swa_param += model_param
            swa_param /= (self.swa_n + 1)
        
        if self.swa_n > self.renewal and self.first_completed:
            print("new model is {} epochs old".format(self.n_since_renewal))
            for model_param, swa_param in zip(self.model.get_weights(), self.swa_model_new):
                swa_param *= self.n_since_renewal
                swa_param += model_param
                swa_param /= (self.n_since_renewal + 1)
            
    def compareAverages(self):
        if isinstance(self.losses['swa'], type(None)):
            self.test_model.set_weights(self.swa_model)
            if self.weighted:
                self.losses['swa'] = self.test_model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], sample_weight=self.test_fold['weights'], verbose=0)
            else:
                self.losses['swa'] = self.test_model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], verbose=0)
        
        self.test_model.set_weights(self.swa_model_new)
        if self.weighted:
            new_loss = self.test_model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], sample_weight=self.test_fold['weights'], verbose=0)
        else:
            new_loss = self.test_model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], verbose=0)
        
        print("Checking renewal swa model, current model: {}, new model: {}".format(self.losses['swa'], new_loss))
        if new_loss < self.losses['swa']:
            print("New model better, replacing\n____________________\n\n")
            self.losses['swa'] = new_loss
            self.swa_n = self.n_since_renewal
            if self.sgd_replacement:
                if isinstance(self.losses['base'], type(None)):
                    if self.weighted:
                        self.losses['base'] = self.model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], sample_weight=self.test_fold['weights'], verbose=0)
                    else:
                        self.losses['base'] = self.model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], verbose=0)
                if self.losses['base'] > new_loss:
                    print("Old average better than current point, starting SGD from old average")
                    self.model.set_weights(self.swa_model)
                    self.n_since_renewal = 0
                else:
                    print("Old average worse than current point, resuming SGD from current point")
                    self.n_since_renewal = 1
            else:
                self.n_since_renewal = 1
            self.swa_model[:] = self.swa_model_new
            self.swa_model_new = self.model.get_weights()
            self.cylcle_since_replacement = 1

        else:
            print("Current model better, renewing\n____________________\n\n")
            self.swa_model_new = self.model.get_weights()
            self.n_since_renewal = 1
            self.test_model.set_weights(self.swa_model)
            self.cylcle_since_replacement += 1
                
    
    def get_losses(self):
        if isinstance(self.losses['swa'], type(None)):
            self.test_model.set_weights(self.swa_model)
            if self.weighted:
                self.losses['swa'] = self.test_model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], sample_weight=self.test_fold['weights'], verbose=0)
            else:
                self.losses['swa'] = self.test_model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], verbose=0)
        
        if isinstance(self.losses['base'], type(None)):
            if self.weighted:
                self.losses['base'] = self.model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], sample_weight=self.test_fold['weights'], verbose=0)
            else:
                self.losses['base'] = self.model.evaluate(self.test_fold['inputs'], self.test_fold['targets'], verbose=0)
        
        return self.losses
