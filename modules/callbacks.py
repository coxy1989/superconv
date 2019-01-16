import math
from typing import Dict, Callable
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow as tf

def _1cycle_mom(iteration_idx:int,
                cyc_iterations:int,
                min_mom:float,
                max_mom:float):
    'TODO: docstring'
    mid = math.floor((cyc_iterations - 1)/2)
    if iteration_idx == mid: return min_mom
    elif iteration_idx == 0 or iteration_idx >= (2 * mid): return max_mom
    else:
        mod = (iteration_idx % mid)
        numerator =  mod if iteration_idx < mid else mid - mod
        return max_mom - (numerator / mid) * (max_mom - min_mom)

def _1cycle_lr(iteration_idx:int,
              cyc_iterations:int,
              ramp_iterations:int,
              min_lr:float,
              max_lr:float):
    'TODO: docstring'
    mid = math.floor((cyc_iterations - 1)/2)
    if iteration_idx == mid: return max_lr
    elif iteration_idx == 0 or iteration_idx == (2 * mid): return min_lr
    elif iteration_idx < cyc_iterations: 
        mod = (iteration_idx % mid)
        numerator =  mod if iteration_idx < mid else mid - mod
        return min_lr + (numerator / mid) * (max_lr - min_lr)
    else:
        idx = iteration_idx - cyc_iterations
        ramp_max = min_lr
        ramp_min = min_lr * 1e-5
        return ramp_max - ((idx + 1) / ramp_iterations) * (ramp_max - ramp_min)

def _inv_lr(iteration_idx:int, gamma:float, power:float,  base_lr:float):
    'ported from: https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L157-L172'
    return base_lr * (1 + gamma * iteration_idx) ** (- power)

class InvSchedulerCallback(tf.keras.callbacks.Callback):
    
    def __init__(self,
                 gamma:float,
                 power:float,
                 base_lr:float):
        'TODO: docstring'        
        self.gamma = gamma
        self.power = power
        self.base_lr = base_lr
        self.iteration = 0
    
    def on_batch_begin(self, batch, logs=None):
        'TODO: docstring'        
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = _inv_lr(self.iteration, self.gamma, self.power, self.base_lr)
        K.set_value(self.model.optimizer.lr, lr)
    
    def on_batch_end(self, batch, logs=None):
        'TODO: docstring'
        self.iteration +=1
       
    @staticmethod
    def plot_schedule(iterations:int, gamma:float, power:float,  base_lr:float):
        'TODO: docstring'   
        xs = range(iterations)
        ys = [_inv_lr(i, gamma, power, base_lr) for i in xs]
        df = pd.DataFrame({'lr' : ys, 'iteration': xs})
        sns.lineplot(x='iteration', y='lr', data=df)

class OneCycleSchedulerCallback(tf.keras.callbacks.Callback):
    
    def __init__(self,
                 cyc_iterations:int,
                 ramp_iterations:int,
                 min_lr:float,
                 max_lr:float,
                 min_mom:float,
                 max_mom:float):
        'TODO: docstring'        
        self.cyc_iterations = cyc_iterations
        self.ramp_iterations = ramp_iterations
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_mom = min_mom
        self.max_mom = max_mom
        self.iteration = 0
    
    def on_batch_begin(self, batch, logs=None):
        'TODO: docstring'        
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if not hasattr(self.model.optimizer, 'momentum'):
            raise ValueError('Optimizer must have a "momentum" attribute.')
        lr = _1cycle_lr(self.iteration, self.cyc_iterations, self.ramp_iterations, self.min_lr, self.max_lr)
        mom = _1cycle_mom(self.iteration, self.cyc_iterations, self.min_mom, self.max_mom)
        K.set_value(self.model.optimizer.lr, lr)
        K.set_value(self.model.optimizer.momentum, mom)   
    
    def on_batch_end(self, batch, logs=None):
        'TODO: docstring'
        self.iteration +=1
        
    @staticmethod
    def plot_schedule(cyc_iterations:int,
                      ramp_iterations:int,
                      min_lr:float,
                      max_lr:float,
                      min_mom:float,
                      max_mom:float):
        xs = range(cyc_iterations + ramp_iterations)
        lr_ys = [_1cycle_lr(i, cyc_iterations, ramp_iterations, min_lr, max_lr) for i in xs]
        mom_ys = [_1cycle_mom(i, cyc_iterations, min_lr, max_lr) for i in xs]
        lr_df = pd.DataFrame({'iteration': xs, 'lr' : lr_ys})
        mom_df = pd.DataFrame({'iteration': xs, 'mom' : mom_ys})
        _ ,(ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 20))
        sns.lineplot(x='iteration', y='lr', data= lr_df, ax=ax1)
        sns.lineplot(x='iteration', y='mom', data= mom_df, ax=ax2)
        
def _triangular_f(it:int, ss:int, min_lr:float, max_lr:float):
    'TODO: docstring'
    cyc = math.floor(it / (ss * 2))
    it_cyc = it - (cyc * 2 * ss)
    mid_dist = math.fabs(it_cyc - ss)
    scalar = mid_dist / ss
    return min_lr + (1 - scalar) * (max_lr - min_lr)

class LRFinderCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, step_size:int, min_lr:float, max_lr:float, evaluate_mod:int, evaluate_fn:Callable):
        'TODO: docstring'        
        super().__init__()
        self.step_size = step_size
        self.lr = min_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.evaluate_mod = evaluate_mod
        self.evaluate_fn = evaluate_fn
        self.lrs = []
        self.its = []
        self.val_lrs = []
        self.val_loss = []
        self.val_acc = []
        self.iteration = 0
    
    def on_batch_begin(self, batch, logs=None):
        'TODO: docstring'
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.lr = _triangular_f(self.iteration, self.step_size, self.min_lr, self.max_lr)
        K.set_value(self.model.optimizer.lr, self.lr)
                
    def on_batch_end(self, batch, logs=None):
        'TODO: docstring'
        self.lrs.append(self.lr)
        self.its.append(self.iteration)
        if self.iteration % self.evaluate_mod == 0:
            self.val_lrs.append(self.lr)
            loss, acc = self.evaluate_fn()
            self.val_loss.append(loss)
            self.val_acc.append(acc)
        self.iteration += 1
        
    def plot_lr_vs_iteration(self):
        'TODO: docstring'        
        df = pd.DataFrame({'lr' : self.lrs, 'iteration': self.its})
        sns.lineplot(x='iteration', y='lr', data=df)
        
    def plot_lr_vs_loss(self):
        'TODO: docstring'        
        df = pd.DataFrame({'lr' : self.val_lrs, 'loss': self.val_loss})
        sns.lineplot(x='lr', y='loss', data=df)
        
    def plot_lr_vs_acc(self):
        'TODO: docstring'        
        df = pd.DataFrame({'lr' : self.val_lrs, 'acc': self.val_acc})
        sns.lineplot(x='lr', y='acc', data=df)
