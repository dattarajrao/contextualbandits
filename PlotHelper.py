'''
Author: Dattaraj Rao (dattarajrao@yahoo.com)
https://www.linkedin.com/in/dattarajrao

Code in suport of the paper "Context-aware recommendations using array of action-based bandit learners" by Dattaraj Rao.
==========================
File: PlotHelper.py
Description:
Helper file with methods to calculate accuracy and plot.
==========================
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# function to handle plotting
def plot_acc_accuracy(mtx, title):
  mtx = np.asarray(mtx, dtype=np.float32)
  print("Number of points = ", len(mtx))
  print("Average accuracy = ", np.mean(mtx))
  threshold = 0.7
  pos_signal = mtx.copy()
  neg_signal = mtx.copy()
  pos_signal[pos_signal <= threshold] = np.nan
  neg_signal[neg_signal > threshold] = np.nan
  plt.style.use(PLOT_STYLE)
  f, ax = plt.subplots(1, figsize=(15,8))
  ax.plot(pos_signal, 'go')
  ax.plot(neg_signal, 'rx')
  #ax.plot(mtx, 'go')
  ax.set_ylim(ymin=0, ymax=1.1)
  ax.title.set_text(title)
  ax.set_ylabel('Accuracy')

# function to calculate accuracy from actual and predicted values
def pred_plot_accuracy(act, pred, title):
  acc = []
  pred = [0 if v is None else v for v in pred]
  act = np.asarray(act)
  pred = np.asarray(pred)
  counter = 0
  for act_val in act:
    acc.append(1 - (abs(pred[counter]-act_val)/5.))
    counter = counter + 1
  plot_acc_accuracy(acc, title)
