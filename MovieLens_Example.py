'''
Author: Dattaraj Rao (dattarajrao@yahoo.com)
https://www.linkedin.com/in/dattarajrao

Code in suport of the paper "Context-aware recommendations using array of action-based bandit learners" by Dattaraj Rao.
==========================
File: MovieLens_Example.py
Description:
This is the example of applying Contextual bandits algorithm to predict movie ratings.
==========================
'''
import numpy as np
import pandas as pd

from ContextualLearner import ContextualLearner
from PlotHelper import pred_plot_accuracy

# load and cleanse the data
df_movielens = pd.read_csv('MovieLens_100k_Normalized.csv')
df_movielens.dropna()
df_movielens = df_movielens.sort_values(by=['unix_timestamp'])

# normalize the features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_movielens['sex'] = le.fit_transform(df_movielens['sex'])
df_movielens['occupation'] = le.fit_transform(df_movielens['occupation'])
df_movielens['zip_code'] = le.fit_transform(df_movielens['zip_code'])

from sklearn.linear_model import SGDRegressor, SGDClassifier

# count records
records = len(df_movielens)
# define context vector fields
context_vector = ['sex', 'age', 'occupation', 'zip_code']
# we will use regressor for this example
cbandit = ContextualLearner(SGDRegressor, None)

# define arrays to capture rolling accuracy
act = []
pred = []

# for each record
for i in range(records):
  # get the row
  record = df_movielens.loc[i]
  # get context vector
  ctx_vec = record[context_vector]
  # simple normalization - all features except sex (0,1)
  ctx_vec[1:] = ctx_vec[1:]/10.
  # for zip_code divide by 100 insetad of 10
  ctx_vec[-1:] = ctx_vec[-1:]/100.
  # get reward scalar value - rating given by user
  rew = record['rating']
  # now we will train a genre based contextual bandit
  # for each genre
  for genre in genres:
    if record[genre] == 1:
      arm = genre
      # first predict reward
      rew_pred = cbandit.predict(ctx_vec.tolist(), arm)
      pred.append(rew_pred)
      act.append(rew)
      # then learn from actual reward and update bandit
      cbandit.train(ctx_vec.tolist(), arm, rew)

# skips 500 values for training errors
pred_plot_accuracy(act[500:], pred[500:], 'Recommendation accuracy of the predicted rating')
