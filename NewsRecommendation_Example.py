'''
Author: Dattaraj Rao (dattarajrao@yahoo.com)
https://www.linkedin.com/in/dattarajrao

Code in suport of the paper "Context-aware recommendations using array of action-based bandit learners" by Dattaraj Rao.
==========================
File: NewsRecommendation_Example.py
Description:
This is the example of applying Contextual bandits algorithm to recommedn news articles.
==========================
'''
import numpy as np
import pandas as pd

from ContextualLearner import ContextualLearner
from PlotHelper import pred_plot_accuracy

# load and cleanse the data
df_data = pd.read_csv('SimulatedArticleData.csv')
df_data.dropna()
# define context vector fields
context_vector = ['Gender', 'Age']
# normalize the features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_data['Gender'] = le.fit_transform(df_data['Gender'])

from sklearn.linear_model import SGDClassifier

# count records
records = len(df_data)
# we will use classifier for this example
# possible rewards are 0 (no-click) or 1 (click)
cbandit = ContextualLearner(SGDClassifier, [0, 1])

# define arrays to capture rolling accuracy
act = []
pred = []

# for each record
for i in range(records):
  # get the row
  record = df_data.loc[i]
  # get context vector
  ctx_vec = record[context_vector]
  # simple normalization for age
  ctx_vec[0] = ctx_vec[0]/100.
  # get recommendation - action or arm
  arm = record['Recommendation']
  # get reward scalar value - rating given by user
  rew = record['Reward']
  # first predict reward
  rew_pred = cbandit.predict(ctx_vec.tolist(), arm)
  # save the prediction and actual reward
  pred.append(rew_pred)
  act.append(rew)
  # then learn from actual reward and update bandit
  cbandit.train(ctx_vec.tolist(), arm, rew)

# skips 500 values for training errors
pred_plot_accuracy(act[500:], pred[500:], 'Recommendation accuracy of the predicted rating')
