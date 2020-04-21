'''
Author: Dattaraj Rao (dattarajrao@yahoo.com)
https://www.linkedin.com/in/dattarajrao

Code in suport of the paper "Context-aware recommendations using array of action-based bandit learners" by Dattaraj Rao.
==========================
File: ContextualLearner.py
Description:
This is the main learner class that builds an array of stochastic gradient descent (SGD) learners
to make predictions on rewards based on actions taken. We keep things very flexible to dynamically
assign a classification or regression model to the learner.
==========================
'''
import numpy as np

class ContextualLearner:
  # define the initial parameters when initializing class
  # parameters:
  #     learnerclass = SGDClassifier (preferred) or SGDRegressor
  #     rew_vec = array of possible rewards  for SGDClassifier and None for SGDRegressor
  def __init__(self, learnerclass, rew_vec):
    self.sgd = None
    self.hist = 50
    self.arm_sgd = {}
    self.dataX = {}
    self.dataY = {}
    self.rew_vec = rew_vec
    self.Learner = learnerclass

  # learn from an individual datapoint
  # parameters:
  #     ctx_vector = vector of context values pre normalized
  #     arm = action selected as a string
  #     reward = scalar reward value
  # returns:
  #     status = True or False if learning was successful
  def train(self, ctx_vector, arm, reward):
    X = []
    Y = []
    if ctx_vector is None or arm is None or reward is None:
      return False
    # if the arm classifier doesn't exist
    if arm not in self.arm_sgd.keys():
      self.arm_sgd[arm] = self.Learner()
      self.dataX[arm] = []
      self.dataY[arm] = []
    # get arm classifier and make prediction
    self.sgd = self.arm_sgd[arm]
    if len(self.dataX[arm]) > self.hist:
      X = self.dataX[arm][:-self.hist]
      Y = self.dataY[arm][:-self.hist]
    X.append(ctx_vector)
    X = np.asarray(X) #.reshape(1, -1)
    Y.append(reward) #= [reward]
    # fit the data point
    if self.rew_vec is not None:
      self.sgd.partial_fit(X, Y, self.rew_vec)
    else:
      self.sgd.partial_fit(X, Y)
    # add to data vectors
    self.dataX[arm].append(ctx_vector)
    self.dataY[arm].append(reward)
    return True

  # predict reward for an individual datapoint
  # parameters:
  #     ctx_vector = vector of context values pre normalized
  #     arm = action selected as a string
  # returns:
  #     reward = scalar reward value
  def predict(self, ctx_vector, arm):
    if ctx_vector is None or arm is None:
      return None
    # if the arm classifier doesn't exist
    if arm in self.arm_sgd.keys():
      # get arm classifier and make prediction
      self.sgd = self.arm_sgd[arm]
      X = ctx_vector
      X = np.asarray(X).reshape(1, -1)
      return self.sgd.predict(X)[0]
    # if nothing found return
    return 0
