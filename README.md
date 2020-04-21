# Context-aware recommendations using array of action-based bandit learners
---
> Author: Dattaraj Rao (dattarajrao@yahoo.com)
> Profile: [Connect with me on LinkedIn!](https://www.linkedin.com/in/dattarajrao)
---

Code in support of the paper "Context-aware recommendations using array of action-based bandit learners" by Dattaraj Rao.
We explore how contextual bandits is viewed an extension of the reinforcement learning (RL) problem and demonstrate a novel algorithm to solve contextual bandits problem using an array of action-based learners. We apply this approach to model an article recommendation system using an array of stochastic gradient descent (SGD) learners to make predictions on rewards based on actions taken.

## List of files:
- ContextualLearner.py - The main class implementing our array of bandit learners algorithm.
- PlotHelper.py - Helper functions for plotting accuracy curves.
- SimulatedArticleData.csv - Simulated dataset with recommendations based on 2 features.
- NewsRecommendation_Example.py - Analysis of above dataset using our bandit algorithm.
- MovieLens_100k_Normalized.csv - Normalized version of MovieLens 100K dataset for analysis.
- MovieLens_Example.py - Analysis of above dataset using our bandit algorithm.

References:
- Lihong Li, Wei Chu, John Langford, Robert E. Schapire, “A Contextual-Bandit Approach to Personalized News Article Recommendation”, arXiv:1003.0146 [cs.LG], Feb. 2010.
- Sebastian Ruder, “An overview of gradient descent optimization algorithms∗”, arXiv:1609.04747v2 [cs.LG], June. 2017.
- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872, http://grouplens.org/datasets/movielens
