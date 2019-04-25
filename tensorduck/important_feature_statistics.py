import numpy as np
import pandas as pd
#import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn import decomposition as skdecomp
from sklearn import ensemble as skensemble
from sklearn import tree as sktree
from sklearn import metrics as skmetrics

import os
import time
import matplotlib.pyplot as plt

from load import DataWorker

def get_color_sequence(y):
    cs = []
    for value in y:
        if value == 0:
            col = "b"
        else:
            col = "r"
        cs.append(col)
    return cs

class WeightedPredictionEnsemble(skensemble.BaggingClassifier):

    def __init__(self, **kwargs):
        self.weight_one = kwargs.pop("weight_one", 1) #default value is 1
        super(WeightedPredictionEnsemble, self).__init__(**kwargs)

    def predict(self, X):
        weight_sum = np.zeros(np.shape(X)[0])
        avg_prediction = np.zeros(np.shape(X)[0])
        for clf in self.estimators_:
            results = clf.predict(X)
            results = results.astype(float)

            weight_sum[np.where(results == 0)] += 1
            weight_sum[np.where(results == 1)] += self.weight_one

            results[np.where(results == 1)] *= self.weight_one
            avg_prediction += results

        final = avg_prediction / weight_sum
        final[np.where(final >= 0.5)] = 1
        final[np.where(final < 1)] = 0

        return final

def sort_features(classifier):

    feature_importance = np.zeros((len(classifier.estimators_),classifier.n_features_))

    for idx,est in enumerate(classifier.estimators_):
        feature_importance[idx,:] = est.feature_importances_

    avg_importance = feature_importance.mean(axis=0)
    sd_importance = feature_importance.std(axis=0)

    sort_indices = np.argsort(avg_importance*-1)

    return sort_indices

def find_top_features(x, y):
    # Previously, we found these hyper parametesr to be pretty good for learning
    # Note, previous models were not found to be sensitive to the max_features parameters
    # Therefore, a smaller value of max_features is used in dimensional reduction
    best_forest = WeightedPredictionEnsemble(base_estimator=sktree.DecisionTreeClassifier(max_features=1.0,
                                                class_weight={0:1,1:0.56}),
                                           n_estimators=1000,
                                           max_samples=150,
                                           weight_one=0.56)

    # extra random trees make a random forest with completely random splits at every node
    extra_random_forest = skensemble.ExtraTreesClassifier(n_estimators=1000, class_weight={0:1,1:0.56})

    extra_random_forest.fit(x, y)

    first_round_features = sort_features(extra_random_forest)
    x_sorted = x[:,first_round_features[:50]]

    extra_random_forest.fit(x_sorted, y)
    second_round_pre_features = sort_features(extra_random_forest) # re-indexed
    second_round_features = [first_round_features[i] for i in second_round_pre_features] # get feature indices in original set

    best_forest.fit(x_sorted, y)
    second_round_pre_features_b = sort_features(best_forest) # re-indexed
    second_round_features_b = [first_round_features[i] for i in second_round_pre_features_b] # get feature indices in original set

    return second_round_features, second_round_features_b


if __name__ == "__main__":
    dw = DataWorker() # load all the data
    x, y, test = dw.get_normalized_production_set()

    collected_top_features = []
    collected_top_features_notrandom = []
    for i in range(10):
        if (i % 10) == 0:
            print("Analyzed %d trees" % i)
        tf, tfb = find_top_features(x,y)
        collected_top_features.append(tf)
        collected_top_features_notrandom.append(tfb)

    np.savetxt("top_features.dat", collected_top_features, fmt="%d")
    np.savetxt("top_features_not_random.dat", collected_top_features_notrandom, fmt="%d")
