import numpy as np
import math
import os
import pandas as pd
from sklearn import preprocessing
from sklearn import cluster

class DataWorker(object):
    def __init__(self):
        """ Load and process training and test data

        Should have .csv files in the directory this is initialized.

        i.e.

        dw = DataWorker()

        dw.train #training data
        dw.target #known results for training data
        dw.test #test inputs

        # return all training data
        x,y, test = dw.get_production_set()

        # small random set for debugging
        x,y, test = dw.get_debug_set()

        # get a regularized training and test set
        x,y, test = dw.get_normalized_production_set()

        # print out the number and fraction of predictions in each class
        dw.print_fraction_predictions(predictions)

        # output to a file called submission.csv
        dw.output_results(y_submit, savename="submission.csv")

        Values are read in with pandas, and then converted to `numpy.ndarrays`.
        This does lose meta information about the columns but this particular
        data set is small enough to work with.

        """
        self.cwd = os.getcwd()
        train = pd.read_csv('train.csv')

        # get the labels
        y = train.target.values
        train.drop(['id', 'target'], inplace=True, axis=1)

        x = train.values

        submission = pd.read_csv('test.csv')
        ids = submission['id'].values
        submission.drop('id', inplace=True, axis=1)

        self.train = x
        self.target = y
        self.test = submission.values #return a numpy value

        self.ids = ids

        self.n_train, self.n_inputs = np.shape(self.train)
        self.n_test = np.shape(self.test)[0]
        self.train_sets = None

    def find_number_classified(self, results):
        """ Outputs the ratio of 1 to 0 from classifcation """

        n_zeros = np.shape(np.where(results == 0))[1]
        n_ones = np.shape(np.where(results == 1))[1]

        return n_zeros, n_ones

    def print_fraction_predictions(self, results):
        zeros, ones = self.find_number_classified(results)
        total = zeros + ones

        print("Classified as zero: %d, %f\n Classified as one: %d, %f" % (zeros, zeros/total, ones, ones/total))

    def get_normalized_production_set(self):
        """ Performs Regularization

        Returns a mean-centered and variance normalized set of features.

        """

        all_features = np.append(self.train, self.test, axis=0)
        regularized = preprocessing.scale(all_features)
        new_training = all_features[:self.n_train,:]
        new_tests = all_features[self.n_train:,:]

        return new_training, self.target, new_tests

    def get_production_set(self):
        """ Return full training, target and test sets """

        return self.train, self.target, self.test

    def get_debug_set(self, n_mini=100):
        """ Get a mini set of test and targets for debugging purposes """

        indices = np.random.choice(self.n_train, size=n_mini)
        new_training_inputs = self.train[indices,:]
        new_training_targets = self.target[indices]

        indices = np.random.choice(self.n_test, size=n_mini)
        new_test_inputs = self.test[indices, :]

        return new_training_inputs, new_training_targets, new_test_inputs

    def output_features(self, feature_list, suffix):
        sub_features = self.train[:, feature_list]
        sub_test_features = self.test[:, feature_list]

        # construct pandas dictionary
        train_dict = {"id": np.arange(self.n_train), "target": self.target}
        test_dict = {"id": self.ids}
        for idx in feature_list:
            train_dict["%d" % idx] = self.train[:, idx]
            test_dict["%d" % idx] = self.test[:, idx]

        output = pd.DataFrame(train_dict)
        output.to_csv("train_%s.csv" % suffix, index=False)
        output = pd.DataFrame(test_dict)
        output.to_csv("test_%s.csv" % suffix, index=False)

    def output_results(self, y_submit, savename="submission.csv"):
        """ Prepare submission file given results"""

        self.print_fraction_predictions(y_submit)
        if np.shape(y_submit)[0] != self.n_test:
            raise IOError("Number of submitted predictions is incorrect. Got size %d, expected size %d" % (np.shape(y_submit)[0], self.n_test))
        output = pd.DataFrame({'id': self.ids, 'target': y_submit})
        output.to_csv(savename, index=False)
