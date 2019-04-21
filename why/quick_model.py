# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from datetime import datetime
import os


class quick_model():

    def __init__(self, test_size=.2, silent=False):
        self.sep = os.sep
        self.test_size = test_size
        self.silent = silent
        if not os.path.exists("output")
            os.mkdir("output") 
        self.__load__()
        self.__make_sets__()

    def __load__(self):
        #
        # Prepare the data
        #

        train = pd.read_csv('..' + self.sep + 'train.csv')
        # get the labels
        self.inputY = train.target.values
        train.drop(['id', 'target'], inplace=True, axis=1)
        self.inputX = train.values

    def __make_sets__(self):
        #
        # Create training and validation sets
        #
        self.x, self.x_test, self.y, self.y_test = train_test_split(self.inputX, self.inputY, test_size=self.test_size, random_state=42, stratify=self.inputY)

        #
        # Create the LightGBM data containers
        #
        self.train_data = lightgbm.Dataset(self.x, label=self.y, silent=self.silent)
        self.test_data = lightgbm.Dataset(self.x_test, label=self.y_test, silent=self.silent)

    def __train__(self):
        #
        # Train the model
        #
        self.myoutput = {}
        self.model = lightgbm.train(self.parameters,
                                    self.train_data,
                                    valid_sets=self.test_data,
                                    evals_result=self.myoutput,
                                    num_boost_round=self.num_boost_round,
                                    early_stopping_rounds=self.early_stopping_rounds)

    def __use_model__(self):
        #
        # Create a submission
        #

        submission = pd.read_csv('..' + self.sep + 'test.csv')
        ids = submission['id'].values
        submission.drop('id', inplace=True, axis=1)

        x = submission.values
        y = self.model.predict(x)

        # note: anything above .5 is rounded up
        binY = [round(i) for i in y]

        time_label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output = pd.DataFrame({'id': ids, 'target': binY})
        output.to_csv("output{0}{1}_submission.csv".format(self.sep, time_label), index=False)

        auc = max(self.myoutput["valid_0"]["auc"])
        params = ",".join(self.model.model_to_string().split("parameters:\n")[1].split("\n\n")[0].split("\n"))
        with open("submission_list.csv", "a") as csv:
            csv.write("{0},{1},{2}\n".format(time_label, auc, params))

        if self.save_graph:
            graph = lightgbm.create_tree_digraph(self.model)
            graph.format = "png"
            graph.render("output{0}{1}".format(self.sep, time_label))

    def run_model(self, num_boost_round=5000, early_stopping_rounds=100, parameters={
            'application': 'binary',
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': 2,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 20,
            'learning_rate': 0.05,
            'max_depth': 2,
            'verbose': 0
            }, save_graph=False):
        self.save_graph = save_graph
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.parameters = parameters
        self.__train__()
        self.__use_model__()
