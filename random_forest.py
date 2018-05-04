import pandas as pd
from decision_tree import DecisionTree
from multiprocessing import Process, Manager
import time


class RandomForest:
    """
    Creates multiple decision trees using bootstrap sampling and predicts output using bagging mechanism
    """

    def __init__(self, training_df, training_result, tree_count=10, min_samples_split=2, max_depth=None):

        self.training_df = training_df
        self.training_result = training_result
        self.tree_count = tree_count
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        tree_manager = Manager()
        self.tree_dict = tree_manager.dict()
        response_manager = Manager()
        self.response_dict = response_manager.dict()

    def fit_trees(self, i):
        """
        Applying Bootstrap sampling and balancing different classes and creating a new
        Decision tree classifier and fitting the training data and adds the reference to a dictionary
        :param i:unique id
        :return:
        """
        bag_col_sample = self.training_df.sample(frac=.95, replace=False, random_state=int(time.time()), axis=1)

        balanced_classes = pd.Series([], name='Response', dtype='int32')

        for col_val in self.training_result.unique():

            if self.training_result[self.training_result == col_val].count() >= 1900:
                balanced_classes = balanced_classes.append(self.training_result[self.training_result == col_val].sample(
                    n=1900, random_state=int(time.time()), replace=False, axis=0))
            else:
                balanced_classes = balanced_classes.append(self.training_result[self.training_result == col_val].sample(
                    n=1200, random_state=int(time.time()), replace=True, axis=0))

        bag = bag_col_sample.join(balanced_classes, how='inner')

        clf = DecisionTree(i, bag, self.min_samples_split, self.max_depth)
        clf.fit()
        self.tree_dict[i] = clf

    def predict_trees(self, i, testing_data):
        """
        Predicting values for all the decision trees and add the response to a dictionary
        :param i: unique id
        :param testing_data:
        :return:
        """
        testing_response = self.tree_dict[i].predict(testing_data)
        self.response_dict[i] = testing_response

    def fit(self):
        """
        Runs multiple processes for fitting data
        :return:
        """
        jobs = []
        count = 0

        for i in range(self.tree_count):
            p = Process(target=self.fit_trees, args=(i + 1,))
            jobs.append(p)
            p.start()

        for j in jobs:
            j.join()
            count += 1
            print('Done Creating Tree No {}'.format(count))

        print('Created Classifier for Random Forest')

    def predict(self, testing_data):
        """
        Runs multiple processes to predict values for each classifier
        :param testing_data: Samples for which value needs to be predicted
        :return:
        """
        jobs = []
        count = 0
        for i in range(self.tree_count):
            p = Process(target=self.predict_trees, args=(i + 1, testing_data))
            jobs.append(p)
            p.start()

        for j in jobs:
            j.join()
            count += 1
            print('Done Predicting values for Tree No {}'.format(count))

        bag_test_response = pd.DataFrame(self.response_dict.values(), dtype='int32')
        m = bag_test_response.mode()
        test_response = m.iloc[0].values.tolist()

        print('Done Predicting all values.')
        return test_response
