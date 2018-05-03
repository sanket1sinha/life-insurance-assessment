import numpy as np
import operator


class DecisionTree:
    """
    Decision Tree Classifier which uses criterion=gini, max_depth=None and min_samples_split=2
    """

    def __init__(self, id, training_df, min_samples_split=2, max_depth=None):

        self.id = id
        self.training_df = training_df
        self.response_column = self.training_df.columns[-1]
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def calc_gini_index(self, training_df, no_of_sample, gini_index_label):
        """
        Calculates gini index and returns the best split node
        :param training_df: Values on which the computation will take place
        :param no_of_sample: Sample size of the data
        :param gini_index_label: computed value for labels using gini index
        :return: a tuple of values which contains either feature name  or the target value
        """
        gini_dic = {}

        if len(training_df[self.response_column].unique()) > 1 and training_df.shape[0] >= 142 and \
                (self.max_depth is None or (self.training_df.shape[1] - training_df.shape[1]) <= self.max_depth):

            for x in training_df.columns:
                if x != self.response_column:
                    c = training_df.groupby([x, self.response_column]).size()
                    t = training_df.groupby([x]).size()
                    w = t / no_of_sample
                    gini_index_attr = (w * (1 - (np.power(c / t, 2).groupby([x]).sum()))).sum()
                    gini_dic[x] = gini_index_label - gini_index_attr

            if len(gini_dic) > 0:
                max_k = max(gini_dic.items(), key=operator.itemgetter(1))[0]
                if gini_dic[max_k] > 0:
                    return 'feature_name', max_k

        return 'target', training_df[self.response_column].mode()[0]

    def partition(self, training_data, split_node, node):
        """
        Computes the partition column using the gini index
        :param training_data: Values on which the computation will take place
        :param split_node: column name on which the split will take place
        :param node: Node object used to construct the tree
        :return:
        """

        for col_value in training_data[split_node].unique():

            df = training_data[training_data[split_node] == col_value]
            df = df.drop(split_node, 1)
            no_of_sample = df.shape[0]

            # Compute gini index for labels
            gini_index_label = 1 - np.power(
                df[self.response_column].value_counts() /
                no_of_sample, 2).sum()

            method, content = self.calc_gini_index(df, no_of_sample, gini_index_label)

            if method == 'feature_name':
                child = Node(feature_name=content, split_value=col_value)
                node.add_child(child)
                self.partition(df, content, child)

            elif method == 'target':
                child = Node(split_value=col_value, leaf_node_value=content)
                node.add_child(child)

    def fit(self):
        """
        Creates Decision tree using gini index and recursively partition the data and saves the
        reference to root node
        :return:
        """
        print('Creating Tree {}....'.format(self.id))

        no_of_sample = self.training_df.shape[0]

        gini_index_label = 1 - np.power(
            self.training_df[self.response_column].value_counts() /
            no_of_sample, 2).sum()

        method, content = self.calc_gini_index(self.training_df, no_of_sample, gini_index_label)

        self.root = Node(content)
        self.partition(self.training_df, content, self.root)

    def predict(self, testing_df):
        """
        Used to traverse the Decision tree using the testing data
        :param testing_df: Uses this data to predict the response
        :return: response of the predicted values are returned
        """
        print('Predicting Values {}....'.format(self.id))

        testing_result_list = []

        for index, row in testing_df.iterrows():

            n = self.root
            r = False

            while n.feature_name:

                split_val = row[n.feature_name]

                for child in n.child:
                    if child.split_value == split_val:
                        n = child
                        break

                if n != child:
                    r = True
                    break

            if r:
                testing_result_list.append(self.training_df[self.response_column].mode().iloc[-1])
            else:
                testing_result_list.append(n.leaf_node_value)

        return testing_result_list


class Node:
    """
    Used to create a node of a multi-way tree
    """

    def __init__(self, feature_name=None, split_value=None, leaf_node_value=None):
        self.feature_name = feature_name
        self.split_value = split_value
        self.child = []
        self.leaf_node_value = leaf_node_value

    def add_child(self, node):
        """
        Adds child to parent node
        :param node: add it as a child to the Node object
        :return:
        """
        self.child.append(node)
