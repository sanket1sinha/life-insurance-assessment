import pandas as pd
import numpy as np
import operator
from tree import Node


class DecisionTree:

    def __init__(self, training_df):

        self.training_df = training_df
        self.no_of_sample = self.training_df.shape[0]

        self.gini_index_label = 1 - np.power(self.training_df[
                            self.training_df.columns[-1]].value_counts() /
                                             (self.no_of_sample - 1), 2).sum()

    def calc_gini_index(self, training_df):

        gini_dic = {}
        col_names = training_df.columns

        if len(set(training_df[training_df.columns[-1]])) > 1:

            for x in training_df.columns:
                if x != col_names[-1]:
                    c = training_df.groupby([x, col_names[-1]]).size()
                    t = training_df.groupby([x]).size()
                    w = t / self.no_of_sample
                    gini_index_attr = (w * (1 - (np.power(c / t, 2).groupby([x]).sum()))).sum()
                    gini_dic[x] = self.gini_index_label - gini_index_attr

            if len(gini_dic) is not 0:
                return 'feature_name', max(gini_dic.items(), key=operator.itemgetter(1))[0]

        else:
            return 'target', training_df[training_df.columns[-1]].iloc[0]

    def partition(self, unique_column_values, split_node, training_df, node):
        dict_splits = {}
        for j in unique_column_values:
            filtered_df = training_df[training_df.values == j]
            filtered_df.drop(split_node, 1)
            dict_splits[j] = filtered_df

        parent_node = split_node
        for key, data_frame in dict_splits.items():
            method, content = self.calc_gini_index(data_frame)


            if method == 'feature_name':
                child = Node(feature_name=content, split_value=key)
                node.add_child(child)
                self.partition(training_df[content].unique(), content, data_frame, child)

            elif method == 'target':
                child = Node(split_value=key, leaf_node_value=content)
                node.add_child(child)


    def create_tree(self):
        method, content = self.calc_gini_index(self.training_df)
        root = Node(content)
        self.partition(self.training_df[content].unique(), content, self.training_df, root)
        print('par:'+root.feature_name)
        # for c in root.child:
        #     if c.feature_name:
        #         print('child1:'+c.feature_name)
        #         print('splitval:'+c.split_value)
        #
        #     else:
        #         print('leaf1:'+c.leaf_node_value)
        #         print('splitval:'+c.split_value)
        #
        #     for m in c.child:
        #         if m.feature_name:
        #             print('child2:'+m.feature_name)
        #             print('splitval:' + m.split_value)
        #
        #         else:
        #             print('leaf2:' + m.leaf_node_value)
        #             print('splitval:'+m.split_value)

df = pd.read_csv("play.csv")

decision_tree = DecisionTree(df)
decision_tree.create_tree()



