import pandas as pd
import numpy as np
import operator
from collections import defaultdict
import tree


class DecisionTree:

    def __init__(self, training_df):

        self.training_df = training_df
        self.no_of_samples = self.training_df.shape[0]

        self.gini_index_label = 1 - np.power(self.training_df[
                            self.training_df.columns[-1]].value_counts() /
                                             (self.no_of_samples - 1), 2).sum()

    def gini_index(self, training_df):

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
                return max(gini_dic.items(), key=operator.itemgetter(1))[0]
        else:
            return training_df[training_df.columns[-1]].iloc[0], "end"

    def partition(self, unique_column_values, split_node, training_df):

        dict_splits ={}
        for j in unique_column_values:
            filtered_df = training_df[training_df.values == j]
            filtered_df.drop(split_node, 1)
            dict_splits[j] = filtered_df
        parent_node = split_node
        for key, data_frame in dict_splits.items():
            split_node = self.gini_index(data_frame, self.gini_index_label)
            print("parent node:" + parent_node)
            print("property:" + key)
            print("child node:" + split_node[0])
            print("\n")
            if split_node[1] != "end":
                self.partition(training_df[split_node[0]].unique(), split_node[0], data_frame)
        return dict_splits



df = pd.read_csv("play.csv")
print(df.columns[-1])
decision_tree = DecisionTree(df)
#Get the best split aka root node
best_split = decision_tree.gini_index()
print(best_split)

#Get the partitions
#X = partition(training_df[best_split[0]].unique(),best_split[0],training_df)


