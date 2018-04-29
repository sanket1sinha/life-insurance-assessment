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

    def partition(self, attributes, best_split, training_df):

        dict_splits ={}
        # Change implementation , get the row directly through pandas
        for index, row in training_df.iterrows():
            for j in attributes:
                if np.any(row.values == j) and j not in dict_splits:
                    df = training_df[training_df[best_split] == j]
                    df = df.drop(best_split, 1)
                    dict_splits[j]= df
        parent_node = best_split
        for key, value in dict_splits.items():
            best_split = self.gini_index(value, self.gini_index_label)
            print("parent node:" + parent_node)
            print("property:" + key)
            print("child node:" + best_split[0])
            print("\n")
            if best_split[1] != 0 and best_split[1] != "end":
                self.partition(training_df[best_split[0]].unique(), best_split[0], value)
        return dict_splits



df = pd.read_csv("play.csv")
print(df.columns[-1])
decision_tree = DecisionTree(df)
#Get the best split aka root node
best_split = decision_tree.gini_index()
print(best_split)

#Get the partitions
#X = partition(training_df[best_split[0]].unique(),best_split[0],training_df)


