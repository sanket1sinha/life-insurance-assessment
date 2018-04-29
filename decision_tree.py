import pandas as pd
import numpy as np
import operator
from collections import defaultdict

def gini_index(training_df,gini_index_label):
    gini_dic ={}
    col_names = training_df.columns
    if len(set(training_df[training_df.columns[-1]])) > 1:
        for x in training_df.columns:
            if x != col_names[-1]:
                c = training_df.groupby([x, col_names[-1]]).size()
                t = training_df.groupby([x]).size()
                w = t / (no_of_lines - 1)
                gini_index_attr = (w * (1 - (np.power(c / t, 2).groupby([x]).sum()))).sum()
                gini_dic[x] = gini_index_label - gini_index_attr
        if len(gini_dic) is not 0:
            return (max(gini_dic.items(), key=operator.itemgetter(1))[0],1)
    else:
        return (training_df[training_df.columns[-1]].iloc[0],"end")


def partition(attributes,best_split,training_df):
    dict_splits = defaultdict(list)
    #Change implementation , get the row directly through pandas
    for index,row in training_df.iterrows():
        for j in attributes:
            if np.any(row.values == j) and j not in dict_splits:
                df = training_df[training_df[best_split] == j]
                df = df.drop(best_split, 1)
                dict_splits[j].append(df)
    parent_node = best_split
    for key, value in dict_splits.items():
        best_split = gini_index(value[0], gini_index_label)
        print("parent node:" + parent_node)
        print("property:"+key)
        print("child node:"+best_split[0])
        print("\n")
        if best_split[1] !=0 and best_split[1] != "end":
            partition(training_df[best_split[0]].unique(), best_split[0], value[0])
    return dict_splits

#First get the data
training_df = pd.read_csv("C:\\Users\\Roshini Seshadri\\Desktop\\play.csv")
no_of_lines =training_df.shape[0]

p = training_df[training_df.columns[-1]].value_counts() / (no_of_lines - 1)

gini_index_label = 1 - np.power(p, 2).sum()

#Get the best split aka root node
best_split = gini_index(training_df,gini_index_label)

#Get the partitions
X = partition(training_df[best_split[0]].unique(),best_split[0],training_df)


