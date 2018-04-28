import pandas as pd
import numpy as np
import operator
from collections import defaultdict

def gini_index(training_df,gini_index_label):
    gini_dic ={}
    for x in training_df.columns:
        if x != col_names[-1]:
            c = training_df.groupby([x, col_names[-1]]).size()
            t = training_df.groupby([x]).size()
            w = t / (no_of_lines - 1)
            gini_index_attr = (w * (1 - (np.power(c / t, 2).groupby([x]).sum()))).sum()
            gini_dic[x] = gini_index_label - gini_index_attr
    return max(gini_dic.items(), key=operator.itemgetter(1))[0]

def partition(attributes,best_split,training_df):
    dict_splits = defaultdict(list)
    #Change implementation , get the row directly through pandas
    for index,row in training_df.iterrows():
        for j in attributes:
            if np.any(row.values == j) and j not in dict_splits:
                df = training_df[training_df[best_split] == j]
                df = df.drop(best_split, 1)
                dict_splits[j].append(df)

    return dict_splits
#First get the data
input=[['age', 'income', 'student', 'creditrating', 'buyscomputer?'], ['l30', 'high', 'no', 'fair', 'no'], ['l30', 'high', 'no', 'excellent', 'no'], ['31to40', 'high', 'no', 'fair', 'yes'], ['g40', 'medium', 'no', 'fair', 'yes'], ['g40', 'low', 'yes', 'fair', 'yes'], ['g40', 'low', 'yes', 'excellent', 'no'], ['31to40', 'low', 'yes', 'excellent', 'yes'], ['l30', 'medium', 'no', 'fair', 'no'], ['l30', 'low', 'yes', 'fair', 'yes'], ['g40', 'medium', 'yes', 'fair', 'yes'], ['l30', 'medium', 'yes', 'excellent', 'yes'], ['31to40', 'medium', 'no', 'excellent', 'yes'], ['31to40', 'high', 'yes', 'fair', 'yes'], ['g40', 'medium', 'no', 'excellent', 'no']]

no_of_lines = 14
col_names = input[0]

training_df = pd.DataFrame(columns=col_names)

for i in range(no_of_lines - 1):
    training_df.loc[i] = input[i]

training_df = training_df.iloc[1:,:]

p = training_df[col_names[-1]].value_counts() / (no_of_lines - 1)
# info_gain_label = np.sum(-(p * np.log2(p)))
gini_index_label = 1 - np.power(p, 2).sum()

#Make it recursive - not done

#Get the best split aka root node
best_split = gini_index(training_df,gini_index_label)

#Get the partitions
X = partition(training_df[best_split].unique(),best_split,training_df)

# For each partition, get the sub attribute
for key, value in X.items():
    best_split = gini_index(value[0], gini_index_label)
    print((best_split, key))
    partition(training_df[best_split].unique(), best_split, value[0])
