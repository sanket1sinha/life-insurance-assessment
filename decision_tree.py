import pandas as pd
import numpy as np
import operator
from tree import Node
from sklearn.preprocessing import Imputer


class DecisionTree:

    def __init__(self, training_df):

        self.training_df = training_df
        self.response_column =  self.training_df.columns[-1]
        self.no_of_sample = self.training_df.shape[0]

        self.gini_index_label = 1 - np.power(self.training_df[
                                        self.response_colum].value_counts() /
                                             (self.no_of_sample - 1), 2).sum()

    def calc_gini_index(self, training_df):
        gini_dic = {}
        p = len(set(training_df[self.response_column]))
        # print('calc_gini:'+str(p))

        if p > 1:

            for x in training_df.columns:
                if x != self.response_column:
                    c = training_df.groupby([x, self.response_column]).size()
                    t = training_df.groupby([x]).size()
                    w = t / self.no_of_sample
                    gini_index_attr = (w * (1 - (np.power(c / t, 2).groupby([x]).sum()))).sum()
                    gini_dic[x] = self.gini_index_label - gini_index_attr

            if len(gini_dic) is not 0:
                return 'feature_name', max(gini_dic.items(), key=operator.itemgetter(1))[0]

        elif p == 1:
            return 'target', training_df[self.response_column].iloc[0]

        else:
            return

    def partition(self, training_df, split_node, node):
        print('partition:'+split_node)
        dict_splits = {}
        # splits = list(map(lambda x: (x, training_df[training_df.values == x].drop(split_node, 1)), unique_column_values))
        for col_value in training_df[split_node].unique():
            df = training_df[training_df[split_node].values == col_value]
            df = df.drop(split_node, 1)

            method, content = self.calc_gini_index(df)

            if method == 'feature_name':
                child = Node(feature_name=content, split_value=col_value)
                node.add_child(child)
                self.partition(df, content, child)

            elif method == 'target':
                child = Node(split_value=col_value, leaf_node_value=content)
                node.add_child(child)
        return


    def create_tree(self):
        method, content = self.calc_gini_index(self.training_df)
        self.root = Node(content)
        self.partition(self.training_df, content, self.root)
        # print('par:' + self.root.feature_name)
        # for c in self.root.child:
        #     if c.feature_name:
        #         print('child1:'+str(c.feature_name))
        #         print('splitval:'+str(c.split_value))
        #
        #     else:
        #         print('leaf1:'+str(c.leaf_node_value))
        #         print('splitval:'+str(c.split_value))
        #
        #     for m in c.child:
        #         if m.feature_name:
        #             print('child2:'+str(m.feature_name))
        #             print('splitval:' + str(m.split_value))
        #
        #         else:
        #             print('leaf2:' + str(m.leaf_node_value))
        #             print('splitval:'+str(m.split_value))

    def predict(self, testing_df):

        for index, row  in testing_df.iterrows():
            n = self.root
            while n.feature_name:
                split_val = row[n.feature_name]
                for child in n.child:
                    if child.split_value == split_val:
                        n = child
                        break
            print(str(index) + n.leaf_node_value)

insurance_features = pd.read_csv("training.csv")
insurance_features_text_cols = insurance_features.select_dtypes(include=['object'])

for col in insurance_features_text_cols:
    categorical_col = pd.Categorical(insurance_features[col])
    insurance_features[col] = categorical_col.codes

# Using imputer as a preprocessing class to replace null values with mean
insurance_features_cleaned = insurance_features[insurance_features.columns].fillna(insurance_features[insurance_features.columns].mean())
insurance_features_cleaned = insurance_features_cleaned.drop('Id',1)

print('done_reading')
decision_tree = DecisionTree(insurance_features_cleaned)
decision_tree.create_tree()
# test = pd.read_csv("play_testing.csv"
#decision_tree.predict(test)



