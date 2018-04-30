import pandas as pd
import numpy as np
import operator
from tree import Node
from sklearn.preprocessing import Imputer


class DecisionTree:

    def __init__(self, training_df):

        self.training_df = training_df
        self.no_of_sample = self.training_df.shape[0]

        self.gini_index_label = 1 - np.power(self.training_df[
                            self.training_df.columns[-1]].value_counts() /
                                             (self.no_of_sample - 1), 2).sum()

    def calc_gini_index(self, training_df):
        print('calc gini')

        gini_dic = {}
        col_names = training_df.columns
        p = len(set(training_df[training_df.columns[-1]]))
        print(training_df)
        print(p)
        if p > 1:

            for x in training_df.columns:
                if x != col_names[-1]:
                    c = training_df.groupby([x, col_names[-1]]).size()
                    t = training_df.groupby([x]).size()
                    w = t / self.no_of_sample
                    gini_index_attr = (w * (1 - (np.power(c / t, 2).groupby([x]).sum()))).sum()
                    gini_dic[x] = self.gini_index_label - gini_index_attr

            if len(gini_dic) is not 0:
                return 'feature_name', max(gini_dic.items(), key=operator.itemgetter(1))[0]

        elif p == 1:
            return 'target', training_df[training_df.columns[-1]].iloc[0]

        else:
            return

    def partition(self, unique_column_values, split_node, training_df, node):
        print('enter predict')

        splits = list(map(lambda x: (x, training_df[training_df.values == x].drop(split_node, 1)), unique_column_values))

        for key, data_frame in splits:
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
        self.root = Node(content)
        self.partition(self.training_df[content].unique(), content, self.training_df, self.root)
        # print('par:' + self.root.feature_name)
        # for c in root.child:
        #     if c.feature_name:
        #         print('child1:'+c.feature_name)
        #         print('splitval:'+c.split_value)
        #
        #     else:
        #         print('leaf1:'+c.leaf_node_value
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
preprocessor = Imputer()
insurance_features_cleaned = preprocessor.fit_transform(insurance_features)
insurance_features = insurance_features.drop('Id',1)
# test = pd.read_csv("play_testing.csv")
print('done_reading')
decision_tree = DecisionTree(insurance_features)
decision_tree.create_tree()
#decision_tree.predict(test)



