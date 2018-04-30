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
                                        self.response_column].value_counts() /
                                             (self.no_of_sample), 2).sum()

    def calc_gini_index(self, training_df):
        gini_dic = {}
        p = len(training_df[self.response_column].unique())
        # print('calc_gini:'+str(p))

        if p > 1 and training_df.shape[0] >= 100:

            for x in training_df.columns:
                if x != self.response_column:
                    c = training_df.groupby([x, self.response_column]).size()
                    t = training_df.groupby([x]).size()
                    w = t / self.no_of_sample
                    gini_index_attr = (w * (1 - (np.power(c / t, 2).groupby([x]).sum()))).sum()
                    gini_dic[x] = self.gini_index_label - gini_index_attr

            if len(gini_dic) > 0:
                max_k = max(gini_dic.items(), key=operator.itemgetter(1))[0]
                if gini_dic[max_k] > 0:
                    return 'feature_name', max_k

        return 'target', training_df[self.response_column].max()

    def partition(self, training_data, split_node, node):

        # splits = list(map(lambda x: (x, training_df[training_df.values == x].drop(split_node, 1)), unique_column_values))
        for col_value in training_data[split_node].unique():
            df = training_data[training_data[split_node] == col_value]
            df = df.drop(split_node, 1)

            method, content = self.calc_gini_index(df)

            if method == 'feature_name':
                child = Node(feature_name=content, split_value=col_value)
                node.add_child(child)
                self.partition(df, content, child)

            elif method == 'target':
                child = Node(split_value=col_value, leaf_node_value=content)
                node.add_child(child)

    def create_tree(self):
        print('Creating Tree')

        method, content = self.calc_gini_index(self.training_df)
        self.root = Node(content)
        self.partition(self.training_df, content, self.root)

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
        print('Predicting Values...')

        counter = 0
        testing_result_df = pd.DataFrame(columns=[self.response_column])
        testing_result_df.index.name = 'Id'

        for index, row in testing_df.iterrows():
            n = self.root
            random = False
            while n.feature_name:
                split_val = row[n.feature_name]
                for child in n.child:
                    if child.split_value == split_val:
                        n = child
                        break
                if n != child:
                    random = True
                    break
            if random:
                testing_result_df.loc[index] = [self.training_df[self.response_column].max()]
                counter+=1
            else:
                testing_result_df.loc[index] = [n.leaf_node_value]
        print(counter)
        return testing_result_df


print('Reading and CLeaning Data')

insurance_training = pd.read_csv("training.csv")
insurance_training_features = insurance_training.iloc[:, :-1]
insurance_training_result = insurance_training['Response']
insurance_testing_features = pd.read_csv("testing.csv")

# Combine training and testing for cleaning data
insurance_features = pd.concat([insurance_training_features,insurance_testing_features])


insurance_features_text_cols = insurance_features.select_dtypes(include=['object'])

for col in insurance_features_text_cols:
    categorical_col = pd.Categorical(insurance_features[col])
    insurance_features[col] = categorical_col.codes



CATEGORICAL_COLUMNS = ["Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6",\
                       "Product_Info_7", "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", "InsuredInfo_1",\
                       "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",\
                       "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7",\
                       "Insurance_History_8", "Insurance_History_9", "Family_Hist_1", "Medical_History_2", "Medical_History_3",\
                       "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7", "Medical_History_8",\
                       "Medical_History_9", "Medical_History_11", "Medical_History_12", "Medical_History_13", "Medical_History_14",\
                       "Medical_History_16", "Medical_History_17", "Medical_History_18", "Medical_History_19", "Medical_History_20",\
                       "Medical_History_21", "Medical_History_22", "Medical_History_23", "Medical_History_25", "Medical_History_26",\
                       "Medical_History_27", "Medical_History_28", "Medical_History_29", "Medical_History_30", "Medical_History_31",\
                       "Medical_History_33", "Medical_History_34", "Medical_History_35", "Medical_History_36", "Medical_History_37",\
                       "Medical_History_38", "Medical_History_39", "Medical_History_40", "Medical_History_41"]

for i in CATEGORICAL_COLUMNS:
    insurance_features[i] = insurance_features[i].fillna(insurance_features[i].mode()[0])

cols = insurance_features.columns.tolist()
remaining_cols = [item for item in cols if item not in CATEGORICAL_COLUMNS]

# Using imputer as a preprocessing class to replace null values with mean
insurance_features[remaining_cols]=insurance_features[remaining_cols].fillna(insurance_features[remaining_cols].mean())
insurance_features = insurance_features.drop('Id',1)


numerical_col=insurance_features.select_dtypes(include=[np.number]).columns.tolist()

for i in numerical_col:
    if i != 'Response':
        insurance_features[i] = pd.cut(insurance_features[i], 10)
        a = pd.Categorical(insurance_features[i])
        insurance_features[i] = a.codes



insurance_training_features_cleaned = insurance_features[:20000]
insurance_testing_features_cleaned = insurance_features[20000:]
insurance_training_features_cleaned = pd.concat([insurance_training_features_cleaned, insurance_training_result], axis=1)

decision_tree = DecisionTree(insurance_training_features_cleaned)
decision_tree.create_tree()
insurance_testing_result = decision_tree.predict(insurance_testing_features_cleaned)
insurance_testing_result_df = pd.DataFrame(insurance_testing_result[insurance_training_result.name].tolist(), index=list(range(20000,30000)), columns=[insurance_training_result.name])
insurance_testing_result_df.index.name = 'Id'
insurance_testing_result_df.to_csv('Response.csv')

