import pandas as pd
import numpy as np
from decision_tree import DecisionTree
from multiprocessing import Process, Manager
import time




class RandomForest:

    @staticmethod
    def worker(i, train_data, train_result, test_data, x):
        # bag = train_data.sample(frac=.95, replace=False, random_state=int(time.time()), axis=1)

        a = pd.Series([], name='Response', dtype='int32')
        for z in train_result.unique():
            if train_result[train_result == z].count() >= 2000:
                a = a.append(train_result[train_result == z].sample(n=2000, random_state=int(time.time()), replace=False, axis=0))
            else:
                a = a.append(train_result[train_result == z].sample(n=2000, random_state=int(time.time()), replace=True, axis=0))
        # bag = train_data.sample(frac=.75, replace=False, random_state=int(time.time()), axis=0)
        bag1 = train_data.join(a, how='inner')
        clf = DecisionTree(bag1)
        clf.create_tree()
        insurance_testing = clf.predict(test_data)
        x[i] = insurance_testing


    @staticmethod
    def sampling(train_data, train_result, test_data):
        tree_count = 50
        jobs = []
        count = 0
        manager = Manager()
        x = manager.dict()
        for i in range(tree_count):
            p = Process(target=RandomForest.worker, args=(i, train_data, train_result, test_data, x))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
            count += 1
            print('Tree No:' + str(count))
        bag_test_response = pd.DataFrame(x.values(), dtype='int32')
        m = bag_test_response.mode()
        test_response = m.iloc[0].values.tolist()
        return test_response


if __name__ == "__main__":
    t1=int(time.time())
    print('Reading and CLeaning Data')

    insurance_training = pd.read_csv("training.csv")
    insurance_training_features = insurance_training.iloc[:, :-1]
    insurance_training_result = insurance_training['Response']
    insurance_testing_features = pd.read_csv("testing.csv")

    # Combine training and testing for cleaning data
    insurance_features = pd.concat([insurance_training_features, insurance_testing_features])
    insurance_features = insurance_features.drop('Id', 1)

    insurance_features_text_cols = insurance_features.select_dtypes(include=['object'])

    for col in insurance_features_text_cols:
        categorical_col = pd.Categorical(insurance_features[col])
        insurance_features[col] = categorical_col.codes

    CATEGORICAL_COLUMNS = ["Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6", \
                           "Product_Info_7", "Employment_Info_2", "Employment_Info_3", "Employment_Info_5",
                           "InsuredInfo_1", \
                           "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6",
                           "InsuredInfo_7", \
                           "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4",
                           "Insurance_History_7", \
                           "Insurance_History_8", "Insurance_History_9", "Family_Hist_1", "Medical_History_2",
                           "Medical_History_3", \
                           "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7",
                           "Medical_History_8", \
                           "Medical_History_9", "Medical_History_11", "Medical_History_12", "Medical_History_13",
                           "Medical_History_14", \
                           "Medical_History_16", "Medical_History_17", "Medical_History_18", "Medical_History_19",
                           "Medical_History_20", \
                           "Medical_History_21", "Medical_History_22", "Medical_History_23", "Medical_History_25",
                           "Medical_History_26", \
                           "Medical_History_27", "Medical_History_28", "Medical_History_29", "Medical_History_30",
                           "Medical_History_31", \
                           "Medical_History_33", "Medical_History_34", "Medical_History_35", "Medical_History_36",
                           "Medical_History_37", \
                           "Medical_History_38", "Medical_History_39", "Medical_History_40", "Medical_History_41"]

    for i in insurance_features.columns:
        if i in CATEGORICAL_COLUMNS:
            insurance_features[i] = insurance_features[i].fillna(insurance_features[i].mode()[0])
        else:
            insurance_features[i] = insurance_features[i].fillna(insurance_features[i].mean())

    NUMERICAL_COLUMNS = insurance_features.select_dtypes(include=[np.number]).columns.tolist()

    for i in NUMERICAL_COLUMNS:
        if i != insurance_training_result.name:
            insurance_features[i] = pd.cut(insurance_features[i], 2)
            a = pd.Categorical(insurance_features[i])
            insurance_features[i] = a.codes

    insurance_training_features_cleaned = insurance_features[:20000]
    insurance_testing_features_cleaned = insurance_features[20000:]

    insurance_testing_result = RandomForest.sampling(
                insurance_training_features_cleaned,insurance_training_result, insurance_testing_features_cleaned)
    insurance_testing_result_df = pd.DataFrame(insurance_testing_result,
                                               index=list(range(20000, 30000)),
                                               columns=[insurance_training_result.name], dtype='int32')
    insurance_testing_result_df.index.name = 'Id'
    insurance_testing_result_df.to_csv('Response.csv')
    t2=int(time.time())
    print(t2-t1)


