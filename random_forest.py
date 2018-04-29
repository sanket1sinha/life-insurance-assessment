class RandomForest:
    train_data = pd.read_csv("training.csv")
    test_data = pd.read_csv("testing.csv")
    train_data=train_data.fillna(train_data.mean())
    test_data=test_data.fillna(test_data.mean())
    cols=test_data.columns.tolist()
    def randomForest(train_data,test_data):
        tree_count = 180
        bag_proportion = .90
        result=[]
        for i in range(tree_count):
            bag = train_data.sample(frac=bag_proportion, replace=True, random_state=i)
            clf = tree.DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
            clf.fit(bag[cols], bag["Response"])
            insurance_testing_result = clf.predict(test_data)
            result.append(insurance_testing_result)
        df=pd.DataFrame(result)
        m=df.mode()
        final=m.iloc[0].values.tolist()
        final=[int(i) for i in final]
        return final
