class RandomForest:
    def randomForest(train_data,test_data):
        tree_count = 180
        bag_proportion = .90
        result=[]
        for i in range(tree_count):
            bag = insurance_training.sample(frac=bag_proportion, replace=True, random_state=i)
            clf = tree.DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
            clf.fit(bag[cols], bag["Response"])
            insurance_testing_result = clf.predict(insurance_testing)
            result.append(insurance_testing_result)
        df=pd.DataFrame(result)
        m=df.mode()
        final=m.iloc[0].values.tolist()
        final=[int(i) for i in final]
        return final
