import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

# Read Data from file
insurance_training = pd.read_csv("training.csv")
insurance_training_features = insurance_training.iloc[:, :-1]
insurance_training_result = insurance_training['Response']
insurance_testing_features = pd.read_csv("testing.csv")

# Combine training and testing for cleaning data
insurance_features = pd.concat([insurance_training_features,insurance_testing_features])



# Checking columns with type object and converting them to categorical
insurance_features_text_cols = insurance_features.select_dtypes(include=['object'])

for col in insurance_features_text_cols:
    categorical_col = pd.Categorical(insurance_features[col])
    insurance_features[col] = categorical_col.codes

# Using imputer as a preprocessing class to replace null values with mean
preprocessor = Imputer()
insurance_features_cleaned = preprocessor.fit_transform(insurance_features)

# Seperating training and testing data
insurance_training_features_cleaned = insurance_features_cleaned[:20000]
insurance_testing_features_cleaned = insurance_features_cleaned[20000:]

# Using RandomForest Classifier to train the model
clf = RandomForestClassifier(n_estimators = 150, max_features = 'sqrt', max_depth = 50, verbose = 1, n_jobs = -1)

clf = clf.fit(insurance_training_features_cleaned, insurance_training_result)

# Predicting values
insurance_testing_result = clf.predict(insurance_testing_features_cleaned)

# Writing testing result to Response.csv
insurance_testing_result_df = pd.DataFrame(insurance_testing_result, index=list(range(20000,30000)), columns=['Response'])
insurance_testing_result_df.index.name = 'Id'
insurance_testing_result_df.to_csv('Response.csv')