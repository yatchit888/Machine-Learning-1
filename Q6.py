import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

def data_loader(file_path):
    """
    Loads the dataset from the given file path and returns a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

# Load data
data = data_loader('https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv')

def data_cleaning(data):
    """
    Performs data cleaning procedures on the DataFrame.
    """
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    ohe.fit(data[categorical_cols])
    data_ohe = ohe.transform(data[categorical_cols])
    feature_names = []
    for i, col in enumerate(categorical_cols):
        feature_names += [f"{col}_{cat}" for cat in ohe.categories_[i]]
    numerical_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    data_final = np.concatenate([data_ohe, data[numerical_cols]], axis=1)
    data_final_df = pd.DataFrame(data_final, columns=feature_names + numerical_cols)
    cleaned_data = data_final_df.dropna()
    return cleaned_data

# Data Cleaning
cleaned_data = data_cleaning(data)

def feature_selection(data):
    """
    Performs feature selection and returns a DataFrame with selected features.
    """
    corr_matrix = data.corr()
    # Select the top 5 features with the highest correlation to the target variable
    top_features = corr_matrix['great_customer_class'].sort_values(ascending=False)[:6]
    selected_features = data[top_features.index]
    selected_features.reset_index(drop=True, inplace=True)
    return selected_features

# Feature Selection
selected_features = feature_selection(cleaned_data)

def train_test_splitter(data):
    """
    Splits the dataset into training and testing sets.
    """
    X = data.drop('great_customer_class', axis=1)
    y = data['great_customer_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_splitter(selected_features)

def train_model(X_train, y_train, model_type):
    """
    Trains the specified model type on the training data.
    """
    if model_type == 'RandomForest':
        model = RandomForestClassifier()
    elif model_type == 'SVM':
        model = SVC()
    elif model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif model_type == 'NaiveBayes':
        model = GaussianNB()
    elif model_type == 'KNN':
        model = KNeighborsClassifier()
    else:
        raise ValueError("Invalid model type. Please choose from: RandomForest, SVM, LogisticRegression, NaiveBayes, KNN")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test data and returns the accuracy.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Train and Evaluate Models
models_to_evaluate = ['RandomForest', 'SVM', 'LogisticRegression', 'NaiveBayes', 'KNN']

def ensemble_learning(models, X_train, y_train, X_test, y_test):
    """
    Uses ensemble learning technique to boost the accuracy of the models.
    """
    ensemble_model = VotingClassifier(estimators=models, voting='hard')
    ensemble_model.fit(X_train, y_train)
    accuracy = ensemble_model.score(X_test, y_test)
    return accuracy

for model_type in models_to_evaluate:
    model = train_model(X_train, y_train, model_type)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy for {model_type}: {accuracy}")

# Ensemble Learning
models = [('RandomForest', RandomForestClassifier()),
          ('SVM', SVC()),
          ('LogisticRegression', LogisticRegression()),
          ('NaiveBayes', GaussianNB()),
          ('KNN', KNeighborsClassifier())]

ensemble_accuracy = ensemble_learning(models, X_train, y_train, X_test, y_test)
print(f"Accuracy with Ensemble Learning: {ensemble_accuracy}")
