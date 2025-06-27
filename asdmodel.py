import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Scalers & Classifiers
from sklearn.preprocessing import Normalizer, QuantileTransformer, PowerTransformer, MaxAbsScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


SCALERS = {
    "Normalizer": Normalizer(),
    "Quantile Transformer": QuantileTransformer(),
    "Power Transformer": PowerTransformer(),
    "MaxAbs Scaler": MaxAbsScaler()
}

CLASSIFIERS = {
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian NB": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF Kernel)": SVC(probability=True),
    "LDA": LinearDiscriminantAnalysis()
}

class ASDModel:
    def __init__(self):
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.pipeline = None
        self.X_columns = []

    def preprocess(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

        self.X_columns = X.columns.tolist()
        X_imputed = self.imputer.fit_transform(X)
        return X_imputed, y

    def encode_input(self, input_dict):
        encoded = {}
        for key in input_dict:
            if key in self.label_encoders:
                encoded[key] = self.label_encoders[key].transform([input_dict[key]])[0]
            else:
                encoded[key] = input_dict[key]
        input_df = pd.DataFrame([encoded])[self.X_columns]
        return self.imputer.transform(input_df)

    def train(self, X, y, scaler_name, clf_name):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        scaler = SCALERS[scaler_name]
        clf = CLASSIFIERS[clf_name]
        self.pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', clf)
        ])
        self.pipeline.fit(X_train, y_train)

    def predict(self, input_array):
        return self.pipeline.predict(input_array)[0]

    def predict_proba(self, input_array):
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            return self.pipeline.predict_proba(input_array)[0][1]
        else:
            return None
