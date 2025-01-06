from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, strategy="mean"):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == "mean":
            self.imputer = SimpleImputer(strategy='mean')
        elif self.strategy == "median":
            self.imputer = SimpleImputer(strategy='median')
        elif self.strategy == "mode":
            self.imputer = SimpleImputer(strategy='most_frequent')
        elif self.strategy == "constant":
            self.imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        else:
            raise ValueError("Invalid strategy")
        self.imputer.fit(X[self.columns])
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.imputer.transform(X[self.columns])
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class CustomMappingEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, map in self.mapping.items():
            X[col] = X[col].map(map)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class CustomDateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.date_columns:
            new_col = col + "_Year"
            X[new_col] = pd.to_datetime(X[col]).dt.year
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class CustomDropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.columns)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features