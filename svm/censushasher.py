import sys
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn import svm
from sklearn.externals import joblib

def categorical_feature_names():
	return ['workclass', 'marital-status', 'education', 'occupation', 'relationship', 'race', 'sex', 'native-country']

def categorical_features(data):
    return data[categorical_feature_names()]

def hashing(data):
    hasher = FeatureHasher(n_features=1000, non_negative=True, input_type="string")
    return hasher.transform(data)

def toDataFrame(array):
    return pd.DataFrame(array)

def transform_categorical_features(census):
    hashed_categorical_values = hashing(categorical_features(census))
    df = toDataFrame(hashed_categorical_values.toarray())
    census.join(df)
    for f in categorical_feature_names() : del census[f]
    return census
