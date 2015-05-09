import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.externals import joblib


def categorical_features():
	return ['workclass', 'marital-status', 'education', 'occupation', 'relationship', 'race', 'sex', 'native-country']

def categorical_indices(data):
	col_names = data.columns.values
	col_index_map = dict(zip(col_names, range(0, len(col_names), 1)))
	return map(lambda f: col_index_map[f+"-enum"], categorical_features())

def categorical_feature_dict(data):
	dic = {}
	for f in categorical_features() :
		k = pd.unique(data[f])
		v = range(0, len(k), 1)
		dic[f] = dict(zip(k, v))

	return dic

def categorize(data, dic) :
	for f in categorical_features() :
		data[f+"-enum"] = data[f].map(lambda x: dic[f][x])
		del data[f]

def encode(data):
#	enc = OneHotEncoder(n_values='auto', categorical_features=categorical_indices(data))
	enc = OneHotEncoder(n_values=100, categorical_features=categorical_indices(data))
	return enc.fit_transform(data)

