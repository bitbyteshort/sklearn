import sys
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn import svm
from sklearn.externals import joblib
import censushasher as ch

def loadModel():
	return joblib.load('/tmp/census-hasher-classifier.pkl')

input_file = sys.argv[1] if (len(sys.argv) >= 2) else '/Users/terryz/Workspace/python/python3/data/census-adult-predict.csv'

census = pd.read_csv(input_file)
X = ch.transform_categorical_features(census)
classifier = loadModel()
y = classifier.predict(X)
print(y)
