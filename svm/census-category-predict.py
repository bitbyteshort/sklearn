import sys
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
import censuscategorical as cg

def loadModel():
	return joblib.load('/tmp/census-categorical-classifier.pkl')

bp_input_file = sys.argv[1] if (len(sys.argv) >= 2) else '/Users/terryz/Workspace/python/python3/data/census-adult-predict.csv'
bp_data = pd.read_csv(bp_input_file)

dic = cg.categorical_feature_dict(bp_data)
cg.categorize(bp_data, dic)
bp_classifier = loadModel()
y = bp_classifier.predict(cg.encode(bp_data))
print(y)
