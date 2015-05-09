import sys
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
import censuscategorical as cg

def training(X, y):
	classifier = svm.SVC()
	classifier.fit(X, y)
	joblib.dump(classifier, '/tmp/census-categorical-classifier.pkl') 

input_file = sys.argv[1] if (len(sys.argv) >= 2) else '/Users/terryz/Workspace/python/python3/data/census-adult-train.csv'

census = pd.read_csv(input_file)
y = census.iloc[:,0]
training_data = census.iloc[:,1:len(census.columns)]
dic = cg.categorical_feature_dict(training_data)
cg.categorize(training_data, dic)
X = cg.encode(training_data)
training(X, y)
print("training done")
