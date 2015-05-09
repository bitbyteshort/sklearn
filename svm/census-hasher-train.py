import sys
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn import svm
from sklearn.externals import joblib
import censushasher as ch
import boto
from boto.s3.connection import S3Connection

input_file = '/tmp/xxx'
#input_file = sys.argv[1] if (len(sys.argv) >= 2) else '/Users/terryz/Workspace/python/python3/data/census-adult-train.csv'

access_key = sys.argv[1]
secret_key = sys.argv[2]
conn = S3Connection(access_key, secret_key)
bucket = conn.get_bucket('eml-datasets')
key = bucket.get_key('terryz/census-adult-train.csv')
key.get_contents_to_filename(input_file)

census = pd.read_csv(input_file)
split_ratio = int(len(census) * 2 / 3)
training_data = census.iloc[:split_ratio, :] 
verification_data = census.iloc[split_ratio:, :]

y = training_data.iloc[:,0]
transformed_training_data = training_data.iloc[:,1:len(training_data.columns)]
X = ch.transform_categorical_features(transformed_training_data)
classifier = svm.SVC()
classifier.fit(X, y)
joblib.dump(classifier, '/tmp/census-hasher-classifier.pkl') 
print("training done")

y = verification_data.iloc[:,0]
transformed_verification_data = verification_data.iloc[:,1:len(verification_data.columns)]
X = ch.transform_categorical_features(transformed_verification_data)
score = classifier.score(X,y)
print("score: %s" % score)
