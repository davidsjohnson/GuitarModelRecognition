from bextract import extractFeatures
import numpy as np 

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#scaler to normalize data between -1 and 1
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

# load training data from saved feature extraction
trainingData = np.load("data/bextract_single.npz")

trainingFeatures = scaler.fit_transform(trainingData['features'])
trainingLabels = trainingData['labels']
trainingSources = trainingData['sources']

# UPDATE with check to make sure test data has the same features and classes...
featureNames = trainingData['featureNames']
classes = trainingData['classes']

# load test data from saved feature extraction
testData = np.load("data/bextract_single_test.npz")

testFeatures = scaler.fit_transform(testData['features'])
testLabels = testData['labels']
testSources = testData['sources']


# select model
# model = KNeighborsClassifier(n_neighbors = 1)
# model = GaussianNB()
model = SVC()
model.fit(trainingFeatures, trainingLabels)

# test model
print model.predict(trainingFeatures[0])
print model.predict(trainingFeatures[len(trainingFeatures)-1])

# evaluate with training set...to see if works.
print model.score(testFeatures, testLabels)
