from bextract import extractFeatures
import numpy as np 
import pickle

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def acousticElectricModel():

	scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

	# load training data from saved feature extraction
	# data = np.load("acoustic_electric.npz")
	featureNames, trainingData, trainingLabels, trainingSources, classes = extractFeatures("acoustic_electric.mf")
	trainingData = scaler.fit_transform(trainingData)

	print "Building Acoustic v. Electric Model..."
	model = SVC()

	scores = cross_validation.cross_val_score(model, trainingData, trainingLabels, cv=5)
	print "Scores:", scores

	# model = DecisionTreeClassifier()
	model.fit(trainingData, trainingLabels)
	print model

	pickle.dump(model, open( "acousticElectricModel.p", "wb" ))
	print "Acoustic v. Electric Model Build!"

def acousticModel():

	scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

	# load training data from saved feature extraction
	featureNames, trainingData, trainingLabels, trainingSources, classes = extractFeatures("acoustic_models.mf")
	trainingData = scaler.fit_transform(trainingData)

	print "Building Acoustic Model..."
	model = SVC()
	# model = DecisionTreeClassifier()
	model.fit(trainingData, trainingLabels)
	print model

	pickle.dump(model, open( "acousticModel.p", "wb" ))
	print "Acoustic Model Build!"

def electricModel():

	scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

	# load training data from saved feature extraction
	featureNames, trainingData, trainingLabels, trainingSources, classes = extractFeatures("electric_models.mf")
	trainingData = scaler.fit_transform(trainingData)

	print "Building Electric Model..."
	model = SVC()
	# model = DecisionTreeClassifier()
	model.fit(trainingData, trainingLabels)

	print model

	pickle.dump(model, open( "electricModel.p", "wb" ))
	print "Electric Model Build!"

def allModel():

	scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

	# load training data from saved feature extraction
	featureNames, trainingData, trainingLabels, trainingSources, classes = extractFeatures("guitar_models.mf")
	trainingData = scaler.fit_transform(trainingData)

	print "Building Electric Model..."
	model = SVC()
	# model = DecisionTreeClassifier()
	model.fit(trainingData, trainingLabels)
	print model

	pickle.dump(model, open( "allModel.p", "wb" ))
	print "All Model Build!"

if __name__ == '__main__':
	acousticElectricModel()
	# acousticModel()
	# electricModel()
	# allModel()