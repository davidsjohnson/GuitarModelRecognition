from bextract import extractFeatures
import numpy as np 
import pickle

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def acousticElectricModel():


	# load training data from saved feature extraction
	# data = np.load("acoustic_electric.npz")
	featureNames, trainingData, trainingLabels, trainingSources, classes = extractFeatures("acoustic_electric.mf")
	print "Len:", len(trainingData[0])

	print "Building Acoustic v. Electric Model..."
	model = SVC()

	scores = cross_validation.cross_val_score(model, trainingData, trainingLabels, cv=5)
	print "Scores:", scores

	model.fit(trainingData, trainingLabels)
	print model

	pickle.dump(model, open( "acousticElectricModel.p", "wb" ))
	print "Acoustic v. Electric Model Build!"

def acousticModel():

	# load training data from saved feature extraction
	featureNames, trainingData, trainingLabels, trainingSources, classes = extractFeatures("acoustic_models.mf")
	print "Len:", len(trainingData[0])

	print "Building Acoustic Model..."
	model = SVC()

	scores = cross_validation.cross_val_score(model, trainingData, trainingLabels, cv=5)
	print "Scores:", scores

	model.fit(trainingData, trainingLabels)
	print model

	pickle.dump(model, open( "acousticModel.p", "wb" ))
	print "Acoustic Model Build!"

def electricModel():

	# load training data from saved feature extraction
	featureNames, trainingData, trainingLabels, trainingSources, classes = extractFeatures("electric_models.mf")
	print "Len:", len(trainingData[0])

	print "Building Electric Model..."
	model = SVC()

	scores = cross_validation.cross_val_score(model, trainingData, trainingLabels, cv=5)
	print "Scores:", scores

	model.fit(trainingData, trainingLabels)

	print model

	pickle.dump(model, open( "electricModel.p", "wb" ))
	print "Electric Model Build!"

def allModel():

	# load training data from saved feature extraction
	featureNames, trainingData, trainingLabels, trainingSources, classes = extractFeatures("guitar_models.mf")
	print "Len:", len(trainingData[0])

	print "Building Electric Model..."
	model = SVC()

	scores = cross_validation.cross_val_score(model, trainingData, trainingLabels, cv=5)
	print "Scores:", scores

	model.fit(trainingData, trainingLabels)
	print model

	pickle.dump(model, open( "allModel.p", "wb" ))
	print "All Model Build!"

if __name__ == '__main__':
	acousticElectricModel()
	acousticModel()
	electricModel()
	# allModel()