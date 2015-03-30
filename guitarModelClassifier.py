from bextract import extractFeatures
import numpy as np 
import pickle

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class AbstractClassifier():

	def predict(self, data):
		'''Takes in features for one audio file and outputs the accumulated 
		   result as a string; either electric or acoustic'''

		# for all features in the audio track 
		# sum results of model prediction (simple count voting system) UPDATE
		totalOutputs = np.zeros(len(self.classes))

		for v in data:
			output = self.model.predict(v)
			totalOutputs[output[0]] += 1

		print totalOutputs
		return self.classes[np.argmax(totalOutputs)]


class AcousticElectricClassifier(AbstractClassifier):

	def __init__(self):

		scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

		# UPDATE
		# loading data to access classes training data from saved feature extraction
		data = np.load("acoustic_electric.npz")
		
		self.trainingData = scaler.fit_transform(data['features'])
		self.trainingLabels = data['labels']
		self.trainingSources = data['sources']

		self.featureNames = data['featureNames']
		self.classes = data['classes']
		print self.classes

		self.model = pickle.load( open( "acousticElectricModel.p", "rb" ) )


class AcousticClassifier(AbstractClassifier):

	def __init__(self):

		scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

		# UPDATE
		# loading data to access classes training data from saved feature extraction
		data = np.load("acoustic_models.npz")
		
		self.trainingData = scaler.fit_transform(data['features'])
		self.trainingLabels = data['labels']
		self.trainingSources = data['sources']

		self.featureNames = data['featureNames']
		self.classes = data['classes']
		print self.classes

		self.model = pickle.load( open( "acousticModel.p", "rb" ) )


class ElectricClassifier(AbstractClassifier):

	def __init__(self):

		scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

		# UPDATE
		# loading data to access classes training data from saved feature extraction
		data = np.load("electric_models.npz")
		
		self.trainingData = scaler.fit_transform(data['features'])
		self.trainingLabels = data['labels']
		self.trainingSources = data['sources']

		self.featureNames = data['featureNames']
		self.classes = data['classes']
		print self.classes

		self.model = pickle.load( open( "electricModel.p", "rb" ) )

class AllModelClassifier(AbstractClassifier):

	def __init__(self):

		scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

		# UPDATE
		# loading data to access classes training data from saved feature extraction
		data = np.load("guitar_models.npz")
		
		self.trainingData = scaler.fit_transform(data['features'])
		self.trainingLabels = data['labels']
		self.trainingSources = data['sources']

		self.featureNames = data['featureNames']
		self.classes = data['classes']
		print self.classes

		self.model = pickle.load( open( "allModel.p", "rb" ) )

class Utils():

	def loadFile(self, filename, label):
		#UPDATE to handle extracting directly from WAV without writing to MF
		mfFile = open("testWav.mf", 'w')
		mfFile.write(filename + "\t" + label)
		mfFile.close()

		return extractFeatures("testWav.mf")


### Unit Testing
if __name__ == '__main__':
		
	utils = Utils()
	scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
	
	featureNames, features, labels, sources, classes = utils.loadFile("acoustic/1/b.wav", "acoustic")

	features = scaler.fit_transform(features)

	aeModel = AcousticElectricClassifier()
	elecModel = ElectricClassifier()
	acousModel= AcousticClassifier()

	aeResult = aeModel.predict(features)

	modelResult = None
	if aeResult =='electric':
		modelResult = elecModel.predict(features)
	elif aeResult == 'acoustic':
		modelResult = acousModel.predict(features)

	if modelResult != None:
		print "Guitar Model:", modelResult
	else:
		print "Invalid Results..."

	print "\nTesting with All Model Classifier (No Hierarchy)"

	allModel = AllModelClassifier()
	allResult = allModel.predict(features)

	print "All Result:", allResult
