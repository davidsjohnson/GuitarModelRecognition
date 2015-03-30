from bextract import extractFeatures
import numpy as np 
import pickle

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class AbstractClassifier():

	def predict(self, data):
		'''Takes in features for one audio file and outputs the accumulated 
		   result as a string; either electric or acoustic'''

		# for all features in the audio track 
		# sum results of model prediction (simple count voting system) UPDATE
		print self.classes
		totalOutputs = np.zeros(len(self.classes))

		for v in data:
			output = self.model.predict(v)
			totalOutputs[output[0]] += 1

		print totalOutputs
		return self.classes[np.argmax(totalOutputs)]


class AcousticElectricClassifier(AbstractClassifier):

	def __init__(self):

		# UPDATE
		# loading data to access classes training data from saved feature extraction
		data = np.load("acoustic_electric.npz")
		
		self.trainingLabels = data['labels']
		self.trainingSources = data['sources']

		self.featureNames = data['featureNames']
		self.classes = data['classes']
		print self.classes

		self.model = pickle.load( open( "acousticElectricModel.p", "rb" ) )


class AcousticClassifier(AbstractClassifier):

	def __init__(self):


		# UPDATE
		# loading data to access classes training data from saved feature extraction
		data = np.load("acoustic_models.npz")
		
		self.trainingLabels = data['labels']
		self.trainingSources = data['sources']

		self.featureNames = data['featureNames']
		self.classes = data['classes']

		self.model = pickle.load( open( "acousticModel.p", "rb" ) )


class ElectricClassifier(AbstractClassifier):

	def __init__(self):


		# UPDATE
		# loading data to access classes training data from saved feature extraction
		data = np.load("electric_models.npz")
		
		self.trainingLabels = data['labels']
		self.trainingSources = data['sources']

		self.featureNames = data['featureNames']
		self.classes = data['classes']
		print self.classes

		self.model = pickle.load( open( "electricModel.p", "rb" ) )

class AllModelClassifier(AbstractClassifier):

	def __init__(self):

		# UPDATE
		# loading data to access classes training data from saved feature extraction
		data = np.load("guitar_models.npz")
		
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


def classify():

	utils = Utils()
	# scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
	
	fileCollection = open("test_data.mf")

	aeModel = AcousticElectricClassifier()
	eModel = ElectricClassifier()
	aModel = AcousticClassifier()

	total = 0
	nCorrect = 0
	nCorrectAll = 0
	totalAll = 0
	for line in fileCollection:
		lineContents = line.split("\t")
		fname = lineContents[0].strip()
		clname = lineContents[1].strip()
		featureNames, features, labels, sources, classes = utils.loadFile(fname, clname)

		aeResult = aeModel.predict(features)

		# # Hack to account for Bextract only return one class when extracting one file...UPDATE
		# if classes[0] == 'electric':
		# 	labels = np.ones(len(features))
		# elif classes[0] == 'acoustic':
		# 	labels = np.zeros(len(features))

		# score = aeModel.model.score(features, labels)


		finalResult = None
		if aeResult == "electric":
			finalResult = eModel.predict(features)

		elif aeResult == "acoustic":
			finalResult = aModel.predict(features)


		if finalResult != None:
			print "This guitar is:", finalResult

		else:
			print "Invalid Results"

		if finalResult == clname:
			nCorrect += 1

		# if classes[0] == 'electric':
		# 	nCorrectAll += outputs[1]
		# 	totalAll = totalAll + outputs[0] + outputs[1]
		# else:
		# 	nCorrectAll += outputs[0]
		# 	totalAll = totalAll + outputs[0] + outputs[1]




	# 	print "File: %s -- Result: %s" % (lineContents[0], aeResult)
	# 	print "Test Score:", score, "\n"
		total +=1

	print "Num Correct:", nCorrect
	print "Total:", total
	print "Accuracy: ", float(nCorrect)/total
	# print "Correct: %d | Total: %d | Accuracy: %f" % (nCorrectAll, totalAll, float(nCorrectAll)/totalAll)



### Unit Testing
if __name__ == '__main__':

	classify()
		
	
