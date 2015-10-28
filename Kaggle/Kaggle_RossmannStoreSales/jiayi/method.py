# method.py 
# this is used to pre process the data, apply different methods to the data, and make plots
from pre_process_file import * #  define the data class, and pre-process the data files
import sklearn 
#from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

class Data():
	def __init__(self,fileNameStore,fileNameTrain):
		self.readData(fileNameStore,fileNameTrain)

	def __fillValue__(self,dataFrame):
		# here self.valuse/columns are all redundant, i use it just to keep the format of data & dataTrain/Store consistent
		self.df=dataFrame
		self.values=dataFrame.values
		self.columns=dataFrame.columns
		self.NR=self.values.shape[0]
		self.NC=self.values.shape[1]
		
	def readData(self,fileNameStore,fileNameTrain):	
		self.fileNameStore=fileNameStore
		self.fileNameTrain=fileNameTrain
		#--process the data files
		#--store	
		dataStore=DataStore(self.fileNameStore)
		dataStore.readData()
		dataStore.modifyStoreType()
		dataStore.modifyAssortment()
		dataStore.modifyCompOpenSince()
		dataStore.modifyPromo2Since()
		#dataStore.fillMissingValues()

		#--train
		dataTrain=DataTrain(self.fileNameTrain)
		dataTrain.readData()
		dataTrain.modifyStateHoliday()
		dataTrain.addDatetime()
		dataTrain.addTimeIndFea(dataStore)
		dataTrain.addTimeDepFea_ordinalDay(dataStore)
		dataTrain.writeDataFrame()

		self.__fillValue__(dataTrain.df)

	#=== select useful features based on the given feature name list===
	def selectFeatures(self,featureNameList):
		columnList=list(self.columns)
		for name in featureNameList:
			if (name not in columnList):
				print "name %s not in our feature list! re-select features"%name
				sys.exit()
		df=self.df[featureNameList]
		self.__fillValue__(df)
	#===can also use sklearn.feature_selection, sklearn.decomposition(.PCA) to select features
#	 sklearn.decomposition.PCA or sklearn.decomposition.RandomizedPCA

	#===normalize every feature, and store the scaler in data, so that we can apply the same scaler to testing set
	def normalizeFeatures(self,flag):
		
		xTrain=self.values
		if(flag=='standard'):
		#every feature will have mean=0,std=1
			self.scaler=sklearn.preprocessing.StandardScaler().fit(xTrain)
			self.valuesScaled=self.scaler.transform(xTrain)
		elif(flag=='minmax'):
		#scale data to [0,1] range
			self.scaler=sklearn.preprocessing.MinMaxScaler()
			self.valuesScaled=self.scaler.fit_transform(xTrain)
		else:
			print "normalizeFeatures under construction, use 'standard' or 'minmax'"
			sys.exit()
	#===apply differnt methods to train the data
	#def applyMethods(self):
	#	est=make_pipeline()

	def __repr__(self):
		string="=======\nfilenames=%s & %s\n%d X %d array\ncolumns: "%(self.fileNameStore,self.fileNameTrain,self.NR,self.NC)
		for i in range(self.NC):
			string=string+" %s "%(self.columns[i])
		string+="\n"
		return string

def main():
	fileStore="../data/store.csv"
	fileTrain="./train_cut.csv"
	data=Data(fileStore,fileTrain)
	dataOri=data #original data
	#
	y=data.df['Sales'].values
	featureNameList=['Store','DayOfWeek','Customers','Open','Promo','SchoolHoliday','PublicHoliday','EasternHolidy','Christmas','Year','DayOfYear','StoreTypeA','StoreTypeB','StoreTypeC','AssortmentNum','Promo2','CompetitionOpen','Promo2Begin']
	data.selectFeatures(featureNameList)
	data.normalizeFeatures('standard')
	x=data.valuesScaled;

	xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.8)
	
	#=== regression
	# fit different polynomials and plot approximations
	#for degree in [0, 1, 2]:
	for degree in [1]:
		est = make_pipeline(sklearn.preprocessing.PolynomialFeatures(degree), LinearRegression())
		est.fit(xTrain, yTrain)
		trainError=mean_squared_error(yTrain,est.predict(xTrain))
		testError=mean_squared_error(yTest,est.predict(xTest))
		#print zip(yTrain,est.predict(xTrain))
		print degree,trainError**0.5,testError**0.5

	#===
	#print data.df.columns
	print data
	print data.df.head()
	print dataOri.df.head()
	

if __name__=="__main__":
	main()

