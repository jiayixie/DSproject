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
from copy import deepcopy

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

	#=== sort the samples based on a given feature
	def sortData(self,featureName):
		columnList=list(self.columns)
		if featureName not in columnList:
			print "sortData: name %s not in our feature list! re-select a feature "%featureName
			sys.exit()
		index=columnList.index(featureName)
		idSort=np.argsort(self.values[:,index])
		self.values=self.values[idSort]
		self.df=pd.DataFrame(self.values,columns=self.columns)

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
	fileTrain="./train_cut.store1.csv"
	#fileTrain="./train_cut.csv"
	#fileTrain="../data/train.csv"
	data=Data(fileStore,fileTrain)
	#data.sortData('OrdinalDay')
	
	dataOri=deepcopy(data) #original data
	#
	y=data.df['Sales'].values
	#featureNameList=['Store','OrdinalDay','DayOfWeek','Customers','Open','Promo','SchoolHoliday','PublicHoliday','EasternHolidy','Christmas','Year','DayOfYear','StoreTypeA','StoreTypeB','StoreTypeC','AssortmentNum','Promo2','CompetitionOpen','Promo2Begin']
	featureNameList=['Store','OrdinalDay','DayOfWeek','Open','Promo','SchoolHoliday','PublicHoliday','EasternHolidy','Christmas','Year','DayOfYear','StoreTypeA','StoreTypeB','StoreTypeC','AssortmentNum','Promo2','CompetitionOpen','Promo2Begin']
	data.selectFeatures(featureNameList)
	data.normalizeFeatures('standard')
	x=data.values
	#x=data.valuesScaled;
	idDayOfYear=featureNameList.index('DayOfYear')
	idYear=featureNameList.index('Year')
	idOday=featureNameList.index('OrdinalDay')

	xTrain,xTest,yTrain,yTest=train_test_split(x,y,train_size=0.8)

	print xTest[:,2]
	print yTest,len(yTest[yTest==0])
	#=== regression
	# do not need scaled features
	#Problem: unstable prediction, prediceted y may >1E10, or <0
	print "begin to do regression"
	# fit different polynomials and plot approximations
	yTrainPredLst=[]
	yTestPredLst=[]
	degLst=[1,2,4]
	for degree in degLst:
		est = make_pipeline(sklearn.preprocessing.PolynomialFeatures(degree), LinearRegression())
		est.fit(xTrain, yTrain)
		yTrainPred=est.predict(xTrain)
		yTestPred=est.predict(xTest)
		print yTestPred[0:10],len(yTestPred[yTestPred==0])
		trainError=mean_squared_error(yTrain,yTrainPred)
		testError=mean_squared_error(yTest,yTestPred)
		#print zip(yTrain,est.predict(xTrain))
		print degree,trainError**0.5,testError**0.5
		yTrainPredLst.append(yTrainPred)
		yTestPredLst.append(yTestPred)

	#=== gradient descent
	# sklearn.linear_model.SGDRegressor
	

	#===
	#==plot training data, test data for year 2013,2014,2015
	#- sort data before plotting
	idX=idOday
	idTrSort=np.argsort(xTrain[:,idX])
	idTeSort=np.argsort(xTest[:,idX])
	xTrain=xTrain[idTrSort]
	yTrain=yTrain[idTrSort]
	xTest=xTest[idTeSort]
	yTest=yTest[idTeSort]
	plt.subplot(2,1,1)
	colorLst=['blue','yellow','green','gray']
	xLst=xTrain
	yLst=yTrain
	yPredLst=yTrainPredLst
	idSort=idTrSort
	plt.plot(xLst[:,idX],yLst,'r.') # the observed data
	for i in range(len(degLst)): # plot the fitted curves
		color=colorLst[i]
		yPredLst[i]=yPredLst[i][idSort]
		plt.plot(xLst[:,idX],yPredLst[i],color)
	plt.title('Training data')
	plt.ylabel('Sales')
	x1,x2,y1,y2=plt.axis()

	plt.subplot(2,1,2)
	xLst=xTest;yLst=yTest;yPredLst=yTestPredLst;idSort=idTeSort
	plt.plot(xLst[:,idX],yLst,'b.') # the observed data
	for i in range(len(degLst)): # plot the fitted curves
		color=colorLst[i]
		yPredLst[i]=yPredLst[i][idSort]
		plt.plot(xLst[:,idX],yPredLst[i],color)
	plt.title('Testing data')
	plt.ylabel('Sales')
	plt.xlabel('OrdinalDay')
	plt.axis((x1,x2,y1,y2))
	plt.show()

	

if __name__=="__main__":
	main()

