# method.py 
#
# this is used to analyze the data, apply different methods to the data, and make plots
from pre_process_file_v2 import * #  define the data class, and pre-process the data files
import sklearn 
#from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from copy import deepcopy
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

class Data():
	def __init__(self,fileNameStore,fileNameTrain):
		self.readData(fileNameStore,fileNameTrain)

	def __fillValue__(self,dataFrame):
		self.df=dataFrame

	def readData(self,fileNameStore,fileNameTrain):	
		self.fileNameStore=fileNameStore
		self.fileNameTrain=fileNameTrain
		#--process the data files
		#--data store	
		dataStore=DataStore(self.fileNameStore)
		dataStore.readData()
		dataStore.modifyStoreType()
		dataStore.modifyAssortment()
		dataStore.modifyCompOpenSince()
		dataStore.modifyPromo2Since()
		#dataStore.fillMissingValues()

		#--data train
		dataTrain=DataTrain(self.fileNameTrain)
		dataTrain.readData()
		dataTrain.modifyStateHoliday()
		dataTrain.addDatetime()
		dataTrain.addTimeIndFea(dataStore)
		dataTrain.addTimeDepFea_ordinalDay(dataStore)
		
		self.__fillValue__(dataTrain.df)

	#===can also use sklearn.feature_selection, sklearn.decomposition(.PCA) to select features
#	 sklearn.decomposition.PCA or sklearn.decomposition.RandomizedPCA

	#===normalize every feature, and store the scaler in data, so that we can apply the same scaler to test set => self.dfScaled, self.scaler
	def normalizeFeatures(self,flag,featureNameList):
		
		#xTrain=self.df.values
		xTrain=self.df[featureNameList].values
		if(flag=='standard'):
		#every feature will have mean=0,std=1
			self.scaler=sklearn.preprocessing.StandardScaler().fit(xTrain)
			valuesScaled=self.scaler.transform(xTrain)
		elif(flag=='minmax'):
		#scale data to [0,1] range
			self.scaler=sklearn.preprocessing.MinMaxScaler()
			valuesScaled=self.scaler.fit_transform(xTrain)
		else:
			print "normalizeFeatures under construction, use 'standard' or 'minmax'"
			sys.exit()
		self.dfScaled=pd.DataFrame(valuesScaled,columns=featureNameList)	
	#===toString====
	def __repr__(self):
		string="=======\nfilenames=%s & %s\n%d X %d array\ncolumns: "%(self.fileNameStore,self.fileNameTrain,self.df.values.shape[0],self.df.values.shape[1])
		for i in range(self.NC):
			string=string+" %s "%(self.df.columns[i])
		string+="\n"
		return string

#==========================================================================
#=== stepWise feature selection ==== forward ====
def featureRedu(dfIn):
	#-generate the x (feature), and y (response)
	y=dfIn['Sales'].values
	#-based on commmon sense, select some feature to start with
	featureNameList=['Store','OrdinalDay','DayOfWeek','Open','Promo','SchoolHoliday','PublicHoliday','EasternHolidy','Christmas','Year','DayOfYear','StoreTypeA','StoreTypeB','StoreTypeC','AssortmentNum','Promo2','CompetitionOpen','Promo2Begin'] #,'Customers'] 
	df=dfIn[featureNameList]
	
	#=== stepwise feature selection
	Nfeature=len(featureNameList)	
	
	fLst=[] # current feature list, name of features, the feature list that will be added in each step
	lastfLst=[] # last feature list, names of features
	recordErrorLst=[] 
	recordNLst=[] # # of features
	recordCoeffLst=[] # this is a 2d list, stores the coefficients from every step's regression result
	Nf=len(featureNameList)
	#error=regression_simp(x,y,0) # regression with no feature, just intercept, need to turn on the fit_intercept in LinearRegression
	#print error

	while(len(lastfLst)<Nf):
		errorMin=1e30
		for ifea in range(Nf):
			fNm=featureNameList[ifea]
			if fNm in lastfLst:
				continue
			flst=lastfLst+[fNm]
			x=df[flst].values
			trainError,coeffLst,NfeatI,NfeatO=regression_simp(x,y,1)
			if trainError<errorMin:
				errorMin=trainError
				fNmMin=fNm
				coeffLstMin=coeffLst
		lastfLst=lastfLst+[fNmMin]
		recordErrorLst.append(errorMin)
		recordNLst.append(len(lastfLst))
		recordCoeffLst.append(coeffLstMin)

	print "\n====Forward StepWise feature reduction===="
	print "intercept=%5.3E"%(recordCoeffLst[Nf-1][0])
	for i in range(Nf):
		print "i=%2d error=%5.3E Nfeature=%d Ceoff=%5.3E feaNm=%-20s"%(i,recordErrorLst[i],recordNLst[i],recordCoeffLst[Nf-1][i+1],lastfLst[i])
		#-the 0th Coeff is for the constant, not for feature, so use [i+1] instead of [i] to access the coefficient
		#print recordCoeffLst[i]
		#print lastfLst[0:i+1]
	#print lastfLst
	plt.close('all')
	plt.plot(recordNLst,recordErrorLst,'r-')
	plt.ylabel("MSE")
	plt.xlabel("# of features")
	plt.savefig("pic/featureRedu.png")
	#plt.show()
	#-return the recorded stepWise change of error(1d), coefficient (2d), and the feature name list (1d list, ordered by the importance of feature)
	#-attention, the len(recordCoeffLst)=len(lastfLst)+1!=len(lastfLst) because the 0th Coeff is for the constant/intercept, not for feature coeff
	return recordErrorLst,recordCoeffLst,lastfLst  

#=== PCA ====== 
#- apply the PCA method to our data features. The output transformed feature would have no real physical meaning?
#- principal component analysis, Linear dimensionality reduction using Singular Value Decomposition of the data and keeping only the most significant singular vectors to project the data to a lower dimensional space.
def do_PCA(ncomp,dfIn):
	featureNameList=['Store','OrdinalDay','DayOfWeek','Open','Promo','SchoolHoliday','PublicHoliday','EasternHolidy','Christmas','Year','DayOfYear','StoreTypeA','StoreTypeB','StoreTypeC','AssortmentNum','Promo2','CompetitionOpen','Promo2Begin'] #,'Customers'] 
	df=dfIn[featureNameList]
	x=df.values
	#preprocess
	pca=PCA(n_components=ncomp,copy=True,whiten=False)
	pca.fit(x)
	xTrans=pca.fit_transform(x)
	pca_score=pca.explained_variance_ratio_ # Percentage of variance explained by each of the selected components [n_components]
	pca_V=pca.components_ # Principal axes in feature space, representing the directions of maximum variance in the data. [n_components, n_features]
	for i in range(ncomp):
		#index=np.argmax(pca_V[i])
		indexLst=np.argsort(map(np.abs,pca_V[i]))
		print pca_V[i][indexLst]
		for j in range(2):
			k=-1-j
			index=indexLst[k]
			print "dominant feature {0} is {1},{2}".format(i,featureNameList[index],pca_V[i][index])

	return xTrans,pca_score,pca_V



#=== the simple regression, polynomial + linear regression
def regression_simp(x,y,deg):
	#est=make_pipeline(sklearn.preprocessing.PolynomialFeatures(deg),LinearRegression())
	est=Pipeline([('PolyFea',sklearn.preprocessing.PolynomialFeatures(deg)),('LinearReg',LinearRegression(fit_intercept=False))]) #-PolynomialFeatures has created a constant term (i.e. intercept), so do not need to fit intercept anymore
	est.fit(x,y)
	NfeatI=est.named_steps['PolyFea'].n_input_features_
	NfeatO=est.named_steps['PolyFea'].n_output_features_
	coeff=est.named_steps['LinearReg'].coef_
	intercept=est.named_steps['LinearReg'].intercept_

	yPred=est.predict(x)
	error=mean_squared_error(y,yPred)
	return error,coeff,NfeatI,NfeatO

#=== regression with Train and Test sets, the output test error can be used for cross validation
def regression(xTrain,xTest,yTrain,yTest,degLst):
 	#-do not need normalized features
	#-Problem: unstable prediction, prediceted y may >1E10
 	print "\n=====begin to do regression===="
 	#-fit with different degrees of polynomial
 	yTrainPredLst=[]
 	yTestPredLst=[]
   	for degree in degLst:
        	est = make_pipeline(sklearn.preprocessing.PolynomialFeatures(degree), LinearRegression())
        	est.fit(xTrain, yTrain)
        	yTrainPred=est.predict(xTrain)
        	yTestPred=est.predict(xTest)
        	trainError=mean_squared_error(yTrain,yTrainPred)
        	testError=mean_squared_error(yTest,yTestPred)
        	#print zip(yTrain,est.predict(xTrain))
        	yTrainPredLst.append(yTrainPred)
        	yTestPredLst.append(yTestPred)
        	print "regression, poly-degree={0}, train_RMS_Err={1}, test_RMS_Err={2}".format(degree,trainError**0.5,testError**0.5)
    
    	return yTrainPredLst,yTestPredLst,trainError,testError

#=== sub function for plotting the regression result ===
def plotRegSub(degLst,idX,idPlot,x,y,yPred,xlabel,ylabel,title):	
	plt.subplot(idPlot)
	colorLst=['blue','gray','green','yellow']
	if (idX>=0):
		idSort=np.argsort(x[:,idX])
		x=x[idSort]
		y=y[idSort]
		plt.plot(x[:,idX],y,'r.')
	else:
		plt.plot(y,'r.')
	for i in range(len(degLst)):
		color=colorLst[i]
		if(idX>=0):
			plt.plot(x[:,idX],yPred[i][idSort],color,label='deg %d'%degLst[i])
		else:
			plt.plot(yPred[i],color,label='deg %d'%degLst[i])
		plt.legend(loc='upper right')	
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	x1,x2,y1,y2=plt.axis()
	return x1,x2,y1,y2

#=== plot the regression result ====
def plotRegression(idX,degLst,*valueLst):
	#==plot the regressin result
	N=len(valueLst)
	if((N-3)*(N-6)!=0):
		print "#==== plotRegression, the length of valueLst need to be 3 (x,y,ypred) or 6 (x1,y1,y1pred,x2,y2,y2pred), but we get {0}".format(N)
		sys.exit()
	#-
	x=valueLst[0];y=valueLst[1];yPred=valueLst[2]
	plt.close('all')
	x1,x2,y1,y2=plotRegSub(degLst,idX,211,x,y,yPred,' ','Sales','Training Data')
	if(N==6):
		x=valueLst[3];y=valueLst[4];yPred=valueLst[5]
		plotRegSub(degLst,idX,212,x,y,yPred,' ','Sales','Test Data')
		#plt.axis((x1,x2,y1,y2))
	#plt.tight_layout()
	plt.savefig("pic/fit_poly.png")
	#plt.show()


def main():
	#===read in train and store data, combine them
	fileStore="../data/store.csv"
	fileTrain="./train_cut.store1.csv"
	#fileTrain="./train_cut.csv"
	#fileTrain="../data/train.csv"
	data=Data(fileStore,fileTrain)
	#--sort the data by date ---
	data.df=data.df.sort(columns=['OrdinalDay'],ascending=True)
	print "\n=====writting the combined data..."
	data.df.to_csv("./combined_train_store.csv")
	
	#=== pre-process the data
	#featureNameList=list(data.df.columns.values)
	featureNameList=['Store','OrdinalDay','DayOfWeek','Open','Promo','SchoolHoliday','PublicHoliday','EasternHolidy','Christmas','Year','DayOfYear','StoreTypeA','StoreTypeB','StoreTypeC','AssortmentNum','Promo2','CompetitionOpen','Promo2Begin','Customers']
	data.normalizeFeatures('standard',featureNameList)

	#=== do PCA ===

	xTrans=pca_score,pca_V=do_PCA(2,data.dfScaled)
	print pca_score
	print pca_V
	"""
	#===do feature reduction
	ErrorReduLst,CoeffReduLstLst,featureReduLst =featureRedu(data.df)
	featureNameList=['Sales']+featureReduLst[0:5]
	#-
	plt.close('all')
	df=data.df[featureNameList]
	scatter_matrix(df,alpha=0.2)
	plt.savefig('pic/scatter_matrix.png')

	#=== other regression to try
	# http://scikit-learn.org/stable/modules/linear_model.html
	# sklearn.linear_model.SGDRegressor
	# sklearn.linear_model.Ridge

    	#===do regression
	y=data.df['Sales'].values
	#featureNameList=['Store','OrdinalDay','DayOfWeek','Open','Promo','SchoolHoliday','PublicHoliday','EasternHolidy','Christmas','Year','DayOfYear','StoreTypeA','StoreTypeB','StoreTypeC','AssortmentNum','Promo2','CompetitionOpen','Promo2Begin'] # 'Customers'
	featureNameList=featureReduLst[0:5]
	#data.normalizeFeatures('standard',featureNameList) #-> self.dfScaled, self.scaler
	x=data.dfScaled[featureNameList].values
	#x=data.df[featureNameList].values
    	xTrain,xTest,yTrain,yTest=train_test_split(x,y,train_size=0.8)
	#-
    	degLst=[1,2]
    	yTrainPredLst,yTestPredLst,trainError,testError=regression(xTrain,xTest,yTrain,yTest,degLst)
	#idX=-1 #-the index of the xvalue I want to use as x-axis. If idX<0, then just plot against the data #
	#idX=list(data.dfScaled.columns).index('OrdinalDay')
	if('OrdinalDay' in featureNameList):
		idX=featureNameList.index('OrdinalDay')
	else:
		idX=-1
	plotRegression(idX,degLst,xTrain,yTrain,yTrainPredLst,xTest,yTest,yTestPredLst)
	#===========
	"""
	
	

if __name__=="__main__":
	main()

