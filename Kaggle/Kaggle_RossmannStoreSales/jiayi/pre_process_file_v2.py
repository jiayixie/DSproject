# this is used to define the store and train data classes, define the pre-processing methods for these two data classes

# TO DO:
# in each method, add one line: if this added feature already exist, then skip

import numpy as np
import pandas as pd
import os
import sys
import datetime

#Store Data
class DataStore:
    def __init__(self,fileName):
        self.fileName=fileName
        if(not os.path.exists(fileName)):
            print "file %s does not exists!"%fileName
            sys.exit()
    
    def readData(self):
	dtype={'Store':int,'DayOfWeek':int}
	#dtype={'DayOfWeek':int}
        self.df=pd.read_csv(self.fileName) #dataForm
        self.NR=self.df.values.shape[0]
        self.NC=self.df.values.shape[1]
   
   
    #===expand StoreType from abcd to 3 independent 0,1 series
    def modifyStoreType(self):
        iST=list(self.df.columns).index('StoreType')
	appList=[] # list that will be appened to the values
        for i in range(self.NR):
            storeType=self.df.values[i][iST]
            if(storeType=='a'):
                l=[1,0,0]
            elif(storeType=='b'):
                l=[0,1,0]
            elif(storeType=='c'):
                l=[0,0,1]
            elif(storeType=='d'):
                l=[0,0,0]
            else:
                print "un-considered situation: missing StoreType "
                sys.exit()
	    appList.append(l)
	to_join=pd.DataFrame(appList,columns=['StoreTypeA','StoreTypeB','StoreTypeC'])
	self.df=self.df.join(to_join)
	self.NC+=3

    #===change Assortment to 1,2,3s
    def modifyAssortment(self):
        iA=list(self.df.columns).index('Assortment')
	appList=[]
        for i in range(self.NR):
            asso=self.df.values[i][iA]
            if(asso=='a'):
                l=1
            elif(asso=='b'):
                l=2
            elif(asso=='c'):
                l=3
	    appList.append(l)
	self.df['AssortmentNum']=appList
        self.NC+=1
    
    #===from CompetitionOpenSinceMonth & CompetitionOpenSinceYear get the CompetitionOpenSince_OrdinalDay  
    def modifyCompOpenSince(self):
	icosM=list(self.df.columns).index('CompetitionOpenSinceMonth')      
	icosY=list(self.df.columns).index('CompetitionOpenSinceYear') 
	appList=[]
	for i in range(self.NR):
	    month=self.df.values[i][icosM]
	    year=self.df.values[i][icosY] #float?
	    day=1; #*** missing value, day of the month
	    if(np.isnan(month) or np.isnan(year)):
	    #***** missing value, assume no open competition, i.e. open in the future
	    	ordinalDay=datetime.date(3000,1,1).toordinal()
	    else:
		date=datetime.date(int(year),int(month),day)
		ordinalDay=date.toordinal()
	    appList.append(ordinalDay)
	self.df['CompetitionOpenSinceOrdinalDay']=np.array([appList]).T
	self.NC+=1

    #=== from Promo2SinceWeek/Year, get Promo2SinceOrdinalDay
    def modifyPromo2Since(self):
	ipsW=list(self.df.columns).index('Promo2SinceWeek')
	ipsY=list(self.df.columns).index('Promo2SinceYear')
	appList=[]
	for i in range(self.NR):
	    week=self.df.values[i][ipsW]
	    year=self.df.values[i][ipsY]
	    day=1 #*** missing value, day of the week, %w where 0 is Sunday and 6 is Saturday. Attention, in datetime, there are a few different definitions for week of the day
	    if(np.isnan(week) or np.isnan(year)): #*** missing value
		ordinalDay=datetime.date(3000,1,1).toordinal()
	    else:
	    	date=datetime.datetime.strptime("%04d-%02d-%d"%(year,week,day),"%Y-%W-%w") # %w weekday, where 0 is Sunday and 6 is Saturday.
  		ordinalDay=date.toordinal()
	    appList.append(ordinalDay)
	self.df['Promo2SinceOrdinalDay']=np.array([appList]).T
	self.NC+=1

    #=== digitize the PrmoInterval month information, Jan->1
    #def modifyPromoInter(self):
#    	ipi=list(self.df.columns).index('PromoInterval')
#	for i in range(self.NR):
#		mlist=self.df.values[i][ipi]
#		if(isinstance(mlist,float)): #*** missing value, no month list
#			l=[]
		
    #=== fill missing values 
#    def fillMissingValue(self):

    #--toString--
    def __repr__(self):
        str="=======\nfilename=%s\n%d X %d array\ncolumns: "%(self.fileName,self.NR,self.NC)
        for i in range(self.NC):
            str=str+" %s "%(self.df.columns[i])
        str+="\n"
        return str

#Train Data
class DataTrain:
    def __init__(self,fileName):
        self.fileName=fileName
        if(not os.path.exists(fileName)):
            print "file %s does not exists!"%fileName
            sys.exit()

    def readData(self):
        self.df=pd.read_csv(self.fileName) #dataForm
        self.NR=self.df.values.shape[0]
        self.NC=self.df.values.shape[1]
	

    #====change StateHoliday from a,b,c,0 to 3 independent 0,1 series
    #====will perform the rm columns at the end
    def modifyStateHoliday(self):
        iSH=list(self.df.columns).index('StateHoliday')
	appList=[]
        for i in range(self.NR):
            
            stateHoliday=self.df.values[i][iSH]
            if stateHoliday=='a':
                l=[1,0,0]
            elif stateHoliday=='b':
                l=[0,1,0]
            elif stateHoliday=='c':
                l=[0,0,1]
            elif stateHoliday=='0': # 0
                l=[0,0,0]
            else:
                # *** Need to change here, probably should use the most-frequent category
                l=[0,0,0]
	    appList.append(l)
	to_join=pd.DataFrame(appList,columns=['PublicHoliday','EasternHolidy','Christmas'])
        self.df=self.df.join(to_join)
	self.NC+=3

    #==== add DayOfYear feature 1-366
    #==== add the prolepic Gregorian ordinal of the data
    def addDatetime(self):        
    	iDate=list(self.df.columns).index('Date')
	appList=[]
	for i in range(self.NR):
	    strdate=self.df.values[i][iDate]
	    date=datetime.datetime.strptime(strdate,"%Y-%m-%d")
	    year=date.timetuple().tm_year
	    month=date.timetuple().tm_mon
	    dayOfYear=date.timetuple().tm_yday
	    ordianalDay=date.toordinal()
	    l=[year,month,dayOfYear,ordianalDay]
	    appList.append(l)
	to_join=pd.DataFrame(appList,columns=['Year','Month','DayOfYear','OrdinalDay'])
	self.df=self.df.join(to_join)
  	self.NC+=4	

    #=== add the time-independent store feature to data train
    #by default, the dataStore are sorted array, dataStore.values[0] is info for store id=1, id~[1,N]. In more general situation, will need to sort Store and fill missing store
    # time indep. features include: StoreType  Assortment  CompetitionDistancePromo2
    def addTimeIndFea(self,dataStore):	
	feaNameList=['StoreTypeA','StoreTypeB','StoreTypeC','AssortmentNum','Promo2']
	indexList=[]
	for name in feaNameList:
		indexList.append(list(dataStore.df.columns).index(name))
	#
	iStore=list(self.df.columns).index('Store') #column number
	appList=[]
	for i in range(self.NR): # loop over all store-date info
	    storeID=self.df.values[i][iStore]
	    l=[]
	    for j in range(len(feaNameList)):
		index=indexList[j]
		l.append(dataStore.df.values[storeID-1][index]) # by default dataStore.values[0]-> infor for store with storeID=1
	    appList.append(l)
	to_join=pd.DataFrame(appList,columns=feaNameList)
	self.df=self.df.join(to_join)
	self.NC+=len(feaNameList)

    #=== add the time-dependent store features to data train 
    # in this method, only the features that are related to ordinalDay are added
    def addTimeDepFea_ordinalDay(self,dataStore):
	feaNameList=['CompetitionOpenSinceOrdinalDay','Promo2SinceOrdinalDay']
 	outNameList=['CompetitionOpen','Promo2Begin'] # value is either 0 or 1
	indexList=[]
	for name in feaNameList:
                indexList.append(list(dataStore.df.columns).index(name))
	iStore=list(self.df.columns).index('Store') #column number
	iOrday=list(self.df.columns).index('OrdinalDay')
	appList=[]
	for i in range(self.NR): # loop over all store-date info in train
	    storeID=self.df.values[i][iStore]
	    ordinalDay=self.df.values[i][iOrday]
	    l=[]
	    for j in range(len(feaNameList)):
		index=indexList[j]
		beginDay=dataStore.df.values[storeID-1][index]
		if(beginDay<=ordinalDay): # on this day(ordinalDay) something is on
			l.append(1)
		else:
			l.append(0)
	    appList.append(l)	
	to_join=pd.DataFrame(appList,columns=outNameList)
        self.df=self.df.join(to_join)
	self.NC+=len(outNameList)

    #==== remove some empty records
    #**** missing value
    #def removeMissingValues(self):

    def __repr__(self):
        str="=======\nfilename=%s\n%d X %d array\ncolumns: "%(self.fileName,self.NR,self.NC)
        for i in range(self.NC):
            str=str+" %s "%(self.df.columns[i])
        str+="\n"
        return str

#------------------------------------------------------------------------
def main():
    #---get store information, and digitize some of the features
    datafile="../data/store.csv"
    dataStore=DataStore(datafile)
    dataStore.readData()
    dataStore.modifyStoreType()
    dataStore.modifyAssortment()
    dataStore.modifyCompOpenSince()
    dataStore.modifyPromo2Since()
    print dataStore
    print dataStore.df.values[0],np.shape(dataStore.df.values[0])
	
    #---
    #datafiletrain="../data/train.csv"
    datafiletrain="./train_cut.csv"
    dataTrain=DataTrain(datafiletrain)
    dataTrain.readData()
    #print dataTrain
    dataTrain.modifyStateHoliday()
    dataTrain.addDatetime()
    print dataTrain.df.values[0],np.shape(dataTrain.df.values[0])
    print dataTrain.df.columns
    dataTrain.addTimeIndFea(dataStore)
    dataTrain.addTimeDepFea_ordinalDay(dataStore)
    print dataTrain
    print dataTrain.df.values[0],np.shape(dataTrain.df.values[0])
    print dataTrain.df.columns
    #dataTrain.writeDataFrame()
    print dataTrain.df.head()

if __name__=="__main__":
    main()

