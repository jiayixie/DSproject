
import numpy as np

"Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"
1,5,2015-07-31,5263,555,1,1,"0","1"
#
"Store","StoreType","Assortment","CompetitionDistance","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2","Promo2SinceWeek","Promo2SinceYear","PromoInterval"
1,"c","a",1270,9,2008,0,,,""
2,"a","a",570,11,2007,1,13,2010,"Jan,Apr,Jul,Oct"


def combine_files(dftrain,dfstore):
	# this is used to combine and digitize the data samples
	valuetrain=dftrain.values.sort(axis=0);
	valuestore=dftrain.values.sort(axis=0);
	columntrain=list(dftrain.columns)
	columnstore=list(dfstore.columns)
	Nrtrain=valuetrain.shape[0]
	Nctrain=valuetrain.shape[1]
	Nrstore=valuestore.shape[0]
	Ncstore=valuestore.shape[1]

	#====change StateHoliday from a,b,c,0 to 3 independent 0,1 series
	#====will perform the rm columns at the end
	for i in range(Nrtrain):
		iSH=columntrain.index('StateHoliday')
		stateHoliday=valuetrain[i][iSH]
		if stateHoliday=='a':
			l=[1,0,0]
		elif stateHoliday=='b':
			l=[0,1,0]
		elif stateHoliday=='c':
			l=[0,0,1]
		else: # 0 or NaN
			l=[0,0,0]
		np.append(valuetrain[i],l)
	np.append(columntrain,['PublicHoliday','EasternHolidy','Christmas'])

	
	#===expand StoreType to 3 independent 0,1 series
	for i in range(Nrtrain):
		iST=columntrain.index('Store')		
				

def main():
	datafile="../data/train.csv"
	#read in data
	dataset=np.genfromtxt(open(datafile,"r"),delimiter=',',dtype='f8')[1:]
	print dataset

if __name__=="__main__":
	main()

