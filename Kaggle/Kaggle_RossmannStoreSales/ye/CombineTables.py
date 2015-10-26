
# Info of a single store
#"Store","StoreType","Assortment","CompetitionDistance","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2","Promo2SinceWeek","Promo2SinceYear","PromoInterval"
class StoreData:
	nstore = 0
	def __init__(this, lineL):
		# split
		#lineL = str.split(',')[0:2]
		# set id
		this.id = int(lineL[0])
		# set type flags
		typename = lineL[1]
		print typename
		this.st_a = 1 if typename=='a' else 0
		this.st_b = 1 if typename=='b' else 0
		this.st_c = 1 if typename=='c' else 0
		# set assortments as a single variable
		this.asst = ord(lineL[2]) - ord('a') + 1
		# compet dist/date
		this.cdis = lineL[3]
		this.cm = lineL[4]
		this.cy = lineL[5]
		# promotion date

		print this.id,this.st_a,this.st_b,this.st_c,this.asst,this.cdis,this.cm


### main ###
import csv

#"Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"
ftrain='./train.csv'
fstore='./store.csv'

data_s_L = []
data_t_L = []

with open(fstore) as fin:
	fin.readline()
	for idx, line in enumerate( csv.reader(fin,skipinitialspace=True) ):
		print idx, line
		sd = StoreData(line)
		if idx==3:
			break
