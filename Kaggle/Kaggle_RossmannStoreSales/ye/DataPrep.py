
# Info of a single store
#"Store","StoreType","Assortment","CompetitionDistance","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2","Promo2SinceWeek","Promo2SinceYear","PromoInterval"
class StoreData(object):
	# load a single line (stored in a list)
	def __init__(self, lineL):
		# split
		#lineL = str.split(',')[0:2]
		# set id
		self.id = int(lineL[0])
		# set type flags
		typename = lineL[1]
		self.st_a = 1 if typename=='a' else 0
		self.st_b = 1 if typename=='b' else 0
		self.st_c = 1 if typename=='c' else 0
		# set assortments as a single variable
		self.asst = ord(lineL[2]) - ord('a') + 1
		# compet dist/date
		try:
			self.cdis = float(lineL[3])
		except ValueError:
			self.cdis = float('nan')
		try:
			self.csm = int(lineL[4])
		except ValueError:
			self.csm = 1
		try:
			self.csy = int(lineL[5])
		except ValueError:
			self.csy = 1900
		# promotion 2 date
		self.promo2 = True if lineL[6] == '1' else False
		if self.promo2:
			self.psw = int(lineL[7])
			self.psy = int(lineL[8])
			self.pmons_T = tuple(str.replace("Sept","Sep") for str in lineL[9].split(','))
		else:
			self.psw = self.psy = float('nan')
			self.pmons_T = ()
		#print self.id,self.st_a,self.st_b,self.st_c,self.asst,self.cdis,self.csm,self.csy,self.promo2,self.psw,self.psy,self.pmons_T

# Info of a single train data
#"Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"
class TrainData(object):
	# load a single line (stored in a list)
	def __init__(self, lineL):
		self.id = int(lineL[0])
		self.dow = int(lineL[1])
		self.date = datetime.datetime.strptime(lineL[2], '%Y-%m-%d').date()
		self.Sales = float(lineL[3])
		self.nC = int(lineL[4])
		self.isopen = True if lineL[5] == '1' else False
		self.promo1 = int(lineL[6])
		self.holi_p = 1 if lineL[7]=='a' else 0
		self.holi_e = 1 if lineL[7]=='b' else 0
		self.holi_c = 1 if lineL[7]=='c' else 0
		self.holi_s = int(lineL[8])
		#print self.id,self.dow,self.date,self.Sales,self.isopen,self.promo1,self.holi_p,self.holi_e,self.holi_c,self.holi_s

# combined/complete daily info of a single store
class DailyRec(TrainData):

	# constructed with a (StoreData, TrainData) pair
	def __init__(self, lineL, sd_L):
		TrainData.__init__( self, lineL )
		sd = filter(lambda data_s:data_s.id==self.id,sd_L)[0]
		self.st_a = sd.st_a
		self.st_b = sd.st_b
		self.st_c = sd.st_c
		self.asst = sd.asst
		self.cdis = sd.cdis if self.__isNotBeforeDate(sd.csy,sd.csm,1) else float('nan')
		self.promo2 = int( sd.promo2 and self.__isNotBeforeWeek(sd.psy,sd.psw) and self.__isWithinMonths(sd.pmons_T) )
		print self.id,self.Sales,self.date,self.dow,self.st_a,self.st_b,self.st_c,self.asst,self.cdis,self.promo1,self.promo2,self.holi_p,self.holi_e,self.holi_c,self.holi_s,self.isopen
		#print "   ",sd.cdis,sd.csy,sd.csm,sd.promo2,sd.psw,sd.psy,sd.pmons_T

	def __isNotBeforeDate(self, y, m, d):
		return ( (self.date - datetime.date(y,m,d)).total_seconds() >= 0 )

	def __isNotBeforeWeek(self, y, w):
		if w == 0:
			date2 = ( datetime.datetime.strptime(y, 1, 1, "%Y %W %w") - datetime.timedelta(days=7) ).date()
		else:
			date2 = datetime.datetime.strptime(str(y) + " " + str(w) + " 1", "%Y %W %w").date()
		return ( (self.date - date2).total_seconds() >= 0 )

	def __isWithinMonths(self, months_T):
		for mname in months_T:
			if self.date.month == datetime.datetime.strptime(mname, "%b").month:
				return True
		return False

### main ###
import csv
import datetime

ftrain='./train.csv'
fstore='./store.csv'

data_s_L = []
data_t_L = []

with open(fstore) as fin:
	fin.readline()
	for idx, line in enumerate( csv.reader(fin,skipinitialspace=True) ):
		data_s_L.append( StoreData(line) )
print "Info of",len( data_s_L ),"stores loaded"

with open(ftrain) as fin:
	fin.readline()
	for idx, line in enumerate( csv.reader(fin,skipinitialspace=True) ):
		#data_t = TrainData(line)
		data_t_L.append( DailyRec( line, data_s_L ) )
print "Info of",len( data_t_L ),"daily records loaded"

