
# load data from file on given columns
def LoadFile( fname, coln_x_L, coln_y, dataX, dataY ):
	#global nbadline
	global nbadline
	with open(fname) as fin:
		for line in fin:
			data = [ float(f) for f in line.split() ]
			if line[-1]==False or True in [ x!=x for x in data ]:
				nbadline += 1
				continue
			dataX.append( [ data[index] for index in coln_x_L ] )
			dataY.append(data[coln_y])



### main ###

# load data
dataX = []; dataY = []
fname="train_DailyRecs.txt"
# 0   1     2   3  4  5   6  7  8   9   10    11    12   13   14   15   16    17
# id sales  y   m  d dow ia ib ic asst cdis prom1 prom2 holp hole holc hols isopen
# 1 5263.0 2015 07 31 5   0  0  1   1 1270.0  1     0   0    0     0     1    True
coln_x_L = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
nbadline = 0
LoadFile(fname, coln_x_L, 1, dataX, dataY)
print len(dataY), " daily records loaded (with ", nbadline, " bad lines discarded)..."
#print dataX[0:10]
#print dataY[0:10]
# try linear regression on day_num-sales
from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(dataX, dataY)
print clf.intercept_, clf.coef_
print clf.score(dataX, dataY)
