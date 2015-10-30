
# 10 walks, each walk can be in 4 directions
import numpy as np
import sys


list=np.array([[1,0],[-1,0],[0,1],[0,-1]])
x0=np.array([0,0])

distCri=3

NAll=0
NdistGE=0
for i1 in range(4): # 1
	#x=np.array([0,0])
	#print "\nStart!",x
 	step1=list[i1]
	#print "Walk 1",x
	for i2 in range(4): # 2
		step2=list[i2]
		#print "Walk 2",x,list[i2]
		for i3 in range(4):
			step3=list[i3]
			#print "Walk 3",x,list[i3]
			for i4 in range(4):
				step4=list[i4]
				#print "Walk 4",x,list[i4]
				for i5 in range(4):
					step5=list[i5]
					#print "Walk 5",x,list[i5]
					for i6 in range(4):
						step6=list[i6]
						#print "Walk 6",x,list[i6]
						for i7 in range(4):
							step7=list[i7]
							#print "Walk 7",x,list[i7]
							for i8 in range(4):
								step8=list[i8]
								#print "Walk 8",x,list[i8]
								for i9 in range(4):
									step9=list[i9]
									#print "Walk 9",x,list[i9]
									for i10 in range(4):
										step10=list[i10]
										#print "Walk 10",x,list[i10]
										NAll+=1
										x=x0+step1+step2+step3+step4+step5+step6+step7+step8+step9+step10
										if((x[0]**2+x[1]**2)>=distCri**2):
											NdistGE+=1

print NAll, NdistGE, float(NdistGE)/float(NAll)
 
