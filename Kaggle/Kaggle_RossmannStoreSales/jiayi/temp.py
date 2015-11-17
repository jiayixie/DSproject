
import os
import os.path
import sys

def count(first,*values):
	print values,len(values)
	print values[0]

A=[1,2,3,4]
B=[3,4,5]
count(1,"a","b")
count(1,A,B)

