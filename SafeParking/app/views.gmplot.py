from flask import render_template,request, make_response, send_from_directory
from app import app
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

import os
import sys

sys.path.append("/Users/cv/DS/Insight/class/Project_crime/program_analyze_data")
import compute_crime_risk as ccr

@app.route('/')
@app.route('/index')
def input():
	return render_template("input.html")


@app.route('/slides') # does not work
def slides():
	return render_template("slides.html")

@app.route('/contact')
def contact():
	return render_template("contact.html")

#merge_static_heatmap.png
#@app.route('/mapimages/<path:filename>') # under test
#@app.route('/<filename>.png')
#@app.route('/<filename>') 
#def return_image(filename):	
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/image/<filename>')
def return_image(filename):	
	#response = make_response(app.send_static_file(filename))
	#filename="/Users/cv/DS/Insight/class/Project_crime/app/static/img/merge_static_heatmap.png"
	#response = make_response(app.send_static_file(filename))
	#response.cache_control.max_age = 0
	#response.mimetype='image/png'
	directory="/Users/cv/DS/Insight/class/Project_crime/app/static/img/"
	#filename="merge_static_heatmap.png"
	response=make_response(send_from_directory(directory,filename))
	return response

"""
@app.route('/mymap.html')
def mymap():
	return render_template('mymap.html')
"""
@app.route('/output')
def output():
	#-- input values
	addressIn=request.args.get('ADDRESS')
	hr=int(request.args.get('TIME'))
	dur=int(request.args.get('DUR'))

	#-- use these values, put it into my code, and produce the results: 
	# 1) risk score of the region nearby (conditional probability). 2) structure of the crimes nearby, random or not, if not, output the top 4 riskest address. 3) the google map html picture. 
	#-- to do: since i do not know javascript, can hardly write function inside html, so i may need 3 html for output: 1. error.html 2. output.random.html 3. output.nonrandom.html (output a table with the riskest addrest)
	
	#-- to do : will need to output some of the correlation function figures
	#-- the 2-piont correlation part is straight forward, no trouble
	#-- does the distribution change a lot from month to month, or year to year?
	#-- what's the safest time to park at this given location
	#-- Q: my city average, is it right? need to look at more locations, see if i can find a place that's below average. A: yes, U of Chicago
	#-- to do: need to find a location that has randomly distributed crime nearby --> Ogden park? BTW, u of chicago is a place with lots of structure,but the correlation function at some distance has value<0, what does that mean?
	#-- to do: do not need to run the random curve every time, just need a one random plot, and use it every time
	#-- to do: need more data, use the 10 year data

	if addressIn==' ':
		addressIn='Chicago,IL' #default address

	addressOut,ratioHr,ratioAll,dfHrLoc,pngNm,flagNoCrime,flagRandom,gmapNm=ccr.getCrimeMap(addressIn,hr,dur)
	#impath="../static/img/"+impath
	print pngNm
	print gmapNm
	impath=gmapNm
	#impath=pngNm

	latlst=[[58.983991,52.395715,51.508742,58.983991],[68,52.395715,51.508742,68]];
	lonlst=[[5.734863,4.888916,-0.120850,5.734863],[7,4.888916,-0.120850,7]];
	colorlst=["#4dffff","#ff33ff"]
	if (flagNoCrime>0): # there is no crime in this area
		message="No historical crime found in %s"%addressIn
		return render_template("error.html")
	#elif flagRandom>0:
	#	#return render_template("outputRandom.html")
	#	return render_template("output.html",addressLst=addressOut,impath=impath)
	else:	
		if ratioAll>1:
			title="HIGH CRIME AREA:"
			message ="the crime rate is %.2f time above the average Chicago crime rate at this time of day."%(ratioAll)
		else:
			title="LOW CRIME AREA:"
			message="the crime rate is %.2f times below the average Chicago crime rate at this time of day"%(1./ratioAll)
		return render_template("output.html",addressLst=addressOut,impath=impath,message=message,titlemsg=title,latC=41.7778691,lonC=-87.60161708,lonlst=lonlst,latlst=latlst,colorlst=colorlst)
		#return render_template("output.gmplot.html",addressLst=addressOut,impath=impath,message=message,titlemsg=title)
	#---call my python functions that computes the crime rate, and makes plot

