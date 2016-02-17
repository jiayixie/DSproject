from flask import render_template,request, make_response, send_from_directory
from app import app
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

import os
import sys

#sys.path.append("/Users/cv/Documents/Jiayi/study/DataScienceProject/SafeParking/prog")
#sys.path.append("/Users/cv/DS/Insight/class/Project_crime/program_analyze_data")
#import compute_crime_risk as ccr
#import compute_crime_risk_googleAPI as ccr
sys.path.append("/home/ubuntu/work/DSproject/SafeParking/prog")
import compute_crime_risk_googleAPI_AWS as ccr
#import compute_crime_risk_googleAPI_v2 as ccr

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

	if addressIn==' ':
		addressIn='Chicago,IL' #default address
	lonC,latC,RSHr,RSAll,dfHrLocdelta,dfHrLoc,flagNoCrime,flagRandom,seglonLst,seglatLst,segcolorLst,imgnmRS,hrS,RSS,dfS,seglonLstS,seglatLstS,segcolorLstS,dfW,seglonLstW,segLatLstW,segcolorLstW,RSW,address=ccr.getCrimeMap(addressIn,hr,dur)
	if (flagNoCrime>0): # there is no crime in this area
		message="No historical crime found in %s"%addressIn
		return render_template("error.html",latC=latC,lonC=lonC)
	else:	
		hrS=int(hrS)
		lonlstH=list(dfHrLoc['Longitude'].values.ravel())	
		latlstH=list(dfHrLoc['Latitude'].values.ravel())	
		lonlstHS=list(dfS['Longitude'].values.ravel())
		latlstHS=list(dfS['Latitude'].values.ravel())
		lonlstHW=list(dfW['Longitude'].values.ravel())
		latlstHW=list(dfW['Latitude'].values.ravel())
		#dfWEven=dfW[dfW['DayOfYear']%2==0]
		#
		title1="Park at %d:00 for %d hours: "%(hr,dur)
		if RSW>50:
			title2="HIGH crime area | Risk Score %d (city avg.=50)"%(RSW)
		else:
			title2="LOW crime area | Risk Score %d (city avg.=50)"%(RSW)
		#	
		#if flagRandom>0:
		#	message="Crimes appear to be randonly distributed, park at your own risk!"
		#else:
		message="Watch out for peak crime area  near"
		#	
		if(hr!=hrS):
			#message2="Suggested parking hour: %d:00 (risk decrease by %.0f%s)"%(hrS,(RSAll-RSS)*100./RSAll,"%")
			message2="Park at %d:00, decrease risk by %.0f%s"%(hrS,(RSAll-RSS)*100./RSAll,"%")
		else:
			message2=" "
			#seglonLstW,segLatLstW,segcolorLstW=
		#return render_template("error.html",latC=latC,lonC=lonC)
		return render_template("output.APItest.v2.html",addressLst=addressIn,message=message,title1=title1,titlemsg=title2,latC=latC,lonC=lonC,lonlstSeg=seglonLst,latlstSeg=seglatLst,colorlst=segcolorLst,lonlstH=lonlstH,latlstH=latlstH,message2=message2,imgnmRS=imgnmRS,hr=hr,hrS=hrS,lonlstHS=lonlstHS,latlstHS=latlstHS,lonlstSegS=seglonLstS,latlstSegS=seglatLstS,colorlstS=segcolorLstS,lonlstHW=lonlstHW,latlstHW=latlstHW,lonlstSegW=seglonLstW,latlstSegW=segLatLstW,colorlstW=segcolorLstW,RSW=RSW,addressPeak=address)
		#return render_template("output.html",addressLst=addressIn,message=message,title1=title1,titlemsg=title2,latC=latC,lonC=lonC,lonlstSeg=seglonLst,latlstSeg=seglatLst,colorlst=segcolorLst,lonlstH=lonlstH,latlstH=latlstH,message2=message2,imgnmRS=imgnmRS)
		#return render_template("output.gmplot.html",addressLst=addressOut,impath=impath,message=message,titlemsg=title)
	#---call my python functions that computes the crime rate, and makes plot

