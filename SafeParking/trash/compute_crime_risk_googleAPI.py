#---this version won't make any plots, plot will be made in html using google API
# TO READ:
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_species_kde.html
# http://www.sciencedirect.com/science/article/pii/S0167923614000268
# https://www.princeton.edu/~fhs/paper327/paper327.pdf
# http://www.emeraldinsight.com/doi/full/10.1108/PIJPSM-04-2013-0039
# -> this is actually useful, accuracy of KDE maps can be evaluated using 3 different metrics: 
#1)hit rate, the percentage of all Time 2 crimes that are contained by predicted hotspots created from Time 1 data
#2)predictive accuracy index: hit rate to the area percentage, or the percentage of the study area that is defined as a hotspot, 
# = (number of present crimes observed in the predicted hotspots/total number of present crimes)/(the area of hotspots/the area of the entire extent (e.g. study area)))
#3)recapture rate index
# two point correlation: http://www.astro.lu.se/Education/utb/ASTM21/ASTM21-P1.pdf (very good explaination!)
# coding: utf-8

# TODO---
# return the segments in a better way. i.e. return N layers of risk boundary, and also return the corresponding color list
# compute an absolute risk score, risk score of city avarege=50. risk score=50+[(#_of_crimes/area/hour_dur-average)/(average)*100]
# need to find a way such that the score wont't exceed [0,100]?? is this necessary? then i need to know the range of risk across all hour and all 2km*2km region
# moving window, and get the window that has lowest risk sum

#get_ipython().magic(u'matplotlib inline')
import sys
import pandas as pd
import geocoding as geo #use pygeocoding instead?
import numpy as np
import scipy as sp
import pymysql as mdb
import pandas.io.sql as psql
from sklearn.neighbors import KernelDensity
import scipy.ndimage as ndimage
from PIL import Image
from StringIO import StringIO
from pygeocoder import Geocoder
from astroML.correlation import bootstrap_two_point_angular
import statsmodels.api as sm
import socket
import get_map as gm
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psycopg2
from matplotlib import _cntr as cntr
#from scipy.interpolate import spline
import operator

import os

FLAGPLOT=0 # plot the KDE and 2-point correlation graph?
DBNM="Chicago_crime_db"
USERNM="cv"
#TABLENM="crimevehicledate_from2001"
TABLENM="crimevehicledate_from2001_clean"
#TABLENM="crimevehicledate_from2001_cleanstreet"
AREAChicago=6.061e+8 # meter square
GMAPRAD=20
GMAPRAD2= 50 #50 # in meter
BANDWIDTH=GMAPRAD2/(180./np.pi*111045) # in rad,8e-6 ~ 50m
IMGDIR="/Users/cv/DS/Insight/class/Project_crime/app/static/img"
#IMGDIR="/Users/cv/DS/Insight/class/Project_crime/app/templates/"
#DAYtotal=400 # total number of days in my database


# In[1260]:

deg2meter=111045.0
meter2deg=1./deg2meter
deg2rad=np.pi/180.
RADIUS=1000. # in meters


# In[1261]:

# the main function, produce the crime map around the given location, and compute the crime rate

def getCrimeMap(addressIn,hr,dhr,rad=1300.): # radius(actually half box length) in meters
    #string = "rm -f %s/mymap_*.html %s/merge_static_heatmap*.png"%(IMGDIR,IMGDIR)
    ##os.system(string)
    geocode=geo.googleGeocoding(addressIn)
    latlon=geo.getGeocodeLatLong(geocode)
    latC=latlon['lat']
    lonC=latlon['lng']
    #print "geocode",geocode
    #print "latlon",latlon
    #print "latC,lonC",latC,lonC
    #???? read get_map.py
    scl,request, gg_img = gm.get_static_google_map(latC,lonC,scale=2) #make map, figure out how big it is!
    #rad=scl*320 #??? scl, scale is meter/pixel?
    
    #-- compute the risk score of this region
    conn,dfHrLocdelta,dfHrLoc,latRange,lonRange,RSHr,RSAll,riskScoreDeltaLst=crimeRiskVal(lonC,latC,hr,dhr)

    
    #---need to get the lat,lon,segments for the suggested hour
    
  
    flagNoCrime=0
    if(len(dfHrLoc)==0):
        flagNoCrime=1
        flagRandom=1
        address="no crime in selected address: %s"%addressIn
        impath=" "
        RSHr=0
        dfHrLoc=0
        return lonC,latC,0,0,0,0,flagNoCrime,flagRandom,0,0,0,0,0,0,0,0,0,0
    #-- plot the risk score image as a function of hour, and also get the suggested parking hour based on moving window average
    imgnmRS,hrSug,RSSug=plotRiskScore(riskScoreDeltaLst,hr,dhr)
    #-- get the peak-crime-spots' kde values, locations (lon&lat&address), and the 2-point correlation value
    flagRandom,seglonLst,seglatLst,segcolorLst=crimeKdeCorr(dfHrLoc,step=100,flagCptCorr=1); # step in meters

    #--- get the whole day's map
    #dfW,seglonLstW,segLatLstW,segcolorLstW=getInfoHrSug(0,24,latRange,lonRange,conn)
    """
    #--cheat----
    if ("University" in addressIn):
	    #print "yes, cheat!!!"
	    flagRandom=0
    elif ("63" in addressIn):
	    flagRandom=1
    """

    print "Number of crimes in this area around this time:",len(dfHrLoc)

    if(hrSug!=hr): # the suggested parking hour is different from
        dfS,seglonLstS,segLatLstS,segcolorLstS=getInfoHrSug(hrSug,dhr,latRange,lonRange,conn)
        return lonC,latC,RSHr,RSAll,dfHrLocdelta,dfHrLoc,flagNoCrime,flagRandom,seglonLst,seglatLst,segcolorLst,imgnmRS,hrSug,RSSug,dfS,seglonLstS,segLatLstS,segcolorLstS
    else:
        return lonC,latC,RSHr,RSAll,dfHrLocdelta,dfHrLoc,flagNoCrime,flagRandom,seglonLst,seglatLst,segcolorLst,imgnmRS,hrSug,RSSug,dfHrLoc,seglonLst,seglatLst,segcolorLst

#-- get the lat, lon, segments for the suggested hour--- my code is a mess, should rewrite it!!
def getInfoHrSug(hrSug,dhr,latRange,lonRange,conn):
    if(dhr>=24):
        sqltime=' hour>0 '
    else:
        tbegin=hrSug
        tend=hrSug+dhr
        ibegin,iend=getTimeId(tbegin,tend)
        if(ibegin<iend):
            sqltime=' hour>=%f AND hour<%f '%(ibegin,iend)
        else:
            #sqltime=' (hour>=%f AND hour<=23) OR (hour>=0 AND hour<%f) '%(ibegin,iend)
            sqltime=' hour>=%f OR hour<%f '%(ibegin,iend)
    
    sqlloc='"Longitude">=%f AND "Longitude"<=%f AND "Latitude">=%f AND "Latitude"<=%f'%(lonRange[0],lonRange[1],latRange[0],latRange[1])
    sql='SELECT "Longitude","Latitude" FROM %s WHERE (%s) AND (%s)'%(TABLENM,sqltime,sqlloc)
    dfS=pd.read_sql(sql,conn)
    flagRandom,seglonLstS,seglatLstS,segcolorLstS=crimeKdeCorr(dfS,step=100,flagCptCorr=0);

    return dfS,seglonLstS,seglatLstS,segcolorLstS


#--plot the risk score figure, and also get the suggested parking hour using moving window average
def plotRiskScore(riskScoreDeltaLst,hr,dur):
    #the riskScoreDeltaLst contains the hour and count of crimes in that hour
    x=riskScoreDeltaLst[0,:]
    y=riskScoreDeltaLst[1,:]
    y=np.append(y[0],y)
    x=np.append(x,x[-1]+1)
    plt.step(x,y,'r-',lw=5)
    plt.axhline(y=50,ls='--',lw=5,color='k')
    plt.xlabel('Hour Of Day',fontsize=24)
    plt.ylabel('Risk Score',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=15)
    if(hr+dur<=24):
        plt.axvspan(hr,hr+dur, facecolor='gray', alpha=0.1,lw=0)
    else:
        plt.axvspan(hr,24,facecolor='gray', alpha=0.1,lw=0)
        plt.axvspan(0,hr+dur-24,facecolor='gray', alpha=0.1,lw=0)
        plt.xlim(0,24)

    #-- moving window average, find the parking time with least risk, assume the duration is the same
    Nwind=dur
    movingAvg=list(np.convolve(riskScoreDeltaLst[1,:], np.ones((Nwind,))/Nwind, mode='valid'))
    min_index, min_value = min(enumerate(movingAvg), key=operator.itemgetter(1))
    hrSug=riskScoreDeltaLst[0,min_index]
    RSSug=min_value
    #
    if(hrSug!=hr):
        if(hrSug+dur<=24):
            plt.axvspan(hrSug,hrSug+dur, facecolor='b', alpha=0.2,lw=0)
        else:
            plt.axvspan(hrSug,24, facecolor='b', alpha=0.2,lw=0)
            plt.axvspan(0,hrSug+dur-24, facecolor='b', alpha=0.2,lw=0)
            plt.xlim(0,24)
    plt.gcf().subplots_adjust(bottom=0.15)
    imgnm="riskScore.png"
    plt.savefig("%s/%s"%(IMGDIR,imgnm))
    plt.close()
    return imgnm,hrSug,RSSug

# getCrimeMap
# In[1262]:

def connect_to_db(dbname,username):
    try:
        conn=psycopg2.connect(database = dbname, user = username)
        cur=conn.cursor()
        return conn,cur
    except psycopg2.Error as e:
        print e.message
        
def readDf(dbname,username,lonmin,lonmax,tablename):
    flag=1
    conn,cur=connect_to_db(dbname,username);
    dfCrime=pd.read_sql_query('SELECT * FROM %s WHERE "Longitude"> %f AND "Longitude"<%f AND "Latitude">%f AND "Latitude"<%f'%(tablename,lonmin,lonmax,latmin,latmax),con=conn)
    if(len(dfCrime)<1):
        print "hey, nothing inside"
        flag=0
    else:
        dfCrime=dfCrime[np.isfinite(dfCrime['Latitude'])].convert_objects(convert_numeric=True)
    return flag,dfCrime


# In[1263]:

def centroid(data):
    h,w = np.shape(data)
    x = np.arange(0,w)
    y = np.arange(0,h)

    X,Y = np.meshgrid(x,y)

    cx = np.sum(X*data)/np.sum(data)
    cy = np.sum(Y*data)/np.sum(data)

    return cx,cy


# In[1264]:

def loc2address(kdeCenterGeo,flagRandom=0):
    address=[]
    if(flagRandom>0):
        address.append("Park at your own risk, crimes appear to be randomly distributed in this area.")
    else:
        for i in range(kdeCenterGeo.shape[0]):
            add=str(Geocoder.reverse_geocode(kdeCenterGeo[i,1],kdeCenterGeo[i,0]))
            address.append(add[:-15])
    return address


# In[1265]:
    

def getTimeId(tbegin,tend):
    if(tbegin>=0):
        ibegin=tbegin
    else:
        ibegin=tbegin+24
    if(tend<=23):
        iend=tend
    else:
        iend=tend-24
    return ibegin,iend
#-- compute the conditional risk probability of this region
def crimeRiskVal(lonC,latC,hr,dhr,rad=RADIUS): #rad in meters
    conn,cur=connect_to_db(DBNM,USERNM)
    #AREAChicago
    area=(2*rad)**2 # area of the nearby region, in meter square
    #ratioArea=area/AREAChicago
    #DAYtotal
     
    #-- based on the given radius (half-length of the box), find the boundary of our region of interest  
    latMin=latC-rad*meter2deg
    latMax=latC+rad*meter2deg
    lonMin=lonC-rad*meter2deg/np.cos(latC*deg2rad)
    lonMax=lonC+rad*meter2deg/np.cos(latC*deg2rad)
    latRange=np.array([latMin,latMax])
    lonRange=np.array([lonMin,lonMax])
    print "range of lon and lat",latRange,lonRange
    #--- compute the HOURtotal, DAYtotal
    sql="SELECT MAX(difhour) FROM %s"%(TABLENM)
    dfmax=pd.read_sql(sql,conn)
    sql="SELECT MIN(difhour) FROM %s"%(TABLENM)
    dfmin=pd.read_sql(sql,conn)
    HOURtotal=dfmax['max'].values[0]-dfmin['min'].values[0]
    DAYtotal=HOURtotal/24.
    
    #--- HEY, the across 24 hour situation is not considered here, e.g., 11pm-1am
    sql="SELECT * FROM %s"%(TABLENM)
    dfAll=pd.read_sql(sql,conn)
    #-average crime rate across Chicago throughout the entire day, count/area/hour
    avgRateAll=len(dfAll)/AREAChicago/HOURtotal 
    #
    if(dhr>=24):
        sqltime=' hour>0 '
    else:
        tbegin=hr
        tend=hr+dhr
        ibegin,iend=getTimeId(tbegin,tend)
        if(ibegin<iend):
            sqltime=' hour>=%f AND hour<%f '%(ibegin,iend)
        else:
            #sqltime=' (hour>=%f AND hour<=23) OR (hour>=0 AND hour<%f) '%(ibegin,iend)
            sqltime=' hour>=%f OR hour<%f '%(ibegin,iend)
    sql='SELECT COUNT(hour) FROM %s WHERE %s'%(TABLENM,sqltime)
    dfHr=pd.read_sql(sql,conn)
    #-average rate of crime at this time period, across the entire city,count/area/hour
    avgRateHr=dfHr['count']/AREAChicago/DAYtotal/dhr
    #
    sqlloc='"Longitude">=%f AND "Longitude"<=%f AND "Latitude">=%f AND "Latitude"<=%f'%(lonMin,lonMax,latMin,latMax)
    sql='SELECT * FROM %s WHERE (%s) AND (%s)'%(TABLENM,sqltime,sqlloc)
    #sql='%s AND "Longitude">=%f AND "Longitude"<=%f AND "Latitude">=%f AND "Latitude"<=%f'%(sql,lonMin,lonMax,latMin,latMax)
    dfHrLoc=pd.read_sql(sql,conn)
    print "length of local crime=",len(dfHrLoc)

    flagHasCrime=1
    if(len(dfHrLoc)==0): # no historical crime at nearby location in this time range
        print "no historical crime at nearby location in this time range"
        flagHasCrime=0
        
    avgRateHrLoc=len(dfHrLoc)/area/DAYtotal/dhr

    #--the risk compared to the city's (lateral) average around this time
    RSHr=np.around(avgRateHrLoc/avgRateHr,2)
    riskScoreHr=50+50*(avgRateHrLoc-avgRateHr)/avgRateHr
    #--the risk compared to the city's (lateral and time) average
    RSAll=np.around(avgRateHrLoc/avgRateAll,2)
    riskScoreAll=50+50*(avgRateHrLoc-avgRateAll)/avgRateAll

    #--compute the risk score for dhr(2?) hours before and 2hour after the selected parking hour + duration
    deltahr=2
    if(dhr+deltahr>=24):
        sqltime=' hour>0 '
    else:
        tbegin=hr-deltahr
        tend=hr+dhr+deltahr
        ibegin,iend=getTimeId(tbegin,tend)
        if(ibegin<iend):
            sqltime=' hour>=%f AND hour<%f '%(ibegin,iend)
        else:
            #sqltime=' (hour>=%f AND hour<=23) OR (hour>=0 AND hour<%f) '%(ibegin,iend)
            sqltime=' hour>=%f OR hour<%f '%(ibegin,iend)
    sql='SELECT hour,COUNT(hour) FROM %s WHERE (%s) AND (%s) GROUP BY hour ORDER BY hour'%(TABLENM,sqltime,sqlloc)
    dfHrLocdelta=pd.read_sql(sql,conn)
    rateLst=dfHrLocdelta['count'].values/area/DAYtotal/1.
    riskScoreDeltaLst=50+50*(rateLst-avgRateAll)/avgRateAll
    riskScoreDeltaLst=np.vstack([dfHrLocdelta['hour'].values,riskScoreDeltaLst])
    print "length of delta local crime=",len(dfHrLocdelta)
    print "risk score:\n  the risk compared to the city's (lateral) average around this time=%f"%(RSHr)
    print "the risk compared to the city's (lateral and time) average=%f"%(RSAll)
    return conn,dfHrLocdelta,dfHrLoc,latRange,lonRange,riskScoreHr,riskScoreAll,riskScoreDeltaLst #ratioHr,ratioAll,




# In[1268]:
def seg2list(seg): # google API only takes list (?) need to change the data before pass it to output.html, the input segment is a combination of list and array: list[array1,array2]
    latlst2=[] #2d list
    lonlst2=[]
    for i in range(len(seg)):
        lonlst=list(seg[i][:,0])
        latlst=list(seg[i][:,1])
        lonlst2.append(lonlst)
        latlst2.append(latlst)
    return lonlst2,latlst2

#def analyzeLocalCrime(df,latRange,lonRange,step=100,bandwidth=3e-6,flag=0):
def crimeKdeCorr(df,step=100,bandwidth=BANDWIDTH,rad=RADIUS,flagCptCorr=0):
    #step in meters, unit of bandwidth depends on the unit of the lon/lat used in KDE
    #what's the unit of bandwidth? if i input lon/lat as radius, then bandwidth is also in radius
    #1 rad=1*180/pi deg=180/pi*111045 meter = 6.36E7 meters
    #3e-6 rad=19 meters
    #input the dataframe that contains only the local crimes
    nCrime=len(df)
    latmin=min(df['Latitude'].values)
    latmax=max(df['Latitude'].values)
    lonmin=min(df['Longitude'].values)
    lonmax=max(df['Longitude'].values)   
    latRange=[latmin,latmax]
    lonRange=[lonmin,lonmax]
    latCent=(latRange[1]-latRange[0])/2+latRange[0]
     
    #print "lat and lon range=",latRange,lonRange    
    #-- generate the grid points for the given lon,lat(deg) range 
    nLatStep=np.floor((latRange[1]-latRange[0])/(step*meter2deg))+1
    nLonStep=np.floor((lonRange[1]-lonRange[0])/(step*meter2deg/np.cos(latCent*deg2rad)))+1    
    latLst=np.linspace(latRange[0],latRange[1],num=nLatStep)
    lonLst=np.linspace(lonRange[0],lonRange[1],num=nLonStep)


    lonGrid,latGrid=np.meshgrid(lonLst,latLst)
    positions=np.vstack([lonGrid.ravel(),latGrid.ravel()])
    values=np.vstack([df['Longitude'].values,df['Latitude'].values])
    kernel=sp.stats.gaussian_kde(values)
    Z=np.reshape(kernel(positions).T,lonGrid.shape)
    #Zthresh=np.copy(Z)
    #locLst=Z>Z.max()*0.8
    locLst=Z.index(Z.max())
    locPeak=np.vstack([lonGrid[locLst],latGrid[locLst]])
    
    #---get conour
    #---create a wider mesh for getting contour
    Nwiden=20; #must be a even number
    widen=step*Nwiden*meter2deg;
    latLstwiden=np.linspace(latRange[0]-widen*0.5,latRange[1]+widen*0.5,num=nLatStep+Nwiden)
    lonLstwiden=np.linspace(lonRange[0]-widen*0.5,lonRange[1]+widen*0.5,num=nLonStep+Nwiden)
    lonGridwiden,latGridwiden=np.meshgrid(lonLstwiden,latLstwiden)
    positionswiden=np.vstack([lonGridwiden.ravel(),latGridwiden.ravel()])
    Zwiden=np.reshape(kernel(positionswiden).T,lonGridwiden.shape)
    #
    X=lonGridwiden;Y=latGridwiden;
    c=cntr.Cntr(X,Y,Zwiden)
    
    seglonLst=[];seglatLst=[];segcolorLst=[]
    ratioLst=[0.2,0.4,0.6,0.8]
    #colorLst=["#00aaff","#ffff00","#ff8000","#ff3300"] # blue,yellow,red,black
    #colorLst=["#66ffff","#66b2ff","#0066ff","#002699"] # all blue
    colorLst=["#66b2ff","#004ce6","#ff6666","#cc0000"]
    for ir in range(len(ratioLst)):
        ratio=ratioLst[ir]
        res=c.trace(Z.max()*ratio)
        nseg=len(res)//2
        segments,codes=res[:nseg],res[nseg:]
        print "number of segments %f=%d"%(ratio,nseg)
        seglon,seglat=seg2list(segments)
        for i in range(nseg):
            seglonLst.append(seglon[i])
            seglatLst.append(seglat[i])
            segcolorLst.append(colorLst[ir])
    print "####test, Peak ceters, in lon,lat",locPeak
    #kdeCenterGeo=locPeak.T
    #print "###### shape",kdeCenterGeo.shape
    #kdeCenterVal=Z[locLst]
    flagRandom=0
    if (flagCptCorr>0):
        #---- compute the two-point angular cross-correlation, has nothing to do with KDE
        #bins = np.linspace(0.000,0.01,num=21) #spacing in degrees, 0.01 deg ~ 1.11 km
        #bins=np.linspace(0.0,rad*meter2deg,num=21)
        bins=np.linspace(0.0,rad*meter2deg,num=10)
        binCenter = 0.5 * (bins[1:] + bins[:-1]) #--the center of the num=20 bins

        corResult = bootstrap_two_point_angular(df['Longitude'],df['Latitude'],bins=bins,method='landy-szalay',Nbootstraps=5)
        #(corr, corr_err,corr_boot) = results


        #-- do a linfit to the correlation curve, x is the bins, y is the corr, weight is the corr_error
        #print "cor results",corResult
        X=sm.add_constant(binCenter)
        linModel = sm.WLS(corResult[0], X, weights=1./corResult[1]**2)
        linFit = linModel.fit()
        #print "binCenter",binCenter
        #print "X",X
        #print "linfit result.params",linFit.params
        #print "linffit result.bse",linFit.bse
        
        if (FLAGPLOT==1):
            #---compute the two-point angular cross-correlation with randome points
            lonRandm=lonRange[0]+np.random.rand(nCrime)*(lonRange[1]-lonRange[0])
            latRandm=latRange[0]+np.random.rand(nCrime)*(latRange[1]-latRange[0])
            corResultRandm=bootstrap_two_point_angular(lonRandm,latRandm,bins=bins,method='landy-szalay',Nbootstraps=5)
            linModelRandm=sm.WLS(corResultRandm[0],X,weights=1./corResultRandm[1]**2)
            linFitRandm=linModelRandm.fit()
        #
            plt.figure(2)
            #--plot the linfit result, and the cor result for both local crimes and random points
            plt.errorbar(binCenter*deg2meter,corResult[0],yerr=corResult[1],fmt='bo') # x-distance in m, y-correlation
            plt.plot(binCenter*deg2meter,binCenter*linFit.params[1]+linFit.params[0],'b-')
            plt.errorbar(binCenter*deg2meter,corResultRandm[0],yerr=corResultRandm[1],fmt='ro')
            plt.plot(binCenter*deg2meter,binCenter*linFitRandm.params[1]+linFitRandm.params[0],'r-')
            plt.xlabel('distance (m)',fontsize=20)
            plt.ylabel('correlation',fontsize=20)
            #plt.show()
            #plt.close()
            plt.savefig("pic/two_point_correlation.png")
            plt.close(2)

        #--check the slope, take the error into consideration
        flagRandom=0
        slopeMax=linFit.params[1]+linFit.bse[1]
        slopeMin=linFit.params[1]-linFit.bse[1]
        slopeMid=linFit.params[1]
        if(slopeMid>0):
            print "####Warning, positive or flat slope! %f+%f=%f"%(linFit.params[1],linFit.bse[1],slopeMax)
            flagRandom=1 # random crimes
        # -25.034908+29.109945=4.075037
        else:
         print   "### negative slope, has structure? %f+%f=%f"%(linFit.params[1],linFit.bse[1],slopeMax)
    return  flagRandom,seglonLst,seglatLst,segcolorLst

#"""    
# In[1270]:
if __name__ == "__main__":
	#addressIn="union park, Chicago, IL " # has some structure
	#addressIn="Ogden park, Chicago, IL" # has little structure - random
	addressIn="University of Chicago, Chicago, IL" # oh, lots of structure, for both 20m and 50m radius/bandwidth
	lonC,latC,RSHr,RSAll,dfHrLocdelta,dfHrLoc,flagNoCrime,flagRandom,seglonLst,seglatLst,segcolorLst,imgnmRS,hrSug,RSSug,dfS,seglonLstS,segLatLstS,segcolorLstS=getCrimeMap(addressIn,9,4)
	print imgnmRS
#print gmapNm
        #print seg2lon
#print address,ratioHr,ratioAll,len(dfHrLoc),flagNoCrime,flagRandom
# In[1271]:
# In[ ]:
#"""



# In[ ]:



