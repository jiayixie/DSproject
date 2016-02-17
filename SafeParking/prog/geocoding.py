import json
import urllib

#get JSON data of address
def googleGeocoding(address):
	"""This function takes an address and returns the latitude and longitude from the Google geocoding API."""
	baseURL = 'http://maps.googleapis.com/maps/api/geocode/json?'
	#geocodeURL = baseURL + 'address=' + address + '&sensor=false'
	#baseURL = 'https://maps.googleapis.com/maps/api/geocode/json?'
	#geocodeURL = baseURL + 'address=' + address + '&key=AIzaSyA1nlPhrmgRKrqQcinoX9JqyEOydiRWoHI'
	geocodeURL = baseURL + 'address=' + address + '&components=administrative_area:IL|country:US'
	#geocodeURL = baseURL + 'address=' + address + '&components=locality:Chicago|administrative_area:IL|country:US'
	print 'geocode URL=  ',geocodeURL
	geocode = json.loads(urllib.urlopen(geocodeURL).read())
	#return geocode
	latlong = geocode['results'][0]['geometry']['location']
	return latlong

import googlemaps as gm
def googleGeocoding2(addr):
	gmaps=gm.Client(key="AIzaSyA1nlPhrmgRKrqQcinoX9JqyEOydiRWoHI")

	geocode_result=gmaps.geocode(address=addr,components={'administrative_area':'IL','country':'US'})
	latlong=geocode_result[0]['geometry']['location']
	return latlong
#inlat = geocode_result[0]['geometry']['location']['lat']
#inlon = geocode_result[0]['geometry']['location']['lng']
#print inlat,inlon
"""
#Extract Lat/Long from JSON
def getGeocodeLatLong(geocodeJSON):
	latlong = geocodeJSON['results'][0]['geometry']['location']
	#print "geocoding lat lon=",latlong
	return latlong
"""
#http://maps.google.com/maps/api/geocode/json?address=1600+Amphitheatre+Parkway,+Mountain+View,+CA&sensor=false
#https://github.com/jainsley/childHood/blob/master/website/googleAPIFunctions.py
