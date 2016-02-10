import json
import urllib

#get JSON data of address
def googleGeocoding(address):
	"""This function takes an address and returns the latitude and longitude from the Google geocoding API."""
	baseURL = 'http://maps.googleapis.com/maps/api/geocode/json?'
	geocodeURL = baseURL + 'address=' + address + '&components=administrative_area:IL|country:US'
	#print 'geocode URL=  ',geocodeURL
	geocode = json.loads(urllib.urlopen(geocodeURL).read())
	return geocode

#Extract Lat/Long from JSON
def getGeocodeLatLong(geocodeJSON):
	"""This function takes the json output of a googleGeocoding function call and
parses it to output the latitude and longitude"""
	latlong = geocodeJSON['results'][0]['geometry']['location']
	return latlong

#https://github.com/jainsley/childHood/blob/master/website/googleAPIFunctions.py
