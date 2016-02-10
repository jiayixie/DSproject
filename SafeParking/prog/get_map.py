import urllib
import cStringIO
#import Image
from PIL import Image 
import numpy as np

#import folium #leaflet to python - http://folium.readthedocs.org/en/latest/

#based on http://hci574.blogspot.com/2010/04/using-google-maps-static-images.html

def get_static_google_map(clat, clon, filename_wo_extension='test-gg', 
	zoom=15, imgsize=(640,640), imgformat="png", maptype="roadmap", markers=None, scale=1,center='Near North Side, Chicago, IL',force_address='no' ):  
	#zoom=14, imgsize=(250,250), imgformat="png", maptype="roadmap", markers=None, scale=1,center='Near North Side, Chicago, IL',force_address='no' ):  

#"""retrieve a map (image) from the static google maps server 
#See: http://code.google.com/apis/maps/documentation/staticmaps/
#Creates a request string with a URL like this:
#http://maps.google.com/maps/api/staticmap?center=Brooklyn+Bridge,New+York,NY&zoom=14&size=512x512&maptype=roadmap
#&markers=color:blue|label:S|40.702147,-74.015794&sensor=false"""
	
	if force_address == 'no':
		center = str(clat)+', '+str(clon)
	else: 
		center = center

	request =  "http://maps.google.com/maps/api/staticmap?" # base URL, append query params, separated by &
	if center != None:
		request += "center=%s&" % center #address or lat/lon

	if center != None: #zoom = 15 is about right for my scale # zoom 0 (all of the world scale ) to 22 (single buildings scale). 
		request += "zoom=%i&" % zoom #this may change

	request += "size=%ix%i&" % (imgsize) # tuple of ints, up to 640 by 640 #size=(640,640) always
	request += "format=%s&" % imgformat # roadmap, satellite, hybrid, terrain
	request += "maptype=%s&" % maptype
	request += "scale=%s&" % scale   #640x640 = 5 in #scale=1 always

	if markers == None:
		request += "markers=" + str(clat)+','+str(clon) +'&'
	else: # add markers (location and style)
		for marker in markers:
			request += "%s&" % marker

	#&markers=color:blue%7Clabel:S%7C62.107733,-145.541936
	#&markers=|label:1|11%20Allstate%20Rd%2C%20Boston%2C%20MA

	#request += "mobile=false&"  # optional: mobile=true will assume the image is shown on a small screen (mobile device)
	request += "sensor=false&"   # must be given, deals with getting loction from mobile device

	#print request

	#calculate scale of map - http://wiki.openstreetmap.org/wiki/Zoom_levels
	ecirc = 6378140.0*2*np.pi #m, equatorial circumference of Earth
	scl = ecirc * np.cos(clat*np.pi/180.)/2**(zoom+8) #m/pix

	#path = '/Users/Sara/Dropbox/cfasgettel/insight/setup/app/static/' # Option 1: save image directly to disk
	#urllib.urlretrieve(request, path+filename_wo_extension+"."+imgformat) 

	#USE THIS INSTEAD
	# Option 2: read into PIL - works on the command line & returns a PNG
	web_sock = urllib.urlopen(request)
	imgdata = cStringIO.StringIO(web_sock.read()) # constructs a StringIO holding the image
	gg_img = Image.open(imgdata)

	return scl, request, gg_img

