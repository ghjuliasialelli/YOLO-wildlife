import os
import pygeoj
from PIL import Image
import numpy as np

S = 7
B = 2
C = 1

size = (4000,3000)

IMAGES = 'images/'
META = 'meta/'

label = {}

image_names = os.listdir(IMAGES)

for file_name in os.listdir(META):
	if file_name.endswith('.geojson'):
		file = pygeoj.load(META+file_name)
		print(file)
		for feature in file:
			bbox = feature.geometry.bbox
			x,y,xt,yt = bbox
			w,h = (xt-x,yt-y)
			c = 1
			l = [x/size[0],y/size[1],w/size[0],h/size[0],c]
			if(feature.properties['IMAGEUUID'] in label):
				label[feature.properties['IMAGEUUID']].append(l)
			else:
				label[feature.properties['IMAGEUUID']] = [l]
print(label)
