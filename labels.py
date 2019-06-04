import os
import pygeoj
import numpy as np
import json 
from collections import namedtuple
from jpegtran import JPEGImage


Bbox = namedtuple('Bbox', 'x y w h c')

IMAGES = 'images/'
META = 'meta/'

LABELS = 'labels.json'

def get_labels():
	if os.path.exists(LABELS):
		with open(LABELS) as file:
			label = json.load(file)
		return label
	label = {}
	#image_names = os.listdir(IMAGES)
	for file_name in os.listdir(META):
		if file_name.endswith('.geojson'):
			file = pygeoj.load(META+file_name)
			for feature in file:
				print(feature.properties['TAGUUID'])
				img = JPEGImage(IMAGES+feature.properties['IMAGEUUID']+'.JPG')
				x1,y1,x2,y2 = feature.geometry.bbox
				w,h = (x2-x1,y2-y1)
				x,y = ((x1+x2)/2,(y1+y2)/2)
				c = 1
				bbox = Bbox(x/img.width,y/img.height,w/img.width,h/img.height,c)
				if(feature.properties['IMAGEUUID'] in label):
					label[feature.properties['IMAGEUUID']].append(bbox)
				else:
					label[feature.properties['IMAGEUUID']] = [bbox]
	
	with open(LABELS) as file:
		json.dump(label,file)
	return label

if __name__ == '__main__':
	label = get_labels()
	print(label)
