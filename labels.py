import os
import pygeoj
from PIL import Image
import numpy as np
import pickle 
from collections import namedtuple

Bbox = namedtuple('Bbox', 'x y w h c')

S = 7
B = 2
C = 1

size = (4000,3000)

IMAGES = 'images/'
META = 'meta/'


def get_labels():
	try:
		with open('labels.txt','rb') as file:
			label = pickle.load(file)
	except:
		label = {}
		#image_names = os.listdir(IMAGES)
		for file_name in os.listdir(META):
			if file_name.endswith('.geojson'):
				file = pygeoj.load(META+file_name)
				for feature in file:
					x,y,xt,yt = feature.geometry.bbox
					w,h = (xt-x,yt-y)
					c = 1
					bbox = Bbox(x/size[0],y/size[1],w/size[0],h/size[0],c)
					if(feature.properties['IMAGEUUID'] in label):
						label[feature.properties['IMAGEUUID']].append(l)
					else:
						label[feature.properties['IMAGEUUID']] = [l]
		with open('labels.txt','wb') as file:
			pickle.dump(label,file)
	return label

if __name__ == '__main__':
	label = get_labels()
	print(label)
