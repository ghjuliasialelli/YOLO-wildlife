import os
import pygeoj
from PIL import Image
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

IMAGES = 'images/'
META = 'meta/'

image_names = os.listdir(IMAGES)

for file_name in os.listdir(META):
	if file_name.endswith('.geojson'):
		file = pygeoj.load(META+file_name)
		for feature in file:
			print(feature)
			image = Image.open(IMAGES+feature.properties['IMAGEUUID']+'.JPG')
			image.show()
			break
		



#Feature(geometry=Geometry(type=Polygon, coordinates=[[[1197.0, 568.0], [1186.0, 568.0], [1179.0, 582.0], [1190.0, 596.0], [1212.0, 597.0], [1229.0, 591.0], [1230.0, 585.0], [1226.0, 579.0], [1224.0, 574.0], [1208.0, 569.0], [1197.0, 568.0]]], bbox=(1179.0, 568.0, 1230.0, 597.0)), properties={'IMAGEUUID': 'f77f4af5a1344b9086b307d2b4ba61ff', 'TAGUUID': 'a9b3a2325dbe4a208bc3ae37eeb8e1e1'})
