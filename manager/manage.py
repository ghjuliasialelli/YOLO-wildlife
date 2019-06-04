from urllib.request import urlopen
import zipfile, os, shutil

import click


SOURCE = 'https://zenodo.org/record/1204408/files/savmap_dataset_v2.zip?download=1' 
IMAGES, META, DATA = 'images', 'meta', 'data.zip'
DOWNSCALED = 'downscaledby{}'
LABELS = 'labels.json'


@click.group()
def cli():
	...

@cli.command()
def dldata():
	# already downloaded
	if os.path.exists(DATA):
		print('already downloaded')
		return

	# stream download to not break Seb's PC
	response = urlopen(SOURCE)
	with open(DATA, 'wb') as a:
		for chunk in iter(lambda: response.read(1 << 25), b''):
			a.write(chunk)

	# extract zip to nice folders
	os.mkdir(IMAGES)
	os.mkdir(META)
	with zipfile.ZipFile(DATA) as z:
		for name in z.namelist():
			# extract images & meta to respective directories
			_dir = IMAGES if name.endswith('JPG') else META
			z.extract(name, path=_dir)

@cli.command()
def clean():
	# delete EVERYTHING
	# break me baby
	shutil.rmtree(IMAGES)
	shutil.rmtree(META)
	os.remove(DATA)

def _images():
	for p, ds, fs in os.walk(IMAGES):
		for fname in fs:
			if not fname.endswith('JPG'):
				continue
			yield os.path.join(p, fname)

@cli.command()
@click.argument('by')
def downscale(by):
	from jpegtran import JPEGImage
	by = int(by)

	if os.path.exists(DOWNSCALED.format(by)):
		print('nothing to do')
		return
	os.mkdir(DOWNSCALED.format(by))

	for imgp in _images():
		img = JPEGImage(imgp)
		new = img.downscale(img.width // by, img.height // by)
		new.save(os.path.join(
			DOWNSCALED.format(by),
			os.path.basename(imgp)))
		print(f'downscaled {os.path.basename(imgp)}')

@cli.command()
def genlabels():
	from collections import namedtuple
	import json

	from jpegtran import JPEGImage
	import pygeoj

	Bbox = namedtuple('Bbox', 'x y w h c')

	if os.path.exists(LABELS):
		with open(LABELS) as file:
			label = json.load(file)
		return label

	label = {}
	for file_name in os.listdir(META):
		if file_name.endswith('.geojson'):
			file = pygeoj.load(os.path.join(META, file_name))
			for feature in file:
				print(feature.properties['TAGUUID'])
				img = JPEGImage(os.path.join(IMAGES, feature.properties['IMAGEUUID']+'.JPG'))
				x1,y1,x2,y2 = feature.geometry.bbox
				w,h = (x2-x1,y2-y1)
				x,y = ((x1+x2)/2,(y1+y2)/2)
				c = 1
				bbox = Bbox(x/img.width,y/img.height,w/img.width,h/img.height,c)
				if(feature.properties['IMAGEUUID'] in label):
					label[feature.properties['IMAGEUUID']].append(bbox)
				else:
					label[feature.properties['IMAGEUUID']] = [bbox]
	
	with open(LABELS, 'w') as file:
		json.dump(label, file)
	return label
