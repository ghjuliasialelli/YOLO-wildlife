from urllib.request import urlopen
import zipfile, os, shutil

import click


SOURCE = 'https://zenodo.org/record/1204408/files/savmap_dataset_v2.zip?download=1' 
IMAGES, META, DATA = 'images', 'meta', 'data.zip'

@click.group()
def cli():
	...

@cli.command()
def dldata():
	# already downloaded
	if os.path.exists(DATA):
		print('already downloaded')
		return

	response = urlopen(SOURCE)
	bindata = response.read()
	
	with open(DATA, 'wb') as a:
		a.write(bindata)
	del bindata  # mem

	os.mkdir(IMAGES)
	os.mkdir(META)
	with zipfile.ZipFile(DATA) as z:
		for name in z.namelist():
			# extract images & meta to respective directories
			_dir = IMAGES if name.endswith('JPG') else META
			z.extract(name, path=_dir)

@cli.command()
def clean():
	shutil.rmtree(IMAGES)
	shutil.rmtree(META)
	os.remove(DATA)
