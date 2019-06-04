from setuptools import setup, find_packages


setup(
	name='savanna', version='0.1',
	packages=find_packages(),
	install_requires=[
		# ml part
		'click', 'Pillow', 'scikit-image',
		'numpy', 'matplotlib', 'scipy',
		'torch', 'torchvision',

		# preprocessing
		'cffi', 'jpegtran',
	],

	entry_points={
		'console_scripts': ['manage = manager:cli'],
	}
)
