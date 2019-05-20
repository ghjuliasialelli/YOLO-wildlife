# fans-de-loan
&lt;3 lolo, Seb, Ghju : la meilleure team 
# riverdale


## lil pipenv tut

To create a pipenv environment:
```sh
pipenv --python 3.7
pipenv shell
pipenv install -e .
```

To download and extract the dataset:
```
manage dldata
```

To remove all data:
```
manage clean
```

#Dataset : SAVMAP Dataset

The dataset proposed here was acquired in May 2014 and contains:

-Raw aerial images (non-rectified) in JPEG format. Additional metadata about individual images (timestamp, latitude, longitude, altitude, etc) can be extracted from the EXIF. Each image is named with a Universally Unique Identifier.

-Polygons indicating the locations of animals tagged during the Micromappers crowdsourcing campaign (please be aware that the polygons contain many false positives and should not be directly used as a "ground truth"). The coordinates of the polygons use the image reference system (i.e. column and row number). Each polygon has a Universally Unique Identifier (TAGUUID) and an associated image (IMAGEUUID). Information about the animal species in each polygon is currently not available. The polygons are provided in the ESRI shapefile and GEOJSON formats.
