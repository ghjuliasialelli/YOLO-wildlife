from typing import Iterable, Tuple


# entirely for collaboration purposes
Bbox = namedtuple('Bbox', 'i1 j1 i2 j2')
Img  =   NewType('Img', np.array)

# the final form of our training data
# Imgs:   4D tensor, a batch of images of fixed size
Imgs =   NewType('Imgs', np.array)    # shape = (bs, w, h, 3)

# Labels: 4D tensor, a batch of labels if fixed size
#
# - labels[l][i][j][0, 1] \in (0, 1) represent the width & height of the
#   (potential) bounding box of center (i, j), in the image's system
#   of coordinates
#
# - labels[l][i][j][2] \in (0, 1) represents the probability that the
#   bounding box (above) is indeed valid
#
# during training, we fix the dimensions of the labels to equal the
# dimensions of the output of the network for fixed size images
Labels = NewType('Labels', np.array)  # shape = (bs, Si, Sj, 3)


def datagen(imgs: Dict[str, Img],
	labels: Dict[str, List[Bbox]], Si: int, Sj: int) ->\
	Iterable[Tuple[Imgs, Labels]]:
	'''@brief generator for infinite data augmentation

	@param imgs the complete image dataset
	@param labels the complete labels dataset
	@param Si number of rows of label tensor
	@param Sj number of cols of label tensor

	@returns a generator for infinite data augmentation
	'''

	while True:
		# make imgs transformations
		# :)

		# make label transformations
		# :)
		yield imgs, labels
