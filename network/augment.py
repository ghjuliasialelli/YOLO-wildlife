from typing import Iterable, Tuple


# entirely for collaboration purposes
Bbox = namedtuple('Bbox', 'i1 j1 i2 j2')
Img  =   NewType('Img', np.array)
Imgs =   NewType('Imgs', np.array)    # shape = (bs, w, h, 3)
Labels = NewType('Labels', np.array)  # shape = (bs, Si, Sj, 3)


def datagen(imgs: Dict[str, Img], labels: Dict[str, List[Bbox]], Si: int, Sj: int) ->\
	Iterable[Tuple[Imgs, Labels]]:
	while True:
		# make imgs transformations
		# :)

		# make label transformations
		# :)
		yield imgs, labels
