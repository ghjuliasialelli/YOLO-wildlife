import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from typing import Iterable, Tuple


# entirely for collaboration purposes
Bbox = namedtuple('Bbox', 'x y w h')
Imgs =   NewType('Img', np.array)     # shape = (bs, w, h, 3)
Labels = NewType('Labels', np.array)  # shape = (bs, S, S, 3)


class YOLO:
	def __init__(self):
		'''initialize convolution layers & set grid size
		parameter
		'''
		...

	def forward(self):
		'''compute output of (batch size) * S * S * 3
		'''
		...

def train(yolo: YOLO, data: Iterable[Tuple[Imgs, Labels]], lr: float, epochs: int):
	'''given an initialized yolo object, train
	'''
	...
