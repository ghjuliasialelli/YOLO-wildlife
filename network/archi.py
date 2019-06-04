import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from typing import Iterable, Tuple
from augment import datagen, Imgs, Labels


class YOLO:
	def __init__(self):
		'''initialize convolution layers & set grid size
		parameters
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



def datagen():
	while True:
		yield imgs, labels
