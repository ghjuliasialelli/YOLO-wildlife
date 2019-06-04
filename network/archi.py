import torch.nn as nn
import torch.optim as optim
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

def train(yolo: YOLO, data: Iterable[Tuple[Imgs, Labels]], lr: float, steps: int):
	'''given an initialized yolo object, train
	'''
	rloss = 0

	lossf = nn.MSELoss()
	opt = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9)
	for i, (imgs, lbls) in enumerate(data):
		if i >= steps:
			break

		# optimization step: compute loss and
		# backpropagate
		opt.zero_grad()
		pred = yolo(imgs)
		loss = lossf(pred, lbls)
		loss.backward()
		opt.step()

		# expectation over loss, with recent loss more
		# important
		rloss = .9 * rloss + .1 * loss.item()
