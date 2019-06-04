import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from typing import Iterable, Tuple
from augment import datagen, Imgs, Labels

nlayers = 3

ndepth = 3

class YOLO(nn.Module):
	def __init__(self):
		'''initialize convolution layers & set grid size
		parameter
		'''        
		super(YOLO, self).__init__()
		self.convs = []
		for i in range(ndepth):
			x = 0
			for j in range(nlayers):
				insize = 3 if i+j == 0 else 64
				outsize = 3 if i+j == ndepth+nlayers else 64
				self.convs.append(nn.Conv2d(
					insize, outsize, 3, padding=1+x,dilation=1+x))
				x = 3*x+2
			if i != ndepth:
				self.convs.append(nn.Conv2d(64,64,3,stride=3))
	
	def forward(self,x):
		'''compute output of (batch size) * S * S * 3
		'''
		for i,conv in enumerate(self.convs):
			# always use relu except last layer use sigmoid
			f = F.sigmoid if i == len(self.convs) else F.relu
			x = f(conv(x)
		return x

def train(yolo: YOLO, data: Iterable[Tuple[Imgs, Labels]], lr: float, epochs: int):
	'''given an initialized yolo object, train
	'''
	...



def datagen():
	while True:
		yield imgs, labels
